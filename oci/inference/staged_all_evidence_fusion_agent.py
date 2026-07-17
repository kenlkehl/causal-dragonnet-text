"""Staged proposal and selection for one all-evidence fusion request.

The wrapper in this module is deliberately model-agnostic.  It accepts an
injected agent exposing ``propose(context)`` and orchestrates several calls
through the existing all-evidence request and response validators. It neither
starts nor imports a language model.

For a proposal-mode request, the wrapper obtains a broad proposal inventory,
two role-specific all-family inventories, and then asks the same injected agent
to select from a conservatively consolidated immutable contract pool.
Only exact duplicates and same-name variants with identical non-role semantics
can combine support; incompatible variants compete without contaminating the
winner. Role-specific requests use fresh sequential evidence IDs, so citations
are mapped back to the original request before they are retained. The returned
value has the same proposal shape expected by the ordinary all-evidence fusion
runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .all_evidence_fusion import (
    ALL_SOURCE_FAMILIES,
    FUSION_PROMPT_VERSION,
    AllEvidenceFusionRequest,
    CandidateContract,
    FusionResult,
    _fusion_request_from_context,
    source_text_temporal_policy_audit,
    validate_all_evidence_fusion_response,
)
from .minimal_staged_selection_postprocessor import (
    MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION,
    postprocess_minimal_staged_selection,
)
from .safe_staged_proposal_union import (
    SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION,
    SafeStagedProposalUnionResult,
    safe_staged_proposal_union,
    safe_staged_proposal_union_identity,
)

STAGED_FUSION_AUDIT_SCHEMA_VERSION = "staged_all_evidence_fusion_audit_v3"
STAGED_SELECTION_BACKFILL_VERSION = MINIMAL_STAGED_SELECTION_POSTPROCESSOR_VERSION
STAGED_SAME_NAME_MERGE_VERSION = SAFE_STAGED_PROPOSAL_UNION_POLICY_VERSION
STAGED_SELECTION_UNION_POSTPROCESSING_VERSION = "safe_union_plus_recall_safe_selection_v4"

_CONFOUNDER_ROLE_FAMILIES = frozenset(ALL_SOURCE_FAMILIES)
_MODIFIER_ROLE_FAMILIES = frozenset(ALL_SOURCE_FAMILIES)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("staged fusion values must be finite and JSON serializable") from exc


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _detached_json_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    detached = json.loads(_canonical_json(value))
    if not isinstance(detached, dict):  # pragma: no cover - guarded by Mapping check
        raise TypeError(f"{label} must be one JSON object")
    return detached


def _reasoning_presence_summary(proposal_agent: Any) -> dict[str, Any]:
    """Summarize separated reasoning fields without retaining their contents."""

    trace = getattr(proposal_agent, "last_response_trace", None)
    if not isinstance(trace, Mapping):
        return {
            "response_trace_available": False,
            "completion_attempt_count": 0,
            "reasoning_content_present_count": 0,
            "reasoning_present_count": 0,
            "any_reasoning_present": False,
        }

    raw_attempts = trace.get("repair_attempts")
    if isinstance(raw_attempts, list) and raw_attempts:
        attempts = [attempt for attempt in raw_attempts if isinstance(attempt, Mapping)]
    else:
        attempts = [trace]
    reasoning_content_count = sum(
        1 for attempt in attempts if bool(attempt.get("reasoning_content"))
    )
    reasoning_count = sum(1 for attempt in attempts if bool(attempt.get("reasoning")))
    return {
        "response_trace_available": True,
        "completion_attempt_count": len(attempts),
        "reasoning_content_present_count": reasoning_content_count,
        "reasoning_present_count": reasoning_count,
        "any_reasoning_present": bool(reasoning_content_count or reasoning_count),
    }


@dataclass(frozen=True)
class _ValidatedProposal:
    spec_json: str
    supporting_evidence_ids: tuple[str, ...]
    supporting_source_families: tuple[str, ...]

    @property
    def spec(self) -> dict[str, Any]:
        return json.loads(self.spec_json)


@dataclass
class _ProposalUnionEntry:
    spec_json: str
    supporting_evidence_ids: list[str]
    supporting_source_families: list[str]
    validated_occurrence_count: int

    @property
    def spec(self) -> dict[str, Any]:
        return json.loads(self.spec_json)


class StagedAllEvidenceFusionAgent:
    """Wrap an injected proposal agent with validated staged fusion.

    Parameters
    ----------
    proposal_agent:
        An injected object with a callable ``propose(context)`` method.  The
        object is used for both proposal and final selection requests.
    final_max_candidates:
        Final selection cap.  The effective cap is also bounded by the cap on
        the original proposal request so the mapped response necessarily fits
        the caller's contract.

    Successful calls append a detached audit record to ``stage_audits``.  The
    record contains request/response hashes, opaque evidence/candidate IDs,
    and counts only; evidence content, extraction contracts, rationales, and
    patient text are never retained in the audit.
    """

    def __init__(self, proposal_agent: Any, *, final_max_candidates: int = 16) -> None:
        propose = getattr(proposal_agent, "propose", None)
        if not callable(propose):
            raise TypeError("proposal_agent must expose callable propose(context)")
        try:
            maximum = int(final_max_candidates)
        except (TypeError, ValueError) as exc:
            raise ValueError("final_max_candidates must be an integer in [1, 64]") from exc
        if not 1 <= maximum <= 64:
            raise ValueError("final_max_candidates must be in [1, 64]")
        self.proposal_agent = proposal_agent
        self.final_max_candidates = maximum
        self._stage_audit_json: list[str] = []

    @property
    def last_stage_audit(self) -> dict[str, Any] | None:
        """Return a detached copy of the most recent successful stage audit."""

        if not self._stage_audit_json:
            return None
        return json.loads(self._stage_audit_json[-1])

    @property
    def stage_audits(self) -> list[dict[str, Any]]:
        """Return detached copies of all successful stage audits."""

        return [json.loads(value) for value in self._stage_audit_json]

    def propose(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Run staged fusion and return one validated proposal-mode response."""

        original = _fusion_request_from_context(context)
        if original.mode != "propose":
            raise ValueError("staged all-evidence fusion requires an original propose-mode context")

        confounder_request, confounder_id_map = self._filtered_proposal_request(
            original,
            role_hint="confounder",
            source_families=_CONFOUNDER_ROLE_FAMILIES,
            stage_name="confounder_role_proposal",
        )
        modifier_request, modifier_id_map = self._filtered_proposal_request(
            original,
            role_hint="effect_modifier",
            source_families=_MODIFIER_ROLE_FAMILIES,
            stage_name="modifier_role_proposal",
        )
        stage_requests: list[tuple[str, AllEvidenceFusionRequest, Mapping[str, str]]] = [
            (
                "full_evidence_proposal",
                original,
                {block.evidence_id: block.evidence_id for block in original.evidence_blocks},
            ),
            (
                "confounder_role_proposal",
                confounder_request,
                confounder_id_map,
            ),
            (
                "modifier_role_proposal",
                modifier_request,
                modifier_id_map,
            ),
        ]

        validated_proposals: list[_ValidatedProposal] = []
        stage_audit: list[dict[str, Any]] = []
        for stage_name, request, evidence_id_map in stage_requests:
            response, result, request_hash, response_hash = self._invoke_validated(request)
            del response
            mapped = self._validated_proposals(result, evidence_id_map=evidence_id_map)
            validated_proposals.extend(mapped)
            stage_audit.append(
                {
                    "stage": stage_name,
                    "request_sha256": request_hash,
                    "response_sha256": response_hash,
                    "evidence_block_count": len(request.evidence_blocks),
                    "source_families": [
                        family
                        for family in ALL_SOURCE_FAMILIES
                        if any(family in block.source_families for block in request.evidence_blocks)
                    ],
                    "validated_proposal_count": len(mapped),
                    "evidence_id_map_to_original": dict(evidence_id_map),
                    "mapped_grounding_evidence_ids": sorted(
                        {
                            evidence_id
                            for proposal in mapped
                            for evidence_id in proposal.supporting_evidence_ids
                        }
                    ),
                    "reasoning_trace_presence": _reasoning_presence_summary(self.proposal_agent),
                }
            )

        safe_union_inputs = self._safe_union_inputs(validated_proposals)
        safe_union_identity = safe_staged_proposal_union_identity()
        safe_union_result = safe_staged_proposal_union(safe_union_inputs)
        safe_union_result.verify(
            candidates=safe_union_inputs,
            expected_implementation_sha256=safe_union_identity.implementation_sha256,
        )
        union, selection_to_representative = self._safe_union_entries(safe_union_result)
        if not union:
            raise ValueError("staged fusion produced no validated candidate contracts")

        effective_cap = min(self.final_max_candidates, original.max_candidates)
        selection_request = self._selection_request(
            original,
            union,
            max_candidates=effective_cap,
        )
        selection_response, selection_result, request_hash, response_hash = self._invoke_validated(
            selection_request
        )
        del selection_response
        if selection_result.mode != "select":  # pragma: no cover - request fixes the mode
            raise ValueError("final staged fusion result was not select-mode")

        remote_response = self._map_selection_to_original_proposals(
            selection_result,
            original_request=original,
        )
        validated_remote_result = validate_all_evidence_fusion_response(
            original,
            remote_response,
        )
        selection_candidate_pool = self._postprocessor_candidate_pool(
            union,
            selected_candidate_ids=selection_result.selected_candidate_ids,
            validated_remote_result=validated_remote_result,
        )
        original_request_source_families = tuple(
            family
            for family in ALL_SOURCE_FAMILIES
            if any(family in block.source_families for block in original.evidence_blocks)
        )
        postprocessed = postprocess_minimal_staged_selection(
            remote_response=remote_response,
            remote_selected_candidate_ids=selection_result.selected_candidate_ids,
            candidate_pool=selection_candidate_pool,
            original_request_source_families=original_request_source_families,
            max_candidates=effective_cap,
        )
        final_response = postprocessed.response
        backfilled_candidate_ids = (
            *postprocessed.mandatory_coverage_candidate_ids,
            *postprocessed.high_confidence_reserve_candidate_ids,
        )
        # The final boundary is intentionally redundant: the mapped response
        # must satisfy the exact original request, not merely the select-mode
        # request from which it was derived.
        final_result = validate_all_evidence_fusion_response(original, final_response)
        detached_response = _detached_json_object(final_response, label="mapped proposal response")

        stage_audit.append(
            {
                "stage": "final_contract_selection",
                "request_sha256": request_hash,
                "response_sha256": response_hash,
                "evidence_block_count": len(selection_request.evidence_blocks),
                "candidate_pool_count": len(selection_request.candidates),
                # Keep the historical fields as explicit aliases for the
                # remote response.  The unambiguous fields below distinguish
                # it from the deterministic post-validation backfill.
                "selected_count": len(selection_result.selected_candidate_ids),
                "selected_candidate_ids": list(selection_result.selected_candidate_ids),
                "remote_selected_count": len(selection_result.selected_candidate_ids),
                "remote_selected_candidate_ids": list(selection_result.selected_candidate_ids),
                "final_selected_count": len(final_result.proposed_specs),
                "backfilled_candidate_ids": list(backfilled_candidate_ids),
                "mandatory_coverage_candidate_ids": list(
                    postprocessed.mandatory_coverage_candidate_ids
                ),
                "high_confidence_reserve_candidate_ids": list(
                    postprocessed.high_confidence_reserve_candidate_ids
                ),
                "selection_postprocessor": postprocessed.audit(),
                "selection_backfill_version": STAGED_SELECTION_BACKFILL_VERSION,
                "selection_union_postprocessing_version": (
                    STAGED_SELECTION_UNION_POSTPROCESSING_VERSION
                ),
                "reasoning_trace_presence": _reasoning_presence_summary(self.proposal_agent),
            }
        )
        audit = {
            "schema_version": STAGED_FUSION_AUDIT_SCHEMA_VERSION,
            "outer_fold": original.outer_fold,
            "split_fingerprint": original.split_fingerprint,
            "original_request_sha256": _content_sha256(original.context()),
            "configured_final_cap": self.final_max_candidates,
            "effective_final_cap": effective_cap,
            "selection_backfill_version": STAGED_SELECTION_BACKFILL_VERSION,
            "selection_union_postprocessing_version": (
                STAGED_SELECTION_UNION_POSTPROCESSING_VERSION
            ),
            "role_specific_proposal_policy": {
                "version": "role_specific_all_evidence_families_v1",
                "eligible_source_families": list(ALL_SOURCE_FAMILIES),
                "neural_query_moments_eligible": True,
                "matched_pair_htr_embedding_and_tfidf_evidence_eligible": True,
            },
            "stages": stage_audit,
            "proposal_union": {
                "validated_proposal_count": len(validated_proposals),
                "unique_contract_count": (
                    len(validated_proposals) - len(safe_union_result.exact_duplicate_candidate_ids)
                ),
                "exact_duplicate_count": len(safe_union_result.exact_duplicate_candidate_ids),
                "same_name_merge": {
                    "version": STAGED_SAME_NAME_MERGE_VERSION,
                    "merged_contract_count": len(
                        safe_union_result.compatible_role_merge_candidate_ids
                    ),
                    "final_candidate_pool_count": len(union),
                },
                "safe_union": self._safe_union_audit(
                    safe_union_result,
                    selection_to_representative=selection_to_representative,
                ),
            },
            "remote_selected_count": len(selection_result.selected_candidate_ids),
            "final_selected_count": len(final_result.proposed_specs),
            "backfilled_candidate_ids": list(backfilled_candidate_ids),
            "returned_proposal_count": len(final_result.proposed_specs),
            "returned_response_sha256": _content_sha256(detached_response),
        }
        self._stage_audit_json.append(_canonical_json(audit))
        return detached_response

    def _invoke_validated(
        self,
        request: AllEvidenceFusionRequest,
    ) -> tuple[dict[str, Any], FusionResult, str, str]:
        request_context = request.context()
        request_hash = _content_sha256(request_context)
        raw_response = self.proposal_agent.propose(request_context)
        response = _detached_json_object(raw_response, label="injected agent response")
        result = validate_all_evidence_fusion_response(request, response)
        return response, result, request_hash, _content_sha256(response)

    @staticmethod
    def _filtered_proposal_request(
        original: AllEvidenceFusionRequest,
        *,
        role_hint: str,
        source_families: frozenset[str],
        stage_name: str,
    ) -> tuple[AllEvidenceFusionRequest, dict[str, str]]:
        retained = [
            block
            for block in original.evidence_blocks
            if block.role_hint == role_hint
            and bool(set(block.source_families).intersection(source_families))
        ]
        if not retained:
            raise ValueError(f"{stage_name} has no matching evidence blocks")

        stage_to_original: dict[str, str] = {}
        evidence: list[dict[str, Any]] = []
        for index, block in enumerate(retained, start=1):
            stage_id = f"evidence_{index:04d}"
            stage_to_original[stage_id] = block.evidence_id
            evidence.append(
                {
                    "evidence_id": stage_id,
                    "source_families": list(block.source_families),
                    "role_hint": block.role_hint,
                    "content": block.content,
                }
            )
        subset_context = {
            "prompt_version": FUSION_PROMPT_VERSION,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "outer_fold": original.outer_fold,
            "split_fingerprint": original.split_fingerprint,
            "mode": "propose",
            "max_candidates": original.max_candidates,
            "evidence": evidence,
        }
        return _fusion_request_from_context(subset_context), stage_to_original

    @staticmethod
    def _validated_proposals(
        result: FusionResult,
        *,
        evidence_id_map: Mapping[str, str],
    ) -> list[_ValidatedProposal]:
        if result.mode != "propose":
            raise ValueError("proposal stage did not produce a propose-mode result")
        grounding = result.response_audit.get("proposal_grounding")
        if not isinstance(grounding, list) or len(grounding) != len(result.proposed_specs):
            raise ValueError("validated proposal grounding is incomplete")

        output: list[_ValidatedProposal] = []
        for spec, row in zip(result.proposed_specs, grounding, strict=True):
            if not isinstance(row, Mapping):
                raise ValueError("validated proposal grounding must contain objects")
            stage_ids = row.get("supporting_evidence_ids")
            families = row.get("supporting_source_families")
            if not isinstance(stage_ids, list) or not isinstance(families, list):
                raise ValueError("validated proposal grounding has an invalid shape")
            try:
                original_ids = tuple(evidence_id_map[str(value)] for value in stage_ids)
            except KeyError as exc:  # pragma: no cover - validator should make this impossible
                raise ValueError("validated proposal cites unmapped evidence") from exc
            output.append(
                _ValidatedProposal(
                    spec_json=_canonical_json(spec),
                    supporting_evidence_ids=original_ids,
                    supporting_source_families=tuple(str(value) for value in families),
                )
            )
        return output

    @staticmethod
    def _safe_union_inputs(
        proposals: Sequence[_ValidatedProposal],
    ) -> list[dict[str, Any]]:
        """Assign opaque occurrence IDs for the strict proposal-union boundary."""

        inputs: list[dict[str, Any]] = []
        for index, proposal in enumerate(proposals, start=1):
            spec = proposal.spec
            CandidateContract(
                spec,
                source_families=proposal.supporting_source_families,
            )
            inputs.append(
                {
                    "candidate_id": f"candidate_{index:04d}",
                    "extraction_spec": spec,
                    "supporting_evidence_ids": list(proposal.supporting_evidence_ids),
                    "supporting_source_families": list(proposal.supporting_source_families),
                    "validated_occurrence_count": 1,
                }
            )
        return inputs

    @staticmethod
    def _safe_union_entries(
        result: SafeStagedProposalUnionResult,
    ) -> tuple[list[_ProposalUnionEntry], dict[str, str]]:
        """Translate authenticated representatives into the selector namespace."""

        result.verify()
        entries: list[_ProposalUnionEntry] = []
        selection_to_representative: dict[str, str] = {}
        for index, candidate in enumerate(result.candidates, start=1):
            selection_id = f"candidate_{index:04d}"
            selection_to_representative[selection_id] = candidate.candidate_id
            detached_candidate = candidate.as_dict()
            entries.append(
                _ProposalUnionEntry(
                    spec_json=_canonical_json(detached_candidate["extraction_spec"]),
                    supporting_evidence_ids=list(candidate.supporting_evidence_ids),
                    supporting_source_families=list(candidate.supporting_source_families),
                    validated_occurrence_count=(candidate.validated_occurrence_count),
                )
            )
        return entries, selection_to_representative

    @staticmethod
    def _safe_union_audit(
        result: SafeStagedProposalUnionResult,
        *,
        selection_to_representative: Mapping[str, str],
    ) -> dict[str, Any]:
        """Return a content-free audit of every proposal occurrence disposition."""

        result.verify()
        return {
            "identity": result.identity.as_dict(),
            "input_sha256": result.input_sha256,
            "output_sha256": result.output_sha256,
            "input_candidate_count": len(result.input_candidate_ids),
            "representative_candidate_ids": list(result.representative_candidate_ids),
            "exact_duplicate_candidate_ids": list(result.exact_duplicate_candidate_ids),
            "compatible_role_merge_candidate_ids": list(result.compatible_role_merge_candidate_ids),
            "omitted_conflict_candidate_ids": list(result.omitted_conflict_candidate_ids),
            "dispositions": [row.as_dict() for row in result.dispositions],
            "conflicts": [row.as_dict() for row in result.conflicts],
            "selection_candidate_to_representative_id": dict(selection_to_representative),
            "incompatible_variant_support_or_roles_propagated": False,
            "semantic_fields_used_for_conflict_ranking": False,
            "patient_rows_or_observed_labels_used": False,
        }

    @staticmethod
    def _selection_request(
        original: AllEvidenceFusionRequest,
        union: Sequence[_ProposalUnionEntry],
        *,
        max_candidates: int,
    ) -> AllEvidenceFusionRequest:
        candidate_names = [str(entry.spec.get("name") or "") for entry in union]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("staged final candidate pool must contain unique names")
        candidates = []
        for index, entry in enumerate(union, start=1):
            contract = CandidateContract(
                entry.spec,
                source_families=entry.supporting_source_families,
            )
            candidates.append(
                {
                    "candidate_id": f"candidate_{index:04d}",
                    "extraction_spec": contract.extraction_spec,
                    "source_families": list(contract.source_families),
                }
            )
        selection_context = {
            "prompt_version": FUSION_PROMPT_VERSION,
            "source_text_temporal_policy": source_text_temporal_policy_audit(),
            "outer_fold": original.outer_fold,
            "split_fingerprint": original.split_fingerprint,
            "mode": "select",
            "max_candidates": max_candidates,
            "evidence": [block.as_prompt_dict() for block in original.evidence_blocks],
            "candidates": candidates,
        }
        return _fusion_request_from_context(selection_context)

    @staticmethod
    def _map_selection_to_original_proposals(
        selection: FusionResult,
        *,
        original_request: AllEvidenceFusionRequest,
    ) -> dict[str, Any]:
        if selection.mode != "select":
            raise ValueError("cannot map a non-selection result")
        notes = selection.response_audit.get("selection_notes")
        if not isinstance(notes, list):
            raise ValueError("validated selection notes are missing")
        notes_by_id: dict[str, Mapping[str, Any]] = {}
        for note in notes:
            if not isinstance(note, Mapping):
                raise ValueError("validated selection notes must contain objects")
            candidate_id = str(note.get("candidate_id") or "")
            notes_by_id[candidate_id] = note

        proposals: list[dict[str, Any]] = []
        for candidate_id, contract in zip(
            selection.selected_candidate_ids,
            selection.selected_contracts,
            strict=True,
        ):
            note = notes_by_id.get(candidate_id)
            if note is None:
                raise ValueError(f"validated selection note missing for {candidate_id}")
            proposal = contract.extraction_spec
            proposal.update(
                {
                    "supporting_evidence_ids": list(note["supporting_evidence_ids"]),
                    "supporting_source_families": list(note["supporting_source_families"]),
                    "rationale": str(note.get("reason") or ""),
                }
            )
            proposals.append(proposal)

        response = {"proposals": proposals}
        # Validate here as well as at the caller boundary so this helper cannot
        # return a response grounded against a different evidence namespace.
        validate_all_evidence_fusion_response(original_request, response)
        return response

    @staticmethod
    def _postprocessor_candidate_pool(
        union: Sequence[_ProposalUnionEntry],
        *,
        selected_candidate_ids: Sequence[str],
        validated_remote_result: FusionResult,
    ) -> list[dict[str, Any]]:
        """Add strictly validated final-note support to selected pool entries.

        Proposal-stage support is intentionally conservative.  A final selector
        sees the complete original evidence namespace and can therefore cite an
        additional block that independently grounds an unchanged candidate.
        The public response validator has already checked those citations for
        namespace, source-family availability, and lexical contract grounding.
        Carrying its audited support into the postprocessor pool prevents the
        later subset check from mistaking that valid broadening for an unsafe
        citation.
        """

        if validated_remote_result.mode != "propose":
            raise ValueError("mapped remote selection must validate as a proposal response")
        selected_ids = tuple(str(value) for value in selected_candidate_ids)
        grounding = validated_remote_result.response_audit.get("proposal_grounding")
        if (
            not isinstance(grounding, list)
            or len(grounding) != len(selected_ids)
            or len(validated_remote_result.proposed_specs) != len(selected_ids)
        ):
            raise ValueError("validated selected-candidate support is incomplete")

        entries_by_id = {
            f"candidate_{index:04d}": entry for index, entry in enumerate(union, start=1)
        }
        support_by_id: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        for candidate_id, spec, row in zip(
            selected_ids,
            validated_remote_result.proposed_specs,
            grounding,
            strict=True,
        ):
            entry = entries_by_id.get(candidate_id)
            if entry is None:
                raise ValueError("validated selection support cites an unknown candidate")
            if _canonical_json(spec) != entry.spec_json:
                raise ValueError("validated selection support changed its candidate contract")
            if not isinstance(row, Mapping):
                raise ValueError("validated selected-candidate support must contain objects")
            evidence_ids = row.get("supporting_evidence_ids")
            source_families = row.get("supporting_source_families")
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or not all(isinstance(value, str) and value for value in evidence_ids)
                or len(evidence_ids) != len(set(evidence_ids))
            ):
                raise ValueError("validated selected-candidate evidence support is invalid")
            if (
                not isinstance(source_families, list)
                or not source_families
                or not all(isinstance(value, str) and value for value in source_families)
                or len(source_families) != len(set(source_families))
                or not set(source_families) <= set(ALL_SOURCE_FAMILIES)
            ):
                raise ValueError("validated selected-candidate family support is invalid")
            if candidate_id in support_by_id:
                raise ValueError("validated selection support contains duplicate candidate IDs")
            support_by_id[candidate_id] = (tuple(evidence_ids), tuple(source_families))

        candidate_pool: list[dict[str, Any]] = []
        for index, entry in enumerate(union, start=1):
            candidate_id = f"candidate_{index:04d}"
            note_evidence, note_families = support_by_id.get(candidate_id, ((), ()))
            evidence_ids = list(dict.fromkeys((*entry.supporting_evidence_ids, *note_evidence)))
            family_set = set(entry.supporting_source_families).union(note_families)
            source_families = [family for family in ALL_SOURCE_FAMILIES if family in family_set]
            candidate_pool.append(
                {
                    "candidate_id": candidate_id,
                    "extraction_spec": entry.spec,
                    "supporting_evidence_ids": evidence_ids,
                    "supporting_source_families": source_families,
                    "validated_occurrence_count": entry.validated_occurrence_count,
                }
            )
        return candidate_pool


__all__ = [
    "STAGED_FUSION_AUDIT_SCHEMA_VERSION",
    "STAGED_SAME_NAME_MERGE_VERSION",
    "STAGED_SELECTION_BACKFILL_VERSION",
    "STAGED_SELECTION_UNION_POSTPROCESSING_VERSION",
    "StagedAllEvidenceFusionAgent",
]
