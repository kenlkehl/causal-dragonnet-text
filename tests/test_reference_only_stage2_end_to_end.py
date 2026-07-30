from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oci.extraction.complete_paged import (
    COMPLETE_PAGED_RESPONSE_SCHEMA,
    COMPLETE_PAGED_TRANSPORT_SCHEMA,
    CompletePageResponse,
    build_complete_page_prompt,
)
import oci.inference.all_evidence_fusion_runner as runner_module
import oci.inference.production_stage1_hierarchy_one_shot as one_shot_module
from oci.inference.adaptive_hierarchical_stage1_reconsideration import (
    AdaptiveCurrentFeature,
    AdaptiveDiagnostic,
    AdaptiveHierarchicalStage1Reconsideration,
    AdaptiveReconsiderationConfig,
    ExactSpentCatalogAuthentication,
    adaptive_hierarchical_stage1_reconsideration_identity,
)
from oci.inference.all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    AS_DOCUMENTED_UNIT,
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
)
from oci.inference.all_evidence_post_extraction_review import (
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    GateAcceptanceDecision,
    apply_post_extraction_review_operations,
    validate_post_extraction_review_response,
)
from oci.inference.all_evidence_fusion import (
    ground_evidence_to_extraction_contract,
)
from oci.inference.approved_hierarchical_discovery_agent import (
    ApprovedHierarchicalDiscoveryAgent,
    direct_numerical_bindings_from_manifest,
)
from oci.inference.production_role_neutral_stage2_handoff import (
    ReferenceOnlyRoleNeutralStage1HandoffPublisher,
)
from oci.inference.production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    RoleNeutralStage1ExecutionPolicy,
    execute_and_publish_role_neutral_stage1,
)
from oci.inference.production_stage1_scope_scheduler import (
    build_canonical_stage1_scope_plan,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from oci.inference.hierarchical_all_architecture_discovery import (
    CROSS_ARCHITECTURE_INTEGRATION_JOB,
    CROSS_ARCHITECTURE_PLANNER_JOB,
    HierarchicalDiscoveryConfig,
)
from oci.inference.hierarchical_discovery_job_cache import (
    AuthenticatedHierarchicalDiscoveryJobCache,
)
from tests.hierarchy_resource_test_support import (
    HIERARCHY_JOB_CACHE_CONFIG,
)
from oci.inference.lossless_stage1_evidence_catalog import (
    build_complete_architecture_chunks,
)
from tests.test_adaptive_hierarchical_stage1_reconsideration import (
    NEW_MISSING_CONSTRUCT,
)
from tests.test_approved_hierarchical_discovery_agent import (
    _MetadataRunner,
    _manifest,
)
from tests.test_direct_upstream_numerical_reference_bank import (
    _write_embedding_component,
    _write_matched_component,
    _write_neural_query_component,
    _write_simple_dense_component,
)
from tests.test_native_role_neutral_payload_catalog_adapter import (
    _native_payloads,
)
from tests.test_lossless_stage1_evidence_catalog import (
    _cumulative_family_payloads,
    _inputs,
)
from tests.test_production_role_neutral_stage2_handoff import (
    _ProviderReadyProducerRecorder,
    _cpu_resource_plan,
    _fit_outcome,
    _fit_text,
    _fit_treatment,
)
from tests.test_production_stage1_hierarchy_one_shot import (
    TEST_ENDPOINT,
    TEST_MODEL,
    _PromptGuard,
    _options,
    _stage2_protocol,
)
from tests.test_production_stage1_role_neutral_execution import (
    _RecordingExecutor,
    _registry,
    _sha,
)


def _content_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fixture_semantic_group(candidate) -> int:
    """Keep two compiler-distinct, evidence-grounded synthetic constructs."""

    source_families = tuple(candidate["source_families"])
    feature_name = str(candidate["feature_name"])
    if not source_families or not feature_name:
        raise AssertionError("fixture candidate lost its source-family provenance")
    return 1 if "_consolidation_slot_" in feature_name else 0


def _fixture_semantic_name(group: int) -> str:
    if group == 0:
        return "baseline_age"
    if group == 1:
        return "egfr_mutation"
    raise AssertionError(f"unexpected fixture semantic group: {group}")


class _NumericalProviderReadyRecorder(_ProviderReadyProducerRecorder):
    """Write provider seals and the genuine direct-reference numerical layouts."""

    def factory(self, expected_component: str):
        parent_factory = super().factory(expected_component)

        def bind(invocation):
            parent = parent_factory(invocation)

            def execute() -> None:
                parent.execute()
                existing = json.loads(
                    (
                        invocation.output_root / "execution_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                projection_proof = existing.get(
                    "stage2_fit_projection_proof"
                )
                scope = invocation.physical_owner
                base = float(10 * (int(scope.canonical_index) + 1))
                if expected_component == "bow":
                    _write_simple_dense_component(
                        component_root=invocation.output_root,
                        scope=scope,
                        families_and_columns=(
                            (
                                BOW_NUISANCE,
                                (
                                    "linear::treatment_nuisance",
                                    "linear::outcome_nuisance",
                                ),
                            ),
                            (
                                BOW_R_LOSS,
                                (
                                    "linear::effect_pseudo_target",
                                    "linear::effect_weighted_r",
                                ),
                            ),
                        ),
                        base=base,
                    )
                elif expected_component == "htr":
                    _write_simple_dense_component(
                        component_root=invocation.output_root,
                        scope=scope,
                        families_and_columns=(
                            (
                                HTR_NEURAL,
                                (
                                    "htr_nuisance::e_hat",
                                    "htr_nuisance::m_hat",
                                    "htr_effect::pseudo_outcome_mse",
                                ),
                            ),
                        ),
                        base=base + 1.0,
                    )
                elif expected_component == "matched_pair":
                    _write_matched_component(
                        invocation.output_root,
                        scope,
                        base=base + 2.0,
                    )
                elif expected_component == "embeddings":
                    _write_embedding_component(
                        invocation.output_root,
                        scope,
                        base=base + 3.0,
                        vocabulary_width=2 + int(scope.canonical_index) % 2,
                    )
                elif expected_component == "tfidf":
                    _write_simple_dense_component(
                        component_root=invocation.output_root,
                        scope=scope,
                        families_and_columns=(
                            (
                                TFIDF_TOPICS,
                                (
                                    "treatment::topic_000",
                                    "outcome::topic_000",
                                    "effect::topic_000",
                                ),
                            ),
                            (
                                TFIDF_ORPHAN_NGRAMS,
                                tuple(
                                    f"residual_tfidf::term_{index}"
                                    for index in range(
                                        2 + int(scope.canonical_index) % 2
                                    )
                                ),
                            ),
                        ),
                        base=base + 4.0,
                    )
                elif expected_component == "neural_query":
                    _write_neural_query_component(
                        invocation.output_root,
                        scope,
                        base=base + 5.0,
                    )
                else:  # pragma: no cover - closed component partition
                    raise AssertionError(expected_component)
                terminal_path = (
                    invocation.output_root / "execution_manifest.json"
                )
                terminal = json.loads(
                    terminal_path.read_text(encoding="utf-8")
                )
                body = {
                    key: value
                    for key, value in terminal.items()
                    if key != "content_sha256"
                }
                body.update(
                    {
                        "plan_scientific_content_sha256": (
                            invocation.plan.scientific_content_sha256
                        ),
                        "physical_owner_scope_id": scope.scope_id,
                        "component": expected_component,
                        "text_truncation_applied": False,
                        "lossy_evidence_selection_applied": False,
                    }
                )
                if expected_component == "bow":
                    assert projection_proof is not None
                    body["stage2_fit_projection_proof"] = projection_proof
                terminal_path.write_text(
                    json.dumps(
                        {**body, "content_sha256": _sha(body)},
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            return BoundRoleNeutralComponentProducer(
                execute=execute,
                authenticate=parent.authenticate,
            )

        return bind


class _IntegrationPromptGuard(_PromptGuard):
    def __init__(
        self,
        *,
        tokenizer_locator: Path,
        model_name: str,
        model_context_window_tokens: int,
    ) -> None:
        del tokenizer_locator, model_context_window_tokens
        super().__init__(model_name=model_name)


class _TwoRoundHierarchyRunner(_MetadataRunner):
    def __init__(
        self,
        *,
        server_urls,
        model_name,
        prompt_nontruncation_guard,
        **_kwargs,
    ) -> None:
        super().__init__()
        endpoint = (
            str(server_urls)
            if isinstance(server_urls, str)
            else str(tuple(server_urls)[0])
        )
        generation_policy = _kwargs["generation_policy"]
        body = {
            "schema_version": "five_fold_fake_stage2_runner_v1",
            "endpoint_urls": [endpoint],
            "model": {"name": model_name, "resolution": "explicit"},
            "retry": {"max_attempts": 1},
            "generation_policy": generation_policy.as_dict(),
            "generation_policy_sha256": generation_policy.content_sha256,
            "generation_policy_resolution": "explicit_closed_policy",
            "prompt_nontruncation_guard": (
                prompt_nontruncation_guard.identity()
            ),
        }
        self._identity = {
            **body,
            "identity_sha256": _content_sha256(body),
        }
        self._endpoint = endpoint
        self._model = model_name
        self._generation_policy = generation_policy
        self._prompt_nontruncation_guard = prompt_nontruncation_guard
        self.adaptive_proposal_calls = 0
        self.adaptive_drop_page_calls = 0
        self.adaptive_noop_page_calls = 0

    def _response(self, job):
        request = json.loads(job.messages[1]["content"])
        request_job = request["job"]
        if request_job == "compare_cross_architecture_candidate_relations":
            anchor_group = _fixture_semantic_group(
                request["anchor_candidate"]
            )
            return {
                "comparisons": {
                    peer["candidate_id"]: {
                        "relation": (
                            "same_construct"
                            if _fixture_semantic_group(peer) == anchor_group
                            else "distinct"
                        ),
                        "reason": (
                            "Retain exactly two compiler-distinct semantic "
                            "constructs in this bounded fixture."
                        ),
                    }
                    for peer in request["peer_candidates"]
                }
            }
        if request_job == "fold_cross_architecture_group_definition":
            prior = request["prior_accumulator"]
            fresh = request["fresh_candidates"]
            groups = {
                _fixture_semantic_group(candidate)
                for candidate in fresh
            }
            if len(groups) != 1:
                raise AssertionError(
                    "compiler-distinct fixture groups were mixed during folding"
                )
            group = next(iter(groups))
            canonical_name = _fixture_semantic_name(group)
            if (
                prior is not None
                and prior["canonical_name"] != canonical_name
            ):
                raise AssertionError(
                    "fixture group accumulator changed semantic identity"
                )
            return {
                "canonical_name": canonical_name,
                "description": (
                    f"Documented {_fixture_semantic_name(group).replace('_', ' ')} "
                    "patient measurement."
                ),
                "unresolved_ambiguity": "",
                "reason": (
                    "Fold every compiler-proven member of this exact fixture "
                    "semantic group."
                ),
            }
        if (
            job.job_kind == CROSS_ARCHITECTURE_PLANNER_JOB
            and request_job == CROSS_ARCHITECTURE_PLANNER_JOB
        ):
            candidates = [
                candidate
                for dossier in request["architecture_dossiers"]
                for candidate in dossier["architecture_candidates"]
            ]
            group_slots = request["identifier_ownership"][
                "identifier_domains"
            ]["planner_group_slots"]
            lookback_slots = request["identifier_ownership"][
                "identifier_domains"
            ]["planner_lookback_slots"]
            if len(candidates) < 2 or len(group_slots) < 2:
                raise AssertionError(
                    "two-feature integration fixture lost candidate coverage"
                )
            return {
                "candidate_assignments": {
                    candidate["candidate_id"]: {
                        "group_slot": group_slots[
                            _fixture_semantic_group(candidate)
                        ],
                    }
                    for candidate in candidates
                },
                "group_slot_definitions": {
                    slot: {
                        "provisional_name": (
                            f"fixture_measure_{index + 1:03d}"
                        ),
                        "reason": (
                            "Exercise two bounded lossless integration groups."
                        ),
                    }
                    for index, slot in enumerate(group_slots)
                },
                "lookback_slot_definitions": {
                    slot: {
                        "selection": "unused",
                        "question": (
                            "No additional raw lookback is required."
                        ),
                        "reason": (
                            "All authenticated evidence remains in the group."
                        ),
                    }
                    for slot in lookback_slots
                },
            }
        if (
            job.job_kind == CROSS_ARCHITECTURE_INTEGRATION_JOB
            and request_job == CROSS_ARCHITECTURE_INTEGRATION_JOB
        ):
            candidates = [
                candidate
                for dossier in request["architecture_context"][
                    "architecture_dossiers"
                ]
                for candidate in dossier["architecture_candidates"]
            ]
            slots = request["identifier_ownership"][
                "identifier_domains"
            ]["integration_slots"]
            if len(candidates) < 2 or len(slots) < 2:
                raise AssertionError(
                    "two-feature integration fixture lost integration slots"
                )
            return {
                "candidate_routes": {
                    candidate["candidate_id"]: {
                        "route": slots[
                            _fixture_semantic_group(candidate)
                        ],
                        "reason": (
                            "Retain this architecture summary in the joint "
                            "fixture feature."
                        ),
                    }
                    for candidate in candidates
                },
                "slot_definitions": {
                    slot: {
                        "canonical_name": (
                            f"fixture_measure_{index + 1:03d}"
                        ),
                        "description": (
                            "A patient measurement in one of two lossless "
                            "cross-architecture fixture groups."
                        ),
                        "unresolved_ambiguity": "",
                    }
                    for index, slot in enumerate(slots)
                },
            }
        if request_job == "plan_adaptive_stage1_reconsideration":
            current = request["current_registry"]
            target = (
                current[0]["feature_name"]
                if current
                else NEW_MISSING_CONSTRUCT
            )
            families = (
                current[0]["source_families"]
                if current
                else [ACTIVE_STAGE1_CONCEPT_FAMILIES[0]]
            )
            return {
                "review_targets": [
                    {
                        "target": target,
                        "problem": (
                            "Exercise one bounded review of the retained registry."
                        ),
                        "relevant_architectures": [families[0]],
                        "requested_evidence_ids": [],
                        "reason": (
                            "Use aggregate diagnostics to review one current contract."
                        ),
                    }
                ],
                "no_lookback_needed": True,
            }
        if request_job == "propose_adaptive_registry_revision":
            self.adaptive_proposal_calls += 1
            current_names = {
                row["feature_name"]
                for row in request["current_registry"]
            }
            if "baseline_age" not in current_names:
                self.adaptive_noop_page_calls += 1
                return {
                    "operations": [],
                    "converged": True,
                }
            self.adaptive_drop_page_calls += 1
            target = "baseline_age"
            diagnostics = request["diagnostics"]
            return {
                "operations": [
                    {
                        "operation": "drop",
                        "targets": [target],
                        "proposed_feature": {},
                        "supporting_evidence_ids": [],
                        "diagnostic_ids": [
                            diagnostics[0]["diagnostic_id"]
                        ],
                        "reason": (
                            "Drop one bounded contract to exercise gate review."
                        ),
                    }
                ],
                "converged": False,
            }
        if request_job == "integrate_cross_architecture_group":
            proposal = request["proposal"]
            return {
                "decision": "accept",
                "canonical_name": proposal["targets"][0],
                "description": "A bounded reviewed patient measurement.",
                "unresolved_ambiguity": "",
                "reason": "Accept the exact compiler-checked review operation.",
            }
        if request_job == "compare_adaptive_candidate_relations":
            return {
                "comparisons": {
                    peer_id: {
                        "relation": "distinct",
                        "reason": "Keep the bounded measurements distinct.",
                    }
                    for peer_id in request["peer_candidate_ids"]
                }
            }
        if request_job == "fold_adaptive_group_definition":
            prior = request["prior_accumulator"]
            first = request["fresh_candidates"][0]
            return {
                "canonical_name": (
                    prior["canonical_name"]
                    if prior is not None
                    else first["feature_name"]
                ),
                "description": (
                    prior["description"]
                    if prior is not None
                    else first["description"]
                ),
                "unresolved_ambiguity": (
                    prior["unresolved_ambiguity"]
                    if prior is not None
                    else first["unresolved_ambiguity"]
                ),
                "reason": "Fold every compiler-proven group member.",
            }
        if request_job == "audit_adaptive_atomic_coverage":
            return {"findings": [], "reviewed_atomic_review": True}
        if request_job == "review_extraction_feature_evidence":
            response = _MetadataRunner._response(self, job)
            response["literal_units"] = []
            return response
        if request_job == "fold_extraction_evidence_definitions":
            response = _MetadataRunner._response(self, job)
            response["representation"]["unit"] = AS_DOCUMENTED_UNIT
            return response
        return _MetadataRunner._response(self, job)

    def run_json(self, *, job):
        request = json.loads(job.messages[1]["content"])
        adaptive = "adaptive" in str(request.get("job", ""))
        generation_parameters = (
            self._generation_policy.for_hierarchical_job(job.job_kind)
        )
        client_path = (
            "proposal_and_post_extraction_review"
            if adaptive
            else "hierarchical_discovery"
        )
        request_audit = self._prompt_nontruncation_guard.validate_request(
            {
                "model": self._model,
                "messages": list(job.messages),
                "temperature": generation_parameters.temperature,
                "max_tokens": generation_parameters.max_tokens,
                "stream": False,
            },
            client_path=client_path,
        )
        response = self._response(job)
        self._prompt_nontruncation_guard.validate_response(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1)),
            request_audit=request_audit,
        )
        request_sha = _content_sha256(job.as_dict())
        response_sha = _content_sha256(response)
        raw = json.dumps(
            response,
            sort_keys=True,
            separators=(",", ":"),
        )
        identity_sha = self._identity["identity_sha256"]
        self.calls.append(job)
        self._metadata.append(
            {
                "job_id": job.job_id,
                "job_kind": job.job_kind,
                "request_sha256": request_sha,
                "runner_identity_sha256": identity_sha,
                "outcome": "success",
                "parsed_response_sha256": response_sha,
                "attempts": [
                    {
                        "attempt_number": 1,
                        "endpoint": self._endpoint,
                        "model": self._model,
                        "response_model": self._model,
                        "finish_reason": "stop",
                        "request_sha256": request_sha,
                        "runner_identity_sha256": identity_sha,
                        "outcome": "success",
                        "retryable": False,
                        "will_retry": False,
                        "usage": {},
                        "content_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                        "raw_transport_bytes": len(
                            raw.encode("utf-8")
                        ),
                        "reasoning_hashes": {},
                        "parsed_response_sha256": response_sha,
                    }
                ],
            }
        )
        return response


class _FakeReviewAgent:
    def __init__(
        self,
        search_config,
        *,
        prompt_nontruncation_guard,
        generation_parameters,
    ) -> None:
        self.search_config = search_config
        self._guard = prompt_nontruncation_guard
        self._generation_parameters = generation_parameters

    def propose(self, context):
        request_audit = self._guard.validate_request(
            {
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "review"}],
                "max_tokens": int(
                    self.search_config.agent_max_tokens
                ),
                "stream": False,
            },
            client_path="proposal_and_post_extraction_review",
        )
        self._guard.validate_response(object(), request_audit=request_audit)
        if (
            int(context["review_round"]) != 1
            or int(context["review_attempt"]) != 1
        ):
            raise AssertionError(
                "fixture reviewer received an unexpected round or retry"
            )
        contracts = list(context["current_contracts"])
        contract_names = [contract["name"] for contract in contracts]
        if (
            len(contracts) != 2
            or len(set(contract_names)) != 2
            or set(contract_names) != {"baseline_age", "egfr_mutation"}
        ):
            raise AssertionError(
                "fixture must enter round-one review with exactly two "
                "compiler-distinct contracts"
            )
        target = next(
            contract
            for contract in contracts
            if contract["name"] == "baseline_age"
        )
        diagnostic = next(
            row
            for row in context["diagnostics"]
            if row.get("feature_name") == target["name"]
        )
        revised = {
            **target,
            "description": (
                "Extract the documented baseline age patient measurement; "
                "return null when absent or ambiguous."
            ),
        }
        supporting_evidence_ids = [
            row["evidence_id"]
            for row in context["sanitized_evidence_catalog"]
            if ground_evidence_to_extraction_contract(
                row,
                revised,
            ).supported
        ]
        if not supporting_evidence_ids:
            raise AssertionError(
                "fixture revision lost exact lexical source-evidence grounding"
            )
        return {
            "schema_version": (
                "all_evidence_post_extraction_review_response_v1"
            ),
            "operations": [
                {
                    "action": "revise",
                    "target_names": [target["name"]],
                    "contract": revised,
                    "supporting_diagnostic_ids": [
                        diagnostic["diagnostic_id"]
                    ],
                    "supporting_evidence_ids": [
                        supporting_evidence_ids[0]
                    ],
                    "reason": (
                        "Exercise one evidence-grounded bounded definition "
                        "revision before opening the untouched gate."
                    ),
                }
            ],
        }


class _FakeCompletePagedExtractor:
    """Schema-valid local transport beneath the real ledger-v2 provider."""

    def __init__(self, **kwargs) -> None:
        self.model_name = str(kwargs["model_name"])
        self._guard = kwargs["prompt_nontruncation_guard"]
        self._max_tokens = int(kwargs["max_tokens"])

    @staticmethod
    def _normalized_value(*, text, feature):
        ordinal = int(
            hashlib.sha256(
                f"{feature.name}\0{text}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        if feature.value_type == "continuous":
            return float(ordinal % 3)
        categories = tuple(feature.categories)
        return categories[ordinal % len(categories)]

    def _transport(self, *, prompt, response):
        request = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        request_audit = self._guard.validate_request(
            request,
            client_path="explicit_feature_extraction",
        )
        self._guard.validate_response(
            object(),
            request_audit=request_audit,
        )
        attempt = {
            "kind": "initial",
            "request_sha256": _content_sha256(request),
            "response_sha256": _content_sha256(response),
            "model": self.model_name,
            "finish_reason": "stop",
        }
        body = {
            "schema_version": COMPLETE_PAGED_TRANSPORT_SCHEMA,
            "transport_retry_count": 0,
            "schema_repair_count": 0,
            "configured_model": self.model_name,
            "attempts": [attempt],
        }
        return {**body, "content_sha256": _content_sha256(body)}

    def extract_complete_page(
        self,
        *,
        text,
        page,
        feature,
        geometry,
    ):
        prompt = build_complete_page_prompt(
            text,
            page=page,
            feature=feature,
            geometry=geometry,
        )
        citation_start = int(page.core_start)
        response = CompletePageResponse.validate(
            {
                "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                "status": "positive",
                "normalized_value": self._normalized_value(
                    text=text,
                    feature=feature,
                ),
                "reason": None,
                "citations": [
                    {
                        "start": citation_start,
                        "end": citation_start + 1,
                        "text": text[
                            citation_start:citation_start + 1
                        ],
                    }
                ],
            },
            text=text,
            page=page,
        )
        return response, self._transport(
            prompt=prompt,
            response=response.as_dict(),
        )

    def reconcile_complete_pages(
        self,
        *,
        text,
        feature,
        children,
    ):
        citations = [
            dict(citation)
            for child in children
            for citation in child["citations"]
        ]
        response = CompletePageResponse.validate(
            {
                "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                "status": "positive",
                "normalized_value": self._normalized_value(
                    text=text,
                    feature=feature,
                ),
                "reason": None,
                "citations": citations,
            },
            text=text,
            page=None,
        ).as_dict()
        prompt = json.dumps(
            {
                "feature_name": feature.name,
                "child_responses": list(children),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return response, self._transport(
            prompt=prompt,
            response=response,
        )

    def cleanup(self) -> None:
        return None


class _FakeStrictForest:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def identity(self):
        return {
            "backend": "fake_strict_causal_forest_v1",
            "strict_causal_forest_only": True,
        }

    def fit_predict(
        self,
        *,
        effect_train,
        control_train,
        treatment,
        outcome,
        effect_heldout,
        control_heldout,
    ):
        self.calls.append(
            {
                "effect_train_shape": tuple(effect_train.shape),
                "control_train_shape": tuple(control_train.shape),
                "treatment_count": len(treatment),
                "outcome_count": len(outcome),
                "effect_heldout_shape": tuple(effect_heldout.shape),
                "control_heldout_shape": tuple(control_heldout.shape),
            }
        )
        signal = np.asarray(effect_heldout[:, 0], dtype=float)
        scale = max(1.0, float(np.max(np.abs(signal))))
        return np.clip(signal / (4.0 * scale), -0.25, 0.25)


def test_reference_fixture_has_exactly_two_compiler_distinct_contracts() -> None:
    candidates = (
        {
            "feature_name": "bow_nuisance_measure",
            "source_families": [BOW_NUISANCE],
        },
        {
            "feature_name": "bow_r_loss_measure",
            "source_families": [BOW_R_LOSS],
        },
        {
            "feature_name": "tfidf_topics_measure_consolidation_slot_002",
            "source_families": [TFIDF_TOPICS],
        },
        {
            "feature_name": "tfidf_topics_measure_consolidation_slot_003",
            "source_families": [TFIDF_TOPICS],
        },
    )
    groups = tuple(
        _fixture_semantic_group(candidate)
        for candidate in candidates
    )
    names = {
        _fixture_semantic_name(group)
        for group in groups
    }

    assert set(groups) == {0, 1}
    assert names == {"baseline_age", "egfr_mutation"}
    assert len(names) == 2


def test_review_schedules_preserve_canonical_outer_order_for_deduplicated_scope() -> None:
    registry = _registry()
    plan = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=_sha(registry),
        global_seed=42,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        gpu_ids=(),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )
    outer = next(
        scope
        for scope in plan.scopes
        if scope.outer_fold == 1 and scope.scope_kind == "full_outer"
    )
    frame = pd.DataFrame(
        {
            "_oci_row_id": list(outer.fit_row_ids),
            "treatment": [
                _fit_treatment(row_id) for row_id in outer.fit_row_ids
            ],
            "outcome": [
                _fit_outcome(row_id) for row_id in outer.fit_row_ids
            ],
        }
    )
    assignments = {
        int(scope.inner_fold): tuple(scope.heldout_row_ids)
        for scope in plan.scopes
        if scope.outer_fold == 1
        and scope.scope_kind == "exact_inner"
        and scope.inner_fold is not None
    }

    class Provider:
        def get_review_partition_assignments(
            self,
            *,
            outer_fold,
            exact_outer_train_row_ids,
        ):
            assert outer_fold == 1
            assert exact_outer_train_row_ids == outer.fit_row_ids
            return assignments

    injected = runner_module._build_injected_review_partition_schedule(
        frame,
        outer_fold=1,
        review_rounds=2,
        minimum_partition_rows=2,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        provider=Provider(),
        provider_identity={"identity_sha256": "a" * 64},
    )
    cumulative = next(
        scope
        for scope in plan.scopes
        if scope.outer_fold == 1
        and scope.scope_kind == "cumulative_spent"
        and scope.context_epoch == 1
    )
    exact_inner_five = next(
        scope
        for scope in plan.scopes
        if scope.outer_fold == 1
        and scope.scope_kind == "exact_inner"
        and scope.inner_fold == 5
    )
    assert (
        injected.row_ids((*injected.initial_spent_fold_ids, injected.gate_fold_ids[0]))
        == cumulative.fit_row_ids
        == exact_inner_five.fit_row_ids
    )

    local = runner_module._build_review_partition_schedule(
        frame,
        outer_fold=1,
        review_rounds=2,
        minimum_partition_rows=2,
        random_state=42,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
    )
    selected_folds = (
        *local.initial_spent_fold_ids,
        local.gate_fold_ids[0],
    )
    selected_rows = {
        row_id
        for fold_id in selected_folds
        for row_id in local.row_ids_by_fold[fold_id]
    }
    assert local.row_ids(selected_folds) == tuple(
        row_id for row_id in outer.fit_row_ids if row_id in selected_rows
    )


def test_reference_fixture_compiler_retains_two_reviewable_contracts(
    tmp_path: Path,
) -> None:
    catalog = runner_module.build_role_neutral_evidence_catalog(_inputs())
    manifest = _manifest(catalog)
    prompt_guard = _IntegrationPromptGuard(
        tokenizer_locator=(tmp_path / "unused-tokenizer").resolve(),
        model_name=TEST_MODEL,
        model_context_window_tokens=4096,
    )
    hierarchy = _TwoRoundHierarchyRunner(
        server_urls=(TEST_ENDPOINT,),
        model_name=TEST_MODEL,
        prompt_nontruncation_guard=prompt_guard,
        generation_policy=_stage2_protocol().generation_policy,
    )
    agent = ApprovedHierarchicalDiscoveryAgent(
        catalog=catalog,
        chunk_plan=build_complete_architecture_chunks(
            catalog,
            max_atoms_per_chunk=2,
            max_bytes_per_chunk=20_000,
        ),
        family_explanations={
            family: f"Interpret complete evidence from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        direct_numerical_manifest=manifest,
        direct_numerical_bindings=(
            direct_numerical_bindings_from_manifest(manifest)
        ),
        runner=hierarchy,
        config=HierarchicalDiscoveryConfig(max_integrated_features=20),
    )

    result = agent.execute(
        approved_wrapper_sha256=agent.precommit.approval_sha256
    )
    contracts = [
        contract.extraction_spec
        for contract in result.compiled_registry.contracts
    ]

    assert [contract["name"] for contract in contracts] == [
        "baseline_age",
        "egfr_mutation",
    ]
    assert all(contract["roles"] for contract in contracts)

    evidence_catalog = [
        atom.as_discovery_item().as_prompt_item()
        for atom in catalog.atoms
    ]
    diagnostics = [
        {
            "diagnostic_id": f"diagnostic_{index:04d}",
            "feature_name": contract["name"],
        }
        for index, contract in enumerate(contracts, start=1)
    ]
    review_generation = (
        _stage2_protocol().generation_policy.feature_proposal_review
    )
    reviewer = _FakeReviewAgent(
        SimpleNamespace(agent_max_tokens=review_generation.max_tokens),
        prompt_nontruncation_guard=prompt_guard,
        generation_parameters=review_generation,
    )
    review_response = reviewer.propose(
        {
            "review_round": 1,
            "review_attempt": 1,
            "current_contracts": contracts,
            "diagnostics": diagnostics,
            "sanitized_evidence_catalog": evidence_catalog,
        }
    )
    validated = validate_post_extraction_review_response(
        review_response,
        current_specs=contracts,
        available_diagnostic_ids=[
            row["diagnostic_id"]
            for row in diagnostics
        ],
        available_diagnostic_targets={
            row["diagnostic_id"]: {row["feature_name"]}
            for row in diagnostics
        },
        available_evidence_ids=[
            row["evidence_id"]
            for row in evidence_catalog
        ],
        available_evidence_catalog=evidence_catalog,
        max_operations=1,
    )
    revised = apply_post_extraction_review_operations(
        contracts,
        validated,
        max_contracts=20,
    )
    assert [row["action"] for row in revised.operation_audit] == [
        "revise"
    ]
    assert [spec["name"] for spec in revised.specs] == [
        "baseline_age",
        "egfr_mutation",
    ]

    adaptive_response = hierarchy._response(
        SimpleNamespace(
            job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
            messages=[
                {"role": "system", "content": "fixture"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "job": "propose_adaptive_registry_revision",
                            "current_registry": [
                                {"feature_name": "baseline_age"}
                            ],
                            "review_plan": {
                                "review_targets": [
                                    {"target": "baseline_age"}
                                ]
                            },
                            "diagnostics": diagnostics,
                        }
                    ),
                },
            ]
        )
    )
    assert adaptive_response["operations"] == [
        {
            "operation": "drop",
            "targets": ["baseline_age"],
            "proposed_feature": {},
            "supporting_evidence_ids": [],
            "diagnostic_ids": ["diagnostic_0001"],
            "reason": (
                "Drop one bounded contract to exercise gate review."
            ),
        }
    ]
    no_drop_response = hierarchy._response(
        SimpleNamespace(
            job_kind=CROSS_ARCHITECTURE_INTEGRATION_JOB,
            messages=[
                {"role": "system", "content": "fixture"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "job": "propose_adaptive_registry_revision",
                            "current_registry": [
                                {"feature_name": "egfr_mutation"}
                            ],
                            "review_plan": {
                                "review_targets": [
                                    {"target": "egfr_mutation"}
                                ]
                            },
                            "diagnostics": diagnostics,
                        }
                    ),
                },
            ],
        )
    )
    assert no_drop_response == {
        "operations": [],
        "converged": True,
    }

    routed_by_name = {
        routed.feature.canonical_name: routed
        for routed in result.completed.routed_features
    }
    adaptive_registry = tuple(
        AdaptiveCurrentFeature(
            feature_name=contract["name"],
            description=routed_by_name[
                contract["name"]
            ].feature.description,
            value_shape_hypothesis=routed_by_name[
                contract["name"]
            ].feature.value_shape_hypothesis,
            source_families=routed_by_name[
                contract["name"]
            ].feature.source_families,
            supporting_evidence_ids=routed_by_name[
                contract["name"]
            ].feature.supporting_evidence_ids,
            definition_summary=contract["description"],
        )
        for contract in contracts
    )
    adaptive_diagnostics = (
        AdaptiveDiagnostic(
            diagnostic_id="diagnostic_0001",
            diagnostic_kind="source_preservation",
            affected_features=tuple(
                contract["name"]
                for contract in contracts
            ),
            summary=(
                "Review exactly one bounded retained feature while the next "
                "gate remains sealed."
            ),
            aggregate_metrics={"observed_count": 12},
        ),
    )
    adaptive_config = AdaptiveReconsiderationConfig(
        max_atoms_per_chunk=2,
        max_bytes_per_chunk=20_000,
    )
    adaptive_builder = AdaptiveHierarchicalStage1Reconsideration(
        catalog=catalog,
        exact_spent_authentication=ExactSpentCatalogAuthentication.create(
            catalog=catalog,
            accumulated_spent_scope_sha256="a" * 64,
            accumulated_spent_row_count=12,
            consumed_gate_fingerprints=("b" * 64,),
            still_sealed_gate_fingerprint="c" * 64,
            upstream_authentication_sha256="d" * 64,
        ),
        family_explanations={
            family: f"Interpret complete evidence from {family}."
            for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
        },
        current_registry=adaptive_registry,
        diagnostics=adaptive_diagnostics,
        config=adaptive_config,
    )
    adaptive_cache = AuthenticatedHierarchicalDiscoveryJobCache(
        root=(tmp_path / "adaptive-cache").resolve(),
        config=HIERARCHY_JOB_CACHE_CONFIG,
    )
    adaptive_execution = adaptive_builder.execute_authenticated(
        runner=hierarchy,
        job_cache=adaptive_cache,
        approved_adaptive_identity=(
            adaptive_hierarchical_stage1_reconsideration_identity(
                adaptive_config
            )
        ),
        approved_runner_identity=hierarchy.identity(),
        approved_cache_identity=adaptive_cache.identity(),
        current_specs=revised.specs,
        max_contracts=20,
    )
    assert [
        spec["name"]
        for spec in adaptive_execution.executable_revision.applied.specs
    ] == ["egfr_mutation"]
    assert hierarchy.adaptive_drop_page_calls > 0
    assert hierarchy.adaptive_noop_page_calls > 0
    assert hierarchy.adaptive_proposal_calls == (
        hierarchy.adaptive_drop_page_calls
        + hierarchy.adaptive_noop_page_calls
    )


def test_authenticated_reference_only_stage2_runs_five_folds_two_reviews_and_seals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = _registry()
    registry_sha256 = _sha(registry)
    plan = build_canonical_stage1_scope_plan(
        registry=registry,
        registry_content_sha256=registry_sha256,
        global_seed=42,
        physical_fit_identity=PHYSICAL_FIT_IDENTITY,
        gpu_ids=(),
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )
    source_catalog = runner_module.build_role_neutral_evidence_catalog(
        _inputs()
    )
    payloads = _cumulative_family_payloads(source_catalog)
    payloads[HTR_NEURAL] = _native_payloads()[HTR_NEURAL]
    recorder = _NumericalProviderReadyRecorder(payloads)
    execution_root = (tmp_path / "role_neutral_execution").resolve()
    execution_manifest = execute_and_publish_role_neutral_stage1(
        root=execution_root,
        plan=plan,
        producer_factories=recorder.factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_cpu_resource_plan(),
            max_parallel_owners=2,
        ),
        executor=_RecordingExecutor(),
    )

    prepared = pd.DataFrame(
        {
            "_oci_row_id": list(
                range(registry["dataset_row_count"])
            ),
            "configured_unit": [
                f"patient-{row_id}"
                for row_id in range(registry["dataset_row_count"])
            ],
            "configured_text": [
                _fit_text(row_id)
                for row_id in range(registry["dataset_row_count"])
            ],
            "configured_treatment": [
                _fit_treatment(row_id)
                for row_id in range(registry["dataset_row_count"])
            ],
            "configured_outcome": [
                _fit_outcome(row_id)
                for row_id in range(registry["dataset_row_count"])
            ],
        }
    )
    prepared_path = (tmp_path / "prepared.parquet").resolve()
    prepared.to_parquet(prepared_path, index=False)
    publication = ReferenceOnlyRoleNeutralStage1HandoffPublisher(
        semantic_member_batch_size=3,
    )(
        target_dir=(tmp_path / "reference_handoff").resolve(),
        prepared=SimpleNamespace(
            stage1_scope_plan=plan,
            registry=registry,
                registry_content_sha256=registry_sha256,
            request_sha256="b" * 64,
            data=prepared.loc[:, ["configured_unit"]],
            options=SimpleNamespace(unit_id_column="configured_unit"),
        ),
        role_neutral_execution_root=execution_root,
        role_neutral_execution_manifest=execution_manifest,
    )

    from oci.inference.direct_upstream_numerical_reference_bank import (
        publish_role_neutral_direct_numerical_reference_bank,
    )

    bank = publish_role_neutral_direct_numerical_reference_bank(
        root=(tmp_path / "direct_numerical_bank").resolve(),
        execution_root=execution_root,
        plan=plan,
        execution_manifest=execution_manifest,
    )
    projections = bank.manifest["projections"]
    assert len(projections) == 40
    assert len({row["logical_scope_id"] for row in projections}) == 40
    deduplicated = [
        row
        for row in projections
        if row["logical_scope_id"] != row["source_transform_scope_id"]
    ]
    assert len(deduplicated) == 5
    assert all(
        row["logical_scope_id"].endswith("hierarchy_epoch_001")
        and row["source_transform_scope_id"].endswith("inner_005")
        and row["logical_and_physical_fit_rows_equal"] is True
        and row["logical_and_physical_fit_row_membership_equal"] is True
        and row["logical_and_physical_fit_row_order_equal"] is True
        and row["physical_owner_row_order_retained"] is True
        for row in deduplicated
    )
    options_root = tmp_path / "one_shot_options"
    options_root.mkdir()
    base_options = _options(options_root)
    options = replace(
        base_options,
        bundle_manifest_path=publication.bundle_manifest_path,
        output_dir=(tmp_path / "stage2_output").resolve(),
        preparation_dir=(tmp_path / "stage2_preparation").resolve(),
        attestation_dir=(tmp_path / "stage2_attestation").resolve(),
        review_rounds=2,
        initial_training_partitions=3,
        interaction_inner_folds=5,
        max_candidates=20,
        stage2_protocol=_stage2_protocol(
            post_extraction_review_min_partition_rows=2,
        ),
        prepared_cohort_path=prepared_path,
        unit_id_column="configured_unit",
        text_column="configured_text",
        treatment_column="configured_treatment",
        outcome_column="configured_outcome",
        outcome_type="binary",
        direct_numerical_bank_manifest_path=bank.manifest_path,
        upstream_review_policy=(
            GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY
        ),
    )

    forest = _FakeStrictForest()
    hierarchy_instances: list[_TwoRoundHierarchyRunner] = []

    class RecordingHierarchy(_TwoRoundHierarchyRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            hierarchy_instances.append(self)

    legacy_calls: list[str] = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            legacy_calls.append(name)
            raise AssertionError(f"forbidden legacy/refit path invoked: {name}")

        return fail

    monkeypatch.setattr(
        one_shot_module,
        "Stage2PromptNonTruncationGuard",
        _IntegrationPromptGuard,
    )
    monkeypatch.setattr(
        one_shot_module,
        "ProductionSingleEndpointJsonDiscoveryJobRunner",
        RecordingHierarchy,
    )
    monkeypatch.setattr(
        one_shot_module,
        "ProductionSingleEndpointFeatureSearchAgent",
        _FakeReviewAgent,
    )
    monkeypatch.setattr(
        one_shot_module,
        "ProductionSingleEndpointVLLMFeatureExtractor",
        _FakeCompletePagedExtractor,
    )
    monkeypatch.setattr(
        one_shot_module,
        "_configured_strict_causal_forest_backend",
        lambda _options: forest,
    )
    monkeypatch.setattr(
        runner_module,
        "load_legacy_full_outer_evidence",
        forbidden("legacy_handoff_loader"),
    )
    monkeypatch.setattr(
        runner_module,
        "load_resealed_tfidf_handoff",
        forbidden("tfidf_handoff_loader"),
    )
    for name in (
        "HistoricalStage1ContextBackend",
        "ContextFitNeuralQueryService",
        "FinalContextFitUpstreamProducer",
    ):
        monkeypatch.setattr(
            one_shot_module,
            name,
            forbidden(name),
        )

    gate_calls: list[tuple[int, ...]] = []

    def accept_gate(
        current_context,
        current_gate,
        current_specs,
        candidate_specs,
        **_kwargs,
    ):
        del current_context, current_specs, candidate_specs
        gate_calls.append(tuple(map(int, current_gate.row_ids)))
        return GateAcceptanceDecision(
            accepted=True,
            reasons=(),
            current={},
            candidate={},
            guards={},
            decision_sha256="8" * 64,
        )

    monkeypatch.setattr(
        runner_module,
        "evaluate_untouched_gate_acceptance",
        accept_gate,
    )

    result = one_shot_module.run_production_stage1_hierarchy_one_shot(
        options
    )

    assert result["status"] == "completed"
    assert result["mode"] == "reference_only_role_neutral_stage2"
    assert len(result["fold_manifest_paths"]) == 5
    assert len(result["fold_prediction_paths"]) == 5
    assert len(forest.calls) == 5
    assert len(gate_calls) == 10
    assert len(hierarchy_instances) == 1
    htr_prompt_preflight_envelope = json.loads(
        (
            options.preparation_dir
            / "htr_stage2_prompt_preflight.json"
        ).read_text(encoding="utf-8")
    )
    htr_prompt_preflight = htr_prompt_preflight_envelope["body"]
    assert (
        htr_prompt_preflight[
            "planned_htr_interpretation_call_count"
        ]
        == 20
    )
    assert htr_prompt_preflight["semantic_aggregate_count"] == 20
    assert (
        htr_prompt_preflight[
            "every_semantic_aggregate_delivered_exactly_once"
        ]
        is True
    )
    assert (
        htr_prompt_preflight["raw_token_arrays_copied_into_prompts"]
        is False
    )
    assert (
        htr_prompt_preflight[
            "endpoint_or_runner_calls_during_preflight"
        ]
        == 0
    )
    assert (
        htr_prompt_preflight["stage2_endpoint_launch_allowed"]
        is True
    )
    assert hierarchy_instances[0].adaptive_drop_page_calls > 0
    assert hierarchy_instances[0].adaptive_noop_page_calls > 0
    assert hierarchy_instances[0].adaptive_proposal_calls == (
        hierarchy_instances[0].adaptive_drop_page_calls
        + hierarchy_instances[0].adaptive_noop_page_calls
    )
    assert legacy_calls == []
    prediction = pd.read_parquet(result["prediction_path"])
    assert list(prediction.columns) == [
        "_oci_row_id",
        "outer_fold",
        "pred_ite_prob",
    ]
    assert len(prediction) == registry["dataset_row_count"]
    assert prediction["_oci_row_id"].is_unique
    assert set(prediction["outer_fold"]) == {1, 2, 3, 4, 5}
    assert np.isfinite(prediction["pred_ite_prob"]).all()
    attestation = json.loads(
        Path(result["attestation_path"]).read_text(encoding="utf-8")
    )
    assert attestation["fold_count"] == 5
    assert attestation["legacy_stage1_loader_invoked"] is False
    assert attestation["tfidf_handoff_loader_invoked"] is False
    assert attestation["independent_stage1_refit_performed"] is False
    assert attestation["structured_or_nonforest_fallback_used"] is False
    assert attestation["oracle_source_opened"] is False
    assert attestation["prepared_cohort"] == {
        "path": str(prepared_path),
        "size": prepared_path.stat().st_size,
        "sha256": hashlib.sha256(
            prepared_path.read_bytes()
        ).hexdigest(),
        "row_count": registry["dataset_row_count"],
        "text_column": "configured_text",
    }
    ledgers = attestation["complete_paged_extraction_ledgers"]
    assert ledgers
    assert [
        row["invocation_index"] for row in ledgers
    ] == list(range(len(ledgers)))
    assert all(
        [payload["kind"] for payload in row["payloads"]]
        == ["page_table", "reconciliation_table"]
        for row in ledgers
    )
    for row in ledgers:
        manifest = json.loads(
            Path(row["manifest"]["path"]).read_text(encoding="utf-8")
        )
        assert manifest["geometry"] == {
            "core_chars": options.complete_page_core_chars,
            "context_chars": options.complete_page_context_chars,
            "max_page_chars": options.complete_page_max_chars,
        }
        assert (
            manifest["planned_page_request_count"]
            == manifest["completed_page_request_count"]
        )
        assert manifest["one_feature_contract_per_page_request"] is True
        assert manifest["raw_note_copies_persisted"] is False
