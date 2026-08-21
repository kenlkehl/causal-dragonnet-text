"""Reciprocal-ColBERT organization of compiled Stage 2 evidence.

The Stage 2 evidence compiler deliberately preserves broad, auditable coverage.
Candidate discovery consumes the compiled packets directly.  This module builds
an independent, auditable evidence hierarchy that later routes named candidates
back to their most relevant compiled evidence.  It breaks card representatives
into short evidence atoms, retrieves cross-architecture neighbors, reranks
mutual neighbors with symmetric document/document ColBERT MeanMaxSim, and
clusters the reciprocal graph.  Optional later rounds encode whole communities,
compare them without the first round's cross-architecture restriction, and
coarsen them before final packet serialization.

Selection is causal-lane aware. The strongest confounder-evidence and
effect-modifier-evidence communities receive independent reserves, overlaps are
deduplicated, and remaining capacity is filled from the global ranking. No
oracle names, labels, or synthetic-data metadata participate in this process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..models.late_interaction import encode_late_interaction_documents


LOGGER = logging.getLogger(__name__)

EVIDENCE_COMMUNITY_SCHEMA_VERSION = "hierarchical_reciprocal_colbert_evidence_v2"
EVIDENCE_COMMUNITY_ARCHITECTURE = "cross_architecture_colbert_community"

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
_SHA_RE = re.compile(r'"text_sha256"\s*:\s*"([0-9a-f]{64})"')
_BOUNDARY_RE = re.compile(r"(?:\n{2,}|<new_note>|(?<=[.!?])\s+(?=[A-Z0-9]))")
_DOMAIN_STOPWORDS = {
    "assessment",
    "alt",
    "ast",
    "bpm",
    "bun",
    "cbc",
    "clinical",
    "cmp",
    "date",
    "diagnosis",
    "disease",
    "dl",
    "findings",
    "follow",
    "high",
    "laboratory",
    "low",
    "medical",
    "mg",
    "min",
    "ml",
    "mmol",
    "normal",
    "note",
    "oncology",
    "patient",
    "plan",
    "plt",
    "reference",
    "result",
    "results",
    "status",
    "study",
    "therapy",
    "treatment",
    "visit",
    "wbc",
}
_STOPWORDS = set(ENGLISH_STOP_WORDS).union(_DOMAIN_STOPWORDS)

DocumentEncoder = Callable[[Sequence[str]], Sequence[np.ndarray]]


@dataclass(frozen=True)
class Stage2EvidenceCommunityConfig:
    """Scientific configuration for one outer-fold evidence graph."""

    model_name: str = "answerdotai/answerai-colbert-small-v1"
    device: str = "cpu"
    max_communities: int = 75
    min_per_causal_lane: int = 30
    max_atom_words: int = 16
    atom_overlap_words: int = 4
    candidate_neighbors: int = 40
    reciprocal_neighbors: int = 5
    louvain_resolution: float = 2.5
    max_exemplars: int = 3
    max_consensus_phrases: int = 20
    inner_fold_saturation: int = 5
    architecture_saturation: int = 4
    # Each value requests another ColBERT community/community round whose
    # Louvain resolution is selected deterministically to approach that count.
    # Targets at or above the current community count are skipped.
    hierarchy_target_communities: tuple[int, ...] = (300, 75)

    def validate(self) -> None:
        if not str(self.model_name).strip():
            raise ValueError("evidence community ColBERT model must be nonempty")
        if not str(self.device).strip():
            raise ValueError("evidence community ColBERT device must be nonempty")
        for field_name, value in (
            ("max_communities", self.max_communities),
            ("min_per_causal_lane", self.min_per_causal_lane),
            ("max_atom_words", self.max_atom_words),
            ("atom_overlap_words", self.atom_overlap_words),
            ("candidate_neighbors", self.candidate_neighbors),
            ("reciprocal_neighbors", self.reciprocal_neighbors),
            ("max_exemplars", self.max_exemplars),
            ("max_consensus_phrases", self.max_consensus_phrases),
            ("inner_fold_saturation", self.inner_fold_saturation),
            ("architecture_saturation", self.architecture_saturation),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"evidence community {field_name} must be an integer")
        if self.max_communities < 1:
            raise ValueError("evidence community max_communities must be positive")
        if self.min_per_causal_lane < 0:
            raise ValueError("evidence community min_per_causal_lane must be nonnegative")
        if 2 * self.min_per_causal_lane > self.max_communities:
            raise ValueError(
                "twice evidence community min_per_causal_lane cannot exceed "
                "max_communities"
            )
        if self.max_atom_words < 4:
            raise ValueError("evidence community max_atom_words must be at least 4")
        if not 0 <= self.atom_overlap_words < self.max_atom_words:
            raise ValueError(
                "evidence community atom_overlap_words must be nonnegative and "
                "smaller than max_atom_words"
            )
        if self.candidate_neighbors < 1:
            raise ValueError("evidence community candidate_neighbors must be positive")
        if not 1 <= self.reciprocal_neighbors <= self.candidate_neighbors:
            raise ValueError(
                "evidence community reciprocal_neighbors must be between 1 and "
                "candidate_neighbors"
            )
        if (
            isinstance(self.louvain_resolution, bool)
            or not isinstance(self.louvain_resolution, (int, float))
            or not math.isfinite(float(self.louvain_resolution))
            or self.louvain_resolution <= 0
        ):
            raise ValueError("evidence community louvain_resolution must be positive")
        if self.max_exemplars < 1:
            raise ValueError("evidence community max_exemplars must be positive")
        if self.max_consensus_phrases < 1:
            raise ValueError("evidence community max_consensus_phrases must be positive")
        if self.inner_fold_saturation < 1:
            raise ValueError("evidence community inner_fold_saturation must be positive")
        if self.architecture_saturation < 1:
            raise ValueError("evidence community architecture_saturation must be positive")
        hierarchy_targets = tuple(self.hierarchy_target_communities)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in hierarchy_targets
        ):
            raise ValueError(
                "evidence community hierarchy_target_communities must contain "
                "positive integers"
            )
        if any(
            later >= earlier
            for earlier, later in zip(hierarchy_targets, hierarchy_targets[1:])
        ):
            raise ValueError(
                "evidence community hierarchy_target_communities must be strictly "
                "decreasing"
            )

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_COMMUNITY_SCHEMA_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True)
class DistilledStage2EvidenceCommunities:
    packets: tuple[dict[str, Any], ...]
    atoms: tuple[dict[str, Any], ...]
    communities: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]
    hierarchy_communities: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    hierarchy_edges: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    return list(
        dict.fromkeys(text for item in values if (text := str(item).strip()))
    )


def _windows(text: str, *, max_words: int, overlap_words: int) -> list[str]:
    """Split a representative at natural boundaries and then by word count."""

    clean = _normalize_space(text.replace("<new_note>", "\n\n<new_note>\n\n"))
    if len(clean.split()) <= max_words:
        return [clean] if clean else []
    segments = [
        value
        for raw in _BOUNDARY_RE.split(text)
        if (value := _normalize_space(raw)) and value != "<new_note>"
    ]
    output: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            output.append(" ".join(pending))
            pending.clear()

    for segment in segments:
        words = segment.split()
        if len(words) > max_words:
            flush()
            step = max(1, max_words - overlap_words)
            for start in range(0, len(words), step):
                window = words[start : start + max_words]
                if window:
                    output.append(" ".join(window))
                if start + max_words >= len(words):
                    break
            continue
        if pending and len(pending) + len(words) > max_words:
            flush()
        pending.extend(words)
    flush()
    return list(dict.fromkeys(value for value in output if value))


def _representative_sha(representative: Mapping[str, Any]) -> str:
    full_sha = str(representative.get("full_text_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", full_sha):
        return full_sha
    return _sha256(str(representative.get("text") or ""))


def _extract_representatives(
    packets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    representatives: list[dict[str, Any]] = []
    outer_fold: int | None = None
    packet_ids: set[str] = set()
    for packet in packets:
        packet_id = str(packet.get("packet_id") or "")
        if not packet_id or packet_id in packet_ids:
            raise ValueError("evidence community input packets require unique packet IDs")
        packet_ids.add(packet_id)
        packet_outer = int(packet["outer_fold"])
        if outer_fold is None:
            outer_fold = packet_outer
        elif outer_fold != packet_outer:
            raise ValueError("one evidence community graph cannot span outer folds")
        content = packet.get("content")
        if not isinstance(content, Mapping):
            continue
        card_support = content.get("support")
        if not isinstance(card_support, Mapping):
            card_support = {}
        raw_representatives = content.get("representative_evidence") or []
        if isinstance(raw_representatives, (str, Mapping)):
            raw_representatives = [raw_representatives]
        if not isinstance(raw_representatives, Sequence):
            continue
        for index, raw_representative in enumerate(raw_representatives):
            if isinstance(raw_representative, Mapping):
                representative = raw_representative
                text = str(representative.get("text") or "").strip()
            else:
                representative = {}
                text = str(raw_representative or "").strip()
            if not text:
                continue
            source_architectures = _string_values(
                representative.get("source_architectures")
                or content.get("source_architectures")
                or packet.get("architecture")
            )
            architecture = str(packet.get("architecture") or "").strip()
            if not architecture:
                raise ValueError(f"evidence packet {packet_id!r} has no architecture")
            representatives.append(
                {
                    "representative_id": f"{packet_id}:representative_{index:02d}",
                    "packet_id": packet_id,
                    "architecture": architecture,
                    "source_architectures": source_architectures or [architecture],
                    "source_families": _string_values(
                        representative.get("source_families")
                        or content.get("source_families")
                    ),
                    "evidence_kind": str(content.get("evidence_kind") or "evidence"),
                    "axes": _string_values(
                        representative.get("evidence_axes")
                        or content.get("evidence_axes")
                        or packet.get("observable_axes")
                    ),
                    "polarities": _string_values(
                        representative.get("polarities") or content.get("polarities")
                    ),
                    "text": text,
                    "text_sha256": _representative_sha(representative),
                    "supporting_context_count": int(
                        representative.get("supporting_context_count") or 0
                    ),
                    "card_inner_folds": sorted(
                        {int(value) for value in card_support.get("inner_folds") or []}
                    ),
                    "card_full_outer_train_support": bool(
                        card_support.get("full_outer_train_support", False)
                    ),
                }
            )
    if outer_fold is None:
        raise ValueError("evidence community input contains no packets")
    if not representatives:
        raise ValueError("evidence community input contains no readable representatives")
    return representatives, outer_fold


def _member_support(
    path: Path,
    target_hashes: set[str],
) -> tuple[dict[str, dict[str, Any]], int]:
    """Stream exact member support for only prompt-visible representatives."""

    aggregated: dict[str, dict[str, Any]] = {}
    scanned = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            scanned += 1
            match = _SHA_RE.search(line)
            if match is None or match.group(1) not in target_hashes:
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            digest = match.group(1)
            support = aggregated.setdefault(
                digest,
                {
                    "inner_folds": set(),
                    "full_outer_train_support": False,
                    "raw_occurrence_count": 0,
                    "member_ids": [],
                },
            )
            references = list(value.get("raw_references") or [])
            support["inner_folds"].update(
                int(reference["inner_fold"])
                for reference in references
                if isinstance(reference, Mapping)
                and reference.get("inner_fold") is not None
            )
            support["full_outer_train_support"] = bool(
                support["full_outer_train_support"]
                or any(
                    isinstance(reference, Mapping)
                    and str(reference.get("scope") or "") == "full_outer_train"
                    for reference in references
                )
            )
            support["raw_occurrence_count"] += len(references)
            member_id = str(value.get("member_id") or "").strip()
            if member_id and member_id not in support["member_ids"]:
                support["member_ids"].append(member_id)
    return (
        {
            digest: {
                **support,
                "inner_folds": sorted(support["inner_folds"]),
            }
            for digest, support in aggregated.items()
        },
        scanned,
    )


def _build_atoms(
    representatives: Sequence[Mapping[str, Any]],
    support_by_sha: Mapping[str, Mapping[str, Any]],
    *,
    max_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for representative in representatives:
        exact_support = support_by_sha.get(str(representative["text_sha256"]))
        if exact_support is None:
            inner_folds = list(representative["card_inner_folds"])
            full_outer = bool(representative["card_full_outer_train_support"])
            raw_occurrences = int(representative["supporting_context_count"])
            member_ids: list[str] = []
            provenance_source = "card_union_fallback"
        else:
            inner_folds = list(exact_support["inner_folds"])
            full_outer = bool(exact_support["full_outer_train_support"])
            raw_occurrences = int(exact_support["raw_occurrence_count"])
            member_ids = _string_values(exact_support.get("member_ids"))
            provenance_source = "exact_member"
        windows = _windows(
            str(representative["text"]),
            max_words=max_words,
            overlap_words=overlap_words,
        )
        for window_index, window in enumerate(windows):
            if not _WORD_RE.search(window):
                continue
            atoms.append(
                {
                    "atom_id": (
                        f"{representative['representative_id']}:atom_{window_index:02d}"
                    ),
                    "representative_id": str(representative["representative_id"]),
                    "representative_text_sha256": str(representative["text_sha256"]),
                    "member_ids": member_ids,
                    "packet_id": str(representative["packet_id"]),
                    "architecture": str(representative["architecture"]),
                    "source_architectures": list(representative["source_architectures"]),
                    "source_families": list(representative["source_families"]),
                    "evidence_kind": str(representative["evidence_kind"]),
                    "axes": list(representative["axes"]),
                    "polarities": list(representative["polarities"]),
                    "inner_folds": inner_folds,
                    "full_outer_train_support": full_outer,
                    "provenance_source": provenance_source,
                    "raw_occurrence_count": raw_occurrences,
                    "text": window,
                }
            )
    if not atoms:
        raise ValueError("evidence community construction produced no readable atoms")
    return atoms


def _encode_documents(
    texts: Sequence[str],
    *,
    config: Stage2EvidenceCommunityConfig,
    document_encoder: DocumentEncoder | None,
    label: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    LOGGER.info(
        "encode Stage 2 evidence-community %s=%s model=%s device=%s",
        label,
        len(texts),
        config.model_name,
        config.device,
    )
    raw_matrices = (
        document_encoder(texts)
        if document_encoder is not None
        else encode_late_interaction_documents(
            texts,
            config.model_name,
            config.device,
            document_chunk_overlap_tokens=min(16, config.max_atom_words),
            strip_common_framing_tokens=True,
        )
    )
    if len(raw_matrices) != len(texts):
        raise RuntimeError(
            "evidence community document encoder returned "
            f"{len(raw_matrices)} matrices for {len(texts)} {label}"
        )
    matrices: list[np.ndarray] = []
    dimensions: set[int] = set()
    for raw_matrix in raw_matrices:
        matrix = np.asarray(raw_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise RuntimeError(
                "evidence community encoder returned invalid token matrix "
                f"shape={matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError("evidence community encoder returned non-finite vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RuntimeError("evidence community encoder returned a zero token vector")
        matrix = matrix / norms
        matrices.append(matrix)
        dimensions.add(int(matrix.shape[1]))
    if len(dimensions) != 1:
        raise RuntimeError("evidence community token matrices have inconsistent dimensions")
    pooled = np.stack([matrix.mean(axis=0) for matrix in matrices]).astype(np.float32)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return matrices, pooled


def _encode_atoms(
    atoms: Sequence[Mapping[str, Any]],
    *,
    config: Stage2EvidenceCommunityConfig,
    document_encoder: DocumentEncoder | None,
) -> tuple[list[np.ndarray], np.ndarray]:
    return _encode_documents(
        [str(atom["text"]) for atom in atoms],
        config=config,
        document_encoder=document_encoder,
        label="atoms",
    )


def _pooled_neighbors(
    pooled: np.ndarray,
    architectures: Sequence[str],
    *,
    candidate_neighbors: int,
    require_cross_architecture: bool = True,
    block_size: int = 256,
) -> list[set[int]]:
    """Generate candidates from centroid-residualized pooled token vectors."""

    centered = pooled - pooled.mean(axis=0, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    count = len(centered)
    neighbors: list[set[int]] = [set() for _ in range(count)]
    architecture_array = np.asarray(list(architectures), dtype=object)
    for begin in range(0, count, block_size):
        end = min(count, begin + block_size)
        scores = centered[begin:end] @ centered.T
        for local_index, row in enumerate(scores):
            index = begin + local_index
            eligible = (
                architecture_array != architecture_array[index]
                if require_cross_architecture
                else np.ones(count, dtype=bool)
            )
            eligible[index] = False
            eligible_indexes = np.flatnonzero(eligible)
            keep = min(candidate_neighbors, len(eligible_indexes))
            if keep == 0:
                continue
            eligible_scores = row[eligible_indexes]
            selected = np.argpartition(eligible_scores, -keep)[-keep:]
            ordered = selected[
                np.lexsort((eligible_indexes[selected], -eligible_scores[selected]))
            ]
            neighbors[index] = set(map(int, eligible_indexes[ordered]))
    return neighbors


def _symmetric_meanmax(left: np.ndarray, right: np.ndarray) -> float:
    similarity = left @ right.T
    return float(
        0.5
        * (
            float(similarity.max(axis=1).mean())
            + float(similarity.max(axis=0).mean())
        )
    )


def _reciprocal_graph(
    matrices: Sequence[np.ndarray],
    pooled_neighbors: Sequence[set[int]],
    *,
    reciprocal_neighbors: int,
) -> tuple[nx.Graph, int]:
    candidate_pairs = sorted(
        (index, neighbor)
        for index, neighbors in enumerate(pooled_neighbors)
        for neighbor in neighbors
        if index < neighbor and index in pooled_neighbors[neighbor]
    )
    LOGGER.info(
        "rerank Stage 2 reciprocal-ColBERT mutual pooled pairs=%s",
        len(candidate_pairs),
    )
    pair_scores: dict[tuple[int, int], float] = {}
    scores_by_node: list[list[tuple[int, float]]] = [
        [] for _ in range(len(matrices))
    ]
    started = time.monotonic()
    for pair_index, (left, right) in enumerate(candidate_pairs, start=1):
        score = _symmetric_meanmax(matrices[left], matrices[right])
        pair_scores[(left, right)] = score
        scores_by_node[left].append((right, score))
        scores_by_node[right].append((left, score))
        if pair_index % 25_000 == 0:
            LOGGER.info(
                "reranked Stage 2 reciprocal-ColBERT pairs=%s/%s seconds=%.1f",
                pair_index,
                len(candidate_pairs),
                time.monotonic() - started,
            )

    ranks: list[dict[int, int]] = []
    for values in scores_by_node:
        ordered = sorted(values, key=lambda item: (-item[1], item[0]))[
            :reciprocal_neighbors
        ]
        ranks.append(
            {neighbor: rank for rank, (neighbor, _score) in enumerate(ordered, start=1)}
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(len(matrices)))
    if pair_scores:
        values = np.asarray(list(pair_scores.values()), dtype=np.float32)
        low, high = map(float, np.quantile(values, [0.1, 0.9]))
    else:
        low, high = 0.0, 1.0
    scale = max(high - low, 1e-6)
    for (left, right), score in pair_scores.items():
        if right not in ranks[left] or left not in ranks[right]:
            continue
        rank_strength = 1.0 - (
            (ranks[left][right] - 1 + ranks[right][left] - 1)
            / max(2.0 * reciprocal_neighbors, 1.0)
        )
        score_strength = min(1.0, max(0.0, (score - low) / scale))
        graph.add_edge(
            left,
            right,
            weight=float(0.7 * rank_strength + 0.3 * score_strength),
            colbert_score=float(score),
            left_rank=int(ranks[left][right]),
            right_rank=int(ranks[right][left]),
        )
    return graph, len(candidate_pairs)


def _cluster_graph(graph: nx.Graph, *, resolution: float, seed: int) -> list[set[int]]:
    isolates = set(nx.isolates(graph))
    connected_nodes = set(graph.nodes) - isolates
    communities: list[set[int]] = []
    if connected_nodes:
        communities.extend(
            map(
                set,
                nx.community.louvain_communities(
                    graph.subgraph(connected_nodes),
                    weight="weight",
                    resolution=resolution,
                    seed=seed,
                ),
            )
        )
    communities.extend([{node} for node in sorted(isolates)])
    return sorted(communities, key=lambda values: (min(values), len(values)))


def _cluster_graph_near_target(
    graph: nx.Graph,
    *,
    target_communities: int,
    seed: int,
) -> tuple[list[set[int]], float]:
    """Choose a deterministic Louvain resolution near a requested community count."""

    if target_communities < 1:
        raise ValueError("hierarchical community target must be positive")
    node_count = graph.number_of_nodes()
    if target_communities >= node_count:
        return [{node} for node in sorted(graph.nodes)], math.inf

    # Resolution/count is usually monotone but not guaranteed for every graph.
    # A fixed logarithmic search is deterministic and robust to small local
    # reversals while keeping the hierarchy policy independent of corpus size.
    resolutions = np.geomspace(0.025, 8.0, num=25).tolist()
    evaluated: list[tuple[tuple[Any, ...], list[set[int]], float]] = []
    for resolution in resolutions:
        partition = _cluster_graph(
            graph,
            resolution=float(resolution),
            seed=int(seed),
        )
        count = len(partition)
        key = (
            abs(count - target_communities),
            0 if count >= target_communities else 1,
            abs(math.log(float(resolution))),
            float(resolution),
        )
        evaluated.append((key, partition, float(resolution)))
    _key, partition, resolution = min(evaluated, key=lambda item: item[0])
    return partition, resolution


def _community_colbert_document(
    record: Mapping[str, Any],
    *,
    atom_text_by_id: Mapping[str, str],
) -> str:
    """Render every underlying evidence atom once for hierarchy routing."""

    texts: list[str] = []
    for atom_id in _string_values(record.get("atom_ids")):
        text = str(atom_text_by_id.get(atom_id) or "").strip()
        if text:
            texts.append(text)
    document = "\n".join(dict.fromkeys(texts))
    if not document:
        raise ValueError(
            f"evidence community {record.get('community_id')!r} has no ColBERT document"
        )
    return document


def _coarsen_community_level(
    records: Sequence[dict[str, Any]],
    *,
    atom_graph: nx.Graph,
    atoms: Sequence[Mapping[str, Any]],
    atom_index_by_id: Mapping[str, int],
    atom_text_by_id: Mapping[str, str],
    config: Stage2EvidenceCommunityConfig,
    target_communities: int,
    hierarchy_level: int,
    seed: int,
    document_encoder: DocumentEncoder | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run one unrestricted community/community ColBERT coarsening round."""

    documents = [
        _community_colbert_document(record, atom_text_by_id=atom_text_by_id)
        for record in records
    ]
    matrices, pooled = _encode_documents(
        documents,
        config=config,
        document_encoder=document_encoder,
        label=f"hierarchy_level_{hierarchy_level}_communities",
    )
    pooled_neighbors = _pooled_neighbors(
        pooled,
        [f"hierarchy_level_{hierarchy_level - 1}"] * len(records),
        candidate_neighbors=config.candidate_neighbors,
        require_cross_architecture=False,
    )
    hierarchy_graph, reranked_pairs = _reciprocal_graph(
        matrices,
        pooled_neighbors,
        reciprocal_neighbors=config.reciprocal_neighbors,
    )
    clusters, resolution = _cluster_graph_near_target(
        hierarchy_graph,
        target_communities=target_communities,
        seed=seed,
    )
    if len(clusters) >= len(records):
        return list(records), [], {
            "hierarchy_level": hierarchy_level,
            "target_communities": int(target_communities),
            "input_communities": len(records),
            "output_communities": len(records),
            "louvain_resolution": resolution,
            "pooled_mutual_pairs_reranked": reranked_pairs,
            "reciprocal_edges": hierarchy_graph.number_of_edges(),
            "status": "no_reduction",
        }

    atom_node_sets: list[set[int]] = []
    child_ids_by_temporary_id: dict[str, list[str]] = {}
    descendants_by_temporary_id: dict[str, list[str]] = {}
    for cluster_index, cluster in enumerate(clusters, start=1):
        temporary_id = f"community_{cluster_index:04d}"
        child_records = [records[index] for index in sorted(cluster)]
        child_ids_by_temporary_id[temporary_id] = [
            str(record["community_id"]) for record in child_records
        ]
        descendants_by_temporary_id[temporary_id] = sorted(
            {
                descendant
                for record in child_records
                for descendant in _string_values(
                    record.get("descendant_leaf_community_ids")
                    or record.get("community_id")
                )
            }
        )
        atom_node_sets.append(
            {
                atom_index_by_id[atom_id]
                for record in child_records
                for atom_id in _string_values(record.get("atom_ids"))
            }
        )

    coarsened = _community_records(
        atom_node_sets,
        atom_graph,
        atoms,
        config=config,
    )
    child_by_id = {str(record["community_id"]): record for record in records}
    for record in coarsened:
        temporary_id = str(record["community_id"])
        record["community_id"] = (
            f"hierarchy_{hierarchy_level:02d}_{temporary_id}"
        )
        record["hierarchy_level"] = int(hierarchy_level)
        record["child_community_ids"] = child_ids_by_temporary_id[temporary_id]
        record["descendant_leaf_community_ids"] = descendants_by_temporary_id[
            temporary_id
        ]
        record["parent_community_id"] = None
        for child_id in record["child_community_ids"]:
            child_by_id[child_id]["parent_community_id"] = str(record["community_id"])

    input_ids = [str(record["community_id"]) for record in records]
    hierarchy_edges = [
        {
            "hierarchy_level": int(hierarchy_level),
            "left_community_id": input_ids[left],
            "right_community_id": input_ids[right],
            "weight": float(data["weight"]),
            "colbert_score": float(data["colbert_score"]),
            "left_rank": int(data["left_rank"]),
            "right_rank": int(data["right_rank"]),
        }
        for left, right, data in sorted(
            hierarchy_graph.edges(data=True),
            key=lambda item: (item[0], item[1]),
        )
    ]
    return coarsened, hierarchy_edges, {
        "hierarchy_level": int(hierarchy_level),
        "target_communities": int(target_communities),
        "input_communities": len(records),
        "output_communities": len(coarsened),
        "louvain_resolution": float(resolution),
        "pooled_mutual_pairs_reranked": int(reranked_pairs),
        "reciprocal_edges": hierarchy_graph.number_of_edges(),
        "input_document_characters": sum(map(len, documents)),
        "status": "completed",
    }


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) > 1
        and not token.isdigit()
        and token.lower() not in _STOPWORDS
    ]


def _ngrams(text: str, *, maximum: int = 3) -> set[str]:
    tokens = _tokens(text)
    output: set[str] = set()
    for width in range(1, maximum + 1):
        output.update(
            " ".join(tokens[index : index + width])
            for index in range(0, len(tokens) - width + 1)
        )
    return output


def _community_consensus(
    nodes: Sequence[int],
    atoms: Sequence[Mapping[str, Any]],
    global_df: Mapping[str, int],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    phrase_atoms: Counter[str] = Counter()
    architectures: dict[str, set[str]] = defaultdict(set)
    folds: dict[str, set[int]] = defaultdict(set)
    for node in nodes:
        atom = atoms[node]
        for phrase in _ngrams(str(atom["text"])):
            phrase_atoms[phrase] += 1
            architectures[phrase].add(str(atom["architecture"]))
            folds[phrase].update(map(int, atom["inner_folds"]))
    ranked: list[tuple[float, str]] = []
    for phrase, count in phrase_atoms.items():
        architecture_count = len(architectures[phrase])
        # A phrase is prompt-visible consensus only when independently present
        # in at least two Stage 1 architectures. Same-architecture repetition
        # remains in the exemplars but cannot label the whole community.
        if architecture_count < 2:
            continue
        inverse_frequency = (
            math.log((1.0 + len(atoms)) / (1.0 + int(global_df.get(phrase, 0))))
            + 1.0
        )
        width_bonus = 1.0 + 0.25 * (len(phrase.split()) - 1)
        score = (
            inverse_frequency
            * math.log1p(count)
            * (1.0 + 0.4 * max(0, architecture_count - 1))
            * width_bonus
        )
        ranked.append((float(score), phrase))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "phrase": phrase,
            "score": score,
            "atom_count": int(phrase_atoms[phrase]),
            "architecture_count": len(architectures[phrase]),
            "inner_folds": sorted(folds[phrase]),
        }
        for score, phrase in ranked[:limit]
    ]


def _causal_lane_values(axis_counts: Mapping[str, int]) -> tuple[float, float, float, list[str]]:
    axes = {axis for axis, count in axis_counts.items() if count > 0}
    confounder = (
        1.0
        if {"treatment", "outcome"}.issubset(axes)
        else 0.55 if axes.intersection({"treatment", "outcome"}) else 0.0
    )
    modifier = (
        1.0
        if {"residual_effect", "matched_pair"}.issubset(axes)
        else 0.65 if axes.intersection({"residual_effect", "matched_pair"}) else 0.0
    )
    lanes: list[str] = []
    if confounder:
        lanes.append(
            "confounder_corroboration"
            if confounder == 1.0
            else "confounder_single_axis"
        )
    if modifier:
        lanes.append(
            "modifier_corroboration"
            if modifier == 1.0
            else "modifier_single_axis"
        )
    if not lanes:
        lanes.append("semantic_or_unclear")
    return confounder, modifier, max(confounder, modifier, 0.2), lanes


def _community_records(
    communities: Sequence[set[int]],
    graph: nx.Graph,
    atoms: Sequence[Mapping[str, Any]],
    *,
    config: Stage2EvidenceCommunityConfig,
) -> list[dict[str, Any]]:
    global_df = Counter(
        phrase for atom in atoms for phrase in _ngrams(str(atom["text"]))
    )
    records: list[dict[str, Any]] = []
    for community_index, node_set in enumerate(communities, start=1):
        nodes = sorted(node_set)
        architecture_counts = Counter(str(atoms[node]["architecture"]) for node in nodes)
        source_architectures = sorted(
            {
                architecture
                for node in nodes
                for architecture in atoms[node]["source_architectures"]
            }
        )
        source_families = sorted(
            {
                family
                for node in nodes
                for family in atoms[node]["source_families"]
            }
        )
        axis_counts = Counter(axis for node in nodes for axis in atoms[node]["axes"])
        polarity_counts = Counter(
            polarity for node in nodes for polarity in atoms[node]["polarities"]
        )
        inner_folds = sorted(
            {fold for node in nodes for fold in atoms[node]["inner_folds"]}
        )
        subgraph = graph.subgraph(nodes)
        edge_weights = [
            float(data["weight"]) for *_edge, data in subgraph.edges(data=True)
        ]
        colbert_scores = [
            float(data["colbert_score"]) for *_edge, data in subgraph.edges(data=True)
        ]
        weighted_degree = dict(subgraph.degree(weight="weight"))
        exemplar_order = sorted(
            nodes,
            key=lambda node: (
                -float(weighted_degree.get(node, 0.0)),
                -len(atoms[node]["inner_folds"]),
                -int(atoms[node]["raw_occurrence_count"]),
                node,
            ),
        )
        exemplar_limit = min(config.max_exemplars, len(nodes))
        chosen: list[int] = []
        seen_architectures: set[str] = set()
        for node in exemplar_order:
            architecture = str(atoms[node]["architecture"])
            if architecture in seen_architectures:
                continue
            chosen.append(node)
            seen_architectures.add(architecture)
            if len(chosen) == exemplar_limit:
                break
        if len(chosen) < exemplar_limit:
            chosen.extend(node for node in exemplar_order if node not in chosen)
            chosen = chosen[:exemplar_limit]

        confounder_value, modifier_value, axis_value, lanes = _causal_lane_values(
            axis_counts
        )
        representative_occurrences: dict[str, int] = {}
        for node in nodes:
            representative_occurrences[str(atoms[node]["representative_id"])] = max(
                representative_occurrences.get(
                    str(atoms[node]["representative_id"]),
                    0,
                ),
                int(atoms[node]["raw_occurrence_count"]),
            )
        records.append(
            {
                "schema_version": EVIDENCE_COMMUNITY_SCHEMA_VERSION,
                "community_id": f"community_{community_index:04d}",
                "atom_ids": [str(atoms[node]["atom_id"]) for node in nodes],
                "atom_count": len(nodes),
                "representative_count": len(
                    {str(atoms[node]["representative_id"]) for node in nodes}
                ),
                "source_packet_ids": sorted(
                    {str(atoms[node]["packet_id"]) for node in nodes}
                ),
                "architectures": dict(sorted(architecture_counts.items())),
                "source_architectures": source_architectures,
                "source_families": source_families,
                "axes": dict(sorted(axis_counts.items())),
                "polarities": dict(sorted(polarity_counts.items())),
                "inner_folds": inner_folds,
                "full_outer_train_support": any(
                    bool(atoms[node]["full_outer_train_support"]) for node in nodes
                ),
                "raw_occurrence_count": sum(representative_occurrences.values()),
                "exact_member_provenance_fraction": float(
                    sum(
                        atoms[node]["provenance_source"] == "exact_member"
                        for node in nodes
                    )
                    / len(nodes)
                ),
                "lanes": lanes,
                "confounder_axis_value": confounder_value,
                "modifier_axis_value": modifier_value,
                "causal_axis_value": axis_value,
                "internal_edges": subgraph.number_of_edges(),
                "edge_density": float(nx.density(subgraph)) if len(nodes) > 1 else 0.0,
                "mean_edge_weight": float(np.mean(edge_weights)) if edge_weights else 0.0,
                "mean_colbert_score": (
                    float(np.mean(colbert_scores)) if colbert_scores else 0.0
                ),
                "consensus_phrases": _community_consensus(
                    nodes,
                    atoms,
                    global_df,
                    limit=config.max_consensus_phrases,
                ),
                "exemplars": [
                    {
                        "atom_id": str(atoms[node]["atom_id"]),
                        "representative_id": str(atoms[node]["representative_id"]),
                        "source_packet_id": str(atoms[node]["packet_id"]),
                        "architecture": str(atoms[node]["architecture"]),
                        "source_architectures": list(
                            atoms[node]["source_architectures"]
                        ),
                        "source_families": list(atoms[node]["source_families"]),
                        "evidence_kind": str(atoms[node]["evidence_kind"]),
                        "evidence_axes": list(atoms[node]["axes"]),
                        "polarities": list(atoms[node]["polarities"]),
                        "inner_folds": list(atoms[node]["inner_folds"]),
                        "full_outer_train_support": bool(
                            atoms[node]["full_outer_train_support"]
                        ),
                        "provenance_source": str(atoms[node]["provenance_source"]),
                        "text": str(atoms[node]["text"]),
                    }
                    for node in chosen
                ],
            }
        )

    sorted_cohesion = np.sort(
        np.asarray([record["mean_edge_weight"] for record in records], dtype=np.float32)
    )
    for record in records:
        cohesion_percentile = float(
            np.searchsorted(
                sorted_cohesion,
                record["mean_edge_weight"],
                side="right",
            )
            / max(1, len(sorted_cohesion))
        )
        fold_value = min(
            1.0,
            len(record["inner_folds"]) / float(config.inner_fold_saturation),
        )
        architecture_value = min(
            1.0,
            len(record["architectures"]) / float(config.architecture_saturation),
        )
        size_value = min(1.0, math.log1p(record["atom_count"]) / math.log(9.0))
        common_score = (
            0.30 * fold_value
            + 0.25 * architecture_value
            + 0.15 * cohesion_percentile
            + 0.10 * size_value
        )
        record["score_components"] = {
            "inner_fold_coverage": fold_value,
            "architecture_diversity": architecture_value,
            "causal_axis_corroboration": float(record["causal_axis_value"]),
            "cohesion_percentile": cohesion_percentile,
            "size_support": size_value,
        }
        record["community_score"] = float(
            common_score + 0.20 * float(record["causal_axis_value"])
        )
        record["confounder_lane_score"] = (
            float(common_score + 0.20 * float(record["confounder_axis_value"]))
            if record["confounder_axis_value"]
            else None
        )
        record["modifier_lane_score"] = (
            float(common_score + 0.20 * float(record["modifier_axis_value"]))
            if record["modifier_axis_value"]
            else None
        )
    records.sort(
        key=lambda record: (
            -float(record["community_score"]),
            -len(record["inner_folds"]),
            -len(record["architectures"]),
            str(record["community_id"]),
        )
    )
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
        record["selected"] = False
        record["selection_lanes"] = []

    for lane, score_key, rank_key in (
        ("confounder", "confounder_lane_score", "confounder_lane_rank"),
        ("modifier", "modifier_lane_score", "modifier_lane_rank"),
    ):
        eligible = [record for record in records if record[score_key] is not None]
        eligible.sort(
            key=lambda record: (
                -float(record[score_key]),
                int(record["rank"]),
                str(record["community_id"]),
            )
        )
        for lane_rank, record in enumerate(eligible, start=1):
            record[rank_key] = lane_rank
        for record in records:
            record.setdefault(rank_key, None)
    return records


def _select_communities(
    records: Sequence[dict[str, Any]],
    *,
    max_communities: int,
    min_per_causal_lane: int,
) -> list[dict[str, Any]]:
    selected_ids: set[str] = set()
    for lane, score_key in (
        ("confounder_reserve", "confounder_lane_score"),
        ("modifier_reserve", "modifier_lane_score"),
    ):
        eligible = sorted(
            (record for record in records if record[score_key] is not None),
            key=lambda record: (
                -float(record[score_key]),
                int(record["rank"]),
                str(record["community_id"]),
            ),
        )[:min_per_causal_lane]
        for record in eligible:
            community_id = str(record["community_id"])
            selected_ids.add(community_id)
            record["selection_lanes"].append(lane)

    for record in records:
        if len(selected_ids) >= min(max_communities, len(records)):
            break
        community_id = str(record["community_id"])
        if community_id in selected_ids:
            continue
        selected_ids.add(community_id)
        record["selection_lanes"].append("global_fill")

    selected = []
    for record in records:
        if str(record["community_id"]) in selected_ids:
            record["selected"] = True
            selected.append(record)
    return selected


def _community_packet(
    record: Mapping[str, Any],
    *,
    outer_fold: int,
    config: Stage2EvidenceCommunityConfig,
    atom_text_by_id: Mapping[str, str],
) -> dict[str, Any]:
    consensus = list(record["consensus_phrases"])
    representative_evidence: list[dict[str, Any]] = []
    if consensus:
        representative_evidence.append(
            {
                "text": "Cross-architecture consensus phrases: "
                + "; ".join(str(item["phrase"]) for item in consensus),
                "evidence_kind": "colbert_community_consensus",
                "source_architectures": list(record["source_architectures"]),
                "evidence_axes": sorted(record["axes"]),
                "inner_folds": list(record["inner_folds"]),
            }
        )
    representative_evidence.extend(dict(exemplar) for exemplar in record["exemplars"])
    packet_id = f"outer_{outer_fold:03d}_{record['community_id']}"
    return {
        "packet_id": packet_id,
        "source": "reciprocal_colbert_evidence_community",
        "architecture": EVIDENCE_COMMUNITY_ARCHITECTURE,
        "outer_fold": int(outer_fold),
        "inner_fold": None,
        "scope": "outer_fold_colbert_distilled_training_evidence",
        "json_path": packet_id,
        "observable_axes": sorted(record["axes"]),
        "content": {
            "schema_version": EVIDENCE_COMMUNITY_SCHEMA_VERSION,
            "community_id": str(record["community_id"]),
            "hierarchy_level": int(record.get("hierarchy_level") or 0),
            "child_community_ids": _string_values(record.get("child_community_ids")),
            "descendant_leaf_community_ids": _string_values(
                record.get("descendant_leaf_community_ids")
            ),
            "community_rank": int(record["rank"]),
            "community_score": float(record["community_score"]),
            "selection_lanes": list(record["selection_lanes"]),
            "causal_lanes": list(record["lanes"]),
            "evidence_axes": sorted(record["axes"]),
            "polarities": sorted(record["polarities"]),
            "source_architectures": list(record["source_architectures"]),
            "source_families": list(record["source_families"]),
            "source_packet_ids": list(record["source_packet_ids"]),
            "support": {
                "atom_count": int(record["atom_count"]),
                "representative_count": int(record["representative_count"]),
                "source_packet_count": len(record["source_packet_ids"]),
                "raw_occurrence_count": int(record["raw_occurrence_count"]),
                "architecture_count": len(record["architectures"]),
                "inner_folds": list(record["inner_folds"]),
                "full_outer_train_support": bool(
                    record["full_outer_train_support"]
                ),
                "exact_member_provenance_fraction": float(
                    record["exact_member_provenance_fraction"]
                ),
            },
            "colbert_community": {
                "model": config.model_name,
                "internal_edges": int(record["internal_edges"]),
                "edge_density": float(record["edge_density"]),
                "mean_edge_weight": float(record["mean_edge_weight"]),
                "mean_colbert_score": float(record["mean_colbert_score"]),
                "score_components": dict(record["score_components"]),
                "confounder_lane_score": record["confounder_lane_score"],
                "modifier_lane_score": record["modifier_lane_score"],
            },
            "consensus_phrases": consensus,
            "representative_evidence": representative_evidence,
            # Candidate discovery never reads this field.  It is the lossless
            # evidence-atom document used by post-discovery ColBERT routing.
            "colbert_document": _community_colbert_document(
                record,
                atom_text_by_id=atom_text_by_id,
            ),
        },
    }


def distill_stage2_evidence_communities(
    packets: Sequence[Mapping[str, Any]],
    *,
    member_manifest_path: Path,
    config: Stage2EvidenceCommunityConfig,
    seed: int,
    document_encoder: DocumentEncoder | None = None,
) -> DistilledStage2EvidenceCommunities:
    """Build and select one outer fold's auditable evidence communities."""

    config.validate()
    representatives, outer_fold = _extract_representatives(packets)
    target_hashes = {str(item["text_sha256"]) for item in representatives}
    support_by_sha, members_scanned = _member_support(
        Path(member_manifest_path),
        target_hashes,
    )
    atoms = _build_atoms(
        representatives,
        support_by_sha,
        max_words=config.max_atom_words,
        overlap_words=config.atom_overlap_words,
    )
    LOGGER.info(
        "prepare Stage 2 evidence communities outer_fold=%s packets=%s "
        "representatives=%s exact_hashes=%s/%s atoms=%s members_scanned=%s",
        outer_fold,
        len(packets),
        len(representatives),
        len(support_by_sha),
        len(target_hashes),
        len(atoms),
        members_scanned,
    )
    matrices, pooled = _encode_atoms(
        atoms,
        config=config,
        document_encoder=document_encoder,
    )
    pooled_neighbors = _pooled_neighbors(
        pooled,
        [str(atom["architecture"]) for atom in atoms],
        candidate_neighbors=config.candidate_neighbors,
    )
    graph, reranked_pairs = _reciprocal_graph(
        matrices,
        pooled_neighbors,
        reciprocal_neighbors=config.reciprocal_neighbors,
    )
    communities = _cluster_graph(
        graph,
        resolution=config.louvain_resolution,
        seed=int(seed),
    )
    records = _community_records(
        communities,
        graph,
        atoms,
        config=config,
    )
    # Later hierarchy rounds re-encode community documents. Release the large
    # first-round token matrices before allocating those next-level matrices.
    del matrices, pooled, pooled_neighbors
    for record in records:
        record["hierarchy_level"] = 0
        record["child_community_ids"] = []
        record["descendant_leaf_community_ids"] = [str(record["community_id"])]
        record["parent_community_id"] = None

    atom_index_by_id = {
        str(atom["atom_id"]): index for index, atom in enumerate(atoms)
    }
    atom_text_by_id = {
        str(atom["atom_id"]): str(atom["text"]) for atom in atoms
    }
    final_records = records
    hierarchy_records: list[dict[str, Any]] = []
    hierarchy_edges: list[dict[str, Any]] = []
    hierarchy_summaries: list[dict[str, Any]] = []
    hierarchy_level = 0
    for requested_round, target in enumerate(
        config.hierarchy_target_communities,
        start=1,
    ):
        if int(target) >= len(final_records):
            hierarchy_summaries.append(
                {
                    "requested_round": requested_round,
                    "hierarchy_level": None,
                    "target_communities": int(target),
                    "input_communities": len(final_records),
                    "output_communities": len(final_records),
                    "status": "target_not_smaller_than_input",
                }
            )
            continue
        proposed_level = hierarchy_level + 1
        coarsened, level_edges, level_summary = _coarsen_community_level(
            final_records,
            atom_graph=graph,
            atoms=atoms,
            atom_index_by_id=atom_index_by_id,
            atom_text_by_id=atom_text_by_id,
            config=config,
            target_communities=int(target),
            hierarchy_level=proposed_level,
            seed=int(seed) + 10_000 * proposed_level,
            document_encoder=document_encoder,
        )
        level_summary["requested_round"] = requested_round
        hierarchy_summaries.append(level_summary)
        if level_summary["status"] != "completed":
            continue
        hierarchy_level = proposed_level
        hierarchy_records.extend(coarsened)
        hierarchy_edges.extend(level_edges)
        final_records = coarsened

    selected = _select_communities(
        final_records,
        max_communities=config.max_communities,
        min_per_causal_lane=config.min_per_causal_lane,
    )
    output_packets = [
        _community_packet(
            record,
            outer_fold=outer_fold,
            config=config,
            atom_text_by_id=atom_text_by_id,
        )
        for record in selected
    ]
    edges = [
        {
            "left_atom_id": str(atoms[left]["atom_id"]),
            "right_atom_id": str(atoms[right]["atom_id"]),
            "weight": float(data["weight"]),
            "colbert_score": float(data["colbert_score"]),
            "left_rank": int(data["left_rank"]),
            "right_rank": int(data["right_rank"]),
        }
        for left, right, data in sorted(
            graph.edges(data=True),
            key=lambda item: (item[0], item[1]),
        )
    ]
    selected_ids = {str(record["community_id"]) for record in selected}
    selected_atom_ids = {
        atom_id
        for record in selected
        for atom_id in record["atom_ids"]
    }
    source_chars = sum(len(str(item["text"])) for item in representatives)
    source_packet_chars = sum(len(_canonical_json(packet)) for packet in packets)
    selected_chars = sum(len(_canonical_json(packet)) for packet in output_packets)
    selected_readable_chars = sum(
        len(str(item.get("text") or ""))
        for packet in output_packets
        for item in packet["content"]["representative_evidence"]
    )
    selected_colbert_chars = sum(
        len(str(packet["content"].get("colbert_document") or ""))
        for packet in output_packets
    )
    summary = {
        "schema_version": EVIDENCE_COMMUNITY_SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "config": config.public_dict(),
        "source_packets": len(packets),
        "source_representatives": len(representatives),
        "source_representative_hashes": len(target_hashes),
        "exact_member_hashes_matched": len(support_by_sha),
        "member_manifest_records_scanned": members_scanned,
        "atoms": len(atoms),
        "pooled_mutual_pairs_reranked": reranked_pairs,
        "reciprocal_edges": graph.number_of_edges(),
        "communities": len(records),
        "hierarchy_communities": len(hierarchy_records),
        "final_communities": len(final_records),
        "final_hierarchy_level": max(
            (int(record.get("hierarchy_level") or 0) for record in final_records),
            default=0,
        ),
        "hierarchy_levels": hierarchy_summaries,
        "selected_communities": len(selected),
        "selected_atoms": len(selected_atom_ids),
        "selected_confounder_lane_communities": sum(
            "confounder_reserve" in record["selection_lanes"] for record in selected
        ),
        "selected_modifier_lane_communities": sum(
            "modifier_reserve" in record["selection_lanes"] for record in selected
        ),
        "selected_lane_overlap": sum(
            {
                "confounder_reserve",
                "modifier_reserve",
            }.issubset(record["selection_lanes"])
            for record in selected
        ),
        "selected_global_fill_communities": sum(
            "global_fill" in record["selection_lanes"] for record in selected
        ),
        "selected_full_inner_fold_coverage": sum(
            len(record["inner_folds"]) >= config.inner_fold_saturation
            for record in selected
        ),
        "source_readable_chars": source_chars,
        "source_packet_chars": source_packet_chars,
        "selected_readable_chars": selected_readable_chars,
        "selected_colbert_chars": selected_colbert_chars,
        "selected_packet_chars": selected_chars,
        "prompt_character_reduction_fraction": (
            1.0 - selected_readable_chars / source_chars if source_chars else 0.0
        ),
        "serialized_packet_reduction_fraction": (
            1.0 - selected_chars / source_packet_chars
            if source_packet_chars
            else 0.0
        ),
        "selected_community_ids": [
            str(record["community_id"])
            for record in final_records
            if str(record["community_id"]) in selected_ids
        ],
    }
    return DistilledStage2EvidenceCommunities(
        packets=tuple(output_packets),
        atoms=tuple(atoms),
        communities=tuple(records),
        edges=tuple(edges),
        summary=summary,
        hierarchy_communities=tuple(hierarchy_records),
        hierarchy_edges=tuple(hierarchy_edges),
    )


__all__ = [
    "DistilledStage2EvidenceCommunities",
    "EVIDENCE_COMMUNITY_ARCHITECTURE",
    "EVIDENCE_COMMUNITY_SCHEMA_VERSION",
    "Stage2EvidenceCommunityConfig",
    "distill_stage2_evidence_communities",
]
