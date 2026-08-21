#!/usr/bin/env python3
"""Prototype a pre-LLM reciprocal-ColBERT graph over Stage 2 evidence.

This is deliberately separate from the production Stage 2 workflow.  It reads
one sealed outer-fold packet file plus its evidence-compiler member manifest,
builds short evidence atoms, retrieves cross-architecture neighbors, reranks
them with symmetric document/document ColBERT MeanMaxSim, and clusters the
reciprocal-neighbor graph.  Oracle metadata is used only after ranking, for
evaluation; it never participates in retrieval, graph construction, clustering,
or community ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from oci.models import late_interaction
from oci.models.concept_embedding_utils import split_text_to_token_chunks


WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
SHA_RE = re.compile(r'"text_sha256"\s*:\s*"([0-9a-f]{64})"')
BOUNDARY_RE = re.compile(r"(?:\n{2,}|<new_note>|(?<=[.!?])\s+(?=[A-Z0-9]))")
DOMAIN_STOPWORDS = {
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
    "mg",
    "medical",
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
STOPWORDS = set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS)


ORACLE_PATTERNS: dict[str, tuple[str, tuple[re.Pattern[str], ...]]] = {
    "age": (
        "confounder",
        (
            re.compile(r"\bage\b", re.I),
            re.compile(r"\baged\b", re.I),
            re.compile(r"\b\d{2,3}[ -]?(?:year|yr)s?[ -]old\b", re.I),
            re.compile(r"\b\d{2,3}[ -]?y/?o\b", re.I),
        ),
    ),
    "sex": (
        "confounder",
        (
            re.compile(r"\bsex\b", re.I),
            re.compile(r"\bgender\b", re.I),
            re.compile(r"\b(?:male|female|man|woman)\b", re.I),
        ),
    ),
    "ecog_performance_status": (
        "confounder",
        (
            re.compile(r"\becog\b", re.I),
            re.compile(r"\bperformance status\b", re.I),
            re.compile(r"\bkarnofsky\b", re.I),
        ),
    ),
    "creatinine_clearance": (
        "confounder",
        (
            re.compile(r"\bcreatinine clearance\b", re.I),
            re.compile(r"\bcrcl\b", re.I),
            re.compile(r"\bcockcroft(?:[ -]gault)?\b", re.I),
        ),
    ),
    "prior_platinum_therapy": (
        "confounder",
        (
            re.compile(
                r"\b(?:prior|previous|adjuvant|history|completed|received)\b"
                r".{0,100}\b(?:platinum|cisplatin|carboplatin)\b",
                re.I | re.S,
            ),
            re.compile(
                r"\b(?:platinum|cisplatin|carboplatin)\b"
                r".{0,100}\b(?:prior|previous|adjuvant|history|completed|received)\b",
                re.I | re.S,
            ),
        ),
    ),
    "histology_type": (
        "effect_modifier",
        (
            re.compile(r"\bhistolog", re.I),
            re.compile(r"\badenocarcinoma\b", re.I),
            re.compile(r"\bsquamous(?: cell)?\b", re.I),
            re.compile(r"\blarge[ -]cell carcinoma\b", re.I),
        ),
    ),
    "egfr_mutation_status": (
        "effect_modifier",
        (
            # Context is required because renal-function ``eGFR`` otherwise
            # aliases to the same lower-cased token as the EGFR cancer gene.
            re.compile(
                r"\begfr\b.{0,60}\b(?:mutation|mutant|wild|exon|positive|negative|"
                r"detected|alteration|status)\b",
                re.I | re.S,
            ),
            re.compile(
                r"\b(?:mutation|mutant|wild|exon|positive|negative|detected|"
                r"alteration|status)\b.{0,60}\begfr\b",
                re.I | re.S,
            ),
            re.compile(
                r"\bepidermal growth factor receptor\b.{0,60}\b(?:mutation|"
                r"mutant|wild|positive|negative|status)\b",
                re.I | re.S,
            ),
        ),
    ),
    "baseline_nlr": (
        "effect_modifier",
        (
            re.compile(r"\bnlr\b", re.I),
            re.compile(r"\bneutrophil(?:[ -]to[ -]|/+)lymphocyte ratio\b", re.I),
        ),
    ),
    "brain_metastases_status": (
        "effect_modifier",
        (
            re.compile(r"\bbrain metast", re.I),
            re.compile(r"\bcerebral metast", re.I),
            re.compile(r"\bintracranial metast", re.I),
            re.compile(r"\bmetast\w*.{0,40}\b(?:brain|cerebral|intracranial)\b", re.I),
        ),
    ),
    "baseline_hemoglobin": (
        "effect_modifier",
        (
            re.compile(r"\bhemoglobin\b", re.I),
            re.compile(r"\bhaemoglobin\b", re.I),
            re.compile(r"\bhgb\b", re.I),
        ),
    ),
}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _windows(text: str, *, max_words: int, overlap_words: int) -> list[str]:
    """Split long representatives at natural boundaries, then by word count."""

    clean = _normalize_space(text.replace("<new_note>", "\n\n<new_note>\n\n"))
    words = clean.split()
    if len(words) <= max_words:
        return [clean] if clean else []

    raw_segments = [_normalize_space(value) for value in BOUNDARY_RE.split(text)]
    segments = [value for value in raw_segments if value and value != "<new_note>"]
    output: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            output.append(" ".join(pending))
            pending.clear()

    for segment in segments:
        segment_words = segment.split()
        if len(segment_words) > max_words:
            flush()
            step = max(1, max_words - overlap_words)
            for start in range(0, len(segment_words), step):
                window = segment_words[start : start + max_words]
                if window:
                    output.append(" ".join(window))
                if start + max_words >= len(segment_words):
                    break
            continue
        if pending and len(pending) + len(segment_words) > max_words:
            flush()
        pending.extend(segment_words)
    flush()
    return list(dict.fromkeys(value for value in output if value))


def _representative_sha(representative: Mapping[str, Any]) -> str:
    full_sha = str(representative.get("full_text_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", full_sha):
        return full_sha
    return _sha256(str(representative.get("text") or ""))


def _member_support(
    path: Path,
    target_hashes: set[str],
) -> dict[str, dict[str, Any]]:
    """Recover only target representatives from a potentially very large manifest."""

    found: dict[str, dict[str, Any]] = {}
    scanned = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            match = SHA_RE.search(line)
            if match is None or match.group(1) not in target_hashes:
                continue
            value = json.loads(line)
            references = list(value.get("raw_references") or [])
            found[match.group(1)] = {
                "inner_folds": sorted(
                    {
                        int(reference["inner_fold"])
                        for reference in references
                        if reference.get("inner_fold") is not None
                    }
                ),
                "full_outer_train_support": any(
                    str(reference.get("scope") or "") == "full_outer_train"
                    for reference in references
                ),
                "raw_occurrence_count": len(references),
                "member_id": str(value.get("member_id") or ""),
            }
    print(
        f"member provenance: matched {len(found)}/{len(target_hashes)} "
        f"representative hashes while scanning {scanned} members",
        flush=True,
    )
    return found


def _extract_representatives(packets_path: Path) -> tuple[list[dict[str, Any]], int]:
    representatives: list[dict[str, Any]] = []
    outer_fold: int | None = None
    for packet in _jsonl(packets_path):
        packet_outer = int(packet["outer_fold"])
        if outer_fold is None:
            outer_fold = packet_outer
        elif outer_fold != packet_outer:
            raise ValueError(f"{packets_path} contains more than one outer fold")
        content = dict(packet.get("content") or {})
        card_support = dict(content.get("support") or {})
        for index, representative in enumerate(content.get("representative_evidence") or []):
            text = str(representative.get("text") or "").strip()
            if not text:
                continue
            representatives.append(
                {
                    "representative_id": f"{packet['packet_id']}:rep_{index:02d}",
                    "packet_id": str(packet["packet_id"]),
                    "architecture": str(packet["architecture"]),
                    "source_architectures": sorted(
                        set(
                            map(
                                str,
                                representative.get("source_architectures")
                                or content.get("source_architectures")
                                or [packet["architecture"]],
                            )
                        )
                    ),
                    "axes": sorted(
                        set(
                            map(
                                str,
                                representative.get("evidence_axes")
                                or content.get("evidence_axes")
                                or packet.get("observable_axes")
                                or [],
                            )
                        )
                    ),
                    "polarities": sorted(
                        set(
                            map(
                                str,
                                representative.get("polarities")
                                or content.get("polarities")
                                or [],
                            )
                        )
                    ),
                    "text": text,
                    "text_sha256": _representative_sha(representative),
                    "supporting_context_count": int(
                        representative.get("supporting_context_count") or 0
                    ),
                    "card_inner_folds": sorted(
                        map(int, card_support.get("inner_folds") or [])
                    ),
                    "card_full_outer_train_support": bool(
                        card_support.get("full_outer_train_support", False)
                    ),
                    "card_exact_member_count": int(
                        card_support.get("exact_member_count") or 0
                    ),
                }
            )
    if outer_fold is None:
        raise ValueError(f"{packets_path} contains no packets")
    return representatives, outer_fold


def _build_atoms(
    representatives: Sequence[Mapping[str, Any]],
    support_by_sha: Mapping[str, Mapping[str, Any]],
    *,
    max_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for representative in representatives:
        sha = str(representative["text_sha256"])
        exact_support = support_by_sha.get(sha)
        if exact_support is None:
            inner_folds = list(representative["card_inner_folds"])
            full_outer = bool(representative["card_full_outer_train_support"])
            provenance_source = "card_union_fallback"
            raw_occurrences = int(representative["supporting_context_count"])
        else:
            inner_folds = list(exact_support["inner_folds"])
            full_outer = bool(exact_support["full_outer_train_support"])
            provenance_source = "exact_member"
            raw_occurrences = int(exact_support["raw_occurrence_count"])
        for window_index, window in enumerate(
            _windows(
                str(representative["text"]),
                max_words=max_words,
                overlap_words=overlap_words,
            )
        ):
            if not WORD_RE.search(window):
                continue
            atoms.append(
                {
                    "atom_id": f"{representative['representative_id']}:win_{window_index:02d}",
                    "representative_id": str(representative["representative_id"]),
                    "packet_id": str(representative["packet_id"]),
                    "architecture": str(representative["architecture"]),
                    "source_architectures": list(representative["source_architectures"]),
                    "axes": list(representative["axes"]),
                    "polarities": list(representative["polarities"]),
                    "inner_folds": inner_folds,
                    "full_outer_train_support": full_outer,
                    "provenance_source": provenance_source,
                    "raw_occurrence_count": raw_occurrences,
                    "text": window,
                }
            )
    return atoms


def _encode_atoms(
    atoms: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    device: str,
) -> tuple[list[np.ndarray], np.ndarray, str]:
    encoder = late_interaction._load_encoder(model_name, device)
    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    overlap = min(16, max(0, int(encoder.document_length) - 3))
    prefix = str(getattr(encoder, "document_encoding_prefix", ""))
    for atom in atoms:
        atom_chunks = split_text_to_token_chunks(
            str(atom["text"]),
            encoder.tokenizer,
            max_seq_length=int(encoder.document_length),
            chunk_overlap_tokens=overlap,
            encoding_prefix=prefix,
        )
        begin = len(chunks)
        chunks.extend(atom_chunks)
        spans.append((begin, len(chunks)))
    print(
        f"encoding {len(atoms)} evidence atoms as {len(chunks)} ColBERT chunks "
        f"on {device}",
        flush=True,
    )
    chunk_matrices = encoder.encode_documents(chunks)
    compatibility_adapter = type(encoder).__name__ == "_StanfordColbertCompatibilityAdapter"
    if compatibility_adapter:
        # Compatibility documents are [CLS] [unused1] content [SEP].  Those
        # common framing tokens otherwise dominate short document/document
        # similarities; production query/document scoring does not have this
        # symmetric-short-document failure mode.
        stripped: list[np.ndarray] = []
        for matrix in chunk_matrices:
            matrix = np.asarray(matrix, dtype=np.float32)
            stripped.append(matrix[2:-1] if len(matrix) > 3 else matrix)
        chunk_matrices = stripped
    matrices = [
        np.concatenate(chunk_matrices[begin:end], axis=0).astype(np.float32)
        for begin, end in spans
    ]
    pooled = np.stack([matrix.mean(axis=0) for matrix in matrices]).astype(np.float32)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return matrices, pooled, type(encoder).__name__


def _pooled_neighbors(
    pooled: np.ndarray,
    architectures: Sequence[str],
    *,
    candidate_k: int,
    block_size: int = 256,
) -> list[set[int]]:
    # Subtract the domain centroid for candidate generation only.  This makes
    # generic NSCLC language less influential without modifying ColBERT scores.
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
            eligible = architecture_array != architecture_array[index]
            eligible[index] = False
            eligible_indexes = np.flatnonzero(eligible)
            keep = min(candidate_k, len(eligible_indexes))
            if keep == 0:
                continue
            eligible_scores = row[eligible_indexes]
            selected = np.argpartition(eligible_scores, -keep)[-keep:]
            ordered = selected[np.argsort(eligible_scores[selected])[::-1]]
            neighbors[index] = set(map(int, eligible_indexes[ordered]))
    return neighbors


def _symmetric_meanmax(left: np.ndarray, right: np.ndarray) -> float:
    similarity = np.asarray(left, dtype=np.float32) @ np.asarray(right, dtype=np.float32).T
    return float(0.5 * (similarity.max(axis=1).mean() + similarity.max(axis=0).mean()))


def _reciprocal_graph(
    matrices: Sequence[np.ndarray],
    pooled_neighbors: Sequence[set[int]],
    *,
    reciprocal_k: int,
) -> tuple[nx.Graph, dict[tuple[int, int], float], int]:
    candidate_pairs = sorted(
        (index, neighbor)
        for index, neighbors in enumerate(pooled_neighbors)
        for neighbor in neighbors
        if index < neighbor and index in pooled_neighbors[neighbor]
    )
    print(
        f"ColBERT reranking {len(candidate_pairs)} mutual pooled-neighbor pairs",
        flush=True,
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
        if pair_index % 5000 == 0:
            elapsed = time.monotonic() - started
            print(
                f"  scored {pair_index}/{len(candidate_pairs)} pairs in {elapsed:.1f}s",
                flush=True,
            )

    final_neighbors: list[list[int]] = []
    ranks: list[dict[int, int]] = []
    for values in scores_by_node:
        ordered = sorted(values, key=lambda item: (-item[1], item[0]))[:reciprocal_k]
        final_neighbors.append([neighbor for neighbor, _score in ordered])
        ranks.append({neighbor: rank for rank, neighbor in enumerate(final_neighbors[-1], 1)})

    graph = nx.Graph()
    graph.add_nodes_from(range(len(matrices)))
    if pair_scores:
        all_scores = np.asarray(list(pair_scores.values()), dtype=np.float32)
        low, high = map(float, np.quantile(all_scores, [0.1, 0.9]))
    else:
        low, high = 0.0, 1.0
    scale = max(high - low, 1e-6)
    for (left, right), score in pair_scores.items():
        if right not in ranks[left] or left not in ranks[right]:
            continue
        rank_strength = 1.0 - (
            (ranks[left][right] - 1 + ranks[right][left] - 1)
            / max(2.0 * reciprocal_k, 1.0)
        )
        score_strength = min(1.0, max(0.0, (score - low) / scale))
        graph.add_edge(
            left,
            right,
            weight=0.7 * rank_strength + 0.3 * score_strength,
            colbert_score=score,
            left_rank=ranks[left][right],
            right_rank=ranks[right][left],
        )
    return graph, pair_scores, len(candidate_pairs)


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


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 1
        and not token.isdigit()
        and token.lower() not in STOPWORDS
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


def _global_ngram_df(atoms: Sequence[Mapping[str, Any]]) -> Counter[str]:
    values: Counter[str] = Counter()
    for atom in atoms:
        values.update(_ngrams(str(atom["text"])))
    return values


def _community_consensus(
    nodes: Sequence[int],
    atoms: Sequence[Mapping[str, Any]],
    global_df: Mapping[str, int],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    atom_counts: Counter[str] = Counter()
    architectures: dict[str, set[str]] = defaultdict(set)
    folds: dict[str, set[int]] = defaultdict(set)
    for node in nodes:
        atom = atoms[node]
        for phrase in _ngrams(str(atom["text"])):
            atom_counts[phrase] += 1
            architectures[phrase].add(str(atom["architecture"]))
            folds[phrase].update(map(int, atom["inner_folds"]))
    total_atoms = len(atoms)
    ranked: list[tuple[float, str]] = []
    for phrase, count in atom_counts.items():
        architecture_count = len(architectures[phrase])
        if count < 2 and architecture_count < 2:
            continue
        document_frequency = int(global_df.get(phrase, 0))
        inverse_frequency = math.log((1.0 + total_atoms) / (1.0 + document_frequency)) + 1.0
        width_bonus = 1.0 + 0.25 * (len(phrase.split()) - 1)
        score = (
            inverse_frequency
            * math.log1p(count)
            * (1.0 + 0.4 * max(0, architecture_count - 1))
            * width_bonus
        )
        ranked.append((score, phrase))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "phrase": phrase,
            "score": score,
            "atom_count": int(atom_counts[phrase]),
            "architecture_count": len(architectures[phrase]),
            "inner_folds": sorted(folds[phrase]),
        }
        for score, phrase in ranked[:limit]
    ]


def _axis_value(axis_counts: Mapping[str, int]) -> tuple[float, list[str]]:
    axes = {axis for axis, count in axis_counts.items() if count > 0}
    lanes: list[str] = []
    values: list[float] = []
    if {"treatment", "outcome"}.issubset(axes):
        lanes.append("confounder_corroboration")
        values.append(1.0)
    elif axes.intersection({"treatment", "outcome"}):
        lanes.append("confounder_single_axis")
        values.append(0.55)
    if {"residual_effect", "matched_pair"}.issubset(axes):
        lanes.append("modifier_corroboration")
        values.append(1.0)
    elif axes.intersection({"residual_effect", "matched_pair"}):
        lanes.append("modifier_single_axis")
        values.append(0.65)
    if not values:
        lanes.append("semantic_or_unclear")
        values.append(0.2)
    return max(values), lanes


def _matches_oracle(name: str, text: str) -> bool:
    return any(pattern.search(text) for pattern in ORACLE_PATTERNS[name][1])


def _community_records(
    communities: Sequence[set[int]],
    graph: nx.Graph,
    atoms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    global_df = _global_ngram_df(atoms)
    records: list[dict[str, Any]] = []
    for community_index, node_set in enumerate(communities, start=1):
        nodes = sorted(node_set)
        architecture_counts = Counter(str(atoms[node]["architecture"]) for node in nodes)
        axis_counts = Counter(axis for node in nodes for axis in atoms[node]["axes"])
        folds = sorted({fold for node in nodes for fold in atoms[node]["inner_folds"]})
        subgraph = graph.subgraph(nodes)
        edge_weights = [float(data["weight"]) for *_edge, data in subgraph.edges(data=True)]
        colbert_scores = [
            float(data["colbert_score"]) for *_edge, data in subgraph.edges(data=True)
        ]
        weighted_degree = dict(subgraph.degree(weight="weight"))
        exemplar_nodes = sorted(
            nodes,
            key=lambda node: (
                -float(weighted_degree.get(node, 0.0)),
                -len(atoms[node]["inner_folds"]),
                -int(atoms[node]["raw_occurrence_count"]),
                node,
            ),
        )
        chosen: list[int] = []
        seen_architectures: set[str] = set()
        for node in exemplar_nodes:
            architecture = str(atoms[node]["architecture"])
            if architecture in seen_architectures and len(chosen) < min(3, len(nodes)):
                continue
            chosen.append(node)
            seen_architectures.add(architecture)
            if len(chosen) == min(6, len(nodes)):
                break
        if len(chosen) < min(6, len(nodes)):
            chosen.extend(
                node
                for node in exemplar_nodes
                if node not in chosen
            )
            chosen = chosen[: min(6, len(nodes))]
        axis_value, lanes = _axis_value(axis_counts)
        consensus_phrases = _community_consensus(nodes, atoms, global_df, limit=20)
        oracle_hits: dict[str, dict[str, Any]] = {}
        consensus_oracle_hits: dict[str, dict[str, Any]] = {}
        for oracle_name, (role, _patterns) in ORACLE_PATTERNS.items():
            hit_nodes = [
                node for node in nodes if _matches_oracle(oracle_name, str(atoms[node]["text"]))
            ]
            if not hit_nodes:
                continue
            hit_architectures = sorted(
                {str(atoms[node]["architecture"]) for node in hit_nodes}
            )
            hit_folds = sorted(
                {fold for node in hit_nodes for fold in atoms[node]["inner_folds"]}
            )
            oracle_hits[oracle_name] = {
                "role": role,
                "atom_count": len(hit_nodes),
                "architectures": hit_architectures,
                "inner_folds": hit_folds,
                "corroborated": (
                    len(hit_nodes) >= 2
                    and len(hit_architectures) >= 2
                    and len(hit_folds) >= 2
                ),
                "example": str(atoms[hit_nodes[0]]["text"]),
            }
            matched_phrases = [
                phrase
                for phrase in consensus_phrases
                if _matches_oracle(oracle_name, str(phrase["phrase"]))
            ]
            if matched_phrases:
                best_phrase = matched_phrases[0]
                consensus_oracle_hits[oracle_name] = {
                    "role": role,
                    "phrase": str(best_phrase["phrase"]),
                    "architecture_count": int(best_phrase["architecture_count"]),
                    "inner_folds": list(best_phrase["inner_folds"]),
                    "distilled": (
                        int(best_phrase["architecture_count"]) >= 2
                        and len(best_phrase["inner_folds"]) >= 2
                    ),
                }
        records.append(
            {
                "community_id": f"community_{community_index:04d}",
                "nodes": nodes,
                "atom_count": len(nodes),
                "architectures": dict(sorted(architecture_counts.items())),
                "axes": dict(sorted(axis_counts.items())),
                "inner_folds": folds,
                "exact_member_provenance_fraction": sum(
                    atoms[node]["provenance_source"] == "exact_member" for node in nodes
                )
                / len(nodes),
                "lanes": lanes,
                "axis_value": axis_value,
                "internal_edges": subgraph.number_of_edges(),
                "edge_density": nx.density(subgraph) if len(nodes) > 1 else 0.0,
                "mean_edge_weight": float(np.mean(edge_weights)) if edge_weights else 0.0,
                "mean_colbert_score": (
                    float(np.mean(colbert_scores)) if colbert_scores else 0.0
                ),
                "consensus_phrases": consensus_phrases,
                "exemplars": [
                    {
                        "atom_id": str(atoms[node]["atom_id"]),
                        "architecture": str(atoms[node]["architecture"]),
                        "axes": list(atoms[node]["axes"]),
                        "inner_folds": list(atoms[node]["inner_folds"]),
                        "text": str(atoms[node]["text"]),
                    }
                    for node in chosen
                ],
                "oracle_hits": oracle_hits,
                "consensus_oracle_hits": consensus_oracle_hits,
            }
        )

    cohesion_values = np.asarray(
        [record["mean_edge_weight"] for record in records], dtype=np.float32
    )
    sorted_cohesion = np.sort(cohesion_values)
    for record in records:
        cohesion_percentile = float(
            np.searchsorted(sorted_cohesion, record["mean_edge_weight"], side="right")
            / max(1, len(sorted_cohesion))
        )
        fold_value = min(1.0, len(record["inner_folds"]) / 5.0)
        architecture_value = min(1.0, len(record["architectures"]) / 4.0)
        size_value = min(1.0, math.log1p(record["atom_count"]) / math.log(9.0))
        record["score_components"] = {
            "inner_fold_coverage": fold_value,
            "architecture_diversity": architecture_value,
            "causal_axis_corroboration": float(record["axis_value"]),
            "cohesion_percentile": cohesion_percentile,
            "size_support": size_value,
        }
        record["community_score"] = (
            0.30 * fold_value
            + 0.25 * architecture_value
            + 0.20 * float(record["axis_value"])
            + 0.15 * cohesion_percentile
            + 0.10 * size_value
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
    return records


def _oracle_evaluation(
    records: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    per_oracle: dict[str, Any] = {}
    for oracle_name, (role, _patterns) in ORACLE_PATTERNS.items():
        matches = [
            record
            for record in records
            if oracle_name in record.get("oracle_hits", {})
        ]
        corroborated = [
            record
            for record in matches
            if record["oracle_hits"][oracle_name]["corroborated"]
        ]
        consensus_matches = [
            record
            for record in records
            if oracle_name in record.get("consensus_oracle_hits", {})
        ]
        distilled = [
            record
            for record in consensus_matches
            if record["consensus_oracle_hits"][oracle_name]["distilled"]
        ]
        best = (
            distilled[0]
            if distilled
            else (
                consensus_matches[0]
                if consensus_matches
                else (corroborated[0] if corroborated else (matches[0] if matches else None))
            )
        )
        per_oracle[oracle_name] = {
            "role": role,
            "first_any_rank": int(matches[0]["rank"]) if matches else None,
            "first_corroborated_rank": (
                int(corroborated[0]["rank"]) if corroborated else None
            ),
            "first_consensus_rank": (
                int(consensus_matches[0]["rank"]) if consensus_matches else None
            ),
            "first_distilled_rank": int(distilled[0]["rank"]) if distilled else None,
            "best_community_id": best["community_id"] if best else None,
            "best_consensus_phrase": (
                best.get("consensus_oracle_hits", {}).get(oracle_name, {}).get("phrase")
                if best
                else None
            ),
            "best_evidence": (
                best.get("oracle_hits", {}).get(oracle_name, {}).get("example")
                if best
                else None
            ),
            "best_architectures": (
                best.get("oracle_hits", {}).get(oracle_name, {}).get("architectures", [])
                if best
                else []
            ),
            "best_inner_folds": (
                best.get("oracle_hits", {}).get(oracle_name, {}).get("inner_folds", [])
                if best
                else []
            ),
        }
    recall_at: dict[str, Any] = {}
    for cutoff in cutoffs:
        first = list(records[:cutoff])
        any_names = {
            name for record in first for name in record.get("oracle_hits", {})
        }
        corroborated_names = {
            name
            for record in first
            for name, hit in record.get("oracle_hits", {}).items()
            if hit["corroborated"]
        }
        consensus_names = {
            name
            for record in first
            for name in record.get("consensus_oracle_hits", {})
        }
        distilled_names = {
            name
            for record in first
            for name, hit in record.get("consensus_oracle_hits", {}).items()
            if hit["distilled"]
        }
        recall_at[str(cutoff)] = {
            "communities_available": len(first),
            "any": {
                "count": len(any_names),
                "total": len(ORACLE_PATTERNS),
                "oracles": sorted(any_names),
            },
            "corroborated": {
                "count": len(corroborated_names),
                "total": len(ORACLE_PATTERNS),
                "oracles": sorted(corroborated_names),
            },
            "consensus": {
                "count": len(consensus_names),
                "total": len(ORACLE_PATTERNS),
                "oracles": sorted(consensus_names),
            },
            "distilled": {
                "count": len(distilled_names),
                "total": len(ORACLE_PATTERNS),
                "oracles": sorted(distilled_names),
            },
        }
    return {"recall_at": recall_at, "per_oracle": per_oracle}


def _rank_consensus_phrases(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_per_community: int = 2,
) -> list[dict[str, Any]]:
    """Create a global, redundancy-suppressed registry of community phrases."""

    candidates: list[dict[str, Any]] = []
    for record in records:
        components = dict(record["score_components"])
        for phrase in record["consensus_phrases"]:
            if int(phrase["architecture_count"]) < 2 or len(phrase["inner_folds"]) < 2:
                continue
            candidates.append(
                {
                    "phrase": str(phrase["phrase"]),
                    "community_id": str(record["community_id"]),
                    "community_rank": int(record["rank"]),
                    "phrase_score": float(phrase["score"]),
                    "atom_count": int(phrase["atom_count"]),
                    "architecture_count": int(phrase["architecture_count"]),
                    "inner_folds": list(phrase["inner_folds"]),
                    "lanes": list(record["lanes"]),
                    "axis_value": float(record["axis_value"]),
                    "cohesion_percentile": float(components["cohesion_percentile"]),
                    "tokens": set(str(phrase["phrase"]).split()),
                }
            )
    if not candidates:
        return []
    phrase_scores = np.sort(
        np.asarray([candidate["phrase_score"] for candidate in candidates], dtype=np.float32)
    )
    for candidate in candidates:
        salience = float(
            np.searchsorted(phrase_scores, candidate["phrase_score"], side="right")
            / len(phrase_scores)
        )
        fold_value = min(1.0, len(candidate["inner_folds"]) / 5.0)
        architecture_value = min(1.0, candidate["architecture_count"] / 4.0)
        candidate["registry_score"] = (
            0.30 * fold_value
            + 0.30 * architecture_value
            + 0.20 * salience
            + 0.10 * candidate["axis_value"]
            + 0.10 * candidate["cohesion_percentile"]
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate["registry_score"],
            -candidate["phrase_score"],
            candidate["phrase"],
            candidate["community_id"],
        )
    )

    selected: list[dict[str, Any]] = []
    per_community: Counter[str] = Counter()
    for candidate in candidates:
        if per_community[candidate["community_id"]] >= maximum_per_community:
            continue
        redundant = False
        for prior in selected:
            intersection = len(candidate["tokens"].intersection(prior["tokens"]))
            union = len(candidate["tokens"].union(prior["tokens"]))
            jaccard = intersection / max(1, union)
            containment = intersection / max(1, min(len(candidate["tokens"]), len(prior["tokens"])))
            if jaccard >= 0.67 or containment >= 1.0:
                redundant = True
                break
        if redundant:
            continue
        selected.append(candidate)
        per_community[candidate["community_id"]] += 1
    output: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected, start=1):
        public = {key: value for key, value in candidate.items() if key != "tokens"}
        public["rank"] = rank
        output.append(public)
    return output


def _phrase_oracle_evaluation(
    phrases: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    per_oracle: dict[str, Any] = {}
    for oracle_name, (role, _patterns) in ORACLE_PATTERNS.items():
        matches = [
            phrase
            for phrase in phrases
            if _matches_oracle(oracle_name, str(phrase["phrase"]))
        ]
        per_oracle[oracle_name] = {
            "role": role,
            "first_rank": int(matches[0]["rank"]) if matches else None,
            "phrase": str(matches[0]["phrase"]) if matches else None,
            "community_id": str(matches[0]["community_id"]) if matches else None,
            "architecture_count": (
                int(matches[0]["architecture_count"]) if matches else None
            ),
            "inner_folds": list(matches[0]["inner_folds"]) if matches else [],
        }
    recall_at: dict[str, Any] = {}
    for cutoff in cutoffs:
        names = {
            oracle_name
            for phrase in phrases[:cutoff]
            for oracle_name in ORACLE_PATTERNS
            if _matches_oracle(oracle_name, str(phrase["phrase"]))
        }
        recall_at[str(cutoff)] = {
            "count": len(names),
            "total": len(ORACLE_PATTERNS),
            "oracles": sorted(names),
        }
    return {"recall_at": recall_at, "per_oracle": per_oracle}


def _metadata_oracle_names(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = value.get("features")
    if not isinstance(raw, list):
        raw = [
            *list(value.get("confounders") or []),
            *list(value.get("effect_modifiers") or []),
        ]
    return {
        str(feature.get("name") or "")
        for feature in raw
        if isinstance(feature, Mapping) and feature.get("name")
    }


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    phrases: Sequence[Mapping[str, Any]],
    phrase_evaluation: Mapping[str, Any],
) -> None:
    lines = [
        "# Reciprocal-ColBERT evidence graph experiment",
        "",
        f"- Outer fold: {summary['outer_fold']}",
        f"- Packets: {summary['packet_count']}",
        f"- Representatives: {summary['representative_count']}",
        f"- Windowed evidence atoms: {summary['atom_count']}",
        f"- Reciprocal graph edges: {summary['graph_edges']}",
        f"- Communities: {summary['community_count']}",
        f"- Exact member provenance: {summary['exact_member_provenance_atoms']}/{summary['atom_count']}",
        "",
        "## Oracle recall (oracle used only for evaluation)",
        "",
        "| Communities | Any lexical evidence | Cross-architecture corroborated | Consensus phrase | Cross-architecture distilled phrase |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cutoff, values in evaluation["recall_at"].items():
        lines.append(
            f"| {cutoff} | {values['any']['count']}/{values['any']['total']} | "
            f"{values['corroborated']['count']}/{values['corroborated']['total']} | "
            f"{values['consensus']['count']}/{values['consensus']['total']} | "
            f"{values['distilled']['count']}/{values['distilled']['total']} |"
        )
    lines.extend(
        [
            "",
            "| Oracle | First any | First corroborated | First consensus | First distilled | Best community |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for oracle_name, values in evaluation["per_oracle"].items():
        lines.append(
            f"| {oracle_name} | {values['first_any_rank'] or '—'} | "
            f"{values['first_corroborated_rank'] or '—'} | "
            f"{values['first_consensus_rank'] or '—'} | "
            f"{values['first_distilled_rank'] or '—'} | "
            f"{values['best_community_id'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Global consensus-phrase registry",
            "",
            "| Phrases | Oracle recall |",
            "|---:|---:|",
        ]
    )
    for cutoff, values in phrase_evaluation["recall_at"].items():
        lines.append(f"| {cutoff} | {values['count']}/{values['total']} |")
    lines.extend(
        [
            "",
            "| Rank | Phrase | Architectures | Folds | Source community |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for phrase in phrases[:75]:
        lines.append(
            f"| {phrase['rank']} | {phrase['phrase']} | "
            f"{phrase['architecture_count']} | {len(phrase['inner_folds'])} | "
            f"{phrase['community_id']} |"
        )
    lines.extend(["", "## Top communities", ""])
    for record in records[:25]:
        phrases = ", ".join(
            value["phrase"] for value in record["consensus_phrases"][:8]
        ) or "(no repeated lexical phrase)"
        exemplar = record["exemplars"][0]["text"] if record["exemplars"] else ""
        oracle_names = ", ".join(record["oracle_hits"]) or "none"
        lines.extend(
            [
                f"### {record['rank']}. {record['community_id']} ({record['community_score']:.3f})",
                "",
                f"- Atoms/architectures/folds: {record['atom_count']} / "
                f"{len(record['architectures'])} / {len(record['inner_folds'])}",
                f"- Lanes: {', '.join(record['lanes'])}",
                f"- Consensus: {phrases}",
                f"- Oracle audit hits: {oracle_names}",
                f"- Medoid-like exemplar: {exemplar}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--members", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model", default="answerdotai/answerai-colbert-small-v1"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-atom-words", type=int, default=16)
    parser.add_argument("--atom-overlap-words", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--reciprocal-k", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for path in (args.packets, args.members, args.metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing experiment directory: {args.output_dir}"
        )
    if args.max_atom_words < 8:
        raise ValueError("--max-atom-words must be at least 8")
    if not 0 <= args.atom_overlap_words < args.max_atom_words:
        raise ValueError("--atom-overlap-words must be in [0, max-atom-words)")
    if args.candidate_k < args.reciprocal_k:
        raise ValueError("--candidate-k must be at least --reciprocal-k")

    started = time.monotonic()
    representatives, outer_fold = _extract_representatives(args.packets)
    metadata_names = _metadata_oracle_names(args.metadata)
    if metadata_names != set(ORACLE_PATTERNS):
        raise ValueError(
            "experiment oracle audit patterns do not match metadata: "
            f"patterns_only={sorted(set(ORACLE_PATTERNS) - metadata_names)}, "
            f"metadata_only={sorted(metadata_names - set(ORACLE_PATTERNS))}"
        )
    support = _member_support(
        args.members,
        {str(value["text_sha256"]) for value in representatives},
    )
    atoms = _build_atoms(
        representatives,
        support,
        max_words=args.max_atom_words,
        overlap_words=args.atom_overlap_words,
    )
    print(
        f"evidence atoms: {len(representatives)} representatives -> {len(atoms)} windows",
        flush=True,
    )
    matrices, pooled, adapter_name = _encode_atoms(
        atoms,
        model_name=args.model,
        device=args.device,
    )
    pooled_neighbors = _pooled_neighbors(
        pooled,
        [str(atom["architecture"]) for atom in atoms],
        candidate_k=args.candidate_k,
    )
    graph, pair_scores, candidate_pairs = _reciprocal_graph(
        matrices,
        pooled_neighbors,
        reciprocal_k=args.reciprocal_k,
    )
    communities = _cluster_graph(
        graph,
        resolution=args.resolution,
        seed=args.seed,
    )
    records = _community_records(communities, graph, atoms)
    evaluation = _oracle_evaluation(records, (25, 50, 75))
    phrases = _rank_consensus_phrases(records)
    phrase_evaluation = _phrase_oracle_evaluation(phrases, (25, 50, 75))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    packet_count = sum(1 for _value in _jsonl(args.packets))
    exact_atoms = sum(atom["provenance_source"] == "exact_member" for atom in atoms)
    summary = {
        "schema_version": "reciprocal_colbert_evidence_graph_experiment_v1",
        "outer_fold": outer_fold,
        "packet_count": packet_count,
        "representative_count": len(representatives),
        "atom_count": len(atoms),
        "exact_member_provenance_atoms": exact_atoms,
        "model": args.model,
        "device": args.device,
        "adapter": adapter_name,
        "max_atom_words": args.max_atom_words,
        "atom_overlap_words": args.atom_overlap_words,
        "candidate_k": args.candidate_k,
        "reciprocal_k": args.reciprocal_k,
        "louvain_resolution": args.resolution,
        "mutual_pooled_candidate_pairs": candidate_pairs,
        "scored_pair_count": len(pair_scores),
        "graph_edges": graph.number_of_edges(),
        "graph_isolates": nx.number_of_isolates(graph),
        "community_count": len(records),
        "elapsed_seconds": time.monotonic() - started,
        "inputs": {
            "packets": str(args.packets.resolve()),
            "members": str(args.members.resolve()),
            "metadata": str(args.metadata.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "communities.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            public_record = {key: value for key, value in record.items() if key != "nodes"}
            handle.write(json.dumps(public_record, ensure_ascii=False) + "\n")
    (args.output_dir / "oracle_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ranked_phrases.json").write_text(
        json.dumps(phrases, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "phrase_oracle_evaluation.json").write_text(
        json.dumps(phrase_evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        args.output_dir / "report.md",
        summary,
        records,
        evaluation,
        phrases,
        phrase_evaluation,
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "oracle_evaluation": evaluation,
                "phrase_oracle_evaluation": phrase_evaluation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
