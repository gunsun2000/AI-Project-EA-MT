from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Optional

from .features import (
    NUMERIC_FEATURE_KEYS,
    build_candidate_feature_vector,
    feature_dict_to_numeric_vector,
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_example_id(example: Any) -> str:
    return (
        _safe_text(getattr(example, "example_id", None))
        or _safe_text(getattr(example, "id", None))
        or _safe_text(getattr(example, "wikidata_id", None))
        or "unknown"
    )


def _extract_source_text(example: Any) -> str:
    return (
        _safe_text(getattr(example, "source", None))
        or _safe_text(getattr(example, "source_text", None))
        or _safe_text(getattr(example, "sentence", None))
        or ""
    )


def _extract_source_span(example: Any) -> str:
    for field_name in ("source_span", "source_span_text", "span_text", "entity_surface", "surface"):
        value = _safe_text(getattr(example, field_name, None))
        if value:
            return value
    return ""


def _extract_qid(candidate: Any) -> Optional[str]:
    for field_name in ("qid", "wikidata_id", "candidate_qid", "id"):
        value = getattr(candidate, field_name, None)
        if value is not None and _safe_text(value):
            return _safe_text(value)
    return None


def _lookup_gold_candidate(resources: Any, canonical_qid: str) -> Optional[Any]:
    if not canonical_qid:
        return None

    for method_name in ("lookup_entity_by_qid", "get_entity_by_qid", "find_entity_by_qid"):
        method = getattr(resources, method_name, None)
        if callable(method):
            try:
                found = method(canonical_qid)
                if found is not None:
                    return found
            except Exception:
                pass
    return None


def _unique_by_qid(candidates: List[Any]) -> List[Any]:
    unique: List[Any] = []
    seen = set()

    for candidate in candidates:
        qid = _extract_qid(candidate)
        key = qid or f"__object__:{id(candidate)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return unique


def _collect_candidates(example: Any, resources: Any) -> List[Any]:
    """
    retrieval 모듈이 아직 없더라도 import 단계에서 깨지지 않도록 동적 import를 사용합니다.
    나중에 retrieval/service.py가 생기면 자동으로 연결됩니다.
    """
    module_names = [
        "src.eamt.retrieval.service",
        "eamt.retrieval.service",
    ]

    for module_name in module_names:
        try:
            retrieval_service = import_module(module_name)
            collect_entity_candidates = getattr(retrieval_service, "collect_entity_candidates", None)

            if callable(collect_entity_candidates):
                candidates = collect_entity_candidates(example, resources)
                if candidates:
                    return list(candidates)
        except Exception:
            pass

    for method_name in ("collect_entity_candidates", "retrieve_candidates", "get_candidates"):
        method = getattr(resources, method_name, None)
        if callable(method):
            try:
                candidates = method(example)
                if candidates:
                    return list(candidates)
            except Exception:
                pass

    return []


def _sort_hard_negatives(source: str, source_span: str, negatives: List[Any]) -> List[Any]:
    """
    Hard Negative Mining:
    - 표면 겹침(surface overlap)이 크고
    - 인기도(popularity)가 높고
    - 모호성(ambiguity)이 높고
    - span match score가 큰 후보를 우선합니다.
    """

    def sort_key(candidate: Any):
        feat = build_candidate_feature_vector(
            source=source,
            candidate=candidate,
            canonical_qid=None,
            source_span=source_span or None,
        )
        return (
            float(feat.get("surface_overlap_score", 0.0)),
            float(feat.get("popularity_score", 0.0)),
            float(feat.get("ambiguity_count", 0.0)),
            float(feat.get("span_match_score", 0.0)),
        )

    return sorted(negatives, key=sort_key, reverse=True)


def build_reranker_train_examples(
    example: Any,
    resources: Any,
    max_negatives: int = 8,
) -> List[Dict[str, Any]]:
    """
    pointwise / pairwise 학습이나 디버깅에 바로 사용할 수 있는 flat 학습 샘플을 생성합니다.

    중요:
    - 학습 feature에는 canonical prior를 넣지 않습니다.
    - 즉 build_candidate_feature_vector(..., canonical_qid=None) 로 호출합니다.
    """
    source = _extract_source_text(example)
    source_span = _extract_source_span(example)
    canonical_qid = _safe_text(getattr(example, "wikidata_id", None)) or None
    example_id = _extract_example_id(example)

    candidates = _collect_candidates(example, resources)
    candidates = _unique_by_qid(list(candidates))

    positives = [c for c in candidates if _extract_qid(c) == canonical_qid]
    negatives = [c for c in candidates if _extract_qid(c) != canonical_qid]
    negatives = _sort_hard_negatives(source, source_span, negatives)

    # gold 후보가 retrieval에 없더라도 학습 샘플이 0 positive가 되지 않도록 방어
    if not positives and canonical_qid is not None:
        kb_record = _lookup_gold_candidate(resources, canonical_qid)
        if kb_record is not None:
            positives = [kb_record]

    selected = _unique_by_qid(positives[:1] + negatives[:max_negatives])
    results: List[Dict[str, Any]] = []

    for idx, cand in enumerate(selected):
        qid = _extract_qid(cand)
        feature_dump = build_candidate_feature_vector(
            source=source,
            candidate=cand,
            canonical_qid=None,  # 학습 시 prior 누수 방지
            source_span=source_span or None,
        )

        results.append(
            {
                "example_id": example_id,
                "group_id": example_id,
                "source": source,
                "source_span": source_span,
                "canonical_qid": canonical_qid,
                "candidate_rank_in_group": idx,
                "retrieval_rank": getattr(cand, "retrieval_rank", None),
                "qid": qid,
                "label": 1 if qid == canonical_qid else 0,
                "features": feature_dump,
                "numeric_features": feature_dict_to_numeric_vector(feature_dump),
                "numeric_feature_keys": list(NUMERIC_FEATURE_KEYS),
            }
        )

    return results


def build_grouped_reranker_train_example(
    example: Any,
    resources: Any,
    max_negatives: int = 8,
) -> Dict[str, Any]:
    """
    listwise / softmax ranking loss용 그룹 형식 샘플을 생성합니다.
    """
    rows = build_reranker_train_examples(example, resources, max_negatives=max_negatives)
    source = _extract_source_text(example)
    source_span = _extract_source_span(example)
    canonical_qid = _safe_text(getattr(example, "wikidata_id", None)) or None
    example_id = _extract_example_id(example)

    candidate_features = [row["features"] for row in rows]

    return {
        "example_id": example_id,
        "group_id": example_id,
        "source": source,
        "source_span": source_span,
        "canonical_qid": canonical_qid,
        "candidate_features": candidate_features,
        "candidates": candidate_features,
        "numeric_features": [row["numeric_features"] for row in rows],
        "labels": [row["label"] for row in rows],
        "qids": [row["qid"] for row in rows],
        "retrieval_ranks": [row["retrieval_rank"] for row in rows],
        "numeric_feature_keys": list(NUMERIC_FEATURE_KEYS),
    }


__all__ = [
    "build_reranker_train_examples",
    "build_grouped_reranker_train_example",
]
