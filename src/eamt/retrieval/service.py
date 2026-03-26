from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from src.DTOlist import CandidateEntity, KBEntityRecord, SpanMatch
from src.eamt.kb.index import (
    lookup_entity_by_qid,
    normalize_surface,
    search_surface_candidates,
)
from .align import SourceSpanMatch, align_source_span, extract_candidate_spans_from_source


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_safe_text(v) for v in value if _safe_text(v)]
    return []


def _extract_source(example: Any) -> str:
    return (
        _safe_text(getattr(example, "source", None))
        or _safe_text(getattr(example, "source_text", None))
        or _safe_text(getattr(example, "sentence", None))
        or ""
    )


def _extract_target_lang(example: Any, resources: Any) -> Optional[str]:
    lang = (
        _safe_text(getattr(example, "target_lang", None))
        or _safe_text(getattr(example, "target_locale", None))
    )
    if lang:
        return lang

    if isinstance(resources, dict):
        return _safe_text(resources.get("target_lang") or resources.get("default_target_lang")) or None

    return (
        _safe_text(getattr(resources, "target_lang", None))
        or _safe_text(getattr(resources, "default_target_lang", None))
        or None
    )


def _extract_gold_qid(example: Any) -> Optional[str]:
    qid = (
        _safe_text(getattr(example, "wikidata_id", None))
        or _safe_text(getattr(example, "qid", None))
    )
    return qid or None


def _get_qid_index(resources: Any) -> Any:
    if isinstance(resources, dict):
        return resources.get("qid_index")
    return getattr(resources, "qid_index", None)


def _get_surface_index(resources: Any) -> Any:
    if isinstance(resources, dict):
        return resources.get("surface_index")
    return getattr(resources, "surface_index", None)


def _to_span_match(match: Optional[SourceSpanMatch]) -> Optional[SpanMatch]:
    if match is None:
        return None
    return SpanMatch(
        source_span=match.matched_text,
        char_start=match.start,
        char_end=match.end,
        match_kind=match.match_kind,
        matched_surface=match.normalized_text,
        match_score=match.match_score,
    )


def _ambiguity_count(surface_index: Any, span_text: str) -> int:
    if not surface_index or not span_text:
        return 0
    normalized = normalize_surface(span_text)
    records = surface_index.get(normalized, [])
    return len(records) if isinstance(records, list) else 0


def _best_record(records: Optional[list[KBEntityRecord]]) -> Optional[KBEntityRecord]:
    if not records:
        return None
    return sorted(records, key=lambda r: getattr(r, "popularity_score", 0.0), reverse=True)[0]


def _record_to_candidate(
    record: KBEntityRecord,
    candidate_source: str,
    source_span: Optional[str],
    span_match: Optional[SpanMatch],
    ambiguity_count: int,
) -> CandidateEntity:
    target_aliases = _safe_list(getattr(record, "target_aliases", []))
    return CandidateEntity(
        qid=_safe_text(record.qid),
        candidate_source=candidate_source,
        target_label=_safe_text(getattr(record, "target_label", None)) or None,
        target_aliases=target_aliases,
        entity_type=_safe_text(getattr(record, "entity_type", None)) or None,
        description=_safe_text(getattr(record, "description", None)) or None,
        popularity_score=float(getattr(record, "popularity_score", 0.0) or 0.0),
        alias_count=len(target_aliases),
        ambiguity_count=ambiguity_count,
        source_span=source_span,
        span_match=span_match,
    )


def _dedupe_by_qid(candidates: Iterable[CandidateEntity]) -> List[CandidateEntity]:
    results: List[CandidateEntity] = []
    seen = set()

    for candidate in candidates:
        qid = _safe_text(getattr(candidate, "qid", None))
        if not qid or qid in seen:
            continue
        seen.add(qid)
        results.append(candidate)

    return results


def collect_entity_candidates(
    example: Any,
    resources: Any,
    top_k: int = 10,
    per_surface_k: int = 5,
    min_char_len: int = 2,
    max_n: int = 5,
) -> List[CandidateEntity]:
    """
    - QID가 있으면 direct lookup 우선
    - source에서 SALT 스타일 span 후보 생성
    - surface 기반 candidate top-K 추가 수집
    """
    source = _extract_source(example)
    target_lang = _extract_target_lang(example, resources)
    gold_qid = _extract_gold_qid(example)

    if not source or not target_lang:
        return []

    qid_index = _get_qid_index(resources)
    surface_index = _get_surface_index(resources)

    collected: List[CandidateEntity] = []

    # 1) anchored QID lookup
    if gold_qid and qid_index is not None:
        records = lookup_entity_by_qid(qid_index, gold_qid, target_lang)
        record = _best_record(records)
        if record is not None:
            raw_match = align_source_span(
                source=source,
                label=_safe_text(getattr(record, "label_en", None)),
                aliases=_safe_list(getattr(record, "aliases_en", [])),
                min_char_len=min_char_len,
            )
            span_match = _to_span_match(raw_match)
            source_span = span_match.source_span if span_match else None
            collected.append(
                _record_to_candidate(
                    record=record,
                    candidate_source="anchored_qid",
                    source_span=source_span,
                    span_match=span_match,
                    ambiguity_count=_ambiguity_count(surface_index, source_span or ""),
                )
            )

    # 2) SALT-style surface search
    if surface_index is not None:
        spans = extract_candidate_spans_from_source(
            source=source,
            min_n=1,
            max_n=max_n,
            min_char_len=min_char_len,
        )

        for span in spans:
            candidates = search_surface_candidates(
                surface_index=surface_index,
                surfaces=[span],
                target_lang=target_lang,
                max_candidates=per_surface_k,
            )

            raw_match = align_source_span(source=source, label=span, aliases=None, min_char_len=min_char_len)
            span_match = _to_span_match(raw_match)

            for candidate in candidates:
                candidate.source_span = span
                candidate.span_match = span_match
                candidate.candidate_source = "surface_search"
                if not getattr(candidate, "ambiguity_count", None):
                    candidate.ambiguity_count = _ambiguity_count(surface_index, span)
                collected.append(candidate)

    collected = _dedupe_by_qid(collected)
    return collected[:top_k]


__all__ = [
    "collect_entity_candidates",
]