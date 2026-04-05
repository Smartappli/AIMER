# Copyright (c) 2026 AIMER contributors.
"""Recommendation engine that proposes candidate models from RAG evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from RAG.omop import build_omop_metadata
from RAG.timm_articles import ensure_timm_article_index_is_fresh, load_timm_article_index

if TYPE_CHECKING:
    from langchain_core.documents import Document

METRIC_HINTS = (
    "auc",
    "dice",
    "sensitivity",
    "specificity",
    "f1",
    "accuracy",
    "recall",
    "precision",
)

TASK_HINTS = {
    "classification": {"classification", "classify", "triage"},
    "segmentation": {"segmentation", "segment", "mask", "contour"},
    "detection": {"detection", "detect", "localization", "lesion"},
}

MODALITY_HINTS = {
    "mri": {"mri", "irm"},
    "ct": {"ct", "scanner", "tomodensitometrie"},
    "xray": {"xray", "x-ray", "radiography", "radio"},
    "ultrasound": {"ultrasound", "echo", "echography"},
    "dermoscopy": {"dermoscopy", "skin", "dermatology"},
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "using",
    "based",
    "medical",
    "clinical",
    "model",
    "models",
    "imaging",
    "best",
    "meilleur",
}


class QueryProfile(BaseModel):
    """Inferred clinical/technical profile extracted from the user query."""

    tasks: list[str]
    modalities: list[str]
    query_tokens: list[str]
    omop_condition_concept_ids: list[int]
    omop_modality_concept_ids: list[int]
    snomed_ct_codes: list[str]


class EvidenceSnippet(BaseModel):
    """One short grounded evidence snippet linked to source metadata."""

    source: str | None
    page: int | None
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)


class RecommendationItem(BaseModel):
    """A single recommended model entry with supporting evidence."""

    model_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[EvidenceSnippet]
    literature_support: int
    experimental_notes: list[str]


class RecommendationResponse(BaseModel):
    """Structured response returned by the recommendation engine."""

    query: str
    query_profile: QueryProfile
    used_filters: dict[str, Any]
    retrieval_mode: str
    safety_notice: str
    recommended_models: list[RecommendationItem]


@dataclass(slots=True)
class _ScoredEvidence:
    score: float = 0.0
    evidence: list[EvidenceSnippet] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Normalize text for approximate keyword matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _tokenize_query(query: str) -> set[str]:
    """Tokenize query while removing generic stopwords."""
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _infer_query_profile(query: str) -> QueryProfile:
    """Infer query intent (task + modality) with a lightweight rule-based parser."""
    lowered = query.lower()
    omop_metadata = build_omop_metadata(query)

    tasks = [
        task
        for task, keywords in TASK_HINTS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    modalities = [
        modality
        for modality, keywords in MODALITY_HINTS.items()
        if any(keyword in lowered for keyword in keywords)
    ]

    condition_ids = [
        value
        for value in omop_metadata.get("omop_condition_concept_ids", [])
        if isinstance(value, int)
    ]
    modality_ids = [
        value
        for value in omop_metadata.get("omop_modality_concept_ids", [])
        if isinstance(value, int)
    ]
    snomed_codes = [
        value
        for value in omop_metadata.get("snomed_ct_codes", [])
        if isinstance(value, str)
    ]

    return QueryProfile(
        tasks=tasks,
        modalities=modalities,
        query_tokens=sorted(_tokenize_query(query)),
        omop_condition_concept_ids=condition_ids,
        omop_modality_concept_ids=modality_ids,
        snomed_ct_codes=snomed_codes,
    )


def _extract_model_name_from_filename(filename: str) -> str:
    """Extract a model-like name from a PDF filename."""
    stem = Path(filename).stem
    head = stem.split(" - ")[0].strip()
    return re.sub(r"\s+", " ", head)


def _build_model_catalog(pdf_directory: Path | None = None) -> dict[str, set[str]]:
    """
    Build a normalized alias map from local PDFs plus the default TIMM article index.

    The TIMM seed ensures the knowledge base contains baseline model paper references
    even when no local PDFs have been ingested yet.
    """
    base_dir = (
        pdf_directory
        if pdf_directory is not None
        else Path(__file__).resolve().parent / "data" / "pdfs"
    )
    ensure_timm_article_index_is_fresh(pdf_directory=base_dir)
    catalog: dict[str, set[str]] = {}
    if base_dir.exists():
        for path in sorted(base_dir.glob("*.pdf")):
            model_name = _extract_model_name_from_filename(path.name)
            aliases = {
                _normalize(model_name),
                _normalize(model_name.split("(")[0]),
                _normalize(model_name.replace("-", " ")),
            }
            aliases = {alias for alias in aliases if alias}
            if aliases:
                catalog[model_name] = aliases

    for article in load_timm_article_index():
        model_name = article["model_name"]
        aliases = {
            _normalize(model_name),
            _normalize(model_name.replace("-", " ")),
            _normalize(model_name.replace("_", " ")),
        }
        aliases = {alias for alias in aliases if alias}
        if aliases:
            catalog.setdefault(model_name, set()).update(aliases)

    return catalog


def _metadata_source(doc: Document) -> str | None:
    """Extract a readable source name from document metadata."""
    source = doc.metadata.get("source")
    if not source:
        return None
    return Path(str(source)).name


def _metadata_page(doc: Document) -> int | None:
    """Extract a normalized page number when available in metadata."""
    page = doc.metadata.get("page") or doc.metadata.get("page_number")
    if isinstance(page, int):
        return page
    return None


def _content_bonus(content_lowered: str, profile: QueryProfile) -> float:
    """Compute bonus based on alignment with task/modality hints."""
    task_bonus = 0.0
    for task in profile.tasks:
        task_bonus += 0.1 if any(k in content_lowered for k in TASK_HINTS[task]) else 0.0

    modality_bonus = 0.0
    for modality in profile.modalities:
        modality_bonus += (
            0.1 if any(k in content_lowered for k in MODALITY_HINTS[modality]) else 0.0
        )

    metric_bonus = 0.15 if any(metric in content_lowered for metric in METRIC_HINTS) else 0.0
    return task_bonus + modality_bonus + metric_bonus


def _score_documents_against_catalog(
    documents: list[Document],
    catalog: dict[str, set[str]],
    profile: QueryProfile,
) -> dict[str, _ScoredEvidence]:
    """Score model candidates based on retrieved chunk evidence."""
    scored: dict[str, _ScoredEvidence] = {}

    for doc in documents:
        content = doc.page_content
        lowered = content.lower()
        normalized = _normalize(content)
        compact = re.sub(r"\s+", " ", content).strip()
        snippet_text = compact[:280]

        for model_name, aliases in catalog.items():
            if not any(alias and alias in normalized for alias in aliases):
                continue

            current = scored.setdefault(model_name, _ScoredEvidence())
            token_overlap = sum(
                1 for token in profile.query_tokens if token in model_name.lower()
            )
            source_bonus = 0.25 if _metadata_source(doc) else 0.0
            delta = 1.0 + _content_bonus(lowered, profile) + source_bonus + (token_overlap * 0.05)
            current.score += delta

            snippet = EvidenceSnippet(
                source=_metadata_source(doc),
                page=_metadata_page(doc),
                snippet=snippet_text,
                relevance=min(1.0, max(0.2, delta / 2.0)),
            )
            if snippet not in current.evidence:
                current.evidence.append(snippet)

    return scored


def _build_experimental_notes(model_name: str, evidence_count: int) -> list[str]:
    """Generate practical experiment notes to guide clinicians and ML teams."""
    return [
        (
            "Valider ce modèle sur un sous-ensemble local et comparer au baseline "
            "clinique avant tout déploiement."
        ),
        (
            "Tracer AUC/Dice/sensibilité-spécificité selon la tâche clinique et "
            "documenter les biais de population."
        ),
        (
            f"Le modèle « {model_name} » est supporté par {evidence_count} extrait(s) "
            "dans la base courante; confirmer la reproductibilité locale."
        ),
    ]


def _safe_retrieve(query: str, *, k: int) -> tuple[list[Document], dict[str, Any], str]:
    """Try retrieval through the existing RAG query module with graceful fallback."""
    try:
        from RAG.query import extract_filters, hybrid_search, rerank_results

        filters = extract_filters(query)
        docs = hybrid_search(query=query, k=max(k * 3, 9), filters=filters)
        reranked = rerank_results(query=query, documents=docs, top_k=max(k * 2, 8))
        return reranked, filters, "hybrid+rerank"
    except Exception:
        return [], {}, "catalog-only-fallback"


def _fallback_recommendations(
    catalog: dict[str, set[str]],
    profile: QueryProfile,
    top_k: int,
) -> list[RecommendationItem]:
    """Provide deterministic lexical fallback when retrieval has no usable evidence."""
    lexical_hits = [
        model
        for model in catalog
        if any(token in model.lower() for token in profile.query_tokens)
    ]
    fallback_models = lexical_hits[:top_k] if lexical_hits else list(catalog.keys())[:top_k]

    return [
        RecommendationItem(
            model_name=model,
            confidence=0.35,
            rationale=(
                "Suggestion exploratoire basée sur le catalogue de la base de "
                "connaissances (pas assez d'extraits récupérés pour scorer)."
            ),
            evidence=[],
            literature_support=0,
            experimental_notes=_build_experimental_notes(model, 0),
        )
        for model in fallback_models
    ]


def recommend_models_for_query(
    query: str,
    *,
    top_k: int = 3,
    documents: list[Document] | None = None,
    pdf_directory: Path | None = None,
) -> RecommendationResponse:
    """Recommend candidate models based on retrieved literature snippets."""
    profile = _infer_query_profile(query)
    catalog = _build_model_catalog(pdf_directory=pdf_directory)

    if documents is None:
        docs, used_filters, retrieval_mode = _safe_retrieve(query, k=top_k)
    else:
        docs, used_filters, retrieval_mode = documents, {}, "injected-documents"

    scored = _score_documents_against_catalog(docs, catalog, profile)

    if not scored:
        recommendations = _fallback_recommendations(catalog, profile, top_k)
    else:
        ranked = sorted(scored.items(), key=lambda item: item[1].score, reverse=True)[:top_k]
        max_score = ranked[0][1].score if ranked else 1.0

        recommendations = []
        for model_name, evidence in ranked:
            confidence = min(0.95, max(0.4, evidence.score / max_score))
            recommendations.append(
                RecommendationItem(
                    model_name=model_name,
                    confidence=round(confidence, 2),
                    rationale=(
                        "Modèle recommandé car fréquemment mentionné dans les passages "
                        "retrouvés, avec des signaux de performance et d'adéquation "
                        "au contexte de requête."
                    ),
                    evidence=evidence.evidence[:3],
                    literature_support=len(evidence.evidence),
                    experimental_notes=_build_experimental_notes(
                        model_name,
                        len(evidence.evidence),
                    ),
                ),
            )

    return RecommendationResponse(
        query=query,
        query_profile=profile,
        used_filters=used_filters,
        retrieval_mode=retrieval_mode,
        safety_notice=(
            "Aide à la décision expérimentale ML uniquement. Ne remplace pas le "
            "jugement médical, ni une validation clinique/prospective."
        ),
        recommended_models=recommendations,
    )
