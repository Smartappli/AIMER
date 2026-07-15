# Copyright (c) 2026 AIMER contributors.
"""Recommendation engine that proposes candidate models from RAG evidence."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from RAG.omop import build_omop_metadata
from RAG.timm_articles import (
    ensure_timm_article_index_is_fresh,
    load_timm_article_index,
)

try:
    from RAG.query import extract_filters, hybrid_search, rerank_results
except ImportError:
    extract_filters = None
    hybrid_search = None
    rerank_results = None

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
    "classification": {
        "classification",
        "classify",
        "classifier",
        "classificatie",
        "classificeren",
        "klassifikation",
        "klassifizieren",
        "triage",
    },
    "segmentation": {
        "segmentation",
        "segment",
        "segmenter",
        "segmentatie",
        "segmenteren",
        "segmentierung",
        "segmentieren",
        "mask",
        "contour",
    },
    "detection": {
        "detection",
        "détection",
        "detect",
        "détecter",
        "detectie",
        "detecteren",
        "erkennung",
        "erkennen",
        "localization",
        "localisation",
        "lokalisatie",
        "lokalisierung",
        "lesion",
        "lésion",
        "laesie",
        "läsion",
    },
}

MODALITY_HINTS = {
    "mri": {"mri", "irm", "mrt", "magnetresonanztomographie"},
    "ct": {
        "ct",
        "ct-scan",
        "scanner",
        "tomodensitometrie",
        "tomodensitométrie",
        "computertomografie",
        "computertomographie",
    },
    "xray": {
        "xray",
        "x-ray",
        "radiography",
        "radiographie",
        "radio",
        "röntgen",
        "rontgen",
        "röntgenfoto",
    },
    "ultrasound": {
        "ultrasound",
        "echo",
        "echography",
        "échographie",
        "echografie",
        "ultraschall",
        "sonografie",
    },
    "dermoscopy": {
        "dermoscopy",
        "dermoscopie",
        "dermatology",
        "dermatologie",
        "dermatoskopie",
        "skin",
        "huid",
        "haut",
    },
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
    "meilleure",
    "pour",
    "avec",
    "dans",
    "quel",
    "quelle",
    "beste",
    "voor",
    "met",
    "welke",
    "der",
    "die",
    "das",
    "und",
    "für",
    "mit",
    "welches",
    "welche",
}

SUPPORTED_LANGUAGES = frozenset({"fr", "en", "nl", "de"})

RATIONALE_TEXT = {
    "fr": (
        "Modèle recommandé car fréquemment mentionné dans les passages retrouvés, "
        "avec des signaux de performance et d'adéquation au contexte de requête."
    ),
    "en": (
        "This model is recommended because it is frequently mentioned in the "
        "retrieved passages, with performance and query-context relevance signals."
    ),
    "nl": (
        "Dit model wordt aanbevolen omdat het vaak voorkomt in de opgehaalde "
        "passages, met aanwijzingen voor prestaties en relevantie voor de vraag."
    ),
    "de": (
        "Dieses Modell wird empfohlen, weil es in den abgerufenen Passagen häufig "
        "erwähnt wird und Leistungs- sowie Kontextsignale zur Anfrage aufweist."
    ),
}

FALLBACK_RATIONALE_TEXT = {
    "fr": (
        "Suggestion exploratoire basée sur le catalogue de la base de connaissances "
        "(pas assez d'extraits récupérés pour scorer)."
    ),
    "en": (
        "Exploratory suggestion based on the knowledge-base catalogue "
        "(not enough retrieved excerpts to calculate a score)."
    ),
    "nl": (
        "Verkennende suggestie op basis van de kennisbankcatalogus "
        "(onvoldoende opgehaalde fragmenten om een score te berekenen)."
    ),
    "de": (
        "Explorative Empfehlung auf Basis des Wissensbankkatalogs "
        "(nicht genügend abgerufene Auszüge für eine Bewertung)."
    ),
}

NO_RECOMMENDATION_TEXT = {
    "fr": (
        "Aucune recommandation étayée n'a été retournée, car les preuves récupérées "
        "ne mentionnent aucun modèle catalogué avec un support exploitable."
    ),
    "en": (
        "No grounded recommendation was returned because the retrieved evidence did "
        "not mention a catalogued model with usable support."
    ),
    "nl": (
        "Er is geen onderbouwde aanbeveling beschikbaar omdat het opgehaalde bewijs "
        "geen gecatalogiseerd model met bruikbare ondersteuning vermeldde."
    ),
    "de": (
        "Es wurde keine belegte Empfehlung zurückgegeben, da die abgerufenen Belege "
        "kein katalogisiertes Modell mit verwertbarer Unterstützung nennen."
    ),
}

SAFETY_NOTICE_TEXT = {
    "fr": (
        "Aide à la décision expérimentale ML uniquement. Ne remplace pas le jugement "
        "médical, ni une validation clinique/prospective."
    ),
    "en": (
        "For experimental ML decision support only. This does not replace medical "
        "judgement or clinical/prospective validation."
    ),
    "nl": (
        "Uitsluitend voor experimentele ML-beslissingsondersteuning. Dit vervangt "
        "geen medisch oordeel of klinische/prospectieve validatie."
    ),
    "de": (
        "Nur zur experimentellen ML-Entscheidungsunterstützung. Dies ersetzt weder "
        "medizinisches Urteil noch eine klinische/prospektive Validierung."
    ),
}

EXPERIMENTAL_NOTES_TEXT = {
    "fr": (
        "Valider ce modèle sur un sous-ensemble local et le comparer au référentiel "
        "clinique avant tout déploiement.",
        "Tracer AUC/Dice/sensibilité-spécificité selon la tâche clinique et documenter "
        "les biais de population.",
        "Le modèle « {model} » est supporté par {count} extrait(s) dans la base "
        "courante; confirmer la reproductibilité locale.",
    ),
    "en": (
        "Validate this model on a local subset and compare it with the clinical "
        "baseline before deployment.",
        "Track AUC/Dice/sensitivity-specificity for the clinical task and document "
        "population bias.",
        "The model “{model}” is supported by {count} excerpt(s) in the current "
        "knowledge base; confirm local reproducibility.",
    ),
    "nl": (
        "Valideer dit model op een lokale subset en vergelijk het vóór implementatie "
        "met de klinische referentie.",
        "Volg AUC/Dice/sensitiviteit-specificiteit voor de klinische taak en "
        "documenteer populatiebias.",
        "Het model ‘{model}’ wordt ondersteund door {count} fragment(en) in de huidige "
        "kennisbank; bevestig de lokale reproduceerbaarheid.",
    ),
    "de": (
        "Validieren Sie dieses Modell vor dem Einsatz an einer lokalen Teilmenge und "
        "vergleichen Sie es mit der klinischen Referenz.",
        "Erfassen Sie AUC/Dice/Sensitivität-Spezifität für die klinische Aufgabe und "
        "dokumentieren Sie Populationsverzerrungen.",
        "Das Modell „{model}“ wird durch {count} Auszug/Auszüge in der aktuellen "
        "Wissensbasis gestützt; bestätigen Sie die lokale Reproduzierbarkeit.",
    ),
}

MIN_TOKEN_LENGTH = 2
MIN_ALIAS_LENGTH = 3
MIN_HYBRID_SEARCH_K = 9
MIN_RERANK_TOP_K = 8
CLASSIFICATION_BASELINE_HINTS = (
    "resnet",
    "efficientnet",
    "convnext",
    "mobilenet",
    "vit",
)


class OpenRAGRuntimeUnavailableError(RuntimeError):
    """Raised when strict OpenRAG retrieval is required but unavailable."""


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
    language: str = "fr"
    query_profile: QueryProfile
    used_filters: dict[str, Any]
    retrieval_mode: str
    safety_notice: str
    recommended_models: list[RecommendationItem]
    no_recommendation_reason: str | None = None


@dataclass(slots=True)
class _ScoredEvidence:
    score: float = 0.0
    evidence: list[EvidenceSnippet] = field(default_factory=list)
    matched_query_tokens: set[str] = field(default_factory=set)


def _is_production() -> bool:
    """Return whether recommendations run in a production environment."""
    environment = os.getenv(
        "AIMER_RAG_ENVIRONMENT",
        os.getenv("ENVIRONMENT", "local"),
    )
    return environment.strip().lower() in {"prod", "production"}


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _resolve_allow_ungrounded(allow_ungrounded: bool | None) -> bool:
    """Resolve whether catalog-only recommendations may be returned."""
    resolved = (
        allow_ungrounded
        if allow_ungrounded is not None
        else _env_bool("RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS", default=False)
    )
    if resolved and _is_production():
        msg = "Ungrounded recommendations are not allowed in production."
        raise RuntimeError(msg)
    return resolved


def _normalize(text: str) -> str:
    """
    Normalize text for approximate keyword matching.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized alphanumeric lowercase string.

    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _tokenize_query(query: str) -> set[str]:
    """
    Tokenize query while removing generic stopwords.

    Args:
        query: User query to tokenize.

    Returns:
        Set of filtered lowercase tokens.

    """
    return {
        token
        for token in re.findall(r"[^\W_]+", query.lower())
        if len(token) > MIN_TOKEN_LENGTH and token not in STOPWORDS
    }


def _infer_query_profile(query: str) -> QueryProfile:
    """
    Infer query intent (task + modality) with a lightweight rule-based parser.

    Args:
        query: User query to analyze.

    Returns:
        Structured profile inferred from the query and OMOP metadata.

    """
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
    """
    Extract a model-like name from a PDF filename.

    Args:
        filename: PDF filename.

    Returns:
        Extracted model name.

    """
    stem = Path(filename).stem
    head = stem.split(" - ")[0].strip()
    return re.sub(r"\s+", " ", head)


def _build_model_catalog(pdf_directory: Path | None = None) -> dict[str, set[str]]:
    """
    Build a normalized alias map from local PDFs plus the default TIMM article index.

    The TIMM seed ensures the knowledge base contains baseline model paper references
    even when no local PDFs have been ingested yet.

    Args:
        pdf_directory: Optional directory containing model PDFs.

    Returns:
        Dictionary mapping canonical model names to normalized aliases.

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
            aliases = {alias for alias in aliases if len(alias) >= MIN_ALIAS_LENGTH}
            if aliases:
                catalog[model_name] = aliases

    for article in load_timm_article_index():
        model_name = article["model_name"]
        aliases = {
            _normalize(model_name),
            _normalize(model_name.split("(")[0]),
            _normalize(model_name.replace("-", " ")),
            _normalize(model_name.replace("_", " ")),
        }
        aliases = {alias for alias in aliases if len(alias) >= MIN_ALIAS_LENGTH}
        if aliases:
            catalog.setdefault(model_name, set()).update(aliases)

    return catalog


def _metadata_source(doc: Document) -> str | None:
    """
    Extract a readable source name from document metadata.

    Args:
        doc: Source document.

    Returns:
        Basename of the source file when available, otherwise ``None``.

    """
    source = doc.metadata.get("source")
    if not source:
        return None
    return Path(str(source)).name


def _metadata_page(doc: Document) -> int | None:
    """
    Extract a normalized page number when available in metadata.

    Args:
        doc: Source document.

    Returns:
        Page number when found and valid, otherwise ``None``.

    """
    page = doc.metadata.get("page") or doc.metadata.get("page_number")
    if isinstance(page, int):
        return page
    return None


def _content_bonus(content_lowered: str, profile: QueryProfile) -> float:
    """
    Compute bonus based on alignment with task/modality hints.

    Args:
        content_lowered: Lowercased document content.
        profile: Query profile inferred from user input.

    Returns:
        Bonus score derived from task, modality, and metric hints.

    """
    task_bonus = 0.0
    for task in profile.tasks:
        task_bonus += (
            0.1 if any(k in content_lowered for k in TASK_HINTS[task]) else 0.0
        )

    modality_bonus = 0.0
    for modality in profile.modalities:
        modality_bonus += (
            0.1 if any(k in content_lowered for k in MODALITY_HINTS[modality]) else 0.0
        )

    metric_bonus = (
        0.15 if any(metric in content_lowered for metric in METRIC_HINTS) else 0.0
    )
    return task_bonus + modality_bonus + metric_bonus


def _metadata_alignment_bonus(doc: Document, profile: QueryProfile) -> float:
    """
    Boost score when document metadata aligns with OMOP/SNOMED query signals.

    Args:
        doc: Source document.
        profile: Query profile inferred from user input.

    Returns:
        Bonus score derived from metadata overlap.

    """
    metadata = doc.metadata or {}
    bonus = 0.0

    condition_ids = metadata.get("omop_condition_concept_ids")
    if isinstance(condition_ids, list):
        overlap = set(profile.omop_condition_concept_ids).intersection(condition_ids)
        if overlap:
            bonus += 0.2 + min(0.1, 0.05 * len(overlap))

    modality_ids = metadata.get("omop_modality_concept_ids")
    if isinstance(modality_ids, list):
        overlap = set(profile.omop_modality_concept_ids).intersection(modality_ids)
        if overlap:
            bonus += 0.2 + min(0.1, 0.05 * len(overlap))

    snomed_codes = metadata.get("snomed_ct_codes")
    if isinstance(snomed_codes, list):
        overlap = set(profile.snomed_ct_codes).intersection(
            code for code in snomed_codes if isinstance(code, str)
        )
        if overlap:
            bonus += 0.15 + min(0.1, 0.05 * len(overlap))

    return bonus


def _score_documents_against_catalog(
    documents: list[Document],
    catalog: dict[str, set[str]],
    profile: QueryProfile,
) -> dict[str, _ScoredEvidence]:
    """
    Score model candidates based on retrieved chunk evidence.

    Args:
        documents: Retrieved documents to analyze.
        catalog: Model alias catalog.
        profile: Query profile inferred from user input.

    Returns:
        Per-model scored evidence aggregated from retrieved documents.

    """
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
            query_tokens_in_doc = {
                token for token in profile.query_tokens if token in lowered
            }
            token_overlap = sum(
                1 for token in profile.query_tokens if token in model_name.lower()
            )
            source_bonus = 0.25 if _metadata_source(doc) else 0.0
            query_coverage_bonus = min(0.3, len(query_tokens_in_doc) * 0.05)
            delta = (
                1.0
                + _content_bonus(lowered, profile)
                + _metadata_alignment_bonus(doc, profile)
                + source_bonus
                + query_coverage_bonus
                + (token_overlap * 0.05)
            )
            current.score += delta
            current.matched_query_tokens.update(query_tokens_in_doc)

            snippet = EvidenceSnippet(
                source=_metadata_source(doc),
                page=_metadata_page(doc),
                snippet=snippet_text,
                relevance=min(1.0, max(0.2, delta / 2.0)),
            )
            if snippet not in current.evidence:
                current.evidence.append(snippet)

    return scored


def _normalize_language(language: str) -> str:
    """Return a supported base language code, defaulting to French."""
    normalized = language.strip().lower().split("-", maxsplit=1)[0]
    return normalized if normalized in SUPPORTED_LANGUAGES else "fr"


def _build_experimental_notes(
    model_name: str,
    evidence_count: int,
    language: str = "fr",
) -> list[str]:
    """
    Generate practical experiment notes to guide clinicians and ML teams.

    Args:
        model_name: Candidate model name.
        evidence_count: Number of supporting evidence snippets.
        language: Requested response language.

    Returns:
        List of practical validation notes.

    """
    localized = EXPERIMENTAL_NOTES_TEXT[_normalize_language(language)]
    return [
        note.format(model=model_name, count=evidence_count)
        for note in localized
    ]


def _confidence_from_score(
    *,
    score: float,
    max_score: float,
    token_coverage: float,
    evidence_count: int,
) -> float:
    """
    Calibrate confidence by combining rank score and grounding signals.

    Args:
        score: Raw score for the current model.
        max_score: Best score among ranked models.
        token_coverage: Fraction of query tokens matched by evidence.
        evidence_count: Number of supporting evidence snippets.

    Returns:
        Confidence score normalized between 0 and 1.

    """
    normalized_score = 0.0 if max_score <= 0 else min(1.0, score / max_score)
    evidence_boost = min(0.15, evidence_count * 0.04)
    confidence = (normalized_score * 0.7) + (token_coverage * 0.2) + evidence_boost
    return min(0.96, max(0.35, confidence))


def _safe_retrieve(query: str, *, k: int) -> tuple[list[Document], dict[str, Any], str]:
    """
    Try retrieval through the existing RAG query module with graceful fallback.

    Args:
        query: User query to retrieve against.
        k: Requested number of final recommendations.

    Returns:
        Tuple containing retrieved documents, applied filters, and retrieval mode.

    """
    if extract_filters is None or hybrid_search is None or rerank_results is None:
        return [], {}, "catalog-only-fallback"

    try:
        filters = extract_filters(query)
        docs = hybrid_search(
            query=query,
            k=max(k * 3, MIN_HYBRID_SEARCH_K),
            filters=filters,
        )
        reranked = rerank_results(
            query=query,
            documents=docs,
            top_k=max(k * 2, MIN_RERANK_TOP_K),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return [], {}, "catalog-only-fallback"
    else:
        return reranked, filters, "hybrid+rerank"


def _safe_retrieve_strict(
    query: str,
    *,
    k: int,
) -> tuple[list[Document], dict[str, Any], str]:
    """Retrieve with strict OpenRAG requirement (no silent fallback)."""
    docs, used_filters, mode = _safe_retrieve(query, k=k)
    if mode == "catalog-only-fallback":
        raise OpenRAGRuntimeUnavailableError(
            "OpenRAG retrieval is required but runtime is not ready "
            "or retrieval failed.",
        )
    return docs, used_filters, mode


def _fallback_recommendations(
    catalog: dict[str, set[str]],
    profile: QueryProfile,
    top_k: int,
    language: str,
) -> list[RecommendationItem]:
    """
    Provide deterministic lexical suggestions for non-clinical exploration.

    Args:
        catalog: Model alias catalog.
        profile: Query profile inferred from user input.
        top_k: Maximum number of models to return.
        language: Normalized response language.

    Returns:
        List of ungrounded recommendation items.

    """
    ranked_models: list[tuple[float, int, str]] = []
    for index, model in enumerate(catalog):
        lowered = model.lower()
        score = sum(1.0 for token in profile.query_tokens if token in lowered)
        if "classification" in profile.tasks:
            for rank, hint in enumerate(CLASSIFICATION_BASELINE_HINTS):
                if hint in lowered:
                    score += max(0.2, 1.2 - (rank * 0.2))
                    break
        ranked_models.append((score, index, model))

    ranked_models.sort(key=lambda item: (-item[0], item[1]))
    if ranked_models and ranked_models[0][0] > 0:
        fallback_models = [model for _, _, model in ranked_models[:top_k]]
    else:
        fallback_models = list(catalog.keys())[:top_k]

    return [
        RecommendationItem(
            model_name=model,
            confidence=0.35,
            rationale=FALLBACK_RATIONALE_TEXT[language],
            evidence=[],
            literature_support=0,
            experimental_notes=_build_experimental_notes(model, 0, language),
        )
        for model in fallback_models
    ]


def _resolve_strict_openrag(strict_openrag: bool | None) -> bool:
    """Resolve strict OpenRAG mode from explicit argument or environment."""
    if _is_production():
        return True
    if strict_openrag is not None:
        return strict_openrag
    return os.getenv("RAG_STRICT_OPENRAG", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def recommend_models_for_query(
    query: str,
    *,
    top_k: int = 3,
    documents: list[Document] | None = None,
    pdf_directory: Path | None = None,
    strict_openrag: bool | None = None,
    allow_ungrounded: bool | None = None,
    language: str = "fr",
) -> RecommendationResponse:
    """
    Recommend candidate models based on retrieved literature snippets.

    Args:
        query: User query describing the clinical or technical need.
        top_k: Maximum number of recommendations to return.
        documents: Optional injected documents, bypassing retrieval.
        pdf_directory: Optional PDF directory used to build the catalog.
        strict_openrag: Require OpenRAG retrieval instead of catalog-only fallback.
        allow_ungrounded: Permit catalog-only exploratory suggestions. This is
            disabled by default and rejected in production.
        language: Response language code (FR, EN, NL, or DE).

    Returns:
        Structured recommendation response with ranked models and evidence.

    """
    resolved_language = _normalize_language(language)
    profile = _infer_query_profile(query)
    catalog = _build_model_catalog(pdf_directory=pdf_directory)

    resolved_strict = _resolve_strict_openrag(strict_openrag)
    resolved_allow_ungrounded = _resolve_allow_ungrounded(allow_ungrounded)

    if documents is None:
        if resolved_strict:
            docs, used_filters, retrieval_mode = _safe_retrieve_strict(query, k=top_k)
        else:
            docs, used_filters, retrieval_mode = _safe_retrieve(query, k=top_k)
    else:
        docs, used_filters, retrieval_mode = documents, {}, "injected-documents"

    scored = _score_documents_against_catalog(docs, catalog, profile)
    no_recommendation_reason = None

    if not scored:
        if resolved_allow_ungrounded:
            recommendations = _fallback_recommendations(
                catalog,
                profile,
                top_k,
                resolved_language,
            )
            retrieval_mode = f"{retrieval_mode}+ungrounded-catalog"
        else:
            recommendations = []
            no_recommendation_reason = NO_RECOMMENDATION_TEXT[resolved_language]
            retrieval_mode = f"{retrieval_mode}+no-grounded-evidence"
    else:
        ranked = sorted(scored.items(), key=lambda item: item[1].score, reverse=True)[
            :top_k
        ]
        max_score = ranked[0][1].score if ranked else 1.0

        recommendations = []
        for model_name, evidence in ranked:
            token_coverage = (
                len(evidence.matched_query_tokens) / len(profile.query_tokens)
                if profile.query_tokens
                else 0.0
            )
            confidence = _confidence_from_score(
                score=evidence.score,
                max_score=max_score,
                token_coverage=token_coverage,
                evidence_count=len(evidence.evidence),
            )
            recommendations.append(
                RecommendationItem(
                    model_name=model_name,
                    confidence=round(confidence, 2),
                    rationale=RATIONALE_TEXT[resolved_language],
                    evidence=evidence.evidence[:3],
                    literature_support=len(evidence.evidence),
                    experimental_notes=_build_experimental_notes(
                        model_name,
                        len(evidence.evidence),
                        resolved_language,
                    ),
                ),
            )

    return RecommendationResponse(
        query=query,
        language=resolved_language,
        query_profile=profile,
        used_filters=used_filters,
        retrieval_mode=retrieval_mode,
        safety_notice=SAFETY_NOTICE_TEXT[resolved_language],
        recommended_models=recommendations,
        no_recommendation_reason=no_recommendation_reason,
    )
