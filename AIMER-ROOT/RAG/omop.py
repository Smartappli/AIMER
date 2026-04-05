# Copyright (c) 2026 AIMER contributors.
"""OMOP + SNOMED CT helpers used to enrich RAG metadata and query filters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClinicalConcept:
    """One concept mapping between clinical language and OMOP/SNOMED."""

    name: str
    omop_concept_id: int
    snomed_ct_code: str
    domain: str
    aliases: tuple[str, ...]


CONCEPTS: tuple[ClinicalConcept, ...] = (
    ClinicalConcept(
        name="Pneumonia",
        omop_concept_id=255848,
        snomed_ct_code="233604007",
        domain="condition",
        aliases=("pneumonia", "pneumonie"),
    ),
    ClinicalConcept(
        name="COVID-19",
        omop_concept_id=37311061,
        snomed_ct_code="840539006",
        domain="condition",
        aliases=("covid", "covid19", "sars-cov-2", "sars cov 2"),
    ),
    ClinicalConcept(
        name="Lung cancer",
        omop_concept_id=254761,
        snomed_ct_code="363358000",
        domain="condition",
        aliases=("lung cancer", "cancer pulmonaire"),
    ),
    ClinicalConcept(
        name="Brain tumor",
        omop_concept_id=436954,
        snomed_ct_code="254935002",
        domain="condition",
        aliases=("brain tumor", "glioma", "tumeur cérébrale"),
    ),
    ClinicalConcept(
        name="Magnetic resonance imaging",
        omop_concept_id=2212808,
        snomed_ct_code="113091000",
        domain="modality",
        aliases=("mri", "irm", "magnetic resonance"),
    ),
    ClinicalConcept(
        name="Computed tomography",
        omop_concept_id=2212865,
        snomed_ct_code="77477000",
        domain="modality",
        aliases=("ct", "scanner", "computed tomography"),
    ),
    ClinicalConcept(
        name="Chest radiography",
        omop_concept_id=2212815,
        snomed_ct_code="168731009",
        domain="modality",
        aliases=("xray", "x-ray", "radiography", "radiographie"),
    ),
    ClinicalConcept(
        name="Ultrasonography",
        omop_concept_id=2213023,
        snomed_ct_code="16310003",
        domain="modality",
        aliases=("ultrasound", "echography", "échographie"),
    ),
    ClinicalConcept(
        name="Segmentation procedure",
        omop_concept_id=4170639,
        snomed_ct_code="385653008",
        domain="procedure",
        aliases=("segmentation", "segment"),
    ),
    ClinicalConcept(
        name="Classification procedure",
        omop_concept_id=4031636,
        snomed_ct_code="308292007",
        domain="procedure",
        aliases=("classification", "classify", "triage"),
    ),
    ClinicalConcept(
        name="Area under ROC curve",
        omop_concept_id=4271761,
        snomed_ct_code="273249006",
        domain="measurement",
        aliases=("auc", "area under curve"),
    ),
    ClinicalConcept(
        name="Sensitivity",
        omop_concept_id=3012888,
        snomed_ct_code="260415000",
        domain="measurement",
        aliases=("sensitivity", "sensibilité"),
    ),
    ClinicalConcept(
        name="Specificity",
        omop_concept_id=3020416,
        snomed_ct_code="260416004",
        domain="measurement",
        aliases=("specificity", "spécificité"),
    ),
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def detect_omop_concepts(text: str) -> list[ClinicalConcept]:
    """Detect known OMOP/SNOMED concepts from free text."""
    normalized = _normalize(text)
    hits: list[ClinicalConcept] = []
    for concept in CONCEPTS:
        if any(alias in normalized for alias in concept.aliases):
            hits.append(concept)
    return hits


def build_omop_metadata(text: str) -> dict[str, list[int] | list[str]]:
    """Build OMOP-compatible metadata fields from plain text content."""
    concepts = detect_omop_concepts(text)
    condition_ids = sorted(
        {concept.omop_concept_id for concept in concepts if concept.domain == "condition"},
    )
    procedure_ids = sorted(
        {concept.omop_concept_id for concept in concepts if concept.domain == "procedure"},
    )
    measurement_ids = sorted(
        {concept.omop_concept_id for concept in concepts if concept.domain == "measurement"},
    )
    modality_ids = sorted(
        {concept.omop_concept_id for concept in concepts if concept.domain == "modality"},
    )
    snomed_codes = sorted({concept.snomed_ct_code for concept in concepts})

    metadata: dict[str, list[int] | list[str]] = {}
    if condition_ids:
        metadata["omop_condition_concept_ids"] = condition_ids
    if procedure_ids:
        metadata["omop_procedure_concept_ids"] = procedure_ids
    if measurement_ids:
        metadata["omop_measurement_concept_ids"] = measurement_ids
    if modality_ids:
        metadata["omop_modality_concept_ids"] = modality_ids
    if snomed_codes:
        metadata["snomed_ct_codes"] = snomed_codes
    return metadata
