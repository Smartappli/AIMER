# OMOP terminology import via Athena

Le service `omop-athena-import` du `docker-compose.yml` charge les fichiers de vocabulaire Athena directement dans les tables standard du schéma vocabulaire OMOP (`${OMOP_VOCAB_SCHEMA}`).

## Fichiers attendus (Athena)

Placer dans `./omop/data/` (ou le dossier configuré via `OMOP_ATHENA_DIR`) :

- `VOCABULARY.csv`
- `DOMAIN.csv`
- `CONCEPT_CLASS.csv`
- `RELATIONSHIP.csv`
- `CONCEPT.csv`
- `CONCEPT_RELATIONSHIP.csv`
- `CONCEPT_ANCESTOR.csv`
- `CONCEPT_SYNONYM.csv`
- `DRUG_STRENGTH.csv`

## Comportement d'import

- Le service vérifie d'abord que les tables OMOP de vocabulaire existent déjà :
  - `vocabulary`
  - `domain`
  - `concept_class`
  - `relationship`
  - `concept`
  - `concept_relationship`
  - `concept_ancestor`
  - `concept_synonym`
  - `drug_strength`
- Les tables sont vidées dans l'ordre compatible avec les contraintes, puis rechargées depuis les fichiers Athena.
- L'import utilise le format Athena tabulé (`DELIMITER E'\t'`) avec en-tête CSV.

Variable configurable :

- `OMOP_ATHENA_DIR` (défaut: `/import-data`)
