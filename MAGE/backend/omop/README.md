# OMOP terminology import (SNOMED CT + ICD-10)

Le service `omop-terminology-import` du `docker-compose.yml` charge automatiquement :

- `./omop/data/snomed_ct.csv`
- `./omop/data/icd10.csv`

vers les tables :

- `${OMOP_VOCAB_SCHEMA}.snomed_ct_raw`
- `${OMOP_VOCAB_SCHEMA}.icd10_raw`

## Format attendu

### SNOMED CT (`snomed_ct.csv`)

```csv
concept_id,concept_name,snomed_code,domain_id,vocabulary_id
```

### ICD-10 (`icd10.csv`)

```csv
concept_id,concept_name,icd10_code,chapter,vocabulary_id
```

Les chemins peuvent être redéfinis via :

- `OMOP_SNOMED_FILE`
- `OMOP_ICD10_FILE`
