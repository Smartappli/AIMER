# Asset and ICT Supplier Register

This register is the repository-level inventory baseline. Production operators
must extend it with deployment-specific owners, locations, contracts, data
flows and supplier identifiers. When AIMER is used by a DORA in-scope financial
entity or as an ICT provider to one, maintain the formal DORA register of
information in the required authority format in addition to this file.

## Asset Inventory

| Asset | Type | Criticality | Data class | Owner | Dependencies | Production boundary |
| --- | --- | --- | --- | --- | --- | --- |
| `aimer-web` | Django web application | High | Account, operational, possible PHI | TBD | PostgreSQL, `aimer-rag`, email, SIEM | Public ingress only |
| `aimer-rag` | RAG API/runtime | High | Query text, embeddings, retrieved documents, possible PHI | TBD | Qdrant, OpenRAG, model/runtime dependencies | Private service |
| `MAGE` | ML/API service | Medium/High | Model metadata and inference payloads | TBD | Model libraries, runtime images | Private service |
| `FARM` | Django/Federated learning service | High | Federated task metadata, possible clinical payloads | TBD | PostgreSQL, Flower, hospital node TLS | Private service |
| PostgreSQL | Database | High | Application data and audit records | TBD | Backup/KMS, private network | Private data tier |
| Qdrant/vector store | Vector database | High | Embeddings and document metadata | TBD | Backup/KMS, private network | Private data tier |
| OpenRAG endpoint | External/internal ICT service | High | Retrieval requests and evidence | TBD | API key, network route | Private or controlled egress |
| SIEM/log platform | Security monitoring | High | Security logs and audit events | TBD | Log forwarders, retention storage | Restricted ops access |
| Container registry | Supply chain | High | Runtime image artifacts | TBD | GitHub Actions, Cosign, SBOM artifacts | Deployment source |
| Backup storage | Resilience | Critical | Encrypted database and object backups | TBD | KMS, restore tooling | Restricted ops access |

## ICT Supplier Register

| Supplier/service | Provides | Critical or important function | Data handled | Contract owner | Location/jurisdiction | Exit plan | Review by |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cloud/infrastructure provider | Compute, network, storage | TBD | TBD | TBD | TBD | TBD | TBD |
| Database managed service | PostgreSQL or storage services | TBD | TBD | TBD | TBD | TBD | TBD |
| Vector database provider | Qdrant or equivalent | TBD | TBD | TBD | TBD | TBD | TBD |
| Model/RAG provider | OpenRAG, model hosting or model API | TBD | TBD | TBD | TBD | TBD | TBD |
| Email provider | Verification and password reset delivery | TBD | Account emails | TBD | TBD | TBD | TBD |
| Container registry | Runtime image hosting | Yes | Image artifacts | TBD | TBD | Mirror or alternate registry | TBD |
| CI/CD provider | Build, tests, release evidence | Yes | Source and artifacts | TBD | TBD | Manual build/deploy path | TBD |
| SIEM provider | Security monitoring and retention | Yes | Security logs | TBD | TBD | Export and alternate collector | TBD |

## Due Diligence Checks

Before onboarding or renewing an ICT supplier, record evidence for:

- Security standard or audit report, such as ISO 27001, SOC 2 or equivalent.
- Data processing agreement and subprocessor list when personal data is handled.
- Encryption at rest and in transit.
- Incident notification commitments and contact route.
- Audit, access, inspection and evidence rights.
- Backup, continuity and disaster recovery commitments.
- Data location and cross-border transfer controls.
- Exit support, data export, deletion and portability.
- Concentration risk and viable alternative provider.

## Change Control

Update this register when:

- A new service, datastore, model endpoint, CI/CD system or supplier is added.
- A supplier starts supporting a critical or important function.
- Data classification or processing location changes.
- A contract, SLA, incident route or exit plan changes.
- A production release materially changes an asset dependency.

