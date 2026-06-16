"""
Government Document Data Pipeline
==================================
Synthetic government document text for federated LLM fine-tuning.
Each client (agency) has its own domain-specific documents.

In production, each agency would ingest real documents via ingest.py.
"""

import numpy as np
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger("pipeline.gov_doc")

# Domain-specific document templates per agency type
AGENCY_TEMPLATES = {
    "healthcare": [
        "Patient {id} admitted with {condition}. Vitals: BP {bp}, HR {hr}, SpO2 {spo2}. Treatment plan: {treatment}.",
        "Discharge summary for patient {id}: diagnosis {condition}, length of stay {los} days, follow-up in {followup} weeks.",
        "Lab results for patient {id}: WBC {wbc}, CRP {crp}, lactate {lactate}. Assessment: {assessment}.",
        "Radiology report: {modality} of {region} shows {finding}. Impression: {impression}.",
        "Clinical note: patient {id} presents with {symptoms}. History of {history}. Plan: {plan}.",
    ],
    "finance": [
        "Transaction alert: account {id} flagged for {reason}. Amount: ${amount}. Risk score: {score}/100.",
        "Compliance review for entity {id}: {status}. AML risk level: {risk}. Last review: {date}.",
        "Suspicious activity report: {count} transactions totaling ${total} over {period} days. Pattern: {pattern}.",
        "KYC verification for customer {id}: identity {verified}. Source of funds: {source}.",
        "Audit finding: {department} shows {finding}. Severity: {severity}. Remediation due: {due_date}.",
    ],
    "urban_planning": [
        "Land use application {id}: zone {zone} to {new_zone}. Plot area: {area} sqm. Status: {status}.",
        "Environmental impact assessment for project {id}: air quality {aq_status}, noise level {noise}dB.",
        "Traffic study for junction {id}: peak hour volume {volume} vehicles. LOS: {los}. Recommendation: {rec}.",
        "Building permit {id}: {floors} floors, {use} use. GFA: {gfa} sqm. Compliance: {compliance}.",
        "Public consultation for {project}: {respondents} responses. Support: {support}%. Key concern: {concern}.",
    ],
    "research": [
        "Grant application {id}: {title}. PI: {pi}. Budget: ${budget}. Duration: {duration} months.",
        "Publication: {title}. Authors: {authors}. Journal: {journal}. Impact factor: {if_score}.",
        "Patent filing {id}: {title}. Inventors: {inventors}. Priority date: {date}. Status: {status}.",
        "Research ethics review for protocol {id}: {title}. Risk level: {risk}. Approval: {approval}.",
        "Collaboration agreement between {org1} and {org2}. Scope: {scope}. IP ownership: {ip}.",
    ],
}

DOMAINS = list(AGENCY_TEMPLATES.keys())


def _fill_template(template: str, rng: np.random.RandomState) -> str:
    """Fill a template with random plausible values."""
    replacements = {
        "{id}": str(rng.randint(10000, 99999)),
        "{condition}": rng.choice(["sepsis", "pneumonia", "cardiac arrest", "stroke", "diabetes complications"]),
        "{bp}": f"{rng.randint(90, 180)}/{rng.randint(60, 110)}",
        "{hr}": str(rng.randint(50, 140)),
        "{spo2}": str(rng.randint(88, 100)),
        "{treatment}": rng.choice(["IV antibiotics", "mechanical ventilation", "surgery", "observation", "medication adjustment"]),
        "{los}": str(rng.randint(1, 30)),
        "{followup}": str(rng.randint(1, 12)),
        "{wbc}": f"{rng.uniform(2, 20):.1f}",
        "{crp}": f"{rng.uniform(0.1, 300):.1f}",
        "{lactate}": f"{rng.uniform(0.5, 8):.1f}",
        "{assessment}": rng.choice(["stable", "improving", "deteriorating", "critical"]),
        "{modality}": rng.choice(["CT", "MRI", "X-ray", "ultrasound"]),
        "{region}": rng.choice(["chest", "abdomen", "head", "spine"]),
        "{finding}": rng.choice(["no acute findings", "consolidation", "mass", "fracture", "effusion"]),
        "{impression}": rng.choice(["normal study", "follow-up recommended", "urgent intervention needed"]),
        "{symptoms}": rng.choice(["fever and cough", "chest pain", "abdominal pain", "confusion", "shortness of breath"]),
        "{history}": rng.choice(["hypertension", "diabetes", "none significant", "previous MI", "COPD"]),
        "{plan}": rng.choice(["admit for observation", "discharge with follow-up", "transfer to ICU", "consult specialist"]),
        "{reason}": rng.choice(["unusual pattern", "high value", "velocity check", "sanctions match"]),
        "{amount}": f"{rng.randint(100, 1000000):,}",
        "{score}": str(rng.randint(1, 100)),
        "{status}": rng.choice(["approved", "pending", "rejected", "under review"]),
        "{risk}": rng.choice(["low", "medium", "high", "critical"]),
        "{date}": f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "{count}": str(rng.randint(2, 50)),
        "{total}": f"{rng.randint(1000, 10000000):,}",
        "{period}": str(rng.randint(1, 90)),
        "{pattern}": rng.choice(["structuring", "layering", "round-tripping", "smurfing"]),
        "{verified}": rng.choice(["verified", "pending verification", "failed"]),
        "{source}": rng.choice(["employment", "business income", "inheritance", "investment returns"]),
        "{department}": rng.choice(["operations", "compliance", "treasury", "IT"]),
        "{severity}": rng.choice(["low", "medium", "high", "critical"]),
        "{due_date}": f"2026-{rng.randint(6,12):02d}-{rng.randint(1,28):02d}",
        "{zone}": rng.choice(["residential", "commercial", "industrial", "mixed-use"]),
        "{new_zone}": rng.choice(["residential", "commercial", "mixed-use", "green space"]),
        "{area}": str(rng.randint(100, 50000)),
        "{aq_status}": rng.choice(["good", "moderate", "unhealthy"]),
        "{noise}": str(rng.randint(40, 90)),
        "{volume}": str(rng.randint(500, 5000)),
        "{rec}": rng.choice(["no changes needed", "signal optimization", "road widening", "traffic calming"]),
        "{floors}": str(rng.randint(1, 50)),
        "{use}": rng.choice(["residential", "commercial", "industrial", "institutional"]),
        "{gfa}": str(rng.randint(500, 100000)),
        "{compliance}": rng.choice(["compliant", "non-compliant", "conditional"]),
        "{project}": rng.choice(["MRT extension", "HDB development", "park connector", "expressway"]),
        "{respondents}": str(rng.randint(50, 5000)),
        "{support}": str(rng.randint(20, 95)),
        "{concern}": rng.choice(["noise", "traffic", "environment", "heritage", "cost"]),
        "{title}": rng.choice(["Novel approach to", "Evaluation of", "Development of", "Assessment of"]) + " " + rng.choice(["drug targets", "policy outcomes", "disease markers", "safety protocols"]),
        "{pi}": rng.choice(["Dr. Tan", "Prof. Lee", "Dr. Chen", "Prof. Lim"]),
        "{budget}": f"{rng.randint(50, 5000):,}K",
        "{duration}": str(rng.randint(6, 60)),
        "{authors}": rng.choice(["Tan et al.", "Lee & Wong", "Chen, Lim, Ng"]),
        "{journal}": rng.choice(["Nature Medicine", "Lancet", "BMJ", "JAMA"]),
        "{if_score}": f"{rng.uniform(1, 90):.1f}",
        "{inventors}": rng.choice(["Tan, Lee", "Chen, Wong", "Lim, Ng"]),
        "{org1}": rng.choice(["A*STAR", "NUS", "NTU", "SUTD"]),
        "{org2}": rng.choice(["MOH", "NEA", "MAS", "GovTech"]),
        "{scope}": rng.choice(["joint research", "technology transfer", "data sharing", "clinical trials"]),
        "{ip}": rng.choice(["shared", "originator", "licensee"]),
        "{approval}": rng.choice(["approved", "conditional", "pending revision"]),
        "{finding}": rng.choice(["gap identified", "process improvement needed", "compliance achieved"]),
    }
    result = template
    for k, v in replacements.items():
        if k in result:
            result = result.replace(k, str(v), 1)
    return result


def generate_documents(
    num_docs: int = 500,
    domain: str = "healthcare",
    seed: int = 42,
) -> List[str]:
    """Generate synthetic documents for a specific domain."""
    rng = np.random.RandomState(seed)
    templates = AGENCY_TEMPLATES.get(domain, AGENCY_TEMPLATES["healthcare"])
    docs = []
    for _ in range(num_docs):
        template = rng.choice(templates)
        docs.append(_fill_template(template, rng))
    return docs


def prepare_federated_data(
    data_path: str = "",
    num_clients: int = 5,
    partition_type: str = "iid",
    alpha: float = 0.5,
    batch_size: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int = 0,
    seed: int = 42,
    local_mode: bool = False,
) -> Tuple[Dict, Dict]:
    """Generate federated document data. Each client gets a different domain."""
    total = max_samples if max_samples > 0 else 500
    rng = np.random.RandomState(seed)

    # Each client gets a different domain (non-IID by nature)
    loaders = {}
    for cid in range(num_clients):
        domain = DOMAINS[cid % len(DOMAINS)]
        docs = generate_documents(total // num_clients, domain, seed=seed + cid)
        n = len(docs)
        n_val = int(n * val_ratio)
        n_train = n - n_val

        loaders[cid] = {
            "train": docs[:n_train],
            "val": docs[n_train:],
            "domain": domain,
        }
        logger.info(f"Client {cid}: {domain}, {n_train} train, {n_val} val docs")

    metadata = {
        "num_clients": num_clients,
        "total_docs": total,
        "domains": [DOMAINS[i % len(DOMAINS)] for i in range(num_clients)],
    }
    return loaders, metadata
