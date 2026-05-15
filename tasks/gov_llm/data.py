"""
Government LLM Data Pipeline
=============================
Pipeline: Generate → Validate → Clean → Partition → Return

3 agencies: Tax Authority, Immigration Agency, Public Health Agency.
Synthetic template-based text for federated LLM fine-tuning.
"""
import random
import logging
from typing import Dict, List

logger = logging.getLogger("pipeline.gov_llm")

AGENCY_NAMES = {0: "Tax Authority", 1: "Immigration Agency", 2: "Public Health Agency"}

_TAX_TEMPLATES = [
    "{tid} taxpayer filed {rtype} return showing income ${inc}K, deductions ${ded}K. {status}.",
    "Audit case {tid}: {rtype} filing, {issue}. {action}.",
    "Revenue assessment {tid}: {rtype}, reported income ${inc}K. {status}.",
]
_TAX_RTYPES = ["individual", "corporate", "partnership", "trust", "estate"]
_TAX_STATUS = ["Flagged for audit", "Accepted", "Under review", "Penalty assessed",
               "Refund approved", "Schedule C discrepancy noted", "Referred to enforcement"]
_TAX_ISSUES = ["unreported offshore income", "excessive business deductions",
               "transfer pricing violation", "charitable donation inflation",
               "payroll tax discrepancy", "cryptocurrency gains unreported"]
_TAX_ACTIONS = ["Full audit initiated", "Desk audit scheduled", "Information request sent",
                "Case closed - no change", "Penalty notice issued", "Criminal referral"]

_IMMIG_TEMPLATES = [
    "{nat} national, age {age}, {vtype} visa application. {doc}. {decision}.",
    "Case {cid}: {nat} applicant for {vtype} permit. {check}. {decision}.",
]
_IMMIG_NATS = ["Chinese", "Indian", "Filipino", "Vietnamese", "Indonesian",
               "Malaysian", "Thai", "Korean", "Japanese", "Australian",
               "British", "American", "Nigerian", "Brazilian", "Pakistani"]
_IMMIG_VTYPES = ["work", "student", "tourist", "dependent", "skilled worker",
                 "investor", "refugee", "permanent resident"]
_IMMIG_DOCS = ["Documents verified", "Interview required", "Background check pending",
               "Medical exam complete", "Sponsor confirmed", "Police clearance received"]
_IMMIG_DECISIONS = ["Approved", "Denied", "Further review", "Conditional approval",
                    "Referred to security", "Expedited processing", "Appeal lodged"]

_HEALTH_TEMPLATES = [
    "Region {reg} reported {n} new {disease} cases. {alert}. Population {pop}K. {response}.",
    "Surveillance report: {disease} in {reg} district. {n} confirmed, {n2} suspected. {response}.",
]
_HEALTH_DISEASES = ["influenza", "dengue", "tuberculosis", "hepatitis B", "COVID-19",
                    "measles", "food poisoning", "respiratory illness", "hand-foot-mouth",
                    "leptospirosis", "chikungunya"]
_HEALTH_REGIONS = ["North", "South", "East", "West", "Central", "Metro", "Rural", "Coastal"]
_HEALTH_ALERTS = ["Cluster detected", "Endemic level", "Outbreak declared",
                  "Under surveillance", "Contact tracing initiated", "Epidemic threshold crossed"]
_HEALTH_RESPONSES = ["Vaccination campaign launched", "Travel advisory issued",
                     "Hospital surge capacity activated", "Quarantine measures in place",
                     "Public health advisory released", "Vector control deployed"]


# ── Step 1: Generate ─────────────────────────────────────────────────

def _gen_tax(seed):
    r = random.Random(seed)
    tpl = r.choice(_TAX_TEMPLATES)
    return tpl.format(
        tid=r.randint(1000, 9999), rtype=r.choice(_TAX_RTYPES),
        inc=r.randint(30, 500), ded=r.randint(5, 100),
        status=r.choice(_TAX_STATUS), issue=r.choice(_TAX_ISSUES),
        action=r.choice(_TAX_ACTIONS),
    )


def _gen_immig(seed):
    r = random.Random(seed)
    tpl = r.choice(_IMMIG_TEMPLATES)
    return tpl.format(
        nat=r.choice(_IMMIG_NATS), age=r.randint(18, 65),
        vtype=r.choice(_IMMIG_VTYPES), doc=r.choice(_IMMIG_DOCS),
        decision=r.choice(_IMMIG_DECISIONS), check=r.choice(_IMMIG_DOCS),
        cid=r.randint(10000, 99999),
    )


def _gen_health(seed):
    r = random.Random(seed)
    tpl = r.choice(_HEALTH_TEMPLATES)
    return tpl.format(
        reg=r.choice(_HEALTH_REGIONS), n=r.randint(5, 500),
        n2=r.randint(1, 200), disease=r.choice(_HEALTH_DISEASES),
        alert=r.choice(_HEALTH_ALERTS), pop=r.randint(50, 5000),
        response=r.choice(_HEALTH_RESPONSES),
    )


_GENERATORS = {0: _gen_tax, 1: _gen_immig, 2: _gen_health}


# ── Step 2: Validate ─────────────────────────────────────────────────

def validate_notes(notes: List[str], agency_name: str) -> int:
    """Check text quality: empty strings, min length, duplicates."""
    n_empty = sum(1 for n in notes if not n or not n.strip())
    n_short = sum(1 for n in notes if len(n) < 20)
    n_unique = len(set(notes))
    n_dupes = len(notes) - n_unique

    logger.info(f"[Validate] {agency_name}: {len(notes)} notes, "
                f"empty={n_empty}, short={n_short}, duplicates={n_dupes}, "
                f"avg_len={sum(len(n) for n in notes)/max(len(notes),1):.0f} chars")

    if n_empty > 0:
        logger.warning(f"[Validate] {agency_name}: {n_empty} empty notes")
    if n_dupes > len(notes) * 0.1:
        logger.warning(f"[Validate] {agency_name}: {n_dupes} duplicates ({n_dupes/len(notes)*100:.1f}%)")
    return n_empty


# ── Step 3: Clean ────────────────────────────────────────────────────

def clean_notes(notes: List[str]) -> List[str]:
    """Remove empty notes, strip whitespace."""
    cleaned = [n.strip() for n in notes if n and n.strip()]
    n_removed = len(notes) - len(cleaned)
    if n_removed > 0:
        logger.info(f"[Clean] Removed {n_removed} empty/whitespace notes")
    return cleaned


# ── Step 4: Generate + Pipeline ──────────────────────────────────────

def generate_agency_data(agency_id, num_notes=200, seed=42):
    """Generate and validate synthetic notes for a government agency."""
    gen = _GENERATORS[agency_id]
    name = AGENCY_NAMES[agency_id]

    # ── Generate ──
    logger.info(f"[Generate] {name}: {num_notes} notes (seed={seed})")
    notes = [gen(seed + i) for i in range(num_notes)]

    # ── Validate ──
    validate_notes(notes, name)

    # ── Clean ──
    notes = clean_notes(notes)

    return notes


def generate_nonmember_data(num_notes=100, seed=9999):
    """Generate held-out notes for MIA testing (not used in training)."""
    r = random.Random(seed)
    notes = []
    for i in range(num_notes):
        gen = r.choice(list(_GENERATORS.values()))
        notes.append(gen(seed + 10000 + i))

    logger.info(f"[Generate] Nonmember set: {len(notes)} notes for MIA testing")
    validate_notes(notes, "Nonmember")
    notes = clean_notes(notes)
    return notes


def get_all_agency_data(num_notes_per_agency=200, seed=42) -> Dict[int, List[str]]:
    """Returns dict {agency_id: list_of_notes} for all 3 agencies."""
    logger.info(f"[Pipeline] Generating data for {len(AGENCY_NAMES)} agencies, "
                f"{num_notes_per_agency} notes each")
    data = {}
    for aid in range(3):
        data[aid] = generate_agency_data(aid, num_notes_per_agency, seed + aid * 1000)
    total = sum(len(v) for v in data.values())
    logger.info(f"[Pipeline] Total: {total} notes across {len(data)} agencies")
    return data
