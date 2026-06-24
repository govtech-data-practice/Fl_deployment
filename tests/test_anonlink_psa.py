"""Test anonlink + clkhash for Private Set Alignment (PSA).

Compares the current exact-match PSI approach with CLK-based fuzzy matching
to demonstrate why PSA is needed for real-world entity alignment.

Scenario: Two hospitals want to align patient records for VFL training.
- Hospital A has: name, DOB, gender
- Hospital B has: name (with typos), DOB (different format), gender
- No shared patient ID exists between the two systems.
"""

import json
import os
import sys
import time
from io import StringIO

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from clkhash import clk
from clkhash.schema import from_json_dict as schema_from_json_dict
import anonlink

# ---------------------------------------------------------------
# 1. Define test data — realistic healthcare quasi-identifiers
# ---------------------------------------------------------------

# Hospital A records (ground truth)
hospital_a = [
    {"name": "John Smith",      "dob": "1985-03-15", "gender": "M"},
    {"name": "Jane Doe",        "dob": "1990-07-22", "gender": "F"},
    {"name": "Robert Johnson",  "dob": "1978-11-03", "gender": "M"},
    {"name": "Maria Garcia",    "dob": "1995-01-30", "gender": "F"},
    {"name": "David Lee",       "dob": "1982-06-18", "gender": "M"},
    {"name": "Sarah Wilson",    "dob": "1988-09-05", "gender": "F"},
    {"name": "Michael Brown",   "dob": "1970-12-25", "gender": "M"},
    {"name": "Emily Davis",     "dob": "1993-04-10", "gender": "F"},
    # Records only in Hospital A (should NOT match)
    {"name": "Chris Taylor",    "dob": "1987-02-14", "gender": "M"},
    {"name": "Lisa Anderson",   "dob": "1991-08-28", "gender": "F"},
]

# Hospital B records — same patients but with noisy identifiers
hospital_b = [
    {"name": "Jon Smith",       "dob": "1985-03-15", "gender": "M"},   # Typo: John -> Jon
    {"name": "Jane Doe",        "dob": "1990-07-22", "gender": "F"},   # Exact match
    {"name": "Rob Johnson",     "dob": "1978-11-03", "gender": "M"},   # Shortened: Robert -> Rob
    {"name": "Maria Garsia",    "dob": "1995-01-30", "gender": "F"},   # Typo: Garcia -> Garsia
    {"name": "David Lee",       "dob": "1982-06-18", "gender": "M"},   # Exact match
    {"name": "Sara Wilson",     "dob": "1988-09-05", "gender": "F"},   # Typo: Sarah -> Sara
    {"name": "Micheal Brown",   "dob": "1970-12-25", "gender": "M"},   # Typo: Michael -> Micheal
    {"name": "Emily Davies",    "dob": "1993-04-10", "gender": "F"},   # Typo: Davis -> Davies
    # Records only in Hospital B (should NOT match)
    {"name": "Kevin Martinez",  "dob": "1984-05-20", "gender": "M"},
    {"name": "Anna Thomas",     "dob": "1996-10-12", "gender": "F"},
]

# Ground truth: first 8 records in each list correspond to the same patients
EXPECTED_MATCHES = 8


# ---------------------------------------------------------------
# 2. Test current PSI approach (exact match — will fail on typos)
# ---------------------------------------------------------------

def test_exact_psi():
    """Current PSI: hash-based exact matching on concatenated fields."""
    from fl_pets.psa.protocol import PSAProtocol

    protocol = PSAProtocol(mode="exact", salt=os.urandom(32))

    # Create composite keys (what you'd have to do without shared IDs)
    keys_a = [f"{r['name']}|{r['dob']}|{r['gender']}" for r in hospital_a]
    keys_b = [f"{r['name']}|{r['dob']}|{r['gender']}" for r in hospital_b]

    hashes_a = protocol.hash_identifiers(keys_a)
    hashes_b = protocol.hash_identifiers(keys_b)

    idx_a, idx_b = PSAProtocol.intersect(hashes_a, hashes_b)

    print("=" * 60)
    print("TEST 1: Current PSI (exact match)")
    print("=" * 60)
    print(f"Hospital A records: {len(hospital_a)}")
    print(f"Hospital B records: {len(hospital_b)}")
    print(f"Expected matches:   {EXPECTED_MATCHES}")
    print(f"Actual matches:     {len(idx_a)}")
    print()

    if idx_a:
        print("Matched pairs:")
        for ia, ib in zip(idx_a, idx_b):
            print(f"  A[{ia}] {hospital_a[ia]['name']:20s} <-> B[{ib}] {hospital_b[ib]['name']}")
    else:
        print("No exact matches found (names have typos).")

    # Show missed matches
    matched_a = set(idx_a)
    missed = [(i, hospital_a[i]["name"], hospital_b[i]["name"])
              for i in range(EXPECTED_MATCHES) if i not in matched_a]
    if missed:
        print(f"\nMissed {len(missed)} matches due to typos:")
        for i, name_a, name_b in missed:
            print(f"  A[{i}] '{name_a}' != B[{i}] '{name_b}'")

    print()
    return len(idx_a)


# ---------------------------------------------------------------
# 3. Test anonlink + clkhash (fuzzy CLK-based matching)
# ---------------------------------------------------------------

def test_clk_psa():
    """PSA with anonlink: Bloom filter CLK encoding + fuzzy matching."""

    # Define the linkage schema — which fields to use and how
    schema_dict = {
        "version": 3,
        "clkConfig": {
            "l": 1024,       # Bloom filter length (bits)
            "xor_folds": 0,  # No folding
            "kdf": {
                "type": "HKDF",
                "hash": "SHA256",
                "info": "cHNhLXRlc3Q=",
                "salt": "c2VjdXJlLXNhbHQ=",  # base64("secure-salt")
                "keySize": 64
            }
        },
        "features": [
            {
                "identifier": "name",
                "format": {"type": "string", "encoding": "utf-8"},
                "hashing": {
                    "comparison": {"type": "ngram", "n": 2},
                    "strategy": {"bitsPerFeature": 300},
                },
            },
            {
                "identifier": "dob",
                "format": {"type": "string", "encoding": "utf-8"},
                "hashing": {
                    "comparison": {"type": "ngram", "n": 1},
                    "strategy": {"bitsPerFeature": 200},
                },
            },
            {
                "identifier": "gender",
                "format": {"type": "enum", "values": ["M", "F"]},
                "hashing": {
                    "comparison": {"type": "exact"},
                    "strategy": {"bitsPerFeature": 100},
                },
            },
        ],
    }
    schema = schema_from_json_dict(schema_dict)

    # Encode records into CLKs (Cryptographic Longterm Keys)
    def records_to_clks(records, schema):
        """Convert records to CLK hashes via clkhash."""
        # Write records as CSV in-memory, then generate CLKs
        import csv
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(["name", "dob", "gender"])
        for r in records:
            writer.writerow([r["name"], r["dob"], r["gender"]])
        buf.seek(0)
        return clk.generate_clk_from_csv(buf, "secret-key", schema)

    t0 = time.time()
    clks_a = records_to_clks(hospital_a, schema)
    clks_b = records_to_clks(hospital_b, schema)
    encode_time = time.time() - t0

    # Run similarity-based matching
    t0 = time.time()

    # Compute candidate pairs using greedy solver
    # Threshold: minimum Dice similarity to consider a match
    THRESHOLD = 0.7
    results = anonlink.candidate_generation.find_candidate_pairs(
        [clks_a, clks_b],
        anonlink.similarities.dice_coefficient_accelerated,
        THRESHOLD
    )

    # Solve for best 1:1 matching (greedy)
    # Returns list of groups; each group is a list of (dataset_idx, record_idx)
    solution = anonlink.solving.greedy_solve(results)
    match_time = time.time() - t0

    # Extract matched pairs (groups with exactly one record from each dataset)
    matched_pairs = []
    for group in solution:
        records_by_dataset = {}
        for dataset_idx, record_idx in group:
            records_by_dataset[dataset_idx] = record_idx
        if 0 in records_by_dataset and 1 in records_by_dataset:
            matched_pairs.append((records_by_dataset[0], records_by_dataset[1]))

    print("=" * 60)
    print("TEST 2: PSA with anonlink + clkhash (fuzzy matching)")
    print("=" * 60)
    print(f"Hospital A records: {len(hospital_a)}")
    print(f"Hospital B records: {len(hospital_b)}")
    print(f"Expected matches:   {EXPECTED_MATCHES}")
    print(f"Actual matches:     {len(matched_pairs)}")
    print(f"Threshold:          {THRESHOLD}")
    print(f"Encode time:        {encode_time:.3f}s")
    print(f"Match time:         {match_time:.3f}s")
    print()

    # Analyse matches
    correct = 0
    false_positives = 0
    print("Matched pairs:")
    for idx_a, idx_b in sorted(matched_pairs):
        name_a = hospital_a[idx_a]["name"]
        name_b = hospital_b[idx_b]["name"]
        is_correct = idx_a < EXPECTED_MATCHES and idx_b < EXPECTED_MATCHES and idx_a == idx_b
        status = "OK" if is_correct else "FALSE POSITIVE"
        if is_correct:
            correct += 1
        else:
            false_positives += 1
        print(f"  A[{idx_a}] {name_a:20s} <-> B[{idx_b}] {name_b:20s}  {status}")

    missed = EXPECTED_MATCHES - correct
    print(f"\nResults:")
    print(f"  Correct matches:  {correct}/{EXPECTED_MATCHES}")
    print(f"  False positives:  {false_positives}")
    print(f"  Missed matches:   {missed}")

    precision = correct / max(correct + false_positives, 1)
    recall = correct / EXPECTED_MATCHES
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"  Precision:        {precision:.3f}")
    print(f"  Recall:           {recall:.3f}")
    print(f"  F1:               {f1:.3f}")
    print()

    return correct


# ---------------------------------------------------------------
# 4. Run both tests and compare
# ---------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Private Set Alignment (PSA) vs Private Set Intersection (PSI)")
    print("Scenario: Two hospitals, no shared patient IDs, names have typos")
    print()

    psi_matches = test_exact_psi()
    clk_matches = test_clk_psa()

    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Ground truth matches:      {EXPECTED_MATCHES}")
    print(f"PSI (exact match):         {psi_matches}/{EXPECTED_MATCHES} ({psi_matches/EXPECTED_MATCHES*100:.0f}% recall)")
    print(f"PSA (CLK fuzzy match):     {clk_matches}/{EXPECTED_MATCHES} ({clk_matches/EXPECTED_MATCHES*100:.0f}% recall)")
    print()
    if clk_matches > psi_matches:
        print("RESULT: PSA with anonlink recovers matches that PSI misses due to typos.")
        print("This is why fuzzy matching is essential for real-world entity alignment")
        print("where parties do not share common identifiers.")
    print()
