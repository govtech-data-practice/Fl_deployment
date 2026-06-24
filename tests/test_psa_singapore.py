"""PSA effectiveness test with realistic Singaporean patient data (small set).

Uses the hand-crafted 20-record dataset from data.generators.sg_synthetic.
"""

import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from data.generators.sg_synthetic import generate_small, add_address_noise
from psa.psa import PSAProtocol, ANONLINK_AVAILABLE

sgh_records, ttsh_records, EXPECTED_MATCHES = generate_small()

# CLK schema for Singapore data (name + DOB + address + gender)
SG_SCHEMA = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "cHNhLXNn", "salt": "c2VjdXJlLXNhbHQ=", "keySize": 64},
    },
    "features": [
        {"identifier": "name",    "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 300}}},
        {"identifier": "dob",     "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 150}}},
        {"identifier": "address", "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 250}}},
        {"identifier": "gender",  "format": {"type": "enum", "values": ["M", "F"]},  "hashing": {"comparison": {"type": "exact"}, "strategy": {"bitsPerFeature": 50}}},
    ],
}
SG_FIELDS = ["name", "dob", "address", "gender"]


def test_exact():
    protocol = PSAProtocol(mode="exact", salt=os.urandom(32))
    keys_a = [f"{r['name']}|{r['dob']}|{r['address']}|{r['gender']}" for r in sgh_records]
    keys_b = [f"{r['name']}|{r['dob']}|{r['address']}|{r['gender']}" for r in ttsh_records]
    hashes_a = protocol.hash_identifiers(keys_a)
    hashes_b = protocol.hash_identifiers(keys_b)
    idx_a, idx_b = PSAProtocol.intersect(hashes_a, hashes_b)

    print("=" * 72)
    print("TEST 1: Exact matching (HMAC hash)")
    print("=" * 72)
    print(f"Expected: {EXPECTED_MATCHES}  |  Matched: {len(idx_a)}")
    if idx_a:
        for ia, ib in zip(idx_a, idx_b):
            print(f"  [{ia:2d}] {sgh_records[ia]['name']:35s} <-> [{ib:2d}] {ttsh_records[ib]['name']}")
    missed = [i for i in range(EXPECTED_MATCHES) if i not in set(idx_a)]
    if missed:
        print(f"\nMissed {len(missed)} patients:")
        for i in missed:
            diffs = []
            if sgh_records[i]["name"] != ttsh_records[i]["name"]:
                diffs.append(f"name: '{sgh_records[i]['name']}' vs '{ttsh_records[i]['name']}'")
            if sgh_records[i]["address"] != ttsh_records[i]["address"]:
                diffs.append("addr diff")
            print(f"  [{i:2d}] {' | '.join(diffs)}")
    print()
    return len(idx_a)


def test_fuzzy(threshold=0.7):
    if not ANONLINK_AVAILABLE:
        print("anonlink not installed — skipping")
        return 0

    import anonlink
    from clkhash import clk
    from clkhash.schema import from_json_dict as schema_from_json_dict
    import csv
    from io import StringIO

    schema = schema_from_json_dict(SG_SCHEMA)

    def encode(records):
        buf = StringIO()
        w = csv.writer(buf)
        w.writerow(SG_FIELDS)
        for r in records:
            w.writerow([r.get(f, "") for f in SG_FIELDS])
        buf.seek(0)
        return clk.generate_clk_from_csv(buf, "sg-psa-key", schema)

    clks_a = encode(sgh_records)
    clks_b = encode(ttsh_records)
    results = anonlink.candidate_generation.find_candidate_pairs(
        [clks_a, clks_b], anonlink.similarities.dice_coefficient_accelerated, threshold)
    solution = anonlink.solving.greedy_solve(results)

    pairs = []
    for group in solution:
        by_ds = {}
        for ds, idx in group:
            by_ds[ds] = idx
        if 0 in by_ds and 1 in by_ds:
            pairs.append((by_ds[0], by_ds[1]))

    correct = sum(1 for ia, ib in pairs if ia < EXPECTED_MATCHES and ib < EXPECTED_MATCHES and ia == ib)

    print("=" * 72)
    print(f"TEST 2: Fuzzy CLK matching (threshold={threshold})")
    print("=" * 72)
    print(f"Expected: {EXPECTED_MATCHES}  |  Matched: {len(pairs)}  |  Correct: {correct}")
    for ia, ib in sorted(pairs):
        is_ok = ia < EXPECTED_MATCHES and ib < EXPECTED_MATCHES and ia == ib
        diff = " *" if sgh_records[ia]["name"] != ttsh_records[ib]["name"] else ""
        print(f"  [{ia:2d}] {sgh_records[ia]['name']:35s} <-> [{ib:2d}] {ttsh_records[ib]['name']:35s} {'OK' if is_ok else 'FP'}{diff}")
    print(f"\n  Precision: {correct/max(len(pairs),1):.3f}  Recall: {correct/EXPECTED_MATCHES:.3f}")
    print()
    return correct


if __name__ == "__main__":
    print("\nPSA — Singapore Healthcare Data (small set)\n")
    exact_n = test_exact()
    fuzzy_n = test_fuzzy()
    print(f"Summary: Exact {exact_n}/{EXPECTED_MATCHES}, Fuzzy {fuzzy_n}/{EXPECTED_MATCHES}\n")
