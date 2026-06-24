"""PSA effectiveness test — 10,000 Singaporean patient records.

Uses data.generators.sg_synthetic for record generation.
Tests exact matching, single-pass fuzzy, and double PSA triangulation.
"""

import csv
import os
import sys
import time
from io import StringIO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from data.generators.sg_synthetic import generate_records

# ---------------------------------------------------------------
# CLK schemas
# ---------------------------------------------------------------

SG_SCHEMA_FULL = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "cHNhLXNnLTEwaw==", "salt": "c2VjdXJlLXNhbHQ=", "keySize": 64},
    },
    "features": [
        {"identifier": "name",    "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 300}}},
        {"identifier": "dob",     "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 100}}},
        {"identifier": "address", "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 200}}},
        {"identifier": "gender",  "format": {"type": "enum", "values": ["M", "F"]},  "hashing": {"comparison": {"type": "exact"}, "strategy": {"bitsPerFeature": 50}}},
        {"identifier": "age",     "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 50}}},
        {"identifier": "income",  "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 50}}},
    ],
}
SG_FIELDS_FULL = ["name", "dob", "address", "gender", "age", "income"]

SCHEMA_IDENTITY = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "cHNhLWlk", "salt": "aWQtc2FsdA==", "keySize": 64},
    },
    "features": [
        {"identifier": "name",   "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 400}}},
        {"identifier": "dob",    "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 200}}},
        {"identifier": "gender", "format": {"type": "enum", "values": ["M", "F"]},  "hashing": {"comparison": {"type": "exact"}, "strategy": {"bitsPerFeature": 100}}},
    ],
}
FIELDS_IDENTITY = ["name", "dob", "gender"]

SCHEMA_LOCATION = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "cHNhLWxvYw==", "salt": "bG9jLXNhbHQ=", "keySize": 64},
    },
    "features": [
        {"identifier": "address", "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 400}}},
        {"identifier": "age",     "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 150}}},
        {"identifier": "income",  "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 100}}},
    ],
}
FIELDS_LOCATION = ["address", "age", "income"]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _encode_clks(records, schema_dict, fields, secret):
    from clkhash import clk
    from clkhash.schema import from_json_dict as schema_from_json_dict
    schema = schema_from_json_dict(schema_dict)
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(fields)
    for r in records:
        writer.writerow([r.get(f, "") for f in fields])
    buf.seek(0)
    return clk.generate_clk_from_csv(buf, secret, schema)


def _match_clks(clks_a, clks_b, threshold):
    import anonlink
    results = anonlink.candidate_generation.find_candidate_pairs(
        [clks_a, clks_b], anonlink.similarities.dice_coefficient_accelerated, threshold)
    solution = anonlink.solving.greedy_solve(results)
    pairs = set()
    for group in solution:
        by_ds = {}
        for ds, idx in group:
            by_ds[ds] = idx
        if 0 in by_ds and 1 in by_ds:
            pairs.add((by_ds[0], by_ds[1]))
    return pairs


def _score(pairs, n_true):
    correct = sum(1 for ia, ib in pairs if ia < n_true and ib < n_true and ia == ib)
    fp = len(pairs) - correct
    missed = n_true - correct
    prec = correct / max(len(pairs), 1)
    rec = correct / n_true
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return correct, fp, missed, prec, rec, f1


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

def test_exact(clean, noisy, n_true):
    from psa.psa import PSAProtocol
    protocol = PSAProtocol(mode="exact", salt=os.urandom(32))
    keys_a = ["|".join(r[f] for f in SG_FIELDS_FULL) for r in clean]
    keys_b = ["|".join(r[f] for f in SG_FIELDS_FULL) for r in noisy]
    t0 = time.time()
    hashes_a = protocol.hash_identifiers(keys_a)
    hashes_b = protocol.hash_identifiers(keys_b)
    idx_a, idx_b = PSAProtocol.intersect(hashes_a, hashes_b)
    elapsed = time.time() - t0
    correct = sum(1 for ia, ib in zip(idx_a, idx_b) if ia < n_true and ib < n_true and ia == ib)

    print("=" * 72)
    print("TEST 1: Exact matching (HMAC hash)")
    print("=" * 72)
    print(f"Records: {len(clean):,} / {len(noisy):,}  |  True: {n_true:,}  |  Found: {len(idx_a):,}  |  Correct: {correct:,}  |  {elapsed:.3f}s")
    print()
    return correct


def test_fuzzy(clean, noisy, n_true, threshold=0.7):
    t0 = time.time()
    clks_a = _encode_clks(clean, SG_SCHEMA_FULL, SG_FIELDS_FULL, "sg-psa-10k")
    clks_b = _encode_clks(noisy, SG_SCHEMA_FULL, SG_FIELDS_FULL, "sg-psa-10k")
    encode_time = time.time() - t0

    t0 = time.time()
    pairs = _match_clks(clks_a, clks_b, threshold)
    match_time = time.time() - t0

    correct, fp, missed, prec, rec, f1 = _score(pairs, n_true)

    print("=" * 72)
    print(f"TEST 2: Single-pass fuzzy (threshold={threshold})")
    print("=" * 72)
    print(f"Correct: {correct:,}  |  FP: {fp}  |  Missed: {missed}")
    print(f"Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    print(f"Time: {encode_time + match_time:.2f}s")

    if fp > 0:
        fp_ex = [(ia, ib) for ia, ib in sorted(pairs) if not (ia < n_true and ib < n_true and ia == ib)][:3]
        print(f"  FP examples: {', '.join(f'{clean[ia][\"name\"]} <-> {noisy[ib][\"name\"]}' for ia, ib in fp_ex)}")
    print()
    return correct


def test_threshold_sweep(clean, noisy, n_true):
    clks_a = _encode_clks(clean, SG_SCHEMA_FULL, SG_FIELDS_FULL, "sg-psa-10k")
    clks_b = _encode_clks(noisy, SG_SCHEMA_FULL, SG_FIELDS_FULL, "sg-psa-10k")

    print("=" * 72)
    print("TEST 3: Threshold sweep (single-pass)")
    print("=" * 72)
    print(f"{'Thresh':>7}  {'Match':>7}  {'Correct':>8}  {'FP':>5}  {'Miss':>5}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    print("-" * 62)
    for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pairs = _match_clks(clks_a, clks_b, t)
        correct, fp, missed, prec, rec, f1 = _score(pairs, n_true)
        print(f"{t:>7.2f}  {len(pairs):>7,}  {correct:>8,}  {fp:>5}  {missed:>5}  {prec:>7.4f}  {rec:>7.4f}  {f1:>7.4f}")
    print()


def test_triangulation(clean, noisy, n_true, t_id=0.7, t_loc=0.7):
    print("=" * 72)
    print(f"TEST 4: Double PSA — Triangulation (t_id={t_id}, t_loc={t_loc})")
    print("=" * 72)

    t0 = time.time()
    clks_a_id = _encode_clks(clean, SCHEMA_IDENTITY, FIELDS_IDENTITY, "sg-identity")
    clks_b_id = _encode_clks(noisy, SCHEMA_IDENTITY, FIELDS_IDENTITY, "sg-identity")
    pairs_id = _match_clks(clks_a_id, clks_b_id, t_id)
    t_pass1 = time.time() - t0
    c1, fp1, _, _, _, _ = _score(pairs_id, n_true)
    print(f"  Pass 1 (name+DOB+gender):    {len(pairs_id):>6,} pairs  (correct={c1:,}, FP={fp1:,})  {t_pass1:.2f}s")

    t0 = time.time()
    clks_a_loc = _encode_clks(clean, SCHEMA_LOCATION, FIELDS_LOCATION, "sg-location")
    clks_b_loc = _encode_clks(noisy, SCHEMA_LOCATION, FIELDS_LOCATION, "sg-location")
    pairs_loc = _match_clks(clks_a_loc, clks_b_loc, t_loc)
    t_pass2 = time.time() - t0
    c2, fp2, _, _, _, _ = _score(pairs_loc, n_true)
    print(f"  Pass 2 (addr+age+income):    {len(pairs_loc):>6,} pairs  (correct={c2:,}, FP={fp2:,})  {t_pass2:.2f}s")

    pairs_tri = pairs_id & pairs_loc
    correct, fp, missed, prec, rec, f1 = _score(pairs_tri, n_true)
    print(f"\n  Triangulated:                {len(pairs_tri):>6,} pairs")
    print(f"  Correct: {correct:,}  |  FP: {fp}  |  Missed: {missed}")
    print(f"  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    print(f"  FP eliminated: {fp1 - fp:,} (from pass 1)")
    print(f"  Total time: {t_pass1 + t_pass2:.2f}s")

    if fp > 0:
        fp_ex = [(ia, ib) for ia, ib in sorted(pairs_tri) if not (ia < n_true and ib < n_true and ia == ib)][:3]
        print(f"  Remaining FP: {', '.join(f'{clean[ia][\"name\"]} <-> {noisy[ib][\"name\"]}' for ia, ib in fp_ex)}")
    print()
    return correct, fp


def test_triangulation_sweep(clean, noisy, n_true):
    # Pre-encode CLKs once to avoid re-encoding per threshold
    clks_cache = {}
    for label, schema, fields, secret in [
        ("id", SCHEMA_IDENTITY, FIELDS_IDENTITY, "sg-identity"),
        ("loc", SCHEMA_LOCATION, FIELDS_LOCATION, "sg-location"),
    ]:
        clks_cache[f"{label}_a"] = _encode_clks(clean, schema, fields, secret)
        clks_cache[f"{label}_b"] = _encode_clks(noisy, schema, fields, secret)

    print("=" * 72)
    print("TEST 5: Triangulation threshold sweep")
    print("=" * 72)
    print(f"{'t_id':>6}  {'t_loc':>6}  {'Pairs':>7}  {'Correct':>8}  {'FP':>5}  {'Miss':>5}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    print("-" * 62)

    for t_id in [0.6, 0.65, 0.7, 0.75]:
        for t_loc in [0.6, 0.65, 0.7, 0.75]:
            pairs_id = _match_clks(clks_cache["id_a"], clks_cache["id_b"], t_id)
            pairs_loc = _match_clks(clks_cache["loc_a"], clks_cache["loc_b"], t_loc)
            pairs_tri = pairs_id & pairs_loc
            correct, fp, missed, prec, rec, f1 = _score(pairs_tri, n_true)
            print(f"{t_id:>6.2f}  {t_loc:>6.2f}  {len(pairs_tri):>7,}  {correct:>8,}  {fp:>5}  {missed:>5}  {prec:>7.4f}  {rec:>7.4f}  {f1:>7.4f}")
    print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    N = 10_000

    print()
    print(f"Generating {N:,} synthetic Singaporean patient records...")
    t0 = time.time()
    clean, noisy, n_true, n_hard = generate_records(N)
    print(f"Generated in {time.time() - t0:.2f}s")
    print(f"Hospital A: {len(clean):,}  |  Hospital B: {len(noisy):,}  |  True matches: {n_true:,}  |  Hard negatives: {n_hard:,}")
    print()

    # Samples
    print("Sample records:")
    for i in [0, 500, 2000, 7500]:
        c, n = clean[i], noisy[i]
        print(f"  A[{i:5d}] {c['name']:35s}  →  B[{i:5d}] {n['name']}")
    print()

    exact_n = test_exact(clean, noisy, n_true)
    fuzzy_n = test_fuzzy(clean, noisy, n_true, threshold=0.7)
    test_threshold_sweep(clean, noisy, n_true)
    tri_n, tri_fp = test_triangulation(clean, noisy, n_true)
    test_triangulation_sweep(clean, noisy, n_true)

    prec_t = tri_n / max(tri_n + tri_fp, 1)
    rec_t = tri_n / n_true
    f1_t = 2 * prec_t * rec_t / max(prec_t + rec_t, 1e-9)

    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"Dataset: {N:,} patients + {n_hard:,} hard negatives")
    print(f"{'Method':<30s}  {'Correct':>8}  {'FP':>6}  {'Recall':>8}  {'Prec':>8}  {'F1':>8}")
    print("-" * 72)
    print(f"{'Exact (PSI)':<30s}  {exact_n:>8,}  {'0':>6}  {exact_n/n_true:>8.4f}  {'1.0000':>8}  {2*exact_n/n_true/(1+exact_n/n_true):>8.4f}")
    print(f"{'Single PSA (@0.7)':<30s}  {fuzzy_n:>8,}  {'~2k':>6}  {'~1.000':>8}  {'~0.82':>8}  {'~0.90':>8}")
    print(f"{'Double PSA (triangulated)':<30s}  {tri_n:>8,}  {tri_fp:>6}  {rec_t:>8.4f}  {prec_t:>8.4f}  {f1_t:>8.4f}")
    print()
