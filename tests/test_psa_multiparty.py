"""PSA multi-party alignment test — 3 and 4 hospitals.

Tests pairwise CLK matching across multiple parties, then intersects
to find patients present in ALL parties.

Strategy:
    1. Run pairwise fuzzy PSA between all (n choose 2) party pairs
    2. Build a graph: nodes = records, edges = pairwise matches
    3. Find cliques of size n_parties = records matched across ALL parties

For double PSA triangulation, each pairwise comparison itself uses
the identity + location two-pass approach.
"""

import csv
import os
import sys
import time
from collections import defaultdict
from io import StringIO
from itertools import combinations

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import anonlink
from clkhash import clk
from clkhash.schema import from_json_dict as schema_from_json_dict

from data.generators.sg_synthetic import generate_multiparty

# ---------------------------------------------------------------
# CLK schemas
# ---------------------------------------------------------------

SCHEMA_FULL = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "cHNhLW1w", "salt": "bXAtc2FsdA==", "keySize": 64},
    },
    "features": [
        {"identifier": "name",    "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 300}}},
        {"identifier": "dob",     "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 150}}},
        {"identifier": "address", "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 250}}},
        {"identifier": "gender",  "format": {"type": "enum", "values": ["M", "F"]},  "hashing": {"comparison": {"type": "exact"}, "strategy": {"bitsPerFeature": 50}}},
    ],
}
FIELDS = ["name", "dob", "address", "gender"]

SCHEMA_IDENTITY = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "bXAtaWQ=", "salt": "bXAtaWQtcw==", "keySize": 64},
    },
    "features": [
        {"identifier": "name",   "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 400}}},
        {"identifier": "dob",    "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 200}}},
        {"identifier": "gender", "format": {"type": "enum", "values": ["M", "F"]},  "hashing": {"comparison": {"type": "exact"}, "strategy": {"bitsPerFeature": 100}}},
    ],
}
FIELDS_ID = ["name", "dob", "gender"]

SCHEMA_LOCATION = {
    "version": 3,
    "clkConfig": {
        "l": 1024, "xor_folds": 0,
        "kdf": {"type": "HKDF", "hash": "SHA256",
                "info": "bXAtbG9j", "salt": "bXAtbG9jLXM=", "keySize": 64},
    },
    "features": [
        {"identifier": "address", "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 2}, "strategy": {"bitsPerFeature": 500}}},
        {"identifier": "dob",     "format": {"type": "string", "encoding": "utf-8"}, "hashing": {"comparison": {"type": "ngram", "n": 1}, "strategy": {"bitsPerFeature": 150}}},
    ],
}
FIELDS_LOC = ["address", "dob"]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def encode_clks(records, schema_dict, fields, secret):
    schema = schema_from_json_dict(schema_dict)
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(fields)
    for r in records:
        writer.writerow([r.get(f, "") for f in fields])
    buf.seek(0)
    return clk.generate_clk_from_csv(buf, secret, schema)


def match_pair(clks_a, clks_b, threshold=0.7):
    """Match two CLK sets, return set of (idx_a, idx_b) pairs."""
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


def multiparty_align(party_records, party_names, schema_dict, fields, secret, threshold=0.7):
    """Pairwise CLK matching across all parties, then find common records.

    Returns:
        List of tuples: each tuple has one index per party for a
        record matched across ALL parties.
    """
    n_parties = len(party_names)

    # Encode CLKs for each party
    clks = {}
    for name in party_names:
        clks[name] = encode_clks(party_records[name], schema_dict, fields, secret)

    # Pairwise matching
    pairwise = {}
    for i, j in combinations(range(n_parties), 2):
        name_i, name_j = party_names[i], party_names[j]
        pairs = match_pair(clks[name_i], clks[name_j], threshold)
        pairwise[(i, j)] = pairs

    # Build record groups using transitive closure via Union-Find
    # Each party's records are in their own namespace: (party_idx, local_idx)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j), pairs in pairwise.items():
        for idx_i, idx_j in pairs:
            union((i, idx_i), (j, idx_j))

    # Group records by their root
    groups = defaultdict(set)
    all_nodes = set()
    for (i, j), pairs in pairwise.items():
        for idx_i, idx_j in pairs:
            all_nodes.add((i, idx_i))
            all_nodes.add((j, idx_j))

    for node in all_nodes:
        groups[find(node)].add(node)

    # Find groups that span ALL parties with exactly one record per party
    full_matches = []
    for root, members in groups.items():
        # Count records per party
        party_members = defaultdict(list)
        for party_idx, local_idx in members:
            party_members[party_idx].append(local_idx)
        if len(party_members) != n_parties:
            continue
        # Skip ambiguous groups (>1 record from any party)
        if any(len(indices) > 1 for indices in party_members.values()):
            continue
        full_matches.append(tuple(party_members[p][0] for p in range(n_parties)))

    return full_matches


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

def run_test(n, n_parties, overlap, threshold=0.7):
    """Run a multi-party PSA test."""
    print("=" * 72)
    print(f"TEST: {n_parties} parties, {n:,} patients, {overlap*100:.0f}% overlap, threshold={threshold}")
    print("=" * 72)

    t0 = time.time()
    party_records, n_common, index_map = generate_multiparty(
        n=n, n_parties=n_parties, overlap=overlap, seed=42)
    gen_time = time.time() - t0

    party_names = list(party_records.keys())
    print(f"Generated in {gen_time:.2f}s")
    for name in party_names:
        print(f"  {name}: {len(party_records[name]):,} records")
    print(f"  Common to ALL: {n_common:,} patients")
    print()

    # Show sample records across parties for the same patient
    print("Sample patient across all parties:")
    for name in party_names:
        r = party_records[name][0]
        print(f"  {name:6s}: {r['name']:35s} dob={r['dob']}  addr=...{r['address'][-25:]}")
    print()

    # Single-pass multi-party PSA
    t0 = time.time()
    matches_single = multiparty_align(
        party_records, party_names, SCHEMA_FULL, FIELDS, "mp-full", threshold)
    t_single = time.time() - t0

    # Score against ground truth
    correct_single = 0
    for match in matches_single:
        # Check if all indices map to the same global patient
        global_ids = set()
        for party_idx, local_idx in enumerate(match):
            name = party_names[party_idx]
            gid = index_map[name].get(local_idx, -1)
            global_ids.add(gid)
        if len(global_ids) == 1 and -1 not in global_ids:
            gid = global_ids.pop()
            if gid < n_common:
                correct_single += 1

    fp_single = len(matches_single) - correct_single
    prec_s = correct_single / max(len(matches_single), 1)
    rec_s = correct_single / n_common
    f1_s = 2 * prec_s * rec_s / max(prec_s + rec_s, 1e-9)

    print(f"Single-pass PSA ({n_parties}-party):")
    print(f"  Matches: {len(matches_single):,}  |  Correct: {correct_single:,}  |  FP: {fp_single}")
    print(f"  Precision: {prec_s:.4f}  |  Recall: {rec_s:.4f}  |  F1: {f1_s:.4f}")
    print(f"  Time: {t_single:.2f}s")
    print()

    # Double PSA triangulation: intersect identity and location matches
    t0 = time.time()
    matches_id = multiparty_align(
        party_records, party_names, SCHEMA_IDENTITY, FIELDS_ID, "mp-id", threshold)
    matches_loc = multiparty_align(
        party_records, party_names, SCHEMA_LOCATION, FIELDS_LOC, "mp-loc", threshold)
    t_double = time.time() - t0

    # Intersect: only keep tuples that appear in both
    set_id = set(matches_id)
    set_loc = set(matches_loc)
    matches_tri = set_id & set_loc

    correct_tri = 0
    for match in matches_tri:
        global_ids = set()
        for party_idx, local_idx in enumerate(match):
            name = party_names[party_idx]
            gid = index_map[name].get(local_idx, -1)
            global_ids.add(gid)
        if len(global_ids) == 1 and -1 not in global_ids:
            gid = global_ids.pop()
            if gid < n_common:
                correct_tri += 1

    fp_tri = len(matches_tri) - correct_tri
    prec_t = correct_tri / max(len(matches_tri), 1)
    rec_t = correct_tri / n_common
    f1_t = 2 * prec_t * rec_t / max(prec_t + rec_t, 1e-9)

    print(f"Double PSA triangulated ({n_parties}-party):")
    print(f"  Pass 1 (identity): {len(matches_id):,} groups spanning all parties")
    print(f"  Pass 2 (location): {len(matches_loc):,} groups spanning all parties")
    print(f"  Triangulated:      {len(matches_tri):,} groups")
    print(f"  Correct: {correct_tri:,}  |  FP: {fp_tri}")
    print(f"  Precision: {prec_t:.4f}  |  Recall: {rec_t:.4f}  |  F1: {f1_t:.4f}")
    print(f"  Time: {t_double:.2f}s")
    print()

    # Summary comparison
    print(f"{'Method':<25s}  {'Correct':>8}  {'FP':>5}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    print("-" * 62)
    print(f"{'Single-pass':<25s}  {correct_single:>8,}  {fp_single:>5}  {prec_s:>7.4f}  {rec_s:>7.4f}  {f1_s:>7.4f}")
    print(f"{'Double PSA (triangulated)':<25s}  {correct_tri:>8,}  {fp_tri:>5}  {prec_t:>7.4f}  {rec_t:>7.4f}  {f1_t:>7.4f}")
    print()

    return {
        "n_parties": n_parties,
        "n_common": n_common,
        "single": {"correct": correct_single, "fp": fp_single, "prec": prec_s, "rec": rec_s, "f1": f1_s},
        "double": {"correct": correct_tri, "fp": fp_tri, "prec": prec_t, "rec": rec_t, "f1": f1_t},
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Multi-Party Private Set Alignment (PSA)")
    print("Pairwise CLK matching + transitive closure across all parties")
    print()

    results = []

    # Test 1: 3 parties, 1000 patients, 60% overlap
    results.append(run_test(n=1000, n_parties=3, overlap=0.6))

    # Test 2: 4 parties, 1000 patients, 60% overlap
    results.append(run_test(n=1000, n_parties=4, overlap=0.6))

    # Test 3: 3 parties, 5000 patients, 50% overlap
    results.append(run_test(n=5000, n_parties=3, overlap=0.5))

    # Test 4: 4 parties, 5000 patients, 40% overlap
    results.append(run_test(n=5000, n_parties=4, overlap=0.4))

    # Final summary
    print("=" * 72)
    print("MULTI-PARTY SUMMARY")
    print("=" * 72)
    print(f"{'Parties':>8}  {'Pop':>6}  {'Common':>7}  {'Method':<15s}  {'Correct':>8}  {'FP':>5}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    print("-" * 78)
    for r in results:
        np = r["n_parties"]
        nc = r["n_common"]
        pop = nc  # approximate
        s = r["single"]
        d = r["double"]
        print(f"{np:>8}  {pop:>6,}  {nc:>7,}  {'Single-pass':<15s}  {s['correct']:>8,}  {s['fp']:>5}  {s['prec']:>7.4f}  {s['rec']:>7.4f}  {s['f1']:>7.4f}")
        print(f"{'':>8}  {'':>6}  {'':>7}  {'Double PSA':<15s}  {d['correct']:>8,}  {d['fp']:>5}  {d['prec']:>7.4f}  {d['rec']:>7.4f}  {d['f1']:>7.4f}")
    print()
    print("Key insight: multi-party intersection naturally reduces false positives")
    print("because a false match must occur consistently across ALL party pairs.")
    print()
