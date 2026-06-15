#!/usr/bin/env python3
"""Validate a data manifest against task requirements.

Usage:
    python validate_manifest.py <manifest.json> [--task <task>]
    python validate_manifest.py ~/fl-deploy/data/fraud/manifest.json --task fraud
"""

import argparse
import sys

from fl_common.data import DataManifest, DataConfig


def main():
    parser = argparse.ArgumentParser(
        description="Validate a data manifest against task requirements."
    )
    parser.add_argument("manifest", help="Path to manifest.json")
    parser.add_argument("--task", help="Task name (overrides manifest task field)")
    args = parser.parse_args()

    manifest = DataManifest.load(args.manifest)
    task = args.task or manifest.task
    config = DataConfig.for_task(task)

    errors = manifest.validate_against(config)

    print(f"Manifest:  {args.manifest}")
    print(f"Task:      {task}")
    print(f"Client:    {manifest.client_id or '(not set)'}")
    print(f"Samples:   {manifest.num_samples}")
    print(f"Format:    {manifest.format}")
    print(f"Checksum:  {manifest.checksum[:16]}...")
    print()

    if errors:
        print("FAILED — validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PASSED — manifest is valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
