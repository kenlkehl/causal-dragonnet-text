#!/usr/bin/env python3
"""Validate causal-dgp-discovery workflow gates for a task directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_GATES = [
    'initial_exploration_gate',
    'fold_construction_gate',
    'bow_evidence_gate',
    'embedding_contrast_gate',
    'htr_evidence_gate',
    'extraction_audit_gate',
    'nuisance_ensemble_gate',
    'candidate_signal_review_gate',
    'extracted_feature_review_gate',
    'parsimony_gate',
    'final_preflight_gate',
    'final_causal_forest_gate',
]

REQUIRED_ARTIFACTS_FOR_FINAL = [
    'workflow_gate_status.json',
    'gate_retry_log.jsonl',
    'ensemble_nuisance_predictions.parquet',
    'candidate_signal_review.jsonl',
    'candidate_feature_review.jsonl',
    'extracted_feature_diagnostics_by_fold.jsonl',
    'parsimony_review_by_fold.jsonl',
    'final_preflight_check.json',
    'ite_estimates.parquet',
]

VALID_STATUSES = {'pending', 'pass', 'retrying', 'blocked_after_retries'}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f'missing required gate ledger: {path}')
    except json.JSONDecodeError as exc:
        raise SystemExit(f'invalid JSON in {path}: {exc}')


def gate_status(record):
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return record.get('status')
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('task_dir', nargs='?', default='.')
    parser.add_argument('--stage', choices=['preflight', 'final'], default='preflight')
    args = parser.parse_args()
    task_dir = Path(args.task_dir)
    ledger_path = task_dir / 'workflow_gate_status.json'
    ledger = load_json(ledger_path)

    errors = []
    for gate in REQUIRED_GATES:
        if gate not in ledger:
            errors.append(f'missing gate: {gate}')
            continue
        status = gate_status(ledger[gate])
        if status not in VALID_STATUSES:
            errors.append(f'{gate}: invalid status {status!r}')
        elif args.stage == 'final' and status != 'pass':
            errors.append(f'{gate}: status {status!r}, expected pass before final ITE reporting')

    if args.stage == 'final':
        for rel in REQUIRED_ARTIFACTS_FOR_FINAL:
            if not (task_dir / rel).exists():
                errors.append(f'missing final artifact: {rel}')

    preflight = task_dir / 'final_preflight_check.json'
    if args.stage == 'final' and preflight.exists():
        data = load_json(preflight)
        if data.get('decision') != 'pass':
            errors.append(f'final_preflight_check decision is {data.get("decision")!r}, expected pass')

    if errors:
        print('Workflow gate validation failed. Retry the failed stages before proceeding:', file=sys.stderr)
        for err in errors:
            print(f'- {err}', file=sys.stderr)
        return 1

    print(f'Workflow gate validation passed for {args.stage}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
