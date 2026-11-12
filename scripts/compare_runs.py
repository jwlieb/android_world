# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compares the benchmark results of two runs, task by task.

Usage:
  python scripts/compare_runs.py BASELINE_RUN_DIR VARIANT_RUN_DIR [--csv OUT]

Each run directory must contain benchmark_tasks.csv (written by run.py).
Metrics are averaged over the instances of each task; errored episodes are
excluded. Only tasks present in both runs are compared row by row.
"""

import argparse
import json
import os
import sys

import pandas as pd

_METRICS = {
    'success': 'success',
    'total_tokens': 'tokens',
    'input_tokens': 'in_tok',
    'output_tokens': 'out_tok',
    'thinking_tokens': 'think_tok',
    'steps': 'steps',
    'run_time_s': 'time_s',
}


def _load(run_dir: str) -> pd.DataFrame:
  path = os.path.join(run_dir, 'benchmark_tasks.csv')
  if not os.path.exists(path):
    sys.exit(f'No benchmark_tasks.csv in {run_dir}')
  df = pd.read_csv(path)
  return df[~df['error'].astype(bool)]


def _config(run_dir: str) -> dict:
  try:
    with open(os.path.join(run_dir, 'benchmark_summary.json')) as f:
      return json.load(f).get('config', {})
  except (OSError, ValueError):
    return {}


def _per_task(df: pd.DataFrame) -> pd.DataFrame:
  agg = df.groupby('task')[list(_METRICS)].mean()
  agg['n'] = df.groupby('task').size()
  return agg


def _pct(new: float, old: float) -> float | None:
  if pd.isna(new) or pd.isna(old) or old == 0:
    return None
  return 100.0 * (new - old) / old


def compare(baseline_dir: str, variant_dir: str) -> pd.DataFrame:
  base = _per_task(_load(baseline_dir))
  var = _per_task(_load(variant_dir))
  common = base.index.intersection(var.index)
  only_base = base.index.difference(var.index)
  only_var = var.index.difference(base.index)
  if len(only_base) or len(only_var):
    print(
        f'Note: tasks only in baseline: {list(only_base)}; only in variant:'
        f' {list(only_var)}. They are left out of the comparison.\n'
    )
  rows = []
  for task in common:
    b, v = base.loc[task], var.loc[task]
    rows.append({
        'task': task,
        'n_base': int(b['n']),
        'n_var': int(v['n']),
        'success_base': b['success'],
        'success_var': v['success'],
        'tokens_base': b['total_tokens'],
        'tokens_var': v['total_tokens'],
        'tokens_change_%': _pct(v['total_tokens'], b['total_tokens']),
        'steps_base': b['steps'],
        'steps_var': v['steps'],
        'time_base_s': b['run_time_s'],
        'time_var_s': v['run_time_s'],
    })
  out = pd.DataFrame(rows).set_index('task')
  if out.empty:
    return out

  # Overall row, over the common tasks (each task weighted equally).
  b_all, v_all = base.loc[common], var.loc[common]
  overall = {
      'n_base': int(b_all['n'].sum()),
      'n_var': int(v_all['n'].sum()),
      'success_base': b_all['success'].mean(),
      'success_var': v_all['success'].mean(),
      'tokens_base': b_all['total_tokens'].mean(),
      'tokens_var': v_all['total_tokens'].mean(),
      'tokens_change_%': _pct(
          v_all['total_tokens'].mean(), b_all['total_tokens'].mean()
      ),
      'steps_base': b_all['steps'].mean(),
      'steps_var': v_all['steps'].mean(),
      'time_base_s': b_all['run_time_s'].mean(),
      'time_var_s': v_all['run_time_s'].mean(),
  }
  out.loc['== MEAN =='] = overall
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument('baseline_dir')
  parser.add_argument('variant_dir')
  parser.add_argument('--csv', help='Also write the comparison to this CSV.')
  args = parser.parse_args()

  for label, run_dir in (('baseline', args.baseline_dir),
                         ('variant', args.variant_dir)):
    cfg = _config(run_dir)
    git = cfg.get('git') or {}
    commit = (git.get('commit') or '?')[:8]
    dirty = ' (uncommitted changes)' if git.get('dirty') else ''
    # Runs from before these options existed had them off.
    options = [
        name
        for name in (
            'm3a_compact_elements', 'm3a_lean_summary', 'm3a_merged_summary'
        )
        if cfg.get(name)
    ]
    if cfg.get('m3a_history_window'):
      options.append(f'm3a_history_window={cfg["m3a_history_window"]}')
    print(
        f'{label:8s}: {run_dir}\n          agent={cfg.get("agent_name")}'
        f' model={cfg.get("model")} seed={cfg.get("llm_seed")}'
        f' commit={commit}{dirty}'
        f' options={",".join(options) or "none"}'
    )
  print()

  result = compare(args.baseline_dir, args.variant_dir)
  if result.empty:
    sys.exit('No tasks in common between the two runs.')
  with pd.option_context(
      'display.max_rows', None,
      'display.max_columns', None,
      'display.width', 200,
      'display.float_format', '{:,.2f}'.format,
  ):
    print(result)
  if args.csv:
    result.to_csv(args.csv)
    print(f'\nWrote {args.csv}')


if __name__ == '__main__':
  main()
