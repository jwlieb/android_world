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

"""Per-task benchmarking: token usage, timing and success, written per run.

Each run directory gets three files, rewritten after every task so partial
runs keep their data:

* benchmark_tasks.csv: one row per task episode.
* benchmark_calls.csv: one row per LLM call (where the tokens go).
* benchmark_summary.json: run configuration and aggregate metrics.
"""

import dataclasses
import datetime
import json
import math
import os
import subprocess
from typing import Any, Optional

from android_world import constants
from android_world.agents import llm_usage
import pandas as pd


TASKS_FILE = 'benchmark_tasks.csv'
CALLS_FILE = 'benchmark_calls.csv'
SUMMARY_FILE = 'benchmark_summary.json'

_TOKEN_FIELDS = (
    'input_tokens',
    'output_tokens',
    'thinking_tokens',
    'cached_tokens',
    'total_tokens',
)


@dataclasses.dataclass(frozen=True)
class Prices:
  """USD prices per million tokens, used to estimate cost."""

  input_per_mtok: float
  output_per_mtok: float
  # Defaults to the input price when not given.
  cached_input_per_mtok: Optional[float] = None

  def cost(self, metrics: dict[str, Any]) -> float:
    cached = metrics.get('cached_tokens', 0)
    uncached = metrics.get('input_tokens', 0) - cached
    cached_price = (
        self.input_per_mtok
        if self.cached_input_per_mtok is None
        else self.cached_input_per_mtok
    )
    # Thinking tokens are billed as output tokens.
    output = metrics.get('output_tokens', 0) + metrics.get('thinking_tokens', 0)
    return (
        uncached * self.input_per_mtok
        + cached * cached_price
        + output * self.output_per_mtok
    ) / 1e6


def task_metrics(
    tracker: llm_usage.UsageTracker,
    timings: dict[str, float],
    max_steps: Optional[int],
    agent_done: Optional[bool],
) -> dict[str, Any]:
  """Builds the benchmark metrics for one task from the usage tracker.

  Args:
    tracker: Tracker holding the calls made during the task.
    timings: Phase durations in seconds: init_s, episode_s, eval_s (any may be
      missing if the task errored out before that phase).
    max_steps: Step budget given to the agent.
    agent_done: Whether the agent declared the task done; None if unknown.

  Returns:
    A dict of metrics; includes the raw call and step-timing lists.
  """
  totals = tracker.totals()
  step_timings = tracker.step_timings
  steps = len(step_timings)
  episode_s = timings.get('episode_s')
  metrics: dict[str, Any] = {
      'max_steps': max_steps,
      'steps': steps,
      'agent_done': agent_done,
      'hit_step_limit': (
          None if max_steps is None else (not agent_done and steps >= max_steps)
      ),
      'init_s': timings.get('init_s'),
      'episode_s': episode_s,
      'eval_s': timings.get('eval_s'),
      'mean_step_s': (
          sum(t.duration_s for t in step_timings) / steps if steps else None
      ),
  }
  metrics.update({k: v for k, v in totals.items() if k != 'models'})
  metrics['models'] = '|'.join(totals['models'])
  metrics['non_llm_episode_s'] = (
      None if episode_s is None else max(episode_s - totals['llm_time_s'], 0.0)
  )
  metrics['tokens_per_step'] = (
      totals['total_tokens'] / steps if steps else None
  )
  metrics['calls'] = [dataclasses.asdict(c) for c in tracker.calls]
  metrics['step_timings'] = [dataclasses.asdict(t) for t in step_timings]
  return metrics


def git_info(repo_dir: str) -> dict[str, Any]:
  """Returns the commit and whether the working tree has local changes."""

  def git(*args: str) -> str:
    return subprocess.run(
        ['git', *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

  try:
    status = git('status', '--porcelain', '--untracked-files=no')
    return {
        'commit': git('rev-parse', 'HEAD'),
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(status),
        'changed_files': status.splitlines(),
    }
  except (OSError, subprocess.CalledProcessError):
    return {'commit': None, 'branch': None, 'dirty': None, 'changed_files': []}


def _load_task_metadata() -> pd.DataFrame:
  path = os.path.join(os.path.dirname(__file__), 'task_metadata.json')
  try:
    df = pd.read_json(path)
  except (OSError, ValueError):
    return pd.DataFrame(columns=['difficulty', 'optimal_steps', 'tags'])
  return df.set_index('task_name')[['difficulty', 'optimal_steps', 'tags']]


def _is_nan(value: Any) -> bool:
  return isinstance(value, float) and math.isnan(value)


class BenchmarkWriter:
  """Collects finished episodes and writes the benchmark files for a run."""

  def __init__(
      self,
      run_dir: str,
      run_config: dict[str, Any],
      prices: Optional[Prices] = None,
  ):
    self.run_dir = run_dir
    self.run_config = run_config
    self.prices = prices
    self.started_at = datetime.datetime.now().isoformat(timespec='seconds')
    self._task_rows: list[dict[str, Any]] = []
    self._call_rows: list[dict[str, Any]] = []
    self._metadata = _load_task_metadata()
    os.makedirs(run_dir, exist_ok=True)
    # When resuming a run from its checkpoint directory, keep earlier rows.
    self._task_rows = self._read_rows(TASKS_FILE)
    self._call_rows = self._read_rows(CALLS_FILE)
    summary_path = os.path.join(run_dir, SUMMARY_FILE)
    if self._task_rows and os.path.exists(summary_path):
      try:
        with open(summary_path) as f:
          self.started_at = json.load(f).get('started_at', self.started_at)
      except (OSError, ValueError):
        pass

  def _read_rows(self, filename: str) -> list[dict[str, Any]]:
    path = os.path.join(self.run_dir, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
      return []
    try:
      return pd.read_csv(path).to_dict('records')
    except (pd.errors.EmptyDataError, ValueError):
      return []

  def add_episode(self, episode: dict[str, Any]) -> None:
    """Adds one finished episode and rewrites the benchmark files."""
    metrics = episode.get(constants.EpisodeConstants.BENCHMARK_METRICS)
    if metrics is None:
      return  # E.g. resumed from a checkpoint written before benchmarking.
    name = episode[constants.EpisodeConstants.TASK_TEMPLATE]
    instance_id = episode.get(constants.EpisodeConstants.INSTANCE_ID)
    exception = episode.get(constants.EpisodeConstants.EXCEPTION_INFO)
    success = episode.get(constants.EpisodeConstants.IS_SUCCESSFUL)

    row: dict[str, Any] = {
        'task': name,
        'instance_id': instance_id,
        'seed': episode.get(constants.EpisodeConstants.SEED),
        'goal': episode.get(constants.EpisodeConstants.GOAL),
        'success': None if _is_nan(success) else float(success),
        'error': exception is not None,
        'error_message': (
            exception.strip().splitlines()[-1] if exception else None
        ),
        'run_time_s': episode.get(constants.EpisodeConstants.RUN_TIME),
    }
    if name in self._metadata.index:
      meta = self._metadata.loc[name]
      row['difficulty'] = meta['difficulty']
      row['optimal_steps'] = pd.to_numeric(
          meta['optimal_steps'], errors='coerce'
      )
      row['tags'] = '|'.join(meta['tags'] or [])
    row.update({
        k: v
        for k, v in metrics.items()
        if k not in ('calls', 'step_timings')
    })
    optimal = row.get('optimal_steps')
    row['steps_over_optimal'] = (
        row['steps'] / optimal
        if optimal and not _is_nan(optimal) and row['steps']
        else None
    )
    if self.prices:
      row['est_cost_usd'] = self.prices.cost(metrics)
    # A resumed run re-runs errored instances; keep only the latest attempt.
    self._task_rows = [
        r for r in self._task_rows
        if not (r['task'] == name and r['instance_id'] == instance_id)
    ]
    self._call_rows = [
        r for r in self._call_rows
        if not (r['task'] == name and r['instance_id'] == instance_id)
    ]
    self._task_rows.append(row)

    step_durations = {
        t['step']: t['duration_s'] for t in metrics.get('step_timings', [])
    }
    for call in metrics.get('calls', []):
      call_row = {'task': name, 'instance_id': instance_id, **call}
      call_row['step_duration_s'] = step_durations.get(call['step'])
      if self.prices:
        call_row['est_cost_usd'] = self.prices.cost(call)
      self._call_rows.append(call_row)
    self.write()

  def write(self) -> None:
    pd.DataFrame(self._task_rows).to_csv(
        os.path.join(self.run_dir, TASKS_FILE), index=False
    )
    pd.DataFrame(self._call_rows).to_csv(
        os.path.join(self.run_dir, CALLS_FILE), index=False
    )
    with open(os.path.join(self.run_dir, SUMMARY_FILE), 'w') as f:
      json.dump(self.summary(), f, indent=2, default=str)

  def summary(self) -> dict[str, Any]:
    return {
        'started_at': self.started_at,
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'config': self.run_config,
        'prices_per_mtok': dataclasses.asdict(self.prices)
        if self.prices
        else None,
        'totals': summarize(pd.DataFrame(self._task_rows)),
        'per_task': per_task(pd.DataFrame(self._task_rows)),
    }


def _mean(series: pd.Series) -> Optional[float]:
  series = pd.to_numeric(series, errors='coerce').dropna()
  return float(series.mean()) if len(series) else None


def summarize(df: pd.DataFrame) -> dict[str, Any]:
  """Aggregate metrics over all task episodes in a run."""
  if df.empty:
    return {'episodes': 0}
  completed = df[~df['error']]
  successes = completed[completed['success'] > 0.5]
  out: dict[str, Any] = {
      'episodes': len(df),
      'completed': len(completed),
      'errors': int(df['error'].sum()),
      'successes': len(successes),
      'success_rate': (
          len(successes) / len(completed) if len(completed) else None
      ),
  }
  for field in _TOKEN_FIELDS:
    out[f'sum_{field}'] = int(df[field].sum())
  out['mean_total_tokens'] = _mean(completed['total_tokens'])
  out['mean_total_tokens_success'] = _mean(successes['total_tokens'])
  out['total_tokens_per_success'] = (
      out['sum_total_tokens'] / len(successes) if len(successes) else None
  )
  out['mean_tokens_per_step'] = _mean(completed['tokens_per_step'])
  out['mean_steps'] = _mean(completed['steps'])
  out['mean_steps_success'] = _mean(successes['steps'])
  if 'steps_over_optimal' in df:
    out['mean_steps_over_optimal_success'] = _mean(
        successes['steps_over_optimal']
    )
  out['llm_calls'] = int(df['llm_calls'].sum())
  out['llm_failed_calls'] = int(df['llm_failed_calls'].sum())
  out['llm_retries'] = int(df['llm_retries'].sum())
  out['sum_run_time_s'] = float(df['run_time_s'].sum())
  out['mean_run_time_s'] = _mean(completed['run_time_s'])
  out['sum_llm_time_s'] = float(df['llm_time_s'].sum())
  if 'est_cost_usd' in df:
    out['sum_est_cost_usd'] = float(df['est_cost_usd'].sum())
    out['est_cost_usd_per_success'] = (
        out['sum_est_cost_usd'] / len(successes) if len(successes) else None
    )
  return out


def per_task(df: pd.DataFrame) -> dict[str, Any]:
  """Aggregate metrics per task template (averaged over instances)."""
  if df.empty:
    return {}
  result = {}
  for name, group in df.groupby('task'):
    completed = group[~group['error']]
    result[name] = {
        'episodes': len(group),
        'errors': int(group['error'].sum()),
        'success_rate': _mean(completed['success']),
        'mean_total_tokens': _mean(completed['total_tokens']),
        'mean_steps': _mean(completed['steps']),
        'mean_run_time_s': _mean(completed['run_time_s']),
    }
  return result
