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

import json
import os
import tempfile

from absl.testing import absltest
from android_world import benchmark
from android_world import constants
from android_world.agents import llm_usage
import pandas as pd

_EC = constants.EpisodeConstants


def _tracker_with_two_steps() -> llm_usage.UsageTracker:
  tracker = llm_usage.UsageTracker()
  for step in range(2):
    tracker.set_step(step)
    tracker.record('m', input_tokens=100, output_tokens=10, latency_s=1.0)
    tracker.record(
        'm', input_tokens=50, output_tokens=5, thinking_tokens=5,
        cached_tokens=20, latency_s=0.5,
    )
    tracker.record_step_time(step, 2.0)
  tracker.set_step(None)
  return tracker


def _episode(name, metrics, success=1.0, exception=None, instance_id=0):
  return {
      _EC.TASK_TEMPLATE: name,
      _EC.INSTANCE_ID: instance_id,
      _EC.SEED: 123,
      _EC.GOAL: 'do the thing',
      _EC.IS_SUCCESSFUL: success,
      _EC.EXCEPTION_INFO: exception,
      _EC.RUN_TIME: 10.0,
      _EC.BENCHMARK_METRICS: metrics,
  }


class UsageTrackerTest(absltest.TestCase):

  def test_call_index_and_totals(self):
    tracker = _tracker_with_two_steps()
    calls = tracker.calls
    self.assertEqual([(c.step, c.call_index) for c in calls],
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    totals = tracker.totals()
    self.assertEqual(totals['input_tokens'], 300)
    self.assertEqual(totals['output_tokens'], 30)
    self.assertEqual(totals['thinking_tokens'], 10)
    self.assertEqual(totals['cached_tokens'], 40)
    # total_tokens is derived when the provider doesn't report it.
    self.assertEqual(totals['total_tokens'], 340)
    self.assertEqual(totals['llm_calls'], 4)
    self.assertEqual(totals['llm_time_s'], 3.0)

  def test_label(self):
    tracker = llm_usage.UsageTracker()
    with tracker.label('action'):
      tracker.record('m')
    tracker.record('m')
    self.assertEqual([c.label for c in tracker.calls], ['action', None])


class TaskMetricsTest(absltest.TestCase):

  def test_task_metrics(self):
    metrics = benchmark.task_metrics(
        _tracker_with_two_steps(),
        {'init_s': 1.0, 'episode_s': 4.0, 'eval_s': 0.5},
        max_steps=10,
        agent_done=True,
    )
    self.assertEqual(metrics['steps'], 2)
    self.assertFalse(metrics['hit_step_limit'])
    self.assertEqual(metrics['total_tokens'], 340)
    self.assertEqual(metrics['tokens_per_step'], 170)
    self.assertEqual(metrics['non_llm_episode_s'], 1.0)
    self.assertLen(metrics['calls'], 4)

  def test_hit_step_limit(self):
    metrics = benchmark.task_metrics(
        _tracker_with_two_steps(), {}, max_steps=2, agent_done=False
    )
    self.assertTrue(metrics['hit_step_limit'])

  def test_prices(self):
    prices = benchmark.Prices(
        input_per_mtok=1.0, output_per_mtok=10.0, cached_input_per_mtok=0.1
    )
    cost = prices.cost({
        'input_tokens': 1_000_000,
        'cached_tokens': 500_000,
        'output_tokens': 100_000,
        'thinking_tokens': 100_000,
    })
    self.assertAlmostEqual(cost, 0.5 + 0.05 + 2.0)


class BenchmarkWriterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.run_dir = tmp.name
    self.metrics = benchmark.task_metrics(
        _tracker_with_two_steps(), {'episode_s': 4.0}, 10, True
    )

  def test_writes_files(self):
    writer = benchmark.BenchmarkWriter(
        self.run_dir,
        {'agent_name': 'm3a_gemini_gcp'},
        benchmark.Prices(input_per_mtok=1.0, output_per_mtok=2.0),
    )
    writer.add_episode(_episode('ContactsAddContact', self.metrics))
    writer.add_episode(_episode('ContactsAddContact', self.metrics,
                                success=0.0, instance_id=1))
    writer.add_episode(_episode(
        'ContactsAddContact', self.metrics, success=float('nan'),
        exception='Traceback...\nRuntimeError: boom', instance_id=2,
    ))

    tasks = pd.read_csv(os.path.join(self.run_dir, benchmark.TASKS_FILE))
    self.assertLen(tasks, 3)
    self.assertEqual(tasks['total_tokens'].tolist(), [340, 340, 340])
    self.assertEqual(tasks.loc[0, 'difficulty'], 'easy')
    self.assertEqual(tasks.loc[2, 'error_message'], 'RuntimeError: boom')
    self.assertIn('est_cost_usd', tasks)

    calls = pd.read_csv(os.path.join(self.run_dir, benchmark.CALLS_FILE))
    self.assertLen(calls, 12)
    self.assertEqual(set(calls['call_index']), {0, 1})

    with open(os.path.join(self.run_dir, benchmark.SUMMARY_FILE)) as f:
      summary = json.load(f)
    totals = summary['totals']
    self.assertEqual(summary['config']['agent_name'], 'm3a_gemini_gcp')
    self.assertEqual(totals['episodes'], 3)
    self.assertEqual(totals['errors'], 1)
    self.assertEqual(totals['successes'], 1)
    self.assertEqual(totals['success_rate'], 0.5)
    self.assertEqual(totals['total_tokens_per_success'], 1020)
    self.assertEqual(
        summary['per_task']['ContactsAddContact']['success_rate'], 0.5
    )

  def test_resume_keeps_rows(self):
    writer = benchmark.BenchmarkWriter(self.run_dir, {})
    writer.add_episode(_episode('ContactsAddContact', self.metrics))
    resumed = benchmark.BenchmarkWriter(self.run_dir, {})
    resumed.add_episode(_episode('SimpleSmsSend', self.metrics))
    tasks = pd.read_csv(os.path.join(self.run_dir, benchmark.TASKS_FILE))
    self.assertEqual(tasks['task'].tolist(),
                     ['ContactsAddContact', 'SimpleSmsSend'])

  def test_rerun_replaces_same_instance(self):
    writer = benchmark.BenchmarkWriter(self.run_dir, {})
    writer.add_episode(_episode(
        'SimpleSmsSend', self.metrics, success=float('nan'),
        exception='Traceback...\nValueError: bad', instance_id=0,
    ))
    writer.add_episode(_episode('SimpleSmsSend', self.metrics, instance_id=1))
    resumed = benchmark.BenchmarkWriter(self.run_dir, {})
    resumed.add_episode(_episode('SimpleSmsSend', self.metrics, instance_id=0))
    tasks = pd.read_csv(os.path.join(self.run_dir, benchmark.TASKS_FILE))
    self.assertEqual(tasks['instance_id'].tolist(), [1, 0])
    self.assertFalse(tasks['error'].any())
    calls = pd.read_csv(os.path.join(self.run_dir, benchmark.CALLS_FILE))
    self.assertLen(calls, 8)

  def test_skips_episode_without_metrics(self):
    writer = benchmark.BenchmarkWriter(self.run_dir, {})
    writer.add_episode(_episode('ContactsAddContact', None))
    self.assertFalse(
        os.path.exists(os.path.join(self.run_dir, benchmark.TASKS_FILE))
    )


if __name__ == '__main__':
  absltest.main()
