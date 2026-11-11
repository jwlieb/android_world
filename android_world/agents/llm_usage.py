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

"""Tracks token usage and latency of LLM calls for benchmarking.

The LLM wrappers in `infer.py` report every call to the process-wide tracker
returned by `get_tracker()`. The suite runner resets it before each task and
reads it afterwards, so any agent built on those wrappers is measured without
changes to the agent itself.
"""

import contextlib
import dataclasses
import threading
from typing import Any, Iterator, Optional


@dataclasses.dataclass
class LlmCall:
  """One logical LLM call (all retry attempts of a single request)."""

  model: str
  # Episode step the call happened in (0-based), None if outside an episode.
  step: Optional[int]
  # Order of the call within its step. For M3A/T3A, 0 = action, 1 = summary.
  call_index: int
  # Optional label set by the caller via `UsageTracker.label()`.
  label: Optional[str] = None
  input_tokens: int = 0
  output_tokens: int = 0
  # Reasoning tokens. Billed as output but not part of the visible text.
  thinking_tokens: int = 0
  # Portion of input_tokens served from the provider's cache.
  cached_tokens: int = 0
  total_tokens: int = 0
  num_images: int = 0
  prompt_chars: int = 0
  output_chars: int = 0
  latency_s: float = 0.0
  attempts: int = 1
  success: bool = True


@dataclasses.dataclass
class StepTiming:
  step: int
  duration_s: float


_TOKEN_FIELDS = (
    'input_tokens',
    'output_tokens',
    'thinking_tokens',
    'cached_tokens',
    'total_tokens',
)


class UsageTracker:
  """Collects LLM calls and step timings for the current task."""

  def __init__(self):
    self._lock = threading.Lock()
    self._calls: list[LlmCall] = []
    self._step_timings: list[StepTiming] = []
    self._step: Optional[int] = None
    self._label: Optional[str] = None

  def reset(self) -> None:
    with self._lock:
      self._calls = []
      self._step_timings = []
      self._step = None
      self._label = None

  def set_step(self, step: Optional[int]) -> None:
    self._step = step

  def record_step_time(self, step: int, duration_s: float) -> None:
    with self._lock:
      self._step_timings.append(StepTiming(step, duration_s))

  @contextlib.contextmanager
  def label(self, name: str) -> Iterator[None]:
    """Labels LLM calls made inside the block, e.g. 'action' or 'summary'."""
    previous = self._label
    self._label = name
    try:
      yield
    finally:
      self._label = previous

  def record(self, model: str, **fields: Any) -> LlmCall:
    """Records one LLM call. `fields` are any LlmCall attributes."""
    with self._lock:
      call_index = sum(1 for c in self._calls if c.step == self._step)
      call = LlmCall(
          model=model,
          step=self._step,
          call_index=call_index,
          label=self._label,
          **fields,
      )
      if not call.total_tokens:
        call.total_tokens = (
            call.input_tokens + call.output_tokens + call.thinking_tokens
        )
      self._calls.append(call)
      return call

  @property
  def calls(self) -> list[LlmCall]:
    with self._lock:
      return list(self._calls)

  @property
  def step_timings(self) -> list[StepTiming]:
    with self._lock:
      return list(self._step_timings)

  def totals(self) -> dict[str, Any]:
    """Aggregates over all calls recorded since the last reset."""
    calls = self.calls
    totals: dict[str, Any] = {f: sum(getattr(c, f) for c in calls)
                              for f in _TOKEN_FIELDS}
    totals['llm_calls'] = len(calls)
    totals['llm_failed_calls'] = sum(1 for c in calls if not c.success)
    totals['llm_retries'] = sum(max(c.attempts - 1, 0) for c in calls)
    totals['llm_time_s'] = sum(c.latency_s for c in calls)
    totals['num_images_sent'] = sum(c.num_images for c in calls)
    totals['prompt_chars'] = sum(c.prompt_chars for c in calls)
    totals['models'] = sorted({c.model for c in calls})
    return totals


_TRACKER = UsageTracker()


def get_tracker() -> UsageTracker:
  return _TRACKER


def add_usage(acc: dict[str, int], usage: dict[str, int]) -> None:
  """Adds token counts from one response into an accumulator dict."""
  for key, value in usage.items():
    acc[key] = acc.get(key, 0) + (value or 0)


def gemini_usage(response: Any) -> dict[str, int]:
  """Extracts token counts from a google-genai GenerateContentResponse."""
  meta = getattr(response, 'usage_metadata', None)
  if meta is None:
    return {}
  return {
      'input_tokens': meta.prompt_token_count or 0,
      'output_tokens': meta.candidates_token_count or 0,
      'thinking_tokens': meta.thoughts_token_count or 0,
      'cached_tokens': meta.cached_content_token_count or 0,
      'total_tokens': meta.total_token_count or 0,
  }


def openai_usage(response_json: dict[str, Any]) -> dict[str, int]:
  """Extracts token counts from an OpenAI chat completions JSON response."""
  usage = (response_json or {}).get('usage') or {}
  completion_details = usage.get('completion_tokens_details') or {}
  prompt_details = usage.get('prompt_tokens_details') or {}
  thinking = completion_details.get('reasoning_tokens') or 0
  completion = usage.get('completion_tokens') or 0
  return {
      'input_tokens': usage.get('prompt_tokens') or 0,
      # OpenAI includes reasoning tokens in completion_tokens; split them out
      # so output/thinking mean the same thing as for Gemini.
      'output_tokens': completion - thinking,
      'thinking_tokens': thinking,
      'cached_tokens': prompt_details.get('cached_tokens') or 0,
      'total_tokens': usage.get('total_tokens') or 0,
  }
