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

from typing import Any
from unittest import mock
from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import m3a
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import representation_utils
from android_world.utils import test_utils
import numpy as np


def _element(**kwargs) -> representation_utils.UIElement:
  """A visible element with a valid on-screen box (screen is 100x100)."""
  defaults = dict(
      is_visible=True,
      bbox_pixels=representation_utils.BoundingBox(10, 20, 10, 20),
      is_clickable=False,
      is_long_clickable=False,
      is_editable=False,
      is_selected=False,
      is_checked=False,
  )
  defaults.update(kwargs)
  return representation_utils.UIElement(**defaults)


class MockMultimodalLlmWrapper(infer.MultimodalLlmWrapper):
  """Mock multimodal LLM wrapper for testing."""

  def __init__(self, mock_responses: list[tuple[str, Any]]):
    self.mock_responses = mock_responses
    self.index = 0
    self.prompts = []

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Any]:
    self.prompts.append(text_prompt)
    if self.index < len(self.mock_responses):
      index = self.index
      self.index += 1
      return self.mock_responses[index][0], None, self.mock_responses[index][1]
    else:
      return infer.ERROR_CALLING_LLM, None, None


class M3AInteractionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_orientation = mock.patch.object(
        adb_utils,
        'get_orientation',
    ).start()
    self.mock_get_physical_frame_boundary = mock.patch.object(
        adb_utils,
        'get_physical_frame_boundary',
    ).start()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_step_method_with_completion(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        'test raw response',
    )])
    self.mock_get_orientation.return_value = 0
    self.mock_get_physical_frame_boundary.return_value = [0, 0, 100, 100]
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step_data = agent.step(goal)
    self.assertTrue(step_data.done)

  def test_step_method_with_invalid_action_output(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([(
        'Output in incorrect format.',
        'test raw response',
    )])
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step_data = agent.step(goal)

    self.assertFalse(step_data.done)
    self.assertIn(
        'Output for action selection is not in the correct format',
        step_data.data['summary'],
    )

  def test_history_recording(self):
    env = test_utils.FakeAsyncEnv()
    llm = MockMultimodalLlmWrapper([
        (
            (
                "Reason: answer question.\nAction: {'action_type': 'answer',"
                " 'text': 'fake answer.'}"
            ),
            'test raw response',
        ),
        (
            'fake summary',
            'test raw response',
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            'test raw response',
        ),
    ])
    self.mock_get_orientation.side_effect = [0, 0, 0]
    self.mock_get_physical_frame_boundary.side_effect = [
        [0, 0, 100, 100],
        [0, 0, 100, 100],
        [0, 0, 100, 100],
    ]
    agent = m3a.M3A(env, llm)

    goal = 'do something'
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)
    self.assertIn('fake summary', step1_data.data['summary'])

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)


class CompactElementsTest(absltest.TestCase):

  def test_compact_line_lists_only_true_flags(self):
    el = _element(
        text='Save', content_description='Save note', is_clickable=True,
        is_checked=True,
    )
    self.assertEqual(
        m3a._generate_compact_ui_element_description(el, 5),
        '5: text="Save" desc="Save note" | clickable, checked',
    )

  def test_compact_line_includes_hint_tooltip_and_all_flags_in_order(self):
    el = _element(
        hint_text='Name', tooltip='tip', is_clickable=True,
        is_long_clickable=True, is_editable=True, is_scrollable=True,
        is_focusable=True, is_focused=True, is_selected=True, is_checked=True,
    )
    self.assertEqual(
        m3a._generate_compact_ui_element_description(el, 0),
        '0: hint="Name" tooltip="tip" | clickable, long_clickable, editable,'
        ' scrollable, focusable, focused, selected, checked',
    )

  def test_compact_line_without_flags_has_no_separator(self):
    el = _element(text='Title')
    self.assertEqual(
        m3a._generate_compact_ui_element_description(el, 2), '2: text="Title"'
    )

  def test_compact_line_quotes_safely_and_keeps_unicode(self):
    el = _element(text='Say "hi"\nnow café')
    self.assertEqual(
        m3a._generate_compact_ui_element_description(el, 1),
        '1: text="Say \\"hi\\"\\nnow café"',
    )

  def test_compact_list_drops_uninformative_and_keeps_indexes(self):
    elements = [
        _element(text='A'),
        _element(),  # No text and not interactive: dropped.
        _element(is_clickable=True),  # Interactive without text: kept.
        _element(is_visible=False, text='hidden'),  # Invalid: dropped.
        _element(content_description='D'),
    ]
    self.assertEqual(
        m3a._generate_ui_elements_description_list(
            elements, (100, 100), compact=True
        ),
        '0: text="A"\n2: | clickable\n4: desc="D"\n',
    )

  def test_default_list_keeps_original_format(self):
    elements = [_element(text='A'), _element()]
    self.assertEqual(
        m3a._generate_ui_elements_description_list(elements, (100, 100)),
        'UI element 0: {"index": 0, "text": "A", "is_clickable": False,'
        ' "is_long_clickable": False, "is_editable": False, "is_selected":'
        ' False, "is_checked": False}\n'
        'UI element 1: {"index": 1, "is_clickable": False,'
        ' "is_long_clickable": False, "is_editable": False, "is_selected":'
        ' False, "is_checked": False}\n',
    )


class M3AFlagsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock.patch.object(adb_utils, 'get_orientation', return_value=0).start()
    mock.patch.object(
        adb_utils, 'get_physical_frame_boundary', return_value=[0, 0, 100, 100]
    ).start()
    self.addCleanup(mock.patch.stopall)
    self.env = test_utils.FakeAsyncEnv()
    self.pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    self.elements = [_element(text='Save', is_clickable=True), _element()]

  def _set_states(self, *element_lists, pixels=None):
    states = [
        interface.State(
            pixels=(self.pixels if pixels is None else pixels[i]).copy(),
            forest=None,
            ui_elements=els,
        )
        for i, els in enumerate(element_lists)
    ]
    mock.patch.object(self.env, 'get_state', side_effect=states).start()

  def test_compact_flag_uses_compact_list_in_action_prompt(self):
    self._set_states(self.elements)
    llm = MockMultimodalLlmWrapper([(
        'Reason: done.\nAction: {"action_type": "status",'
        ' "goal_status": "complete"}',
        'raw',
    )])
    agent = m3a.M3A(self.env, llm, compact_ui_elements=True)
    agent.step('goal')
    prompt = llm.prompts[0]
    self.assertIn('\n0: text="Save" | clickable\n', prompt)
    self.assertNotIn('UI element 0', prompt)
    self.assertIn(m3a.COMPACT_ELEMENTS_NOTE, prompt)

  def test_default_action_prompt_unchanged(self):
    self._set_states(self.elements)
    llm = MockMultimodalLlmWrapper([(
        'Reason: done.\nAction: {"action_type": "status",'
        ' "goal_status": "complete"}',
        'raw',
    )])
    agent = m3a.M3A(self.env, llm)
    agent.step('goal')
    prompt = llm.prompts[0]
    self.assertIn('UI element 0: {"index": 0, "text": "Save"', prompt)
    self.assertNotIn(m3a.COMPACT_ELEMENTS_NOTE, prompt)

  def test_compact_flag_uses_compact_lists_in_summary_prompt(self):
    after = [_element(text='Saved')]
    self._set_states(self.elements, after)
    llm = MockMultimodalLlmWrapper([
        ('Reason: tap.\nAction: {"action_type": "click", "index": 0}', 'raw'),
        ('summary', 'raw'),
    ])
    agent = m3a.M3A(
        self.env, llm, compact_ui_elements=True, wait_after_action_seconds=0
    )
    agent.step('goal')
    summary_prompt = llm.prompts[1]
    self.assertIn('0: text="Save" | clickable', summary_prompt)
    self.assertIn('0: text="Saved"', summary_prompt)
    self.assertNotIn('UI element 0', summary_prompt)
    self.assertIn(m3a.COMPACT_ELEMENTS_NOTE, summary_prompt)


if __name__ == '__main__':
  absltest.main()
