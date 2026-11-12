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

import dataclasses

from absl.testing import absltest
from android_world.agents import screen_diff
from android_world.env import representation_utils
import numpy as np

_SCREEN = (100, 100)


def _el(text=None, box=(10, 20, 10, 20), **kwargs):
  return representation_utils.UIElement(
      text=text,
      is_visible=True,
      bbox_pixels=representation_utils.BoundingBox(*box),
      **kwargs,
  )


class ElementKeyTest(absltest.TestCase):

  def test_same_content_same_key(self):
    a = _el('Save', is_clickable=True)
    self.assertEqual(
        screen_diff.element_key(a),
        screen_diff.element_key(dataclasses.replace(a)),
    )

  def test_key_ignores_position(self):
    self.assertEqual(
        screen_diff.element_key(_el('Save', box=(0, 5, 0, 5))),
        screen_diff.element_key(_el('Save', box=(50, 60, 50, 60))),
    )

  def test_key_changes_with_text_or_flag(self):
    base = screen_diff.element_key(_el('Save'))
    self.assertNotEqual(base, screen_diff.element_key(_el('Saved')))
    self.assertNotEqual(
        base, screen_diff.element_key(_el('Save', is_checked=True))
    )
    self.assertNotEqual(
        base, screen_diff.element_key(_el('Save', is_focused=True))
    )

  def test_key_is_stable_64_bit_int(self):
    key = screen_diff.element_key(_el('Save'))
    self.assertIsInstance(key, int)
    self.assertLess(key, 2**64)
    # Not Python's salted hash(): fixed across processes.
    self.assertEqual(key, screen_diff.element_key(_el('Save')))


class FingerprintTest(absltest.TestCase):

  def test_similarity_is_exact_multiset_jaccard(self):
    a = screen_diff.ScreenFingerprint.from_elements(
        [_el('A'), _el('B'), _el('B')], _SCREEN
    )
    b = screen_diff.ScreenFingerprint.from_elements(
        [_el('B'), _el('C')], _SCREEN
    )
    # Intersection {B:1} = 1; union {A:1, B:2, C:1} = 4.
    self.assertAlmostEqual(a.similarity(b), 0.25)
    self.assertAlmostEqual(a.similarity(a), 1.0)

  def test_empty_screens_are_identical(self):
    empty = screen_diff.ScreenFingerprint.from_elements([], _SCREEN)
    self.assertAlmostEqual(empty.similarity(empty), 1.0)

  def test_invalid_elements_are_ignored(self):
    hidden = representation_utils.UIElement(text='X', is_visible=False)
    a = screen_diff.ScreenFingerprint.from_elements([_el('A'), hidden], _SCREEN)
    b = screen_diff.ScreenFingerprint.from_elements([_el('A')], _SCREEN)
    self.assertEqual(a.digest(), b.digest())

  def test_digest_depends_on_layout(self):
    a = screen_diff.ScreenFingerprint.from_elements([_el('A')], _SCREEN)
    moved = screen_diff.ScreenFingerprint.from_elements(
        [_el('A', box=(30, 40, 30, 40))], _SCREEN
    )
    self.assertNotEqual(a.digest(), moved.digest())
    self.assertAlmostEqual(a.similarity(moved), 1.0)

  def test_digest_ignores_element_order(self):
    a = screen_diff.ScreenFingerprint.from_elements(
        [_el('A'), _el('B', box=(30, 40, 30, 40))], _SCREEN
    )
    b = screen_diff.ScreenFingerprint.from_elements(
        [_el('B', box=(30, 40, 30, 40)), _el('A')], _SCREEN
    )
    self.assertEqual(a.digest(), b.digest())


class DiffTest(absltest.TestCase):

  def test_added_removed_and_unchanged_with_indexes(self):
    before = [_el('Title'), _el('Old'), _el('Row'), _el('Row')]
    after = [_el('Title'), _el('Row'), _el('New')]
    d = screen_diff.diff(before, after, _SCREEN)
    self.assertEqual([(i, e.text) for i, e in d.added], [(2, 'New')])
    self.assertEqual(
        [(i, e.text) for i, e in d.removed], [(1, 'Old'), (3, 'Row')]
    )
    self.assertEqual(d.unchanged_count, 2)

  def test_diff_skips_invalid_elements(self):
    hidden = representation_utils.UIElement(text='X', is_visible=False)
    d = screen_diff.diff([hidden], [_el('A')], _SCREEN)
    self.assertEqual([i for i, _ in d.added], [0])
    self.assertEqual(d.removed, [])


class ScreensUnchangedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    self.els = [_el('A'), _el('B', box=(30, 40, 30, 40))]

  def _unchanged(self, after_els, after_pixels):
    return screen_diff.screens_unchanged(
        self.els, after_els, self.pixels, after_pixels, _SCREEN
    )

  def test_identical_screens(self):
    self.assertTrue(self._unchanged(list(self.els), self.pixels.copy()))

  def test_small_pixel_noise_is_ignored(self):
    noisy = self.pixels.copy()
    noisy[0:5, 0:10] = 255  # 50 of 10,000 pixels = 0.5% (e.g. the clock).
    noisy[50:60, 50:60] = 5  # Below the per-channel threshold.
    self.assertTrue(self._unchanged(list(self.els), noisy))

  def test_large_pixel_change_counts(self):
    changed = self.pixels.copy()
    changed[0:20, 0:10] = 255  # 2% of pixels.
    self.assertFalse(self._unchanged(list(self.els), changed))

  def test_moved_element_counts(self):
    moved = [_el('A'), _el('B', box=(60, 70, 60, 70))]
    self.assertFalse(self._unchanged(moved, self.pixels.copy()))

  def test_different_pixel_shapes_count_as_changed(self):
    self.assertFalse(
        self._unchanged(list(self.els), np.zeros((50, 50, 3), np.uint8))
    )


if __name__ == '__main__':
  absltest.main()
