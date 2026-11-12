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

"""Hash-based comparison of two screens' UI elements.

Every comparison here is per screen: it looks at one before/after pair and
keeps nothing between steps. Memory and time therefore depend on how many
elements a screen has (tens to a few hundred), never on episode length.

Elements are compared by exact 64-bit content hashes held in multisets
(collections.Counter). At this size exact matching is both cheap and free of
false positives, which matters: a Bloom filter's false positive here would
silently hide a changed element from the model. If cross-step comparisons are
ever needed at scale (e.g. matching the current screen against every earlier
screen of a very long episode), ScreenFingerprint is the one place to swap in
MinHash signatures with LSH.
"""

import collections
import dataclasses
import hashlib
import json

from android_world.agents import m3a_utils
from android_world.env import representation_utils
import numpy as np

UIElement = representation_utils.UIElement

# A pixel counts as changed when any channel moves by more than this.
PIXEL_CHANNEL_THRESHOLD = 8
# Top rows ignored when comparing pixels: the status bar (clock, signal and
# notification icons) changes on its own. On a 1080x2400 Pixel 6 the status
# bar is ~5% of the height.
STATUS_BAR_FRACTION = 0.06
# Below the status bar, screens whose changed-pixel fraction is under this
# count as unchanged. 0.1% of 1080x2400 is ~2.6k px (about 50x50): room for a
# blinking cursor, while a drawn stroke or a toggled checkbox icon that the
# accessibility tree misses still counts as a change.
MAX_CHANGED_PIXEL_FRACTION = 0.001

# Content that identifies an element. Position (bbox) is deliberately left
# out so an element that moves, e.g. after a scroll, is still the same element.
_CONTENT_FIELDS = (
    'text',
    'content_description',
    'hint_text',
    'tooltip',
    'class_name',
    'resource_name',
    'is_checked',
    'is_checkable',
    'is_clickable',
    'is_editable',
    'is_enabled',
    'is_focused',
    'is_focusable',
    'is_long_clickable',
    'is_scrollable',
    'is_selected',
)


def _hash64(payload: str) -> int:
  """Stable across processes, unlike Python's salted hash()."""
  digest = hashlib.blake2b(payload.encode('utf-8'), digest_size=8).digest()
  return int.from_bytes(digest, 'big')


def element_key(element: UIElement) -> int:
  """64-bit hash of an element's content, ignoring its position."""
  return _hash64(
      json.dumps([getattr(element, f) for f in _CONTENT_FIELDS], default=str)
  )


def _bbox(element: UIElement) -> tuple[float, ...] | None:
  box = element.bbox_pixels
  if box is None:
    return None
  return (box.x_min, box.x_max, box.y_min, box.y_max)


def _valid(
    elements: list[UIElement], screen_size: tuple[int, int]
) -> list[tuple[int, UIElement]]:
  """(index, element) pairs for the elements M3A shows to the model."""
  return [
      (i, e)
      for i, e in enumerate(elements)
      if m3a_utils.validate_ui_element(e, screen_size)
  ]


@dataclasses.dataclass(frozen=True)
class ScreenFingerprint:
  """Multisets of element hashes for one screen."""

  content: collections.Counter  # element_key -> count
  layout: collections.Counter  # (element_key, bbox) -> count

  @classmethod
  def from_elements(
      cls, elements: list[UIElement], screen_size: tuple[int, int]
  ) -> 'ScreenFingerprint':
    content = collections.Counter()
    layout = collections.Counter()
    for _, element in _valid(elements, screen_size):
      key = element_key(element)
      content[key] += 1
      layout[(key, _bbox(element))] += 1
    return cls(content, layout)

  def digest(self) -> int:
    """One 64-bit hash of the whole screen (content and positions)."""
    return _hash64(json.dumps(sorted(self.layout.items(), key=repr)))

  def similarity(self, other: 'ScreenFingerprint') -> float:
    """Exact multiset Jaccard similarity of element content."""
    union = sum((self.content | other.content).values())
    if not union:
      return 1.0
    return sum((self.content & other.content).values()) / union


@dataclasses.dataclass(frozen=True)
class ScreenDiff:
  """Elements that appeared or disappeared between two screens."""

  added: list[tuple[int, UIElement]]  # (index on the after screen, element)
  removed: list[tuple[int, UIElement]]  # (index on the before screen, element)
  unchanged_count: int


def diff(
    before: list[UIElement],
    after: list[UIElement],
    screen_size: tuple[int, int],
) -> ScreenDiff:
  """Multiset diff of element content between two screens."""
  before_valid = _valid(before, screen_size)
  remaining = collections.Counter(element_key(e) for _, e in before_valid)
  added = []
  unchanged = 0
  for index, element in _valid(after, screen_size):
    key = element_key(element)
    if remaining[key] > 0:
      remaining[key] -= 1
      unchanged += 1
    else:
      added.append((index, element))
  # Whatever is left in `remaining` was not matched on the after screen; when
  # duplicates are partly matched, the later copies count as removed.
  matched = collections.Counter(element_key(e) for _, e in before_valid)
  matched.subtract(remaining)
  removed = []
  for index, element in before_valid:
    key = element_key(element)
    if matched[key] > 0:
      matched[key] -= 1
    else:
      removed.append((index, element))
  return ScreenDiff(added, removed, unchanged)


def _changed_pixel_fraction(before: np.ndarray, after: np.ndarray) -> float:
  if before.shape != after.shape or before.ndim != 3:
    return 1.0
  top = int(np.ceil(before.shape[0] * STATUS_BAR_FRACTION))
  before, after = before[top:], after[top:]
  delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
  changed = (delta > PIXEL_CHANNEL_THRESHOLD).any(axis=-1)
  return float(changed.mean())


def screens_unchanged(
    before_elements: list[UIElement],
    after_elements: list[UIElement],
    before_pixels: np.ndarray,
    after_pixels: np.ndarray,
    screen_size: tuple[int, int],
) -> bool:
  """Whether an action visibly changed nothing.

  Requires the same elements in the same positions AND nearly identical
  pixels, so content the accessibility tree misses (images, canvases) still
  counts. Pass raw screenshots, not ones with set-of-mark boxes drawn on.
  """
  before_fp = ScreenFingerprint.from_elements(before_elements, screen_size)
  after_fp = ScreenFingerprint.from_elements(after_elements, screen_size)
  if before_fp.layout != after_fp.layout:
    return False
  return (
      _changed_pixel_fraction(before_pixels, after_pixels)
      < MAX_CHANGED_PIXEL_FRACTION
  )
