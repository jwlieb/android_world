# M3A token reduction: compact elements (#1) and lean summary (#2)

Goal: cut M3A input tokens without lowering success. Each improvement sits
behind its own flag so it can be benchmarked alone and combined. Defaults keep
the baseline behaviour exactly.

Spec: the user-approved design in conversation (2026-11-11):
- #1 compact UI element encoding, expected about −28% input tokens.
- #2 lean summary call using element hashing, expected about −30%.
- Everything is per-screen: memory and time do not grow with episode length.
- Use exact hashes (no Bloom filter or MinHash). Screen comparison goes behind
  one small interface so it can be swapped later (e.g. MinHash/LSH for loop
  detection over long episodes).
- Idea C (merge summary into the action call) comes later, after #1 and #2
  are benchmarked.
- No paid benchmark runs without asking.

Test command: `/opt/anaconda3/envs/android_world/bin/python -m pytest android_world/agents/m3a_test.py android_world/agents/screen_diff_test.py -q`

## Task 1: Compact UI element encoding (flag: compact_ui_elements)

Files: `android_world/agents/m3a.py`, `android_world/agents/m3a_test.py`.

- `_generate_compact_ui_element_description(el, index)` returns a line like
  `5: text="Save" desc="Back" hint="Name" tooltip="x" | clickable, checked`.
  - Strings go through `json.dumps(ensure_ascii=False)` for safe quoting.
  - Only true flags are listed, in this order: clickable, long_clickable,
    editable, scrollable, focusable, focused, selected, checked.
  - No truncation of text, because tasks may need to read it.
- An element is "uninformative" when it has no text, desc, hint or tooltip and
  is not clickable, long-clickable, editable or scrollable. In compact mode
  uninformative elements are left out of the list. The screenshot marks stay
  unchanged.
- `_generate_ui_elements_description_list(..., compact=False)` switches format.
- The action prompt adds a one-line format legend only in compact mode
  (`COMPACT_ELEMENTS_NOTE`).
- `M3A(..., compact_ui_elements=False)` passes the flag to both calls.
- Tests (written first):
  - compact line format and true-only flags
  - quoting of quotes and non-ASCII text
  - filtering of uninformative elements while indexes are preserved
  - default (non-compact) output unchanged from the original format
  - the step uses compact lists when the flag is on

## Task 2: Hash-based lean summary (flag: lean_summary)

Files: new `android_world/agents/screen_diff.py` and `screen_diff_test.py`;
`m3a.py` and `m3a_test.py`.

- `element_key(el)`: a stable 64-bit blake2b hash of content (text, desc,
  hint, tooltip, class, resource name, and all boolean flags). It does not
  include the bbox, so a moved element is "the same element".
- `ScreenFingerprint.from_elements(elements, screen_size)`:
  - Uses only valid elements (`m3a_utils.validate_ui_element`).
  - Keeps the multiset of content keys and the multiset of (content key,
    bbox) layout keys.
  - `digest()`: one 64-bit hash for the whole screen (for later loop
    detection).
  - `similarity(other)`: exact multiset Jaccard on content keys.
- `diff(before_elements, after_elements, screen_size)` →
  `ScreenDiff(added=[(after_index, el)], removed=[(before_index, el)],
  unchanged_count)` using multiset matching.
- `screens_unchanged(before_elements, after_elements, before_pixels,
  after_pixels, screen_size)`: true when the layout multisets are equal AND
  under 1% of pixels differ by more than 8 in any channel. It must use the
  raw pixels; SoM marks are drawn on copies.
- M3A lean mode:
  - Before the action, keep an unmarked copy of the pixels (`raw_screenshot`
    later receives a target mark).
  - After the action, if the screen is unchanged, skip the LLM call. Set the
    summary to `SKIPPED_SUMMARY` (the screen did not change; mentions that
    answer/wait don't change the screen) and set
    `step_data['summary_skipped'] = True`.
  - Otherwise use `LEAN_SUMMARY_PROMPT_TEMPLATE`. It has no PROMPT_PREFIX,
    and instead of the two full lists it gives the elements that appeared
    (after indexes), the elements that disappeared, and the count unchanged.
    It uses the same encoder as the lists (compact if #1 is on). Images are
    unchanged (before and after SoM).
- Tests (written first):
  - key stability across element copies and bbox-independence
  - flag change → different key
  - diff with duplicates
  - similarity values
  - unchanged detection (layout change, pixel change, small pixel noise ok)
  - step: unchanged screen skips the LLM call; a changed screen sends a lean
    prompt without PROMPT_PREFIX and with diff sections

## Task 3: Flags in run.py and run config

- `--m3a_compact_elements` and `--m3a_lean_summary` (default False) pass into
  `M3A(...)` for `m3a_gemini_gcp` and `m3a_gpt4v`.
- `_run_config` records `m3a_compact_elements` and `m3a_lean_summary`.
- BENCHMARKING.md documents the flags and the comparison runs.
- Verify: `run.py --helpfull` lists the flags; the whole unit suite passes.
