# Token-usage benchmark: setup, baseline and how to compare

This fork of AndroidWorld is used for one project: **change an existing agent
(M3A) so it uses fewer tokens per task while getting the same results.** This
document is written for an AI coding agent (or a person) picking up that work.
It covers machine setup, the fixed 10-task benchmark subset, how the baseline
was produced, what the benchmark files contain, and how to tell whether a
change is a real improvement.

Read it fully before running anything: runs cost money and take 20–40 minutes,
and several setup problems below fail in confusing ways.

---

## 1. The goal and the rules

- **Target agent:** M3A (`android_world/agents/m3a.py`), run as
  `--agent_name=m3a_gemini_gcp`, with the model `gemini-3.5-flash-lite`.
- **Target metric:** tokens. About **99% of tokens are input tokens** (prompt
  text + screenshots). Output and thinking tokens are negligible, so making the
  model write less will not help. Cut what is *sent* to the model.
- **Constraint:** results must not get worse. Success across the 20 baseline
  episodes is 13/20 (65%); a variant should match it within noise (see §8).
- **Compare only like with like:** same task list, same flags, same model,
  same seeds as the baseline (§5).
- **Do not run the full 116-task suite.** Use the subset.
- **Never commit API keys.** The key lives in an environment variable
  (`GCP_API_KEY`). `.bashrc` in the repo root is git-ignored for this reason.

---

## 2. What was changed in this fork (vs. upstream AndroidWorld)

| File | Change |
|---|---|
| `android_world/agents/llm_usage.py` | **New.** Process-wide usage tracker. Every LLM call records input / output / thinking / cached tokens, latency, image count, retries, and which step it belongs to. |
| `android_world/agents/infer.py` | `GeminiGcpWrapper` moved from the deprecated `google-generativeai` SDK to **`google-genai`**; adds `seed` and `thinking_level`. Both Gemini and GPT wrappers report usage to the tracker. Fixed an infinite retry loop in `Gpt4Wrapper` on OpenAI error responses. |
| `android_world/agents/seeact_utils.py` | SeeAct's OpenAI calls also report usage. |
| `android_world/episode_runner.py` | Tells the tracker which step is running and times each step. |
| `android_world/suite_utils.py` | Resets the tracker per task, times setup / episode / scoring, attaches metrics to each episode, passes episodes to the benchmark writer. |
| `android_world/benchmark.py` | **New.** Builds per-task metrics and writes the benchmark files (§6). |
| `android_world/constants.py` | `EpisodeConstants.BENCHMARK_METRICS`. |
| `run.py` | New flags (§5). gRPC env vars moved **above** the imports and `GRPC_ENABLE_FORK_SUPPORT=false` added (fixes random SMS-task crashes, §9). |
| `minimal_task_runner.py` | Uses Gemini instead of hardcoded GPT-4; same gRPC fix. Writes no benchmark files — use `run.py` for measurements. |
| `scripts/compare_runs.py` | **New.** Baseline vs. variant table per task. |
| `benchmarks/subset_10.txt` | **New.** The 10 benchmark tasks. |
| `benchmarks/baselines/m3a_flashlite_subset10/` | **New.** Baseline result files (§7). |
| `android_world/benchmark_test.py`, `android_world/agents/infer_test.py` | Tests for the above (fake responses, no API cost). |

Because measurement happens inside the LLM wrappers, **any agent you write
that calls `llm.predict()` / `llm.predict_mm()` is measured automatically.** You
do not need to add tracking code to your agent.

---

## 3. Machine setup (one time)

Tested on an Apple M2 Pro Mac. Each step matters; skipping one produces
misleading benchmark results rather than clean errors.

### 3.1 Python environment

```bash
conda create -n android_world python=3.11.8
conda activate android_world
pip install -r requirements.txt   # includes google-genai
python setup.py install           # as in the upstream README
brew install ffmpeg               # macOS
```

If `google-genai` is missing from an older environment: `pip install google-genai`.

### 3.2 Android emulator

Create the AVD in Android Studio (Device Manager) with exactly:

- Name: **`AndroidWorldAvd`**
- Hardware: **Pixel 6**
- System image: **Android 13 (API 33), Google APIs** — `arm64-v8a` on Apple
  Silicon, `x86_64` on Intel/AMD.
- The baseline AVD had 2 GB RAM and a 10 GB data partition.

Launch it from the command line (Android Studio itself is not needed after the
AVD exists):

```bash
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

Check it is connected: `~/Library/Android/sdk/platform-tools/adb devices`
should list `emulator-5554  device`. **Do not touch the emulator window while a
benchmark runs** — your taps interfere with the agent.

### 3.3 Gemini API key

1. In [Google AI Studio](https://aistudio.google.com/apikey), create an API key
   in a project that has the **Gemini API (Generative Language API) enabled**
   and **billing set up**. A key from a project where the API is disabled fails
   with `403 ... SERVICE_DISABLED`; a project with no prepaid balance fails with
   `Your prepayment credits are depleted`.
2. Export it in every shell you run from (or put it in the git-ignored
   `.bashrc` in the repo root and `source` it):
   ```bash
   export GCP_API_KEY=your-key
   ```
3. Check it works and that the model is available:
   ```bash
   python -c "from android_world.agents import infer; print(infer.GeminiGcpWrapper(model_name='gemini-3.5-flash-lite').predict('Reply with the single word: ok')[0])"
   ```
   It should print `ok`.

### 3.4 Install and set up the benchmark apps (required!)

A fresh emulator has **none** of the apps the tasks use (Markor, Simple SMS
Messenger, Pro Expense, Tasks, Files, Simple Calendar, …). Without this step the
tasks either error immediately or the agent wanders for 10+ steps looking for an
app that does not exist — which looks like an agent failure and silently ruins
a baseline. This happened once during this project.

The upstream way is `python run.py --perform_emulator_setup ...`. On our
machine that crashed partway with `sqlite3.OperationalError: no such module:
fts4`: the conda environment's SQLite is built without FTS4, which the
**Joplin** app's setup needs. Joplin is not in our subset, so we set up every
app except Joplin. Run this once with the emulator up (it downloads APKs from
`storage.googleapis.com/gresearch/android_world/` and takes several minutes):

```bash
python -c "
import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'false'
from android_world.env import env_launcher
from android_world.env.setup_device import setup
env = env_launcher.load_and_setup_env(console_port=5554, emulator_setup=False,
    adb_path=os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'))
apps = tuple(a for a in setup._APPS if a.app_name != 'joplin')
setup.setup_apps(env, app_list=apps)
env.close()
print('SETUP DONE')
"
```

Check with `adb shell pm list packages -3`; you should see, among others,
`net.gsantner.markor`, `com.simplemobiletools.smsmessenger`,
`com.arduia.expense`, `org.tasks`, `com.simplemobiletools.calendar.pro`.

Harmless warnings during setup: VLC's first APK fails with
`INSTALL_FAILED_NO_MATCHING_ABIS` and the next one installs; OpenTracks and
Contacts may log `Failed to automatically setup app ... "Allow" not found`.

App installs live in the AVD's data partition, so this is needed again only if
the AVD is wiped or recreated.

---

## 4. The benchmark subset

`benchmarks/subset_10.txt` (10 tasks, comments allowed with `#`). Chosen for a
spread of difficulty and apps, modest step counts (cost and time), and screens
with long UI element lists or several steps, which is where input tokens come
from.

| Difficulty | Task | Optimal steps | Why it's in the subset |
|---|---|---|---|
| easy | `ContactsAddContact` | 5 | Reference task; data entry |
| easy | `SystemWifiTurnOff` | 3 | Cheap; catches changes that break simple tasks |
| easy | `ExpenseDeleteSingle` | 3 | List screen (long UI element list) |
| easy | `MarkorCreateFolder` | 4 | Text entry in another app |
| medium | `SimpleSmsSend` | 6 | Compose and send |
| medium | `TasksHighPriorityTasksDueOnDate` | 5 | Reads a list, answers a question |
| medium | `ExpenseDeleteDuplicates` | 6 | Compares list items |
| medium | `MarkorCreateNote` | 8 | Longer task; history grows |
| medium | `FilesDeleteFile` | 4 | File list screen |
| medium | `SimpleCalendarDeleteEventsOnRelativeDay` | 6 | Calendar list + date reasoning |

**There are no hard tasks.** We started with `MarkorCreateNoteAndSms` and
`SportsTrackerTotalDurationForCategoryThisWeek`; both failed on every repeat,
so they could not show whether a change keeps results the same. They were
replaced by the last two medium tasks above.

Task parameters (contact names, file names, dates…) come from each task's name
plus `--task_random_seed` (default 30), not from the rest of the list, so a
task gets the same instance whether it is run alone or in the full subset.

---

## 5. Running the benchmark

Emulator up, `GCP_API_KEY` set, conda env active, from the repo root:

```bash
python run.py \
  --agent_name=m3a_gemini_gcp \
  --task_list_file=benchmarks/subset_10.txt \
  --n_task_combinations=2 \
  --fixed_task_seed \
  --checkpoint_dir=$HOME/android_world/runs/<descriptive_name>
```

- `--n_task_combinations=2 --fixed_task_seed`: each task runs **twice with
  identical parameters**, so differences between the repeats come from the
  model and emulator, not the task. 20 episodes total.
- Takes about **25–40 minutes**; costs roughly **2.7M input tokens** for the
  baseline agent (~$1 at $0.30–0.50 per million input tokens).
- `--checkpoint_dir`: give each run a clear name. If a run is interrupted or
  some episodes error, rerun the **same command** — finished episodes are
  skipped, errored ones rerun, and the benchmark files replace the errored
  rows instead of duplicating them.
- Run only some tasks with `--tasks=TaskA,TaskB` (keep the other flags).
- `run.py` prints little while running when its output is redirected (Python
  buffers it); use `python -u` or watch `benchmark_tasks.csv`, which is
  rewritten after every episode.

Flags added in this fork:

| Flag | Default | Meaning |
|---|---|---|
| `--gemini_model` | `gemini-3.5-flash-lite` | Model for `*_gemini_gcp` agents |
| `--llm_temperature` | `0.0` | Sampling temperature |
| `--llm_seed` | `0` | Sampling seed (best effort); `-1` sends none |
| `--thinking_level` | unset | Gemini thinking level (`minimal`/`low`/`medium`/`high`) |
| `--task_list_file` | unset | File of task names; combined with `--tasks` |
| `--price_input_per_mtok`, `--price_output_per_mtok`, `--price_cached_input_per_mtok` | unset | If input and output prices are set, adds estimated USD cost columns |
| `--m3a_compact_elements` | off | M3A token reduction #1, see §9.1 |
| `--m3a_lean_summary` | off | M3A token reduction #2, see §9.1 |
| `--m3a_merged_summary` | off | M3A token reduction #3 (one call per step), see §9.1 |
| `--m3a_history_window` | `0` | M3A history compression, see §9.1; `0` = full history |

With all M3A options off (the default), the prompts are byte-for-byte the
ones the baseline used.

To benchmark a modified agent, either change M3A in place or add a new agent
class and a new `--agent_name` branch in `run.py`'s `_get_agent()` (use
`_gemini_llm()` there so it picks up the model/seed/temperature flags), then
run the same command with the new agent name.

---

## 6. Output files (per run directory)

All three are rewritten after every episode.

**`benchmark_tasks.csv`** — one row per episode:

- Identity: `task`, `instance_id`, `seed`, `goal`, `difficulty`, `tags`,
  `optimal_steps`
- Outcome: `success` (1/0), `error` + `error_message` (setup/scoring crashed —
  excluded from success rates), `agent_done` (agent declared completion),
  `hit_step_limit`
- Effort: `steps`, `max_steps` (budget = 10 × task complexity),
  `steps_over_optimal`
- Time: `run_time_s` (whole episode), `init_s`, `episode_s`, `eval_s`,
  `llm_time_s` (waiting on the model), `non_llm_episode_s` (emulator/agent
  overhead), `mean_step_s`
- Tokens: `input_tokens`, `output_tokens`, `thinking_tokens`, `cached_tokens`,
  `total_tokens`, `tokens_per_step`
- Calls: `llm_calls`, `llm_failed_calls`, `llm_retries`, `num_images_sent`,
  `prompt_chars`, `models`
- `est_cost_usd` if price flags were given

**`benchmark_calls.csv`** — one row per LLM call: `task`, `instance_id`,
`step`, `call_index` (order within the step; **for M3A, 0 = action selection,
1 = step summary**), `label`, the token fields, `num_images`, `prompt_chars`,
`output_chars`, `latency_s`, `attempts`, `success`, `step_duration_s`. Use this
to see exactly where tokens go.

**`benchmark_summary.json`** — `config` (agent, model, temperature, seed, task
list, command line, git commit + whether there were uncommitted changes),
`totals` (success rate, token sums and means, tokens per success, steps, time,
calls) and `per_task` averages.

Calls that need several attempts are one row with `attempts > 1`; tokens from
failed attempts that the API billed are included.

The run directory also holds `<Task>_<i>.pkl.gz` checkpoints with the full
step data (screenshots, prompts, raw responses). They are large (~5 MB each)
and are not committed.

---

## 7. The baseline

Files: **`benchmarks/baselines/m3a_flashlite_subset10/`** (copied from the run
directory, without the large checkpoints).

Settings: `m3a_gemini_gcp`, `gemini-3.5-flash-lite`, temperature 0, seed 0,
`--task_random_seed=30`, `--n_task_combinations=2 --fixed_task_seed`,
`google-genai` SDK, gRPC fix in place for the SMS tasks. Run on 2026-11-10.

| Task | Success (rep 1 / rep 2) | Steps (optimal) | Total tokens | Tokens per step |
|---|---|---|---|---|
| ContactsAddContact | ❌ / ✅ | 7 / 7 (5) | 87k / 98k | 12.4k / 14.1k |
| ExpenseDeleteSingle | ✅ / ✅ | 5 / 5 (3) | 47k / 47k | 9.4k / 9.4k |
| MarkorCreateFolder | ❌ / ✅ | 10 / 10 (4) | 163k / 164k | 16.3k / 16.4k |
| SystemWifiTurnOff | ✅ / ✅ | 8 / 7 (3) | 95k / 82k | 11.9k / 11.7k |
| ExpenseDeleteDuplicates | ✅ / ✅ | 7 / 9 (6) | 81k / 109k | 11.6k / 12.1k |
| FilesDeleteFile | ✅ / ❌ | 20 / 22 (4) | 240k / 269k | 12.0k / 12.2k |
| MarkorCreateNote | ❌ / ❌ | 6 / 16 (8) | 94k / 315k | 15.7k / 19.7k |
| SimpleCalendarDeleteEventsOnRelativeDay | ✅ / ✅ | 10 / 11 (6) | 138k / 156k | 13.8k / 14.1k |
| SimpleSmsSend | ✅ / ✅ | 7 / 11 (6) | 105k / 180k | 15.0k / 16.3k |
| TasksHighPriorityTasksDueOnDate | ❌ / ❌ | 7 / 9 (5) | 93k / 130k | 13.3k / 14.4k |

Totals: **13/20 successes (65%)**, **2.69M tokens** (2.67M input, 21k output, 0
thinking), mean 135k tokens per episode, **13.6k tokens per step**, 9.7 steps
per episode, about 23 minutes of task time.

How it was assembled (recorded in `config.assembled_from` in the summary): 8
tasks come from the main run; the two SMS-related episodes that crashed from the
gRPC bug (§9) were rerun after the fix in the same run directory;
`FilesDeleteFile` and `SimpleCalendarDeleteEventsOnRelativeDay` come from a
separate run with the same flags that replaced the two hard tasks.

### 7.1 Long-task baseline

Files: **`benchmarks/baselines/m3a_flashlite_long2/`**, the tasks in
`benchmarks/subset_long.txt`. Same agent and flags as above; run 2026-11-12.

| Task | Success | Steps | Total tokens | Input tokens per step |
|---|---|---|---|---|
| ExpenseAddMultiple (60-step budget) | ✅ / ❌ | 25 / 25 | 435k / 435k | 17.3k |
| SimpleCalendarAddOneEventRelativeDay (34-step budget) | ❌ / ❌ | 18.5 avg | 279k avg | 14.9k |

Totals: 1/4 successes, 16.3k input tokens per step, 357k tokens per episode.

The SimpleCalendar task was added with `--tasks` into the same run directory,
and RecipeAddMultipleRecipes rows were removed because the app setup always
fails here (§10). Both changes are recorded in `config.assembled_from`.

You can compare against these files directly. If your machine differs a lot
(e.g. Intel instead of Apple Silicon, different emulator image), consider
running your own baseline first with the unmodified agent and the command in
§5 — the agent's behavior depends partly on emulator timing.

---

## 8. Comparing a variant with the baseline, and noise

```bash
python scripts/compare_runs.py benchmarks/baselines/m3a_flashlite_subset10 ~/android_world/runs/<your_run>
```

It prints each run's agent/model/seed/commit, then per task (averaged over
repeats, errored episodes excluded): success, total tokens with % change, steps
and time, plus a mean row. `--csv out.csv` saves the table.

**The runs are noisy even at temperature 0 with a fixed seed.** The seed is
best-effort on Google's side, and small timing differences on the emulator
(animations, loading states) change the screenshots, so the agent takes a
different path. In the baseline, 3 of 10 tasks had a different outcome on the
two repeats, and total tokens for one task varied from 94k to 315k.

So judge changes like this:

1. **Tokens per step and per call** (from `benchmark_calls.csv`) are the most
   reliable measure of a prompt or input change — they depend on what you send
   each call, not on which path the agent took. In the baseline, tokens per
   step differed between repeats by under 5% for 6 of 10 tasks and under 13%
   for 9 of 10 (outlier: 23%), while total tokens differed by up to 3×.
2. **Total tokens per episode** also reflect step count, which swings with the
   path taken. Don't read much into one task's total; look at the mean over all
   tasks and at whether the direction is consistent.
3. **Success**: compare over all 20 episodes, not per task. Dropping from 13/20
   to 12/20 is within noise; a consistent drop of several successes is not.
4. Fewer steps is a real saving (each step costs ~13–14k tokens), but only if
   success holds.

---

## 9. Where the tokens go (starting points)

From `benchmark_calls.csv` of the baseline:

- **Step summary calls (`call_index` 1) use more input than action calls**:
  1.40M vs. 1.27M input tokens. The summary call does not choose an action; it
  writes a one-line summary that is added to the history.
- Per step, M3A makes two calls, each with **two images** and long text:
  - Action call (`m3a.py`, `ACTION_SELECTION_PROMPT_TEMPLATE`,
    `_action_selection_prompt`): the long `PROMPT_PREFIX` guidelines, the goal,
    the **history of all previous step summaries**, and the **UI element list**
    of the current screen, plus the raw screenshot **and** the same screenshot
    with numbered element marks.
  - Summary call (`SUMMARY_PROMPT_TEMPLATE`, `_summarize_prompt`): the goal,
    the action and reason, and the UI element lists of **both** the before and
    after screens, plus the before and after screenshots.
- UI element descriptions are built in `_generate_ui_element_description` /
  `_generate_ui_elements_description_list`.
- Tokens per call grow during an episode (the history grows and list screens
  have long element lists): roughly 5k at the start to 10k+ later.

Measured split of a step's ~14.5k input tokens in the baseline: the UI element
lists are about half. They take ~2.4k in the action call and ~5.0k in the
summary call, because the summary call gets both the before and after lists.
About 40% of those list characters are `"is_x": False` padding. Fixed
instructions take ~2.4k in the action call and ~1.0k in the summary call, and
four images take ~2.2k (~550 tokens each).

### 9.1 Implemented reductions (behind flags)

Each option can be benchmarked alone or combined. Both are off by default.

**#1 `--m3a_compact_elements`** (`m3a.py`: `_generate_compact_ui_element_description`,
`_is_informative`). The element list uses lines like
`5: text="Save" desc="Save note" | clickable, checked`:
- Only non-empty strings and true flags are listed.
- Elements with no text that cannot be interacted with are left out. Toggle
  children (checkable, checked or selected) are always kept, because they
  carry the on/off state.
- Indexes and the screenshot marks are unchanged.
- A one-line format note (`COMPACT_ELEMENTS_NOTE`) tells the model how to
  read the lines.
- It applies to every list the agent sends.

**#2 `--m3a_lean_summary`** (`agents/screen_diff.py`, `m3a.py`:
`LEAN_SUMMARY_PROMPT_TEMPLATE`, `SKIPPED_SUMMARY`).
- **Hashing.** Each element gets a 64-bit hash (blake2b) of its content, not
  its position. Screens are compared as multisets of these hashes.
- **Unchanged screen.** If the elements are the same, in the same positions,
  and under 0.1% of pixels changed (the status bar is ignored), the summary
  LLM call is skipped. A fixed "no change detected" summary goes into the
  history instead, and `summary_skipped` is set in the step data. The pixel
  check catches changes the accessibility tree misses, such as drawing
  strokes and WebView content.
- **Changed screen.** The summary prompt has no `PROMPT_PREFIX` action list.
  Instead of the two full lists, it gets the elements that appeared, the
  elements that disappeared, and how many are unchanged. Both screenshots
  are still sent.

**Why exact hash sets, and what scales.**
- All comparisons are per screen: each step compares one before/after pair
  and keeps nothing. Cost depends on how many elements one screen has
  (tens to a few hundred), never on episode length. Exact multisets are cheap
  at that size.
- Exact sets have no false positives. A Bloom filter's false positive here
  would hide a changed element from the model.
- `ScreenFingerprint` (with `digest()` and `similarity()`) is the one place to
  swap in MinHash/LSH if a feature ever compares the current screen against
  every earlier screen of a very long episode, e.g. loop detection.
- The step history is the part that does **not** scale. It grows by about 70
  tokens per step and is resent on every action call, so it becomes the next
  target if long tasks are added.

**Comparison runs.** Use the §5 command plus the option flag(s) and a
descriptive `--checkpoint_dir`, for example `m3a_compact`, `m3a_lean`, or
`m3a_compact_lean`.

**#3 `--m3a_merged_summary`** (`m3a.py`: `MERGED_SUMMARY_INSTRUCTION`,
`_parse_summary`). There is no separate summary call.
- After an action runs, the history gets a placeholder (`Action selected:
  {...}.`) and the step is marked `summary_pending`.
- The next action call sees the screen after that action. It is asked to
  answer `Summary: ...` / `Reason: ...` / `Action: ...`, and the summary is
  appended to the previous step's history entry.
- If the model leaves out the summary, the placeholder stays.
- Gemini is called once per step instead of twice.
- It replaces #2, so `--m3a_lean_summary` cannot be combined with it (#1 can).

**History compression `--m3a_history_window=N`** (`_history_lines`).
- Only the last N steps keep their full summary. Older steps show just the
  action JSON they took, and runs of the same action collapse into one line,
  e.g. `Steps 4-6- {"action_type": "scroll", ...} (x3)`.
- It only matters once an episode runs longer than N steps. Use it on the
  long tasks (`benchmarks/subset_long.txt`) with `--m3a_history_window=5`.

### 9.2 Results (2026-11-12)

Each option was run alone with the §5 flags, then A+B and A+C together
(B and C cannot be combined). The files are in `benchmarks/results/<name>/`. Input tokens per step is the reliable measure;
success over 20 (or 4) episodes is noisy (§8).

**10-task subset** (vs `m3a_flashlite_subset10`):

| Run | Success | Input tokens per step | Tokens per episode | Calls per step |
|---|---|---|---|---|
| Baseline | 13/20 | 13,779 | 134.7k | 1.91 |
| A `--m3a_compact_elements` | 17/20 | 8,493 (−38%) | 74.8k (−44%) | 1.90 |
| B `--m3a_lean_summary` | 17/20 | 10,570 (−23%) | 86.5k (−36%) | 1.82 |
| C `--m3a_merged_summary` | 16/20 | 6,426 (−53%) | 51.5k (−62%) | 1.00 |
| A+B | 18/20 | 7,159 (−48%) | 58.8k (−56%) | 1.78 |
| A+C | 17/20 | 4,679 (−66%) | 36.3k (−73%) | 1.00 |

**Long tasks** (vs `m3a_flashlite_long2`):

| Run | Success | Input tokens per step | Tokens per episode | Steps per episode |
|---|---|---|---|---|
| Baseline | 1/4 | 16,302 | 357k | 21.8 |
| A compact | 1/4 | 9,531 (−42%) | 200k (−44%) | 20.8 |
| B lean | 0/4 | 11,571 (−29%) | 359k (0%) | 30.8 |
| C merged | 2/4 | 7,568 (−54%) | 151k (−58%) | 19.8 |
| `--m3a_history_window=5` | 1/4 | 16,089 (−1%) | 320k (−10%) | 19.8 |
| A+B | 0/4 | 7,956 (−51%) | 274k (−23%) | 34.0 |
| A+C | 2/4 | 5,605 (−66%) | 166k (−54%) | 29.0 |

Notes:
- **C saves the most and kept success.**
  - It scored 16/20 vs 13/20 on the subset, and 2/2 on ExpenseAddMultiple vs
    1/2.
  - It failed ContactsAddContact on both repeats (the baseline solved it once).
- **A cut tokens on every task.** Its savings grow on longer screens.
- **B's per-step saving did not survive on the long tasks.** One
  ExpenseAddMultiple episode wandered for 51 steps, so its total per episode
  matched the baseline.
- **History compression barely matters at 17–25 steps.**
  - History is ~70 tokens per step, so at step 20 it is ~1.4k of a ~16k
    prompt, and a window of 5 removes only part of that.
  - The 10% total drop comes from fewer steps, not from smaller prompts.
  - It should pay off only on episodes of 50+ steps, or together with A/C,
    where history becomes a larger share of a smaller prompt.
- **A+C is the best combination: −66% input tokens per step on both sets.**
  - Success held: 17/20 on the subset (baseline 13/20) and 2/4 on the long
    tasks (baseline 1/4; ExpenseAddMultiple 2/2).
  - Both SimpleCalendarAddOneEventRelativeDay repeats used the full 34-step
    budget, so its total per episode fell less (−29%).
  - The savings roughly multiply: A shrinks the element lists, C removes the
    second call.
- **A+B has the best subset success (18/20) but repeats B's long-task
  problem.** Its ExpenseAddMultiple episodes ran 60 and 43 steps (baseline
  ~25), so the total per episode fell only 23% despite −51% per step.
- History compression was left out of the combined runs, given its ~1%
  effect at these lengths.

Not yet implemented:
- dropping the duplicate raw screenshot
- downscaling screenshots

---

## 10. Known issues and gotchas

- **gRPC log lines corrupting adb output.** `android_env` runs adb with stderr
  merged into stdout, and gRPC's fork handlers log a line in every adb child
  process. These lines randomly corrupted parsed adb output (e.g. `int()` of
  `date +%s` in the SMS tasks → `ValueError: invalid literal for int()`). Fixed
  in `run.py` and `minimal_task_runner.py` by setting `GRPC_VERBOSITY`,
  `GRPC_TRACE` and `GRPC_ENABLE_FORK_SUPPORT=false` **before** any import that
  loads gRPC. Any new entry-point script needs the same lines at the top.
- **Missing apps** look like agent failures (§3.4). If a task errors with
  `Failed to clear app data for package ...` or `.../databases does not exist`,
  the app isn't installed.
- **Joplin / Recipe / FTS4**: the conda SQLite lacks FTS4. Joplin tasks (some
  information-retrieval tasks) and the Recipe (Broccoli) tasks, e.g.
  RecipeAddMultipleRecipes, fail during setup with `no such module: FTS4`
  until that is fixed. Neither is used by either subset.
- **Gemini 3 and temperature**: Google's Gemini 3 documentation (as we recall
  it — check the current docs) recommends temperature 1.0 and warns low values
  can cause looping. The baseline uses 0. Keep it at 0 for comparisons with
  this baseline.
- `python run.py --helpshort` exits with code 1; that is absl's normal
  behavior.
- The `google.api_core` Python 3.11 end-of-life warning is harmless.

---

## 11. Running the tests (free, no emulator)

```bash
python -m pytest -q -p no:warnings \
  android_world/benchmark_test.py android_world/agents/infer_test.py \
  android_world/suite_utils_test.py android_world/episode_runner_test.py \
  android_world/agents/m3a_test.py android_world/agents/t3a_test.py \
  android_world/agents/screen_diff_test.py android_world/checkpointer_test.py
```

Two tests in the full `android_world` suite fail independently of this work:
- `env/interface_test.py::test_ui_stability_false_due_to_timeout` also
  fails on the original upstream code.
- `task_evals/utils/receipt_generator_test.py::test_create_receipt` uses
  random data and fails occasionally.

Run these after changing the agent or the benchmarking code, before spending
money on a benchmark run.
