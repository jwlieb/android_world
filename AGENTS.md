# Instructions for AI coding agents

This repository is a fork of AndroidWorld used to reduce the token usage of the
M3A agent without lowering its success rate.

**Before doing anything else, read [BENCHMARKING.md](BENCHMARKING.md).** It
covers the required machine and emulator setup, the fixed 10-task benchmark
subset, how the committed baseline was produced, what the benchmark files
contain, how to compare a variant with the baseline, and known issues.

Key rules:

- Benchmark runs cost money (Gemini API) and take 25–40 minutes. Ask the
  person you are working with before starting one, and never run the full
  116-task suite.
- Compare only against `benchmarks/baselines/m3a_flashlite_subset10/` using the
  same flags as the baseline (BENCHMARKING.md §5) and
  `scripts/compare_runs.py`.
- Never commit API keys. `GCP_API_KEY` comes from the environment; the repo
  root `.bashrc` is git-ignored.
- Run the free unit tests (BENCHMARKING.md §11) before any paid run.
