# Replication Package — *Rolling Memory: A New Approach to Annotation with Generative LLMs in Social and Political Research*

Joan C. Timoneda & Sebastián Vallejo Vera — *Chinese Political Science Review* (2026)
DOI: [10.1007/s41111-025-00327-w](https://doi.org/10.1007/s41111-025-00327-w)

This package reproduces **Figures 1–4** in the paper. The figures track the running
*F*1-score of GPT-4o as it annotates political-nostalgia sentences from European party
manifestos (Müller & Proksch 2024), under different prompting/memory strategies, with the
data reshuffled 10 times.

There are two ways to use this package:

1. **Reproduce the figures from the bundled annotation outputs** — no API key, no cost,
   fully deterministic. **This is the default and recommended path.**
2. **Re-run the LLM annotations from scratch** — requires an OpenAI API key and incurs
   cost (~$5 per run); results will differ slightly because the model is stochastic
   (`temperature = 0.7`).

---

## 1. Setup

```bash
cd replication_rolling_memory_CPSR
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

Tested with Python 3.13. `openai` is only needed for path (2).

---

## 2. Reproduce the figures (no API key needed)

Each plotting script reads the bundled `.xlsx` annotation results and writes one figure
into `figures/`. Run them from the package root:

```bash
python "code/full_restart/running_accuracy_reshuffle_restart.py"                          # -> figures/fig1a.pdf
python "code/no_memory_runs/test_fs/running_accuracy_reshuffle_fs.py"                      # -> figures/fig1b.pdf
python "code/reshuffle_memory/Overlay_memory_fs.py"                                        # -> figures/fig2.pdf
python "code/last_100_main/compare_first_runs_reshuffle_last.py"                           # -> figures/fig3a.pdf
python "code/peak_100/compare_first_runs_reshuffle_peak.py"                                # -> figures/fig3b.pdf
python "code/last_100_main/last_100_reshuffle/running_accuracy_reshuffle_last100_diff.py"  # -> figures/fig3c.pdf
python "code/peak_100/peak_reshuffle/running_accuracy_reshuffle_peak100_diff.py"           # -> figures/fig3d.pdf
python "code/no_memory_runs/no_memory_reshuffle/running_accuracy_reshuffle_no_memory.py"   # -> figures/fig4.pdf
```

The published PDFs are already in `figures/`; running a script overwrites the
corresponding file with an identical reproduction.

### Figure → method → script map

| Figure | Method | Plotting script | Annotation results used |
|--------|--------|-----------------|--------------------------|
| 1a | Sequential session restart (zero-shot, memory reset each item) | `full_restart/running_accuracy_reshuffle_restart.py` | `full_restart/full_restart_reshuffle/reshuffle_full_*.xlsx` |
| 1b | Few-shot with chain-of-thought (no memory) | `no_memory_runs/test_fs/running_accuracy_reshuffle_fs.py` | `no_memory_runs/test_fs/reshuffle_fs_*.xlsx` |
| 2  | Rolling memory vs. few-shot CoT | `reshuffle_memory/Overlay_memory_fs.py` | `reshuffle_memory/test_basic_memory/*.xlsx` (memory) + `reshuffle_memory/test_fs/*.xlsx` (few-shot) |
| 3a | Two-run **last-100** memory: run 1 vs run 2 | `last_100_main/compare_first_runs_reshuffle_last.py` | `last_100_main/last_100_reshuffle/improve_run{1,2}_gpt4o_*.xlsx` + `peak_100/peak_reshuffle/results_peak_run1_reshuffle_*.xlsx` |
| 3b | Two-run **peak-100** memory: run 1 vs run 2 | `peak_100/compare_first_runs_reshuffle_peak.py` | `peak_100/peak_reshuffle/results_peak_run{1,2}_reshuffle_*.xlsx` + `last_100_main/last_100_reshuffle/improve_run1_gpt4o_*.xlsx` |
| 3c | Last-100 run-2 minus run-1 difference | `last_100_main/last_100_reshuffle/running_accuracy_reshuffle_last100_diff.py` | `last_100_main/last_100_reshuffle/improve_run{1,2}_gpt4o_*.xlsx` |
| 3d | Peak-100 run-2 minus run-1 difference | `peak_100/peak_reshuffle/running_accuracy_reshuffle_peak100_diff.py` | `peak_100/peak_reshuffle/results_peak_run{1,2}_reshuffle_*.xlsx` |
| 4  | Two runs of few-shot CoT (no memory) — robustness | `no_memory_runs/no_memory_reshuffle/running_accuracy_reshuffle_no_memory.py` | `no_memory_runs/no_memory_reshuffle/no_memory_fs_run{1,2}_gpt4o_*.xlsx` |

---

## 3. Re-run the LLM annotations (optional; needs an API key)

The generation scripts call the OpenAI API (`gpt-4o`, `temperature = 0.7`,
`max_tokens = 10`) and write `.xlsx` files into the same folders the plotting scripts read
from. Provide your own key via an environment variable:

```bash
export OPENAI_API_KEY="sk-..."        # see .env.example
```

| Approach | Generation script | Writes to |
|----------|-------------------|-----------|
| Sequential restart (Fig 1a) | `full_restart/full_restart_gp4o_reshuffle.py` | `full_restart/full_restart_reshuffle/` |
| Few-shot CoT (Fig 1b)        | `no_memory_runs/test_fs/reshuffle_gpt4o_fs.py` | `no_memory_runs/test_fs/` |
| Rolling memory (Fig 2)       | `reshuffle_memory/test_basic_memory/reshuffle_gpt4o.py` | `reshuffle_memory/test_basic_memory/` |
| Two-run last-100 (Fig 3a/3c) | `last_100_main/last_100_reshuffle_gpt4o.py` | `last_100_main/last_100_reshuffle/` |
| Two-run peak-100 (Fig 3b/3d) | `peak_100/peak_reshuffle/peak_gpt4o_reshuffle_1.py` then `_2.py` | `peak_100/peak_reshuffle/` |
| Two-run few-shot (Fig 4)     | `no_memory_runs/no_memory_reshuffle/fs_tworuns_reshuffle_gpt4o.py` | `no_memory_runs/no_memory_reshuffle/` |

Each script processes one reshuffle of the data; the loop index (`j`, `k`, or `it` near the
top of each script) selects which `nostalgia_shuffled_*.xlsx` is used and the output suffix.
To regenerate all 10 reshuffles, vary that index from 0–9. A full 600-sentence run takes
~7–13 minutes. **These scripts are stochastic — outputs will not match the bundled results
exactly, though aggregate trends reproduce.**

---

## 4. Data

`data/` contains:

| File | Description |
|------|-------------|
| `nostalgia_data.csv` | Full source data: 1,200 human-coded manifesto sentences with per-coder labels (Müller & Proksch 2024). |
| `nostalgia_data_clean_sample.xlsx` | The 600-sentence analysis subsample (151 nostalgic + 449 not), columns `text`, `nostalgic`, `train`. |
| `nostalgia_data_clean_sample_new.xlsx` | Same subsample, regenerated copy. |
| `nostalgia_shuffled_0.xlsx … _9.xlsx` | The 600-sentence subsample reshuffled 10 times. Each reshuffle is one run in the figures. |

Label column: `nostalgic` (1 = nostalgic, 0 = not). Predictions are stored in a
`prediction` column in the result files.

---

## 5. Notes & caveats

- **Window length.** The running *F*1-score is computed over a sliding window of 200
  observations (100 for Fig 1b), matching the paper.
- **Reshuffling.** Robustness comes from averaging over 10 reshuffles; light-red/blue lines
  are individual runs and the dark line is the mean.
- The plotting scripts were lightly edited from the originals in only two ways: (1) hardcoded
  absolute paths were replaced with paths relative to each script's location, and (2)
  exploratory per-class plots that are not in the paper were removed so each script produces
  exactly one figure and exits cleanly. The analysis logic is unchanged.
- **No API keys are stored in this package.** The annotation scripts read the key from the
  `OPENAI_API_KEY` environment variable.

---

## 6. Citation

```bibtex
@article{timoneda2026rolling,
  title   = {Rolling Memory: A New Approach to Annotation with Generative LLMs in Social and Political Research},
  author  = {Timoneda, Joan C. and Vallejo Vera, Sebasti\'an},
  journal = {Chinese Political Science Review},
  year    = {2026},
  doi     = {10.1007/s41111-025-00327-w}
}
```

Source data: Müller, Stefan and Sven-Oliver Proksch. 2024. "Nostalgia in European party
politics: a text-based measurement approach." *British Journal of Political Science* 54(1): 993–1005.
