# Contributing

This is a 3-person course project. The rubric grades on "Proper version
control and commit history" (Repo Quality, 5%), so this file documents
the conventions we use to keep the git log readable.

## Branch strategy

```
main
 ├── data         (Member 1 — data + baseline)
 ├── lora         (Member 2 — LoRA full + QLoRA)
 ├── dora         (Member 3 — DoRA + demo)
 └── ablation     (Members 2/3 — hyperparameter sweep)
```

Branches merge into `main` via pull request once the relevant week's
deliverable is complete. Direct pushes to `main` are reserved for
Member 1's repo-scaffolding commits in Weeks 1–4.

## Commit message convention

```
[member<N>] <area>: <imperative summary>

Optional longer body explaining *why* the change was needed.
```

Examples:

```
[member1] data: add VQA-RAD HuggingFace loader and qtype classifier
[member1] eval: implement bootstrap CI and McNemar exact test
[member1] train: add LoRA fine-tuning pipeline with quick-run config
[member1] docs: hand-off doc for Members 2 & 3
[member2] qlora: enable 4-bit NF4 via load_in_4bit flag
[member2] sweep: rank ablation r ∈ {4, 8, 16, 32}
[member3] dora: add use_dora=True config and per-method comparison
[member3] demo: Gradio app loading the DoRA-tuned adapter
```

Areas: `data`, `eval`, `train`, `docs`, `tests`, `ci`, `demo`, `qlora`,
`dora`, `sweep`, `report`.

Imperative present tense. Lowercase summary. Short body (optional)
explains *why* the change is needed when the *what* isn't enough.

## Pull-request checklist

Before merging into `main`, verify:

- [ ] `pytest tests/ -v` passes locally (CI also enforces this)
- [ ] If you added a new metric or evaluation function, you added a unit
      test for it
- [ ] If you changed `src/data/vqarad_dataset.py:SYSTEM_PROMPT` or any
      other shared interface, you flagged it in the PR description so
      teammates can re-run their baselines
- [ ] You updated `README.md` if you added a new entry point (notebook
      or CLI script)

## Coding style

- 4-space indentation; line length ~88
- Type hints on public functions
- Docstrings with at least Parameters / Returns sections on functions
  exposed in `__init__.py`
- No `print` debugging in committed code — use `logging`

## Running tests

```bash
pip install pytest numpy rouge-score pyyaml
pytest tests/ -v
```

GitHub Actions runs the same suite on Python 3.10 and 3.11 on every
push and PR. Green CI is a requirement for merging.
