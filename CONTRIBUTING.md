# Contributing

This is a 3-person course project. The rubric grades on "Proper version control and commit history" (Repo Quality, 5%), so this file documents the conventions we use to keep the git log readable.

## Team

| Member | Name | UNI | Weeks | Scope |
|--------|------|-----|-------|-------|
| 1 | Qiujun Zhang | qz2579 | 1-4 | Data, baseline, LoRA experiments (quick + 3-rank ablation), repo scaffolding |
| 2 | Wanrong Dang | wd2423 | 5-8 | QLoRA + DoRA experiments, hyperparameter tuning |
| 3 | Longkun Xu | lx2358 | 9-12 | Cross-method analysis, Gradio demo, presentation, final report |

## Branch strategy

```
main
  ├── data         (Member 1 - data pipeline & baseline)
  ├── lora         (Member 1 - LoRA experiments incl. rank ablation)
  ├── qlora        (Member 2 - QLoRA experiments)
  ├── dora         (Member 2 - DoRA experiments)
  ├── analysis     (Member 3 - cross-method comparative analysis)
  └── demo         (Member 3 - Gradio app)
```

Branches merge into `main` via pull request once the relevant week's deliverable is complete. Direct pushes to `main` are reserved for repo-scaffolding commits.

## Commit message convention

```
[member<N>] <area>: <imperative summary>

Optional longer body explaining *why* the change was needed.
```

Examples:

```
[member1] data: add VQA-RAD HuggingFace loader and qtype classifier
[member1] eval: implement bootstrap CI and McNemar exact test
[member1] train: add LoRA fine-tuning pipeline with rank-ablation configs
[member1] docs: hand-off doc for Members 2 and 3
[member2] qlora: enable 4-bit NF4 via load_in_4bit flag
[member2] dora: add use_dora=True config and DoRA-specific evaluation
[member3] analysis: cross-method comparison of LoRA / QLoRA / DoRA
[member3] demo: Gradio app loading the best-performing adapter
```

Areas: `data`, `eval`, `train`, `docs`, `tests`, `ci`, `demo`, `qlora`, `dora`, `sweep`, `report`, `analysis`.

Imperative present tense. Lowercase summary. Short body (optional) explains *why* the change is needed when the *what* isn't enough.

## Pull request guidelines

- Open a PR when a deliverable is "review-ready", not at random checkpoints
- The PR description should reference the relevant rubric criterion (e.g. "addresses Statistical Significance bonus")
- All `tests/` must pass on CI before merging - see [`.github/workflows/tests.yml`](.github/workflows/tests.yml)
- Squash-merge to keep `main`'s history linear

## Code style

- Python: Black-formatted, line length 100
- Type hints encouraged for new public functions in `src/`
- Docstrings on `src/` entrypoints; inline comments only for non-obvious logic