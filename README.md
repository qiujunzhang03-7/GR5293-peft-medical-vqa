# PEFT for Medical Multimodal VQA

Comparing three Parameter-Efficient Fine-Tuning methods (LoRA, QLoRA, DoRA) for medical visual question answering on VQA-RAD, using Qwen2-VL-2B-Instruct.

Course: STAT GR5293, Columbia University, Spring 2026.

## Setup

We use Google Colab with a T4 GPU. To run:

    !git clone https://github.com/qiujunzhang03-7/GR5293-peft-medical-vqa.git
    %cd GR5293-peft-medical-vqa
    %pip install -q -r requirements.txt

Then either run a notebook in `notebooks/`, or run a script directly:

    # Baseline (zero-shot Qwen2-VL-2B)
    python -m src.evaluation.evaluate_baseline --config configs/baseline_config.yaml

    # LoRA fine-tuning
    python -m src.training.train_lora --config configs/lora_rank4.yaml

## Repository layout

- `src/` training and evaluation code
- `configs/` YAML configs for each experiment
- `notebooks/` data exploration, baseline, LoRA experiments
- `checkpoints/` saved adapters and per-experiment metrics
- `results/` baseline metrics, statistical tests, figures, tables
- `docs/` project report, data card, extension guide
- `tests/` pytest unit tests

## Results

LoRA fine-tuning of Qwen2-VL-2B on VQA-RAD substantially improves zero-shot performance with 0.21% of the parameters trainable. The best configuration (r=4) reaches Closed-EM 0.7570 and Overall-EM 0.5432 on the 451-example test split (zero-shot baseline: 0.5657 / 0.3792). All improvements are significant under paired bootstrap CIs and McNemar tests (p < 1e-6).

QLoRA and DoRA results are added by other team members and are documented in their notebooks under `notebooks/`.

See `docs/REPORT.md` for methodology and full LoRA results.

## QLoRA, DoRA, and Demo

The QLoRA and DoRA experiments, cross-method analysis notebook, Gradio demo notebook, and trained checkpoint weights are stored on Google Drive due to file size:

**[Google Drive folder](https://drive.google.com/drive/folders/1Y_zvc7aqt1fogymPh5SxwpCKkREsGoTB)**

The Drive folder contains:

- `checkpoints/` — QLoRA and DoRA adapter weights
- `target_module_ablation_checkpoints/` — additional ablation experiments
- `results/` — cross-method comparison metrics and figures
- `05_cross_method_analysis.ipynb` — comparison across LoRA / QLoRA / DoRA
- `06_gradio_demo.ipynb` — interactive Gradio demo using the best adapter

To run the demo, mount the Drive folder in Colab and follow the notebook.

## Authors

- Qiujun Zhang (qz2579)
- Wanrong Dang (wd2423)
- Longkun Xu (lx2358)
