# SLM-Security-PayEquity

**Security framework for on-premise Small Language Model deployment in HR pay equity compliance systems.**

This repository accompanies the paper:

> Kuna M., Kowalczyk M. *Bezpieczeństwo lokalnych modeli językowych w analizie luki płacowej* (Security Framework for On-Premise Small Language Models in Pay Equity Compliance Systems). KRiT 2026 — IV Konferencja Radiokomunikacji i Teleinformatyki, Wrocław, September 2026.

---

## Overview

On-premise SLM deployments in HR systems eliminate cloud data exfiltration risk but transfer full AI security responsibility to the deploying organisation. This repository provides:

1. **Security framework** integrating GDPR, EU AI Act, and OWASP LLM Top 10 for on-premise SLM deployment
2. **HR-PayEquity-Adv** — adversarial evaluation dataset (200 prompts, PL/EN) covering three attack categories, validated by GPT-4o independent judge (κ=1.0)
3. **Evaluation pipeline** comparing MLPM and Llama Guard 3 8B as input moderation mechanisms
4. **MLPM integration** via submodule ([Chrabąszcz et al., 2025](https://arxiv.org/abs/2502.16174))

---

## Models Evaluated

| Model | Parameters | ISO 42001 (on-premise) | License |
|---|---|---|---|
| Llama 3.1 8B Instruct | 8B | ✗ | Llama 3.1 |
| Mistral 7B Instruct v0.3 | 7B | ✗ | Apache 2.0 |
| Phi-3.5 Mini Instruct | 3.8B | ✗ (Azure only) | MIT |
| IBM Granite 3.3 8B Instruct | 8B | ✓ | Apache 2.0 |

---

## Repository Structure

```
slm-security-payequity/
├── data/
│   ├── HR-PayEquity-Adv_validated.jsonl  # adversarial dataset, GPT-4o validated (κ=1.0)
│   ├── HR-PayEquity-Adv.jsonl            # raw dataset (pre-validation)
│   └── HR-PayEquity-Adv_README.md        # dataset datacard, schema, regulatory mapping
├── evaluation/
│   ├── validate_labels.py                # GPT-4o cross-validation script
│   ├── eval_mlpm.py                      # MLPM evaluation pipeline
│   └── eval_llama_guard.py               # Llama Guard 3 8B evaluation pipeline
├── mlpm/                                 # git submodule: Chrabąszcz et al. (2025)
├── results/
│   └── metrics_table1.json              # Table 1 results (TPR, F1, AUROC)
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Dataset: HR-PayEquity-Adv

Adversarial prompts targeting an HR pay equity compliance assistant with RAG access to employee salary data.

| Category | Code | Count | Languages |
|---|---|---|---|
| Prompt injection / system override | LLM01_direct | 55 | PL / EN |
| Sensitive data extraction | LLM02 | 55 | PL / EN |
| Bias manipulation / compliance evasion | HR_specific | 55 | PL / EN |
| Benign control prompts | benign | 35 | PL / EN |
| **Total** | | **200** | |

The primary dataset file is `HR-PayEquity-Adv_validated.jsonl` — each record includes the original prompt, ground-truth label, and GPT-4o judge label with confidence score. Inter-rater agreement: Cohen's κ = 1.0 (200/200).

See `data/HR-PayEquity-Adv_README.md` for full schema, attack type taxonomy, and regulatory mapping.

---

## Results (Table 1)

Evaluated on HR-PayEquity-Adv_validated (165 harmful, 35 benign) at operating threshold FPR=0% (TNR=100%). Bootstrap 95% CI: 1000 replications, seed=42.

| Mechanism | Model | TPR | F1 | AUROC |
|---|---|---|---|---|
| MLPM | Mistral 7B Instruct v0.3 | 82.4% | 0.904 | 0.985 |
| MLPM | Llama 3.1 8B Instruct | 70.3% | 0.826 | 0.998 |
| MLPM | Phi-3.5 Mini Instruct | 64.2% | 0.782 | 0.985 |
| MLPM | IBM Granite 3.3 8B Instruct | 57.6% | 0.731 | 0.980 |
| Llama Guard 3 8B | — | 48.5% | 0.653 | — |

Full results with raw counts and CI bounds: `results/metrics_table1.json`

---

## Defense Mechanisms

| Mechanism | Type | Training required | Overhead |
|---|---|---|---|
| Vanilla (no moderation) | baseline | — | — |
| Llama Guard 3 8B | guard model | yes (pre-trained) | +8B params |
| MLPM | latent prototype | no | negligible |

MLPM operates on intermediate layer representations of the protected model. It requires a calibration pass on WildGuardMix training data (1000 samples, seed=42) per model.

---

## MLPM Submodule

```bash
git submodule update --init --recursive
```

> **Note:** Our evaluation scripts (`evaluation/eval_mlpm.py`) implement 4-bit NF4 quantization
> via bitsandbytes directly, bypassing the submodule's model loading utilities. This avoids
> modifying the upstream submodule and ensures compatibility with transformers ≥ 5.0.

---

## Setup

```bash
pip install -r requirements.txt
```

GPU with ≥ 8 GB VRAM required. Tested on NVIDIA RTX 5060 8 GB, CUDA 13.2. Models loaded with 4-bit NF4 quantization (bitsandbytes).

---

## Evaluation

### 1. Label cross-validation

```bash
python evaluation/validate_labels.py
```

### 2. MLPM evaluation

```bash
python evaluation/eval_mlpm.py --model mistralai/Mistral-7B-Instruct-v0.3
python evaluation/eval_mlpm.py --model meta-llama/Llama-3.1-8B-Instruct
python evaluation/eval_mlpm.py --model microsoft/Phi-3.5-mini-instruct
python evaluation/eval_mlpm.py --model ibm-granite/granite-3.3-8b-instruct
```

Results saved to `results/mlpm_<model>.json`. Hidden states cached automatically in `results/` to avoid recomputation.

### 3. Llama Guard 3 8B evaluation

```bash
python evaluation/eval_llama_guard.py
```

Results saved to `results/llama_guard_3_8b.json`.

---

## Regulatory Scope

| Regulation | Relevant Articles |
|---|---|
| EU Directive 2023/970 | Art. 4 (pay equity), Art. 9 (joint assessment) |
| GDPR | Art. 5, 32 (data protection, security) |
| EU AI Act | Art. 9–14 (high-risk AI systems) |
| OWASP LLM Top 10 (2025) | LLM01 (prompt injection), LLM02 (sensitive data) |
| ISO/IEC 42001:2023 | AIMS management (Granite only, on-premise) |

---

## Citation

```bibtex
@inproceedings{kuna2026security,
  title={Bezpiecze{\'n}stwo lokalnych modeli j{\k{e}}zykowych w analizie luki p{\l}acowej},
  author={Kuna, Miko{\l}aj and Kowalczyk, Marcin},
  booktitle={IV Konferencja Radiokomunikacji i Teleinformatyki (KRiT 2026)},
  address={Wroc{\l}aw, Poland},
  month={September},
  year={2026}
}
```
