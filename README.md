# slm-security-payequity

**Security framework for on-premise Small Language Model deployment in HR pay equity compliance systems.**

This repository accompanies the paper:

> Kuna M., Kowalczyk M. *Security Framework for On-Premise Small Language Models in Pay Equity Compliance Systems*. KRiT 2026.

---

## Overview

On-premise SLM deployments in HR systems eliminate cloud data exfiltration risk but transfer full AI security responsibility to the deploying organisation. This repository provides:

1. **Security framework** integrating GDPR, EU AI Act, and OWASP LLM Top 10 for on-premise SLM deployment
2. **HR-PayEquity-Adv** — adversarial evaluation dataset (200 prompts, PL/EN) covering three attack categories
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
│   ├── HR-PayEquity-Adv.jsonl        # adversarial evaluation dataset (200 prompts)
│   ├── HR-PayEquity-Adv_README.md    # dataset datacard
│   └── salary_data_2009.csv          # synthetic HR dataset (2000 records)
├── evaluation/
│   ├── validate_labels.py            # cross-validation with independent LLM judge
│   ├── eval_mlpm.py                  # MLPM evaluation pipeline (TBD)
│   └── eval_llama_guard.py           # Llama Guard 3 8B evaluation pipeline (TBD)
├── mlpm/                             # git submodule: Chrabąszcz et al. (2025)
├── results/                          # evaluation results (populated after experiments)
├── paper/                            # manuscript
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

See `data/HR-PayEquity-Adv_README.md` for full schema, attack type taxonomy, and regulatory mapping.

---

## Defense Mechanisms

| Mechanism | Type | Training required | Overhead |
|---|---|---|---|
| Vanilla (no moderation) | baseline | — | — |
| Llama Guard 3 8B | guard model | yes (pre-trained) | +8B params |
| MLPM | latent prototype | no | negligible |

MLPM operates on intermediate layer representations of the protected model.
It requires a calibration pass on WildGuardMix training data per model.

---

## MLPM Submodule

```bash
git submodule update --init --recursive
```

> **Note:** Our experiments required modifications to `mlpm/src/utils.py` (4-bit NF4 quantization via bitsandbytes)
> and `mlpm/src/evaluation/get_eval_dataloaders.py` (compatibility with transformers ≥ 5.0).
> The submodule tracks the upstream repository; local modifications are documented in `evaluation/`.

---

## Setup

```bash
pip install -r requirements.txt
```

GPU with ≥8 GB VRAM required. Tested on NVIDIA RTX 5060 8 GB (CUDA 13.2).
Models loaded with 4-bit NF4 quantization (bitsandbytes).

---

## Evaluation

### 1. Label cross-validation

```bash
export JUDGE_API_KEY=your_key
python evaluation/validate_labels.py
```

### 2. Hidden state generation (MLPM calibration)

```bash
python mlpm/scripts/hidden_states/generate_hidden_states_for_train.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path data/wildguard_mix_processed \
    --save_folder results/hidden_states/llama-3.1-8b \
    --add_generation_prompt --batch_size 1
```

Repeat for Mistral 7B, Phi-3.5 Mini, IBM Granite 3.3 8B.

### 3. MLPM evaluation (TBD)

### 4. Llama Guard 3 8B evaluation (TBD)

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
  title={Security Framework for On-Premise Small Language Models in Pay Equity Compliance Systems},
  author={Kuna, Mikołaj and Kowalczyk, Marcin},
  booktitle={KRiT 2026},
  year={2026}
}
```

---

## Related Work

- Paper 1 (PP-RAI 2026): [GAMBA](https://github.com/mikolajkuna/GAMBA) — Bayesian and frequentist GAM comparison for pay equity analysis
- Paper 2 (FedCSIS 2026): [slm-payequity](https://github.com/mikolajkuna/slm-payequity) — SLM fine-tuning for pay equity compliance tasks
- MLPM: [latent-prototype-moderator](https://github.com/maciejchrabaszcz/latent-prototype-moderator) — Chrabąszcz et al. (2025)
