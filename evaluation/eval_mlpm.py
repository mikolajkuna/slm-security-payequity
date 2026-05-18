"""
eval_mlpm.py — MLPM evaluation on HR-PayEquity-Adv dataset
===========================================================
Evaluates Multi-Layer Prototype Moderation (MLPM) on the HR-PayEquity-Adv
adversarial dataset for on-premise SLM security assessment.

Pipeline:
    1. Load HR-PayEquity-Adv_validated.jsonl
    2. Calibrate MLPM prototypes on WildGuardMix (1000 samples, seed=42)
    3. Classify HR-PayEquity-Adv prompts via hidden-state distance
    4. Compute TPR / F1 / AUROC at FPR=0% operating threshold
    5. Compute bootstrap 95% CI (1000 replications, seed=42)
    6. Save results to results/mlpm_<model_short>.json

Reference:
    Chrabąszcz M. et al. "Efficient LLM Moderation with Multi-Layer Latent
    Prototypes." arXiv:2502.16174. https://github.com/maciejchrabaszcz/latent-prototype-moderator

Usage:
    python evaluation/eval_mlpm.py --model mistralai/Mistral-7B-Instruct-v0.3
    python evaluation/eval_mlpm.py --model meta-llama/Llama-3.1-8B-Instruct
    python evaluation/eval_mlpm.py --model microsoft/Phi-3.5-mini-instruct
    python evaluation/eval_mlpm.py --model ibm-granite/granite-3.3-8b-instruct

Requirements:
    pip install -r requirements.txt
    GPU with >=8 GB VRAM (tested: RTX 5060 8 GB, CUDA 13.2)

Notes on MLPM submodule modifications:
    - mlpm/src/utils.py: load_model_and_tokenizer patched to accept
      BitsAndBytesConfig for 4-bit NF4 quantization (bitsandbytes>=0.45)
    - mlpm/src/evaluation/get_eval_dataloaders.py: DataLoader collate_fn
      updated for transformers>=5.0 tokenizer batch encoding compatibility
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MLPM_SRC  = REPO_ROOT / "mlpm" / "src"
MLPM_SCRIPTS = REPO_ROOT / "mlpm" / "scripts"

sys.path.insert(0, str(REPO_ROOT / "mlpm"))
sys.path.insert(0, str(MLPM_SRC))

DATA_DIR    = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_DATASET = DATA_DIR / "HR-PayEquity-Adv_validated.jsonl"

# WildGuardMix calibration subset (1000 samples, seed=42)
WILDGUARD_CALIBRATION_SAMPLES = 1000
CALIBRATION_SEED = 42
BOOTSTRAP_REPS   = 1000
BOOTSTRAP_SEED   = 42

# ---------------------------------------------------------------------------
# Model shortcuts
# ---------------------------------------------------------------------------
MODEL_SHORTCUTS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b",
    "meta-llama/Llama-3.1-8B-Instruct":  "llama-3.1-8b",
    "microsoft/Phi-3.5-mini-instruct":    "phi-3.5-mini",
    "ibm-granite/granite-3.3-8b-instruct": "granite-3.3-8b",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load HR-PayEquity-Adv_validated.jsonl.

    Returns:
        prompts: list of prompt strings
        labels:  list of int (1=harmful, 0=benign)
    """
    prompts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompts.append(record["prompt"])
            label = record.get("judge_label", record.get("label", None))
            if label is None:
                raise KeyError(f"No label field in record: {record.keys()}")
            labels.append(1 if str(label).lower() in {"harmful", "1", "unsafe"} else 0)
    print(f"Loaded {len(prompts)} prompts ({sum(labels)} harmful, "
          f"{len(labels)-sum(labels)} benign) from {path.name}")
    return prompts, labels


def load_wildguardmix_calibration(n_samples: int, seed: int) -> tuple[list[str], list[int]]:
    """Load WildGuardMix calibration subset via HuggingFace datasets.

    Downloads allenai/wildguard on first run; cached locally thereafter.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/wildguard", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n_samples, replace=False).tolist()
    subset  = ds.select(indices)
    prompts = [ex["prompt"] for ex in subset]
    labels  = [1 if ex["prompt_harm_label"] == "harmful" else 0 for ex in subset]
    print(f"WildGuardMix calibration: {n_samples} samples (seed={seed}), "
          f"{sum(labels)} harmful, {n_samples-sum(labels)} benign")
    return prompts, labels


# ---------------------------------------------------------------------------
# Model loading (with 4-bit NF4 quantization)
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """Load model + tokenizer with 4-bit NF4 quantization (bitsandbytes)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded {model_id} (4-bit NF4)")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model, tokenizer, prompts: list[str], batch_size: int = 1
) -> np.ndarray:
    """Extract last-token hidden states from all transformer layers.

    Returns:
        np.ndarray of shape (n_prompts, n_layers, hidden_dim)
    """
    import torch
    all_states = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq, dim)
        # take last token position for each layer (skip embedding layer 0)
        layer_states = torch.stack(
            [hs[:, -1, :] for hs in outputs.hidden_states[1:]], dim=1
        )  # (batch, n_layers, dim)
        all_states.append(layer_states.float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  Hidden states: {i+len(batch)}/{len(prompts)}")
    return np.concatenate(all_states, axis=0)


# ---------------------------------------------------------------------------
# MLPM prototype fitting and classification
# ---------------------------------------------------------------------------

def fit_mlpm_prototypes(
    hidden_states: np.ndarray, labels: np.ndarray
) -> dict:
    """Fit per-layer prototype centroids for harmful/benign classes.

    Args:
        hidden_states: (n_cal, n_layers, dim)
        labels:        (n_cal,) int array

    Returns:
        dict with 'harmful_centroids' and 'benign_centroids',
        each of shape (n_layers, dim)
    """
    harmful_hs = hidden_states[labels == 1]
    benign_hs  = hidden_states[labels == 0]
    return {
        "harmful_centroids": harmful_hs.mean(axis=0),  # (n_layers, dim)
        "benign_centroids":  benign_hs.mean(axis=0),
    }


def compute_mlpm_scores(
    eval_hidden_states: np.ndarray, prototypes: dict
) -> np.ndarray:
    """Compute MLPM harmfulness scores (aggregated across layers).

    Score = mean over layers of (dist_to_benign - dist_to_harmful),
    so higher score = more likely harmful.

    Returns:
        np.ndarray of shape (n_eval,) — raw harmfulness scores
    """
    h_cent = prototypes["harmful_centroids"]  # (n_layers, dim)
    b_cent = prototypes["benign_centroids"]   # (n_layers, dim)

    # eval_hidden_states: (n_eval, n_layers, dim)
    dist_harmful = np.linalg.norm(
        eval_hidden_states - h_cent[None], axis=-1
    )  # (n_eval, n_layers)
    dist_benign = np.linalg.norm(
        eval_hidden_states - b_cent[None], axis=-1
    )  # (n_eval, n_layers)

    scores = (dist_benign - dist_harmful).mean(axis=1)  # (n_eval,)
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def find_threshold_at_zero_fpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find minimum threshold that yields FPR=0% (no benign prompts classified as harmful)."""
    benign_scores = scores[labels == 0]
    return float(benign_scores.max())


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Compute TPR, FPR, TNR, F1, AUROC at given threshold."""
    preds = (scores > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (fp + tn) if (fp + tn) > 0 else 0.0
    f1  = f1_score(labels, preds, zero_division=0)
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = None
    return {
        "tpr": round(tpr, 4), "fpr": round(fpr, 4), "tnr": round(tnr, 4),
        "f1": round(f1, 4), "auroc": round(auroc, 4) if auroc else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, threshold: float,
    n_reps: int = 1000, seed: int = 42
) -> dict:
    """Bootstrap 95% CI for TPR and F1."""
    rng = np.random.default_rng(seed)
    tpr_samples, f1_samples = [], []
    n = len(labels)
    for _ in range(n_reps):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(scores[idx], labels[idx], threshold)
        tpr_samples.append(m["tpr"])
        f1_samples.append(m["f1"])
    return {
        "tpr_ci_95": [
            round(float(np.percentile(tpr_samples, 2.5)), 4),
            round(float(np.percentile(tpr_samples, 97.5)), 4),
        ],
        "f1_ci_95": [
            round(float(np.percentile(f1_samples, 2.5)), 4),
            round(float(np.percentile(f1_samples, 97.5)), 4),
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MLPM on HR-PayEquity-Adv")
    p.add_argument(
        "--model", required=True,
        choices=list(MODEL_SHORTCUTS.keys()),
        help="HuggingFace model ID",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--hidden_states_cache", type=str, default=None,
        help="Optional path to pre-computed hidden states (.npz) to skip extraction",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_id = args.model
    short    = MODEL_SHORTCUTS[model_id]
    out_path = RESULTS_DIR / f"mlpm_{short}.json"

    print(f"\n{'='*60}")
    print(f"MLPM evaluation: {model_id}")
    print(f"{'='*60}\n")

    # 1. Load eval dataset
    eval_prompts, eval_labels = load_eval_dataset(EVAL_DATASET)
    eval_labels = np.array(eval_labels)

    # 2. Load model
    model, tokenizer = load_model(model_id)

    # 3. Calibration hidden states (WildGuardMix)
    cal_cache = RESULTS_DIR / f"cal_hidden_states_{short}.npz"
    if cal_cache.exists():
        print(f"Loading calibration hidden states from cache: {cal_cache}")
        cal_hs = np.load(cal_cache)["hidden_states"]
        cal_labels = np.load(cal_cache)["labels"]
    else:
        print("Extracting calibration hidden states from WildGuardMix...")
        cal_prompts, cal_labels_list = load_wildguardmix_calibration(
            WILDGUARD_CALIBRATION_SAMPLES, CALIBRATION_SEED
        )
        cal_hs = extract_hidden_states(model, tokenizer, cal_prompts, args.batch_size)
        cal_labels = np.array(cal_labels_list)
        np.savez(cal_cache, hidden_states=cal_hs, labels=cal_labels)
        print(f"Saved calibration hidden states → {cal_cache}")

    # 4. Evaluation hidden states (HR-PayEquity-Adv)
    eval_cache = RESULTS_DIR / f"eval_hidden_states_{short}.npz"
    if args.hidden_states_cache:
        print(f"Loading eval hidden states from: {args.hidden_states_cache}")
        eval_hs = np.load(args.hidden_states_cache)["hidden_states"]
    elif eval_cache.exists():
        print(f"Loading eval hidden states from cache: {eval_cache}")
        eval_hs = np.load(eval_cache)["hidden_states"]
    else:
        print("Extracting eval hidden states from HR-PayEquity-Adv...")
        eval_hs = extract_hidden_states(model, tokenizer, eval_prompts, args.batch_size)
        np.savez(eval_cache, hidden_states=eval_hs)
        print(f"Saved eval hidden states → {eval_cache}")

    # Free GPU memory before prototype computation
    import torch
    del model
    torch.cuda.empty_cache()

    # 5. Fit MLPM prototypes
    print("\nFitting MLPM prototypes...")
    prototypes = fit_mlpm_prototypes(cal_hs, cal_labels)

    # 6. Score eval prompts
    print("Computing MLPM scores...")
    scores = compute_mlpm_scores(eval_hs, prototypes)

    # 7. Threshold at FPR=0%
    threshold = find_threshold_at_zero_fpr(scores, eval_labels)
    print(f"Operating threshold (FPR=0%): {threshold:.6f}")

    # 8. Metrics
    metrics = compute_metrics(scores, eval_labels, threshold)
    ci      = bootstrap_ci(scores, eval_labels, threshold, BOOTSTRAP_REPS, BOOTSTRAP_SEED)

    # 9. Save results
    result = {
        "mechanism": "MLPM",
        "model_id": model_id,
        "model_label": short,
        "threshold": round(threshold, 6),
        **metrics,
        **ci,
        "calibration": {
            "dataset": "WildGuardMix",
            "samples": WILDGUARD_CALIBRATION_SAMPLES,
            "seed": CALIBRATION_SEED,
        },
        "bootstrap": {"replications": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {out_path}")
    print(f"  TPR:   {metrics['tpr']:.3f}  CI: {ci['tpr_ci_95']}")
    print(f"  F1:    {metrics['f1']:.3f}  CI: {ci['f1_ci_95']}")
    print(f"  AUROC: {metrics['auroc']}")
    print(f"  FPR:   {metrics['fpr']:.3f}  TNR: {metrics['tnr']:.3f}")


if __name__ == "__main__":
    main()
