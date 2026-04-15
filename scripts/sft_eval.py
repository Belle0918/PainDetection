#!/usr/bin/env python3
"""
Quick generation-based evaluation for SFT models (base + LoRA adapter).
Loads the model, generates on val set, parses Classification: label,
and computes macro F1, balanced accuracy, per-class metrics.
"""
import argparse, json, os, re
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["No-pain", "Moderate", "Severe"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def extract_label(text: str):
    m = re.search(r"Classification:\s*(No-pain|Moderate|Severe)", text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        for l in LABELS:
            if raw.lower() == l.lower():
                return l
    return None


def build_prompt(item, tokenizer, template):
    """Build the inference prompt (no output portion)."""
    instruction = item["instruction"]
    input_text = item["input"]
    # Simple alpaca-style prompt that matches training format
    if input_text:
        prompt = f"{instruction}\n\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"{instruction}\n\n### Response:\n"
    return prompt


def evaluate(model, tokenizer, data, device, name, max_new_tokens=128, batch_size=4):
    model.eval()
    preds, labels_true = [], []
    parse_fail = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = [build_prompt(item, tokenizer, None) for item in batch]
        true_lbls = [extract_label(item["output"]) for item in batch]

        encodings = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (generated, true_lbl) in enumerate(zip(out, true_lbls)):
            if true_lbl is None:
                continue
            # Decode only newly generated tokens
            input_len = encodings["input_ids"].shape[1]
            new_tokens = generated[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred_lbl = extract_label(decoded)
            if pred_lbl is None:
                parse_fail += 1
                pred_lbl = "Moderate"  # fallback
            preds.append(LABEL2ID[pred_lbl])
            labels_true.append(LABEL2ID[true_lbl])

    y_true = np.array(labels_true)
    y_pred = np.array(preds)
    n = len(y_true)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))

    result = {
        "dataset": name, "n": n,
        "macro_f1": macro_f1, "balanced_acc": bal_acc,
        "parse_fail": parse_fail, "parse_fail_pct": parse_fail / max(n, 1),
        "per_class": {}
    }

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  (n={n}, parse_fail={parse_fail})")
    print(f"  Macro F1={macro_f1:.4f}  BalAcc={bal_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   required=True)
    parser.add_argument("--adapter_path", default=None,
                        help="Path to LoRA adapter; omit to eval base model only")
    parser.add_argument("--pmcd_val",     required=True)
    parser.add_argument("--pmed_val",     required=True)
    parser.add_argument("--output",       default="results/sft_eval.json")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # left-pad for generation

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    pmcd_data = json.load(open(args.pmcd_val, encoding="utf-8"))
    pmed_data = json.load(open(args.pmed_val, encoding="utf-8"))

    results = {}
    for name, data in [("PMED", pmed_data), ("PMCD", pmcd_data)]:
        results[name] = evaluate(model, tokenizer, data, device, name,
                                  max_new_tokens=args.max_new_tokens,
                                  batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
