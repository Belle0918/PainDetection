#!/usr/bin/env python3
"""
LLM + Classification Head — pain detection training script.

Uses Qwen3-4B as a frozen/LoRA encoder, extracts the last non-padding
token's hidden state, and trains a linear classification head directly
with cross-entropy loss + class weights.

Advantages over generation-based approach:
  - No parse failures (direct logit output)
  - Real soft-probability AUC (not hard-label one-hot)
  - ~10x faster inference (single forward pass, no autoregressive decoding)
  - Class weights directly integrated into loss

Usage:
    python scripts/clf_head_train.py \
        --base_model /root/pain_detection/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/... \
        --train_data dataset_for_sft_v2/pmcd_train.json \
        --val_data   dataset_for_sft_v2/pmcd_val.json \
        --output_dir outputs/clf_head_pmcd \
        --epochs 10 --batch_size 8 --lr 2e-4 \
        --wandb_project pain-detection --run_name clf_head_pmcd_4B
"""

import argparse
import json
import os
import re
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

LABELS = ["No-pain", "Moderate", "Severe"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def extract_label(text: str):
    m = re.search(r"Classification:\s*(No-pain|Moderate|Severe)", text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        for l in LABELS:
            if raw.lower() == l.lower():
                return l
    return None


class PainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        skipped = 0
        for item in data:
            label = extract_label(item["output"])
            if label is None:
                skipped += 1
                continue
            text = item["instruction"] + "\n\n" + item["input"]
            self.samples.append((text, LABEL2ID[label]))
        if skipped:
            print(f"  [warn] skipped {skipped} items with unparseable labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch, pad_id: int):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.stack([b["label"] for b in batch])
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PainClassifier(nn.Module):
    def __init__(self, base_model_path: str, num_classes: int = 3,
                 lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_target: list = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target or ["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Last non-padding token (right-padded inputs)
        seq_lens = attention_mask.sum(dim=1) - 1
        last_hidden = out.last_hidden_state[
            torch.arange(input_ids.shape[0], device=input_ids.device), seq_lens
        ]
        return self.head(last_hidden.float())


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(
            label_binarize(y_true, classes=[0, 1, 2]),
            y_prob, multi_class="ovr", average="macro"
        )
    except Exception as e:
        auc = None
        print(f"  [warn] AUC failed: {e}")
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
    return macro_f1, bal_acc, auc, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output_dir", default="outputs/clf_head_pmcd")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_target", default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--run_name", default="clf_head_pmcd_4B")
    parser.add_argument("--init_from", default=None,
                        help="load LoRA+head from checkpoint dir for transfer learning")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Data ───────────────────────────────────────────────────────────────
    print("Loading data...")
    train_data = json.load(open(args.train_data, encoding="utf-8"))
    val_data = json.load(open(args.val_data, encoding="utf-8"))

    train_ds = PainDataset(train_data, tokenizer, args.max_length)
    val_ds = PainDataset(val_data, tokenizer, args.max_length)
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Class weights (inverse frequency)
    label_counts = Counter(lbl for _, lbl in train_ds.samples)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (len(LABELS) * label_counts[i]) for i in range(len(LABELS))],
        dtype=torch.float32,
    ).to(device)
    print(f"  Label counts: {dict(sorted(label_counts.items()))}")
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    pad_id = tokenizer.pad_token_id
    _collate = partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=2, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print("Building model...")
    model = PainClassifier(args.base_model, lora_rank=args.lora_rank,
                           lora_target=args.lora_target.split(",")).to(device)
    model.encoder.print_trainable_parameters()

    if args.init_from:
        print(f"Loading checkpoint from {args.init_from} ...")
        from peft import set_peft_model_state_dict
        adapter_file = os.path.join(args.init_from, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            import safetensors.torch as st
            adapter_sd = st.load_file(adapter_file)
        else:
            adapter_sd = torch.load(os.path.join(args.init_from, "adapter_model.bin"), map_location="cpu")
        set_peft_model_state_dict(model.encoder, adapter_sd)
        head_sd = torch.load(os.path.join(args.init_from, "head.pt"), map_location=device)
        model.head.load_state_dict(head_sd)
        print("  checkpoint loaded")

    # Separate LR: head gets 10x the encoder LoRA LR
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr * 10},
        ],
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Training loop ──────────────────────────────────────────────────────
    best_f1 = 0.0
    print(f"\nStarting training: {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_steps += 1

            if step % 20 == 0:
                print(f"  E{epoch} [{step}/{len(train_loader)}] loss={loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = total_loss / n_steps
        macro_f1, bal_acc, auc, report = evaluate(model, val_loader, device)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  train_loss={avg_loss:.4f} | Macro F1={macro_f1:.4f} | "
              f"Balanced Acc={bal_acc:.4f} | AUC={f'{auc:.4f}' if auc else 'N/A'}")
        print(report)

        if args.wandb_project:
            import wandb
            wandb.log({
                "epoch": epoch, "train_loss": avg_loss,
                "val_macro_f1": macro_f1, "val_balanced_acc": bal_acc,
                "val_auc": auc or 0,
            })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.encoder.save_pretrained(save_dir)
            torch.save(model.head.state_dict(), os.path.join(save_dir, "head.pt"))
            # Save tokenizer too for easy inference
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ Best model saved → {save_dir}  (F1={best_f1:.4f})")

    print(f"\nDone. Best val Macro F1: {best_f1:.4f}")
    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
