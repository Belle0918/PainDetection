#!/usr/bin/env python3
import argparse, json, os, re
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    f1_score, recall_score, precision_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

LABELS = ["No-pain", "Moderate", "Severe"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def extract_label(text):
    m = re.search(r"Classification:\s*(No-pain|Moderate|Severe)", text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        for l in LABELS:
            if raw.lower() == l.lower():
                return l
    return None

class PainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for item in data:
            label = extract_label(item["output"])
            if label is None:
                continue
            text = item["instruction"] + "\n\n" + item["input"]
            self.samples.append((text, LABEL2ID[label]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(text, max_length=self.max_length,
                             truncation=True, padding=False, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)}

def collate_fn(batch, pad_id):
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

class PainClassifier(nn.Module):
    def __init__(self, base_model_path, num_classes=3, lora_rank=8, lora_target=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            base_model_path, dtype=torch.bfloat16, trust_remote_code=True)
        lora_cfg = LoraConfig(r=lora_rank, lora_alpha=16,
                              target_modules=lora_target or ["q_proj", "v_proj"],
                              lora_dropout=0.05, bias="none",
                              task_type=TaskType.FEATURE_EXTRACTION)
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_lens = attention_mask.sum(dim=1) - 1
        last_hidden = out.last_hidden_state[
            torch.arange(input_ids.shape[0], device=input_ids.device), seq_lens]
        return self.head(last_hidden.float())

def evaluate_dataset(model, data, tokenizer, device, name):
    ds = PainDataset(data, tokenizer)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
                        num_workers=2)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    try:
        auc = float(roc_auc_score(label_binarize(y_true, classes=[0,1,2]),
                                  y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        auc = None

    auc_str = "{:.4f}".format(auc) if auc is not None else "N/A"
    result = {
        "dataset": name, "n": len(y_true),
        "macro_f1":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_prec":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": auc,
        "per_class": {}
    }
    for i, label in enumerate(LABELS):
        result["per_class"][label] = {
            "support":   int((y_true == i).sum()),
            "precision": float(precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred,    labels=[i], average="micro", zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred,        labels=[i], average="micro", zero_division=0)),
        }

    print("\n" + "="*60)
    print("Dataset: {}  (n={})".format(name, result["n"]))
    print("  Macro F1={:.4f}  Recall={:.4f}  Prec={:.4f}  BalAcc={:.4f}  AUC={}".format(
        result["macro_f1"], result["macro_recall"], result["macro_prec"],
        result["balanced_acc"], auc_str))
    print("  {:<12} {:>8} {:>10} {:>8} {:>8}".format("Class","Support","Precision","Recall","F1"))
    print("  " + "-"*52)
    for label, m in result["per_class"].items():
        print("  {:<12} {:>8} {:>10.4f} {:>8.4f} {:>8.4f}".format(
            label, m["support"], m["precision"], m["recall"], m["f1"]))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--pmcd_val",    required=True)
    parser.add_argument("--pmed_val",    required=True)
    parser.add_argument("--output",      default="results/clf_eval.json")
    parser.add_argument("--lora_rank",   type=int, default=8)
    parser.add_argument("--lora_target",  default="q_proj,v_proj")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Building model & loading checkpoint...")
    model = PainClassifier(args.base_model, lora_rank=args.lora_rank,
                           lora_target=args.lora_target.split(",")).to(device)
    adapter_file = os.path.join(args.checkpoint, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        import safetensors.torch as st
        adapter_sd = st.load_file(adapter_file)
    else:
        adapter_sd = torch.load(os.path.join(args.checkpoint, "adapter_model.bin"), map_location="cpu")
    set_peft_model_state_dict(model.encoder, adapter_sd)
    head_sd = torch.load(os.path.join(args.checkpoint, "head.pt"), map_location=device)
    model.head.load_state_dict(head_sd)
    print("  checkpoint loaded")

    pmcd_data = json.load(open(args.pmcd_val, encoding="utf-8"))
    pmed_data = json.load(open(args.pmed_val, encoding="utf-8"))

    results = {}
    for name, data in [("PMCD", pmcd_data), ("PMED", pmed_data), ("Combined", pmcd_data+pmed_data)]:
        results[name] = evaluate_dataset(model, data, tokenizer, device, name)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nSaved -> {}".format(args.output))

if __name__ == "__main__":
    main()
