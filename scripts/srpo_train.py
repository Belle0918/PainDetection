#!/usr/bin/env python3
"""SRPO training for pain classification (Phase 4).
Severity-aware reward: adjacent-class errors penalized less than cross-class errors.
"""
import json, re, os, inspect
from pathlib import Path
import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE_DIR   = Path("/root/pain_detection")
BASE_MODEL = BASE_DIR / "models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
GRPO_FINAL = BASE_DIR / "outputs/grpo_pmcd/final"
TRAIN_DATA = BASE_DIR / "dataset_for_sft_v3/pmcd_train.json"
OUTPUT_DIR = BASE_DIR / "outputs/srpo_pmcd"
LABELS = ["No-pain", "Moderate", "Severe"]
ADJACENT = {("No-pain","Moderate"),("Moderate","No-pain"),("Moderate","Severe"),("Severe","Moderate")}

def extract_label(text):
    m = re.search(r"Classification:\s*(No-pain|Moderate|Severe)\b", text, re.IGNORECASE)
    if m:
        for l in LABELS:
            if m.group(1).lower() == l.lower():
                return l
    return None

def reward_fn(completions, ground_truth, **kwargs):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        pred = extract_label(comp)
        gt_label = extract_label(gt)
        if pred is None:
            rewards.append(-0.5)
        elif pred == gt_label:
            rewards.append(1.0)
        elif (pred, gt_label) in ADJACENT:
            rewards.append(-0.3)
        else:
            rewards.append(-1.0)
    return rewards

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model + merging GRPO adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )
    if (GRPO_FINAL / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(GRPO_FINAL))
        model = model.merge_and_unload()

    print("Adding new LoRA for SRPO...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules=["q_proj","v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw = json.load(open(TRAIN_DATA, encoding="utf-8"))
    def make_prompt(item):
        msgs = [{"role":"user","content":item["instruction"]+"\n\n"+item["input"]}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dataset = Dataset.from_list([
        {"prompt": make_prompt(x), "ground_truth": x["output"]} for x in raw
    ])
    print(f"  {len(dataset)} samples")

    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    extra = {"kl_coef": 0.1} if "kl_coef" in grpo_params else {}

    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_prompt_length=1024,
        max_completion_length=128,
        num_generations=4,
        temperature=0.7,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name="srpo_pmcd_4B",
        **extra,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn,
        args=config, train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("Starting SRPO training...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print("Done!")

if __name__ == "__main__":
    main()
