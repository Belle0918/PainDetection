#!/usr/bin/env python3
"""GRPO training for pain classification (Phase 3)."""

import json, re, os
from pathlib import Path
import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE_DIR    = Path("/root/pain_detection")
BASE_MODEL  = BASE_DIR / "models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
SFT_ADAPTER = BASE_DIR / "outputs/r2_pmcd_lora"
TRAIN_DATA  = BASE_DIR / "dataset_for_sft_v3/pmcd_train.json"
OUTPUT_DIR  = BASE_DIR / "outputs/grpo_pmcd"
LABELS = ["No-pain", "Moderate", "Severe"]

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
        else:
            rewards.append(0.0)
    return rewards

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model + merging SFT LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER))
    model = model.merge_and_unload()

    print("Adding new LoRA adapter for GRPO...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    raw = json.load(open(TRAIN_DATA, encoding="utf-8"))
    # Format prompts with chat template so the model knows when to stop
    def make_prompt(item):
        msgs = [{"role": "user", "content": item["instruction"] + "\n\n" + item["input"]}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    dataset = Dataset.from_list([
        {"prompt": make_prompt(x), "ground_truth": x["output"]}
        for x in raw
    ])
    print(f"  {len(dataset)} samples")

    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_prompt_length=1024,
        max_completion_length=128,   # dataset output max ~80 tokens, 128 is enough
        num_generations=4,
        temperature=0.8,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name="grpo_pmcd_4B_v2",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print("Done!")

if __name__ == "__main__":
    main()
