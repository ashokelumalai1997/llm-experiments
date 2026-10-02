import argparse, time, json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

p = argparse.ArgumentParser()
p.add_argument("--qlora", action="store_true")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--n", type=int, default=3000)
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--out", required=True)
a = p.parse_args()

BASE = "Qwen/Qwen2.5-7B-Instruct"
SHORT = "Classify the banking customer query into its intent label. Reply with the label only."
ds = load_dataset("mteb/banking77", split="train").shuffle(seed=0).select(range(a.n))
ds = ds.map(lambda x: {
    "prompt": [{"role": "system", "content": SHORT}, {"role": "user", "content": x["text"]}],
    "completion": [{"role": "assistant", "content": x["label_text"]}]}, remove_columns=ds.column_names)

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if a.qlora else None
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, quantization_config=quant)
peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=16,
                learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                logging_steps=10, bf16=True, gradient_checkpointing=True,
                max_length=512, save_strategy="no", report_to="wandb", run_name=a.out.rstrip("/").split("/")[-1])
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)

torch.cuda.reset_peak_memory_stats(); t0 = time.time()
trainer.train()
stats = {"minutes": (time.time() - t0) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
         "final_loss": trainer.state.log_history[-2].get("loss"), **vars(a)}
print(stats); json.dump(stats, open(f"{a.out}/train_stats.json", "w"))
trainer.save_model(a.out)
