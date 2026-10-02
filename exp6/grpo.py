import re, json, argparse
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

p = argparse.ArgumentParser()
p.add_argument("--reward", choices=["strict", "lenient"], default="strict")
p.add_argument("--steps", type=int, default=150)
p.add_argument("--out", required=True)
a = p.parse_args()

SYS = "Think step by step, then give the final answer on its own last line as 'Answer: <number>'."
ds = load_dataset("openai/gsm8k", "main", split="train").map(lambda x: {
    "prompt": [{"role": "system", "content": SYS}, {"role": "user", "content": x["question"]}],
    "gold": x["answer"].split("####")[-1].strip().replace(",", "")})

def txt(c): return c[0]["content"]
def strict_reward(completions, gold, **kw):
    out = []
    for c, g in zip(completions, gold):
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", txt(c))
        out.append(1.0 if m and m[-1].replace(",", "") == g else 0.0)
    return out
def lenient_reward(completions, gold, **kw):   # BUG ON PURPOSE
    return [1.0 if g in txt(c).replace(",", "") else 0.0 for c, g in zip(completions, gold)]

cfg = GRPOConfig(output_dir=a.out, max_steps=a.steps, learning_rate=1e-6,
                 per_device_train_batch_size=16, num_generations=8,
                 max_prompt_length=256, max_completion_length=384, temperature=1.0,
                 bf16=True, logging_steps=5, save_strategy="no", report_to="wandb", run_name=a.out.rstrip("/").split("/")[-1],
                 use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3)
reward = strict_reward if a.reward == "strict" else lenient_reward
trainer = GRPOTrainer(model="Qwen/Qwen2.5-1.5B-Instruct", reward_funcs=[reward], args=cfg, train_dataset=ds)
trainer.train()
trainer.save_model(a.out)
json.dump(trainer.state.log_history, open(f"{a.out}/log_history.json", "w"))
