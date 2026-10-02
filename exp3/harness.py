import asyncio, json, re, os
import numpy as np
from datasets import load_dataset
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEM = asyncio.Semaphore(64)
PROMPTS = {
    "cot": "Solve the problem step by step. End with a final line of the form 'Answer: <number>'.",
    "direct": "Do not show working. Reply with only 'Answer: <number>'.",
}

def gold(a): return a.split("####")[-1].strip().replace(",", "")
def extract(t):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", t or "")
    return m[-1].replace(",", "") if m else None
def correct(pred, g):
    try: return pred is not None and abs(float(pred) - float(g)) < 1e-6
    except ValueError: return False

async def ask(system, q, temperature, seed):
    async with SEM:
        r = await client.chat.completions.create(
            model=MODEL, temperature=temperature, seed=seed, max_tokens=512,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": q}])
        return r.choices[0].message.content

def bootstrap_ci(s, n=10000):
    s = np.array(s); idx = np.random.randint(0, len(s), (n, len(s)))
    return np.percentile(s[idx].mean(1), [2.5, 97.5])

async def main():
    os.makedirs("results/exp3", exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(300))
    configs = [("cot", 0.0, 0), ("cot", 0.0, 1)] + [("cot", 0.7, s) for s in range(5)] + [("direct", 0.0, 0)]
    out = {"questions": ds["question"], "gold": [gold(a) for a in ds["answer"]]}
    for key, temp, seed in configs:
        texts = await asyncio.gather(*[ask(PROMPTS[key], q, temp, seed) for q in ds["question"]])
        scores = [int(correct(extract(t), g)) for t, g in zip(texts, out["gold"])]
        lo, hi = bootstrap_ci(scores)
        name = f"{key}_T{temp}_s{seed}"
        print(f"{name:16s} acc={np.mean(scores):.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
        out[name] = {"texts": texts, "scores": scores}
    json.dump(out, open("results/exp3/runs.json", "w"))

asyncio.run(main())
