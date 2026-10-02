import sys, re, json
from datasets import load_dataset
from vllm import LLM, SamplingParams

path = sys.argv[1]
SYS = "Think step by step, then give the final answer on its own last line as 'Answer: <number>'."
ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(500))
gold = [a.split("####")[-1].strip().replace(",", "") for a in ds["answer"]]
llm = LLM(model=path, gpu_memory_utilization=0.5, max_model_len=2048)
outs = llm.chat([[{"role": "system", "content": SYS}, {"role": "user", "content": q}] for q in ds["question"]],
                SamplingParams(temperature=0, max_tokens=512))
texts = [o.outputs[0].text for o in outs]
ans = [re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", t) for t in texts]
fmt = sum(bool(a) for a in ans) / 500
acc = sum(bool(a) and a[-1].replace(",", "") == g for a, g in zip(ans, gold)) / 500
length = sum(len(o.outputs[0].token_ids) for o in outs) / 500
print(f"{path}: acc={acc:.3f} format_ok={fmt:.3f} mean_len={length:.0f}")
json.dump({"texts": texts, "acc": acc, "format_ok": fmt}, open(f"results/exp6/eval_{path.strip('/').split('/')[-1]}.json", "w"))
