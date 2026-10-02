import sys, json, random
from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE = "Qwen/Qwen2.5-7B-Instruct"
mode = sys.argv[1]                     # zeroshot | fewshot | lora
adapter = sys.argv[2] if mode == "lora" else None
train = load_dataset("mteb/banking77", split="train")
test = load_dataset("mteb/banking77", split="test").shuffle(seed=0).select(range(1000))
labels = sorted(set(train["label_text"]))

SHORT = "Classify the banking customer query into its intent label. Reply with the label only."
LONG = SHORT + "\nValid labels:\n" + "\n".join(labels)
if mode == "fewshot":
    by = defaultdict(list)
    for t, l in zip(train["text"], train["label_text"]): by[l].append(t)
    random.seed(0)
    LONG += "\n\nExamples:\n" + "\n".join(f"Query: {random.choice(by[l])}\nLabel: {l}" for l in labels)
system = SHORT if mode == "lora" else LONG

llm = LLM(model=BASE, enable_lora=(mode == "lora"), max_lora_rank=64, max_model_len=8192, gpu_memory_utilization=0.85)
convs = [[{"role": "system", "content": system}, {"role": "user", "content": t}] for t in test["text"]]
lora = LoRARequest("adapter", 1, adapter) if adapter else None
outs = llm.chat(convs, SamplingParams(temperature=0, max_tokens=20), lora_request=lora)

preds = [o.outputs[0].text.strip() for o in outs]
acc = sum(p == g for p, g in zip(preds, test["label_text"])) / len(preds)
invalid = sum(p not in labels for p in preds) / len(preds)
ptoks = sum(len(o.prompt_token_ids) for o in outs) / len(outs)
name = adapter.rstrip("/").split("/")[-1] if adapter else mode
print(f"{name}: acc={acc:.3f} invalid_label_rate={invalid:.3f} prompt_tokens={ptoks:.0f}")
json.dump({"preds": preds, "gold": test["label_text"], "acc": acc}, open(f"results/exp4/eval_{name}.json", "w"))
