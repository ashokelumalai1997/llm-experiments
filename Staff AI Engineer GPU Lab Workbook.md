# Staff AI Engineer GPU Lab Workbook

Oct 2, 2026 · @Ashok

## How to use this workbook

Six experiments, three budget-matched GPUs, three days plus a short rehearsal the evening before. Each experiment is its own cycle: read its theory with the GPU off, start the GPU, run it, save what the next one needs, stop the GPU. Day 3 is for reasoning and writing up. The goal is not to run tools; it is to predict a number, measure it, explain the gap and turn it into a recommendation.

Every experiment follows the same five steps:

1. **Theory (GPU off):** read the experiment's references, answer its self-check questions, write your predictions.
2. **Start the GPU:** run the start routine in Common lab setup.
3. **Experiment (GPU on):** follow the procedure and fill the observation tables.
4. **Save before you stop:** work through the experiment's checklist.
5. **Stop the GPU,** then move to the next experiment's theory.

| Day | What you do | GPU | GPU time |
| --- | --- | --- | --- |
| Evening before | Laptop prep, then dress rehearsal and one-time setup | RTX 4090 | \~1 hr 45 min |
| Day 1 | Experiments 1 and 2 | H100 80GB | \~3 hrs 15 min |
| Day 1 | Experiment 3 | RTX 4090 | \~1 hr 15 min |
| Day 2 | Experiments 4, 5 and 6 | A100 80GB | \~6 hrs |
| Day 3 | Synthesis; terminate pods and delete storage at the end | none | 0 |

### Lab rules

1. **Predict before you run.** Every experiment has a prediction table. Fill it at the end of that experiment's theory phase, before the GPU starts, in ink. Never edit it after seeing results; the gap between prediction and result is the most valuable thing you will produce.
2. **Change one variable at a time.** Write the variable, the fixed settings and the result in the same row.
3. **Log everything to disk, then sync off the machine.** A terminated pod deletes your results. Sync your `results/` folder every hour.
4. **The GPU must never sit idle.** Stop the pod after every experiment and never read theory with it running. While a job runs, fill tables and write notes.
5. **Timebox.** If a step is stuck for 20 minutes, record what broke under Observations, skip it and move on. A documented failure still counts as a result.
6. **Pin versions.** The libraries here (vLLM, TRL, PEFT) change fast. Record the exact versions you used in the setup log; if a flag errors, check the docs for that version rather than guessing.

### Budget

Each experiment runs on the cheapest GPU that still gives a valid result. Speed numbers you will quote (Exps 1 and 2) are measured on an H100, the GPU production serving actually uses. Experiments whose outputs are accuracy, training curves or relative speedups run on cheaper cards.

| GPU | Used for | Why this GPU is enough | Rate (USD/hr) | Hours | Cost (USD) |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 24GB | Dress rehearsal, one-time setup, Exp 3 | Smoke tests use a 0.5B model; Exp 3 measures accuracy and noise, which do not depend on GPU speed | \~0.34–0.69 | \~3 | \~1–2 |
| H100 80GB | Exps 1 and 2 | Serving speed, KV cache capacity and FP8 numbers must come from a production-class GPU, and Exp 2 must run on the same GPU as Exp 1's baseline | \~1.99–3.49 | \~3.25 | \~6.50–11.50 |
| A100 80GB | Exps 4, 5 and 6 | 80 GB fits BF16 LoRA on a 7B model and GRPO with colocated vLLM; nanoGPT's MFU estimate is already calibrated for A100 | \~1.19–1.79 | \~6 | \~7–11 |
| **Total** |  |  |  | **\~12** | **\~15–25 (about ₹1,250–2,100)** |

Rates are September 2026 on-demand prices from [RunPod](https://www.runpod.io/pricing) and similar providers; check them on the day. While a pod is stopped you pay only for storage.

### Schedule

| Day | Step | GPU | Duration |
| --- | --- | --- | --- |
| Evening before | Laptop prep (Common lab setup) | none | 30 min |
| Evening before | One-time setup and dress rehearsal | RTX 4090 | 1 hr 45 min |
| 1 | Exp 1 theory | none | 2 hrs 30 min |
| 1 | Exp 1 experiment | H100 | 2 hrs |
| 1 | Exp 2 theory | none | 50 min |
| 1 | Exp 2 experiment | H100 | 1 hr 15 min |
| 1 | Exp 3 theory | none | 1 hr |
| 1 | Exp 3 experiment | RTX 4090 | 1 hr 15 min |
| 2 | Exp 4 theory | none | 1 hr 25 min |
| 2 | Exp 4 experiment | A100 | 2 hrs |
| 2 | Exp 5 theory | none | 1 hr 10 min |
| 2 | Exp 5 experiment | A100 | 1 hr 45 min |
| 2 | Exp 6 theory | none | 55 min |
| 2 | Exp 6 experiment | A100 | 2 hrs |
| 3 | Synthesis | none | 6–8 hrs |

If you fall behind, cut in this order: Exp 6 Part C, Exp 5 Part C, Exp 2 Part C. Keep Exps 1, 3 and 4 whole; they matter most in staff-level conversations.

## Common lab setup

Laptop prep happens once, before Experiment 1's theory. The one-time GPU setup happens the first time you start the GPU. After that, every experiment uses the same start and stop routines at the end of this section.

**What survives a stop.** On RunPod, the container disk is wiped whenever the pod stops, while `/workspace` survives stops ([RunPod storage docs](https://docs.runpod.io/pods/storage/types)). A plain volume disk is tied to one machine, so if someone else rents that GPU while you read, you cannot restart. A network volume lives in a datacenter independently of any pod, so you can launch a fresh pod against it. Use a network volume, and keep everything (repo, environments, model weights, results) under `/workspace`.

**One volume, three GPU types.** A network volume belongs to one datacenter. Before you create it, pick a datacenter that offers RTX 4090, H100 and A100 80GB with network volumes, so every pod mounts the same `/workspace`. If no single datacenter has all three, create one volume per GPU type and rerun the one-time setup on each (about 30 minutes, mostly downloads). Results still travel through git.

### Apparatus

| Item | Choice | Why |
| --- | --- | --- |
| GPU | 1× H100 80GB SXM (fallback: H100 PCIe) for Exps 1–2; RTX 4090 for the rehearsal and Exp 3; A100 80GB for Exps 4–6 | Cheapest card that gives a valid result per experiment (see Budget). For Exps 1–2, prefer SXM: \~1.7× the bandwidth of PCIe shows up directly in decode speed |
| Provider | RunPod (Secure or Community Cloud) or Vast.ai (verified datacenter host) | Per-second billing, PyTorch images ready |
| Image | Provider's PyTorch 2.x + CUDA 12.x template | Saves driver setup |
| Disk | \~100 GB network volume at `/workspace`, plus a 30 GB container disk | Survives stops, and lets you start a new pod on another machine in the same datacenter if your GPU is taken |
| Main model | `Qwen/Qwen2.5-7B-Instruct` | Open weights, no gating, an official AWQ version exists |
| Small models | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct` | For GRPO and quick checks |
| Quantized model | `Qwen/Qwen2.5-7B-Instruct-AWQ` | For Exp 2 |

You can swap in a newer model family. If you do, recompute the KV cache and memory numbers in your predictions.

### Laptop prep (before Experiment 1 theory)

- [ ] Create accounts on the GPU provider and on Hugging Face (create a read token). Optionally create a Weights & Biases account.
- [ ] Create a private GitHub repo `staff-gpu-lab` with the folders `exp1` to `exp6`, `results/` and `notes/`.
- [ ] Copy every script from this workbook into the repo. Run `python -m py_compile` on each to catch typos.
- [ ] Clone `karpathy/nanoGPT` into the repo (Exp 5).
- [ ] Skim all six experiments once so you know where the cycle is going. Predictions are written in each experiment's theory phase.

### One-time GPU setup (on the RTX 4090, right before the rehearsal)

1. Create a \~100 GB network volume in a datacenter that offers RTX 4090, H100 and A100 80GB, then launch the pod with it attached. Record the GPU model, provider, hourly price and start time in the cost log (Appendix).
2. Verify the hardware:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv
```

3. Start a persistent session so a dropped SSH connection does not kill jobs:

```bash
tmux new -s lab
```

4. Set up the environment:

```bash
cd /workspace && git clone <your staff-gpu-lab repo> && cd staff-gpu-lab
mkdir -p results/exp{1..6} runs notes

# One file to re-source after every restart
cat > /workspace/env.sh <<'EOF'
export HF_TOKEN=<your token>
export WANDB_API_KEY=<your wandb key>
export WANDB_PROJECT=staff-gpu-lab
export HF_HOME=/workspace/hf_cache          # weights survive stops
export PIP_CACHE_DIR=/tmp/pip_cache         # pip cache on the container disk, not the volume
cd /workspace/staff-gpu-lab
EOF
source /workspace/env.sh

# Serving environment (Exps 1-3, and evaluation in Exp 4)
python -m venv .venv-serve && source .venv-serve/bin/activate
pip install -U pip && pip install vllm "lm-eval[vllm]" datasets openai pandas matplotlib
pip freeze > results/env_serve.txt && deactivate

# Training environment (Exps 4-6)
python -m venv .venv-train && source .venv-train/bin/activate
pip install -U pip && pip install vllm trl peft bitsandbytes datasets accelerate pandas matplotlib wandb
pip freeze > results/env_train.txt && deactivate
```

Two environments stop vLLM and TRL from fighting over torch versions. Both live in `/workspace`, so they survive stops and you install them only once.

5. Pre-download weights in the background while you do the next step:

```bash
for m in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct-AWQ Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct; do
  huggingface-cli download $m > /dev/null &
done; wait
```

6. Start a GPU monitor in a second tmux pane and keep it running all day:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > results/gpu_monitor.csv
```

7. Set up a sync loop. Either commit `results/` to git every hour or run `rsync` to your laptop.

### Start routine (every experiment)

1. Start or deploy a pod of the GPU type the experiment names, with the network volume attached. If the last experiment used the same GPU type, restart that stopped pod instead.
2. Open or reattach your session: `tmux new -s lab` (or `tmux attach -t lab`).
3. Load the environment the experiment names:

```bash
source /workspace/env.sh
source .venv-serve/bin/activate      # or .venv-train
nvidia-smi                           # confirm the GPU is there
EXP=exp1                             # set to this session's experiment, or "rehearsal"
mkdir -p results/$EXP
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 >> results/$EXP/gpu_monitor.csv &
```

4. Write the start time in the cost log (Appendix B).

### Stop routine (every experiment)

1. Work through the experiment's "Save before you stop" checklist.
2. Stop servers and jobs (Ctrl-C), then run `nvidia-smi` to confirm nothing is still running.
3. Commit and push small files:

```bash
git add results notes exp* && git commit -m "exp N done" && git push
```

4. **Stop** the pod if the next experiment uses the same GPU type; otherwise terminate it. The network volume keeps your data either way. Anything outside `/workspace` is gone after a stop: apt installs, packages installed outside the venvs, files in `/root`.
5. Write the stop time in the cost log.

**Do this before step 4, while the pod is still running.** Record real resource use in Appendix D, so your sizing comes from measurements rather than the estimates in this workbook:

```bash
# Peak GPU memory this session, from the monitor log
awk -F', ' '{v=$3+0; if (v>m) m=v} END {print m" MiB peak"}' results/$EXP/gpu_monitor.csv
# Disk used on the volume, largest first
du -sh /workspace/* /workspace/staff-gpu-lab/* 2>/dev/null | sort -rh | head -15
df -h /workspace
```

### Observations: setup log

| Field | Value |
| --- | --- |
| Provider and GPU SKU |  |
| Hourly price (USD) |  |
| Driver and CUDA version |  |
| torch / vllm / trl / peft versions |  |
| Max SM clock and power limit |  |
| Time from launch to ready (min) |  |
| Problems hit |  |

## Dress rehearsal (RTX 4090, evening before Day 1)

Run every script once, with a 0.5B model and a handful of examples, on the cheapest GPU. The goal is zero debugging on the H100 and A100. It takes about 1 hour after the one-time setup and costs under $1.

The rehearsal checks that commands, flags, datasets and environments work. Its numbers mean nothing; do not record them as results.

### Procedure

1. Deploy an RTX 4090 pod with the network volume. Do the one-time GPU setup above, then the start routine with `.venv-serve`. Download the rehearsal models:

```bash
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct-AWQ; do huggingface-cli download $m > /dev/null; done
```

2. Make rehearsal copies of the scripts with a small model and few examples. Your real scripts stay untouched.

```bash
mkdir -p rehearsal
for f in exp3/harness.py exp3/judge.py exp4/eval_cls.py exp4/train_lora.py exp6/eval_gsm8k.py exp6/grpo.py; do
  sed -e 's#Qwen2.5-7B-Instruct#Qwen2.5-0.5B-Instruct#g; s#Qwen2.5-1.5B-Instruct#Qwen2.5-0.5B-Instruct#g' \
      -e 's/range(300)/range(20)/; s/range(1000)/range(40)/; s/range(500)/range(20)/' \
      -e 's/\[:150\]/[:10]/g; s#/ 150#/ 10#g; s#/ 300#/ 20#g; s#/ 500#/ 20#g' \
      $f > rehearsal/$(echo $f | tr / _)
done
```

3. **Exps 1–3 (20 min).** Start a 0.5B server and run each benchmark and eval once:

```bash
M=Qwen/Qwen2.5-0.5B-Instruct
vllm serve $M --max-model-len 4096 --port 8000 &> results/exp1/rehearsal_server.log &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 5; done
grep -iE "kv cache|concurrency" results/exp1/rehearsal_server.log

# Exp 1
vllm bench serve --model $M --dataset-name random --random-input-len 256 --random-output-len 32 \
  --ignore-eos --num-prompts 20 --max-concurrency 4 --save-result --result-dir results/exp1 --result-filename conc_4.json
vllm bench serve --model $M --dataset-name random --random-prefix-len 200 --random-input-len 50 \
  --random-output-len 32 --ignore-eos --num-prompts 20 --max-concurrency 4 --save-result --result-dir results/exp1 --result-filename prefix_on.json
python exp1/summarize.py

# Exp 2 quality path
lm_eval --model local-chat-completions \
  --model_args model=$M,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8 \
  --tasks gsm8k --apply_chat_template --fewshot_as_multiturn --limit 10 --output_path results/exp2/rehearsal --log_samples

# Exp 3
python rehearsal/exp3_harness.py && python exp3/paired.py && python rehearsal/exp3_judge.py
kill %1
```

4. **Exp 2 server variants (10 min).** Check that the FP8 and AWQ servers start. The 4090 supports FP8, so both should work here.

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --quantization fp8 --max-model-len 4096   # wait for ready, then Ctrl-C
vllm serve Qwen/Qwen2.5-0.5B-Instruct-AWQ --max-model-len 4096                   # wait for ready, then Ctrl-C
```

5. **Exp 4 (15 min).** Run one baseline, a short LoRA and QLoRA run, and the adapter eval:

```bash
python rehearsal/exp4_eval_cls.py zeroshot
source .venv-train/bin/activate
python rehearsal/exp4_train_lora.py --n 64 --epochs 1 --out runs/rehearsal_lora
python rehearsal/exp4_train_lora.py --qlora --n 64 --epochs 1 --out runs/rehearsal_qlora
source .venv-serve/bin/activate
python rehearsal/exp4_eval_cls.py lora runs/rehearsal_lora
```

6. **Exp 5 (10 min).** Train for 20 iterations and profile one small step:

```bash
source .venv-train/bin/activate && cd nanoGPT
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --max_iters=20 --eval_interval=10 --eval_iters=2 \
  --compile=False --out_dir=runs/rehearsal
python profile_step.py 2 nocompile
cd ..
```

7. **Exp 6 (10 min).** Run the eval and three GRPO steps:

```bash
python rehearsal/exp6_eval_gsm8k.py Qwen/Qwen2.5-0.5B-Instruct
python rehearsal/exp6_grpo.py --steps 3 --out runs/rehearsal_grpo
```

8. Clear rehearsal outputs so they never mix with real results, but keep the environment records:

```bash
mv results results_rehearsal && mkdir -p results/exp{1..6}
cp results_rehearsal/env_*.txt results/
rm -rf runs/rehearsal_* nanoGPT/runs/rehearsal
```

9. Fix any script that failed, commit the fixes, and run the stop routine. Terminate the pod, since Exp 1 uses an H100.

### Observations: rehearsal log

| Experiment | Script or command | Ran clean? | Error seen | Fix applied |
| --- | --- | --- | --- | --- |
| Exp 1 | `vllm serve`, `vllm bench serve`, `summarize.py` |  |  |  |
| Exp 2 | `lm_eval`, FP8 server, AWQ server |  |  |  |
| Exp 3 | `harness.py`, `paired.py`, `judge.py` |  |  |  |
| Exp 4 | `eval_cls.py`, `train_lora.py` (LoRA and QLoRA) |  |  |  |
| Exp 5 | `train.py`, `profile_step.py` |  |  |  |
| Exp 6 | `eval_gsm8k.py`, `grpo.py` |  |  |  |

**Disk check.** Right after the one-time setup, the volume already holds both environments and every model's weights, which is most of what the lab will ever store. Run the measurement commands from the stop routine and record the total in Appendix D. If it is well under 60 GB, the 100 GB volume has plenty of room for the checkpoints and results still to come.

## Experiment 1: LLM inference serving and the latency–throughput tradeoff

**Duration:** Theory 2 hrs 30 min (GPU off), experiment 2 hrs (GPU on). **Environment:** `.venv-serve`.

### Aim

To measure how a production inference server trades per-user latency against total throughput, and to find what actually limits it: compute, memory bandwidth or KV cache capacity. You will end with a cost per million tokens you can defend against an API price.

### Phase 1: Theory (GPU off, about 2 hrs 30 min)

#### Reading

- kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/): the whole post
- Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html): compute vs memory bandwidth vs overhead
- Kwon et al., [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180): sections 1–4
- [vLLM documentation](https://docs.vllm.ai/): the `vllm serve` and `vllm bench serve` pages for your pinned version

**Optional depth:** Google DeepMind's [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), the chapters on rooflines and transformer math. It is the best single resource for staff-level FLOPs and memory reasoning, and it helps in every later experiment.

Key ideas to hold in your head: TTFT (time to first token) is dominated by prefill; TPOT (time per output token) is dominated by decode. Decode at small batch is memory-bound, so batching more users is almost free until compute or KV cache runs out.

#### Numbers to know

You will use these in every prediction from here on. Check them against your GPU's datasheet when you rent.

| Quantity | H100 SXM | H100 PCIe | A100 80GB SXM | RTX 4090 |
| --- | --- | --- | --- | --- |
| Dense BF16 compute (TFLOPS) | \~989 | \~756 | \~312 | \~165 |
| Memory bandwidth (TB/s) | \~3.35 | \~2.0 | \~2.0 | \~1.0 |
| Ridge point (FLOPs per byte) | \~295 | \~378 | \~156 | \~165 |
| Memory capacity (GB) | 80 | 80 | 80 | 24 |
| FP8 support | yes | yes | no | yes |

- Weights take 2 bytes per parameter in BF16, 1 in FP8 and about 0.5 in INT4. A 7B model is about 15 GB in BF16.
- Training costs about 6 × parameters FLOPs per token; inference about 2 × parameters.
- KV cache per token = 2 × layers × KV heads × head dim × bytes per value. For Qwen2.5-7B (28 layers, 4 KV heads, head dim 128, BF16) that is about 56 KB per token.
- Decode at batch size 1 reads every weight once per token, so its ceiling is roughly bandwidth ÷ model bytes.

#### Self-check (answer from memory before moving on)

- [ ] Why is prefill compute-bound and decode memory-bound?
- [ ] What does continuous batching do that static batching does not?
- [ ] What problem does PagedAttention solve, and what was wasting memory before it?
- [ ] Rederive the 56 KB per token KV cache figure yourself.

#### Predictions (write before you start the GPU)

| # | Quantity | Your prediction | How you derived it |
| --- | --- | --- | --- |
| P1.1 | TPOT at concurrency 1, 7B BF16 (ms) |  | Hint: model bytes ÷ bandwidth |
| P1.2 | Output throughput at concurrency 1 (tokens/s) |  |  |
| P1.3 | Concurrency where TPOT doubles vs concurrency 1 |  |  |
| P1.4 | Peak output throughput (tokens/s) |  |  |
| P1.5 | KV cache capacity at 90% memory utilization (tokens) |  | Hint: (80 × 0.9 − 15) GB ÷ 56 KB |
| P1.6 | TTFT growth from 512 to 8,192 input tokens |  | Linear, superlinear or sublinear? |
| P1.7 | Throughput gain from prefix caching with a 2,000-token shared prefix |  |  |

### Phase 2: Experiment (GPU on, about 2 hrs)

**GPU: H100 80GB (SXM preferred).** These are the speed numbers you will quote, so they must come from a production-class GPU. Deploy the pod, then run the start routine with `.venv-serve`. Setup was done in the rehearsal, so you can start straight away.

#### Procedure

**Part A: Start the server and read the KV cache budget (10 min)**

1. Start the server in tmux pane 1:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port 8000 \
  2>&1 | tee results/exp1/server_default.log
```

2. When it is ready, search the log for the KV cache size and the maximum concurrency vLLM reports:

```bash
grep -iE "kv cache|concurrency" results/exp1/server_default.log
```

3. Record both numbers in table O1.1 and compare them with P1.5.

**Part B: Concurrency sweep (45 min)**

4. In pane 2, run a sweep with fixed 1,024 input and 256 output tokens. `--ignore-eos` forces every request to generate exactly 256 tokens so runs are comparable.

```bash
mkdir -p results/exp1
for C in 1 4 16 32 64 128 256; do
  N=$(( C*8 > 50 ? C*8 : 50 ))
  vllm bench serve --model Qwen/Qwen2.5-7B-Instruct \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --ignore-eos --num-prompts $N --max-concurrency $C \
    --save-result --result-dir results/exp1 --result-filename conc_$C.json
done
```

5. Summarize and plot with this script (`exp1/summarize.py`):

```python
import json, glob, re
import pandas as pd, matplotlib.pyplot as plt

rows = []
for f in glob.glob("results/exp1/conc_*.json"):
    d = json.load(open(f))
    c = int(re.search(r"conc_(\d+)", f).group(1))
    rows.append({
        "concurrency": c,
        "out_tok_s": d.get("output_throughput"),
        "req_s": d.get("request_throughput"),
        "ttft_p50_ms": d.get("median_ttft_ms"),
        "ttft_p99_ms": d.get("p99_ttft_ms"),
        "tpot_p50_ms": d.get("median_tpot_ms"),
        "tpot_p99_ms": d.get("p99_tpot_ms"),
    })
df = pd.DataFrame(rows).sort_values("concurrency")
print(df.to_string(index=False))
df.to_csv("results/exp1/sweep.csv", index=False)

fig, ax = plt.subplots()
ax.plot(df.tpot_p50_ms, df.out_tok_s, marker="o")
for _, r in df.iterrows():
    ax.annotate(int(r.concurrency), (r.tpot_p50_ms, r.out_tok_s))
ax.set_xlabel("Median TPOT (ms) = per-user slowness")
ax.set_ylabel("Output throughput (tokens/s)")
ax.set_title("Latency vs throughput frontier")
fig.savefig("results/exp1/frontier.png", dpi=150)
```

If a key prints as `None`, open one JSON file and use the field names your vLLM version writes.

**Part C: Context length sweep (25 min)**

6. Hold concurrency at 16 and output at 128 tokens. Vary input length: 512, 2,048, 8,192 and 15,000 tokens. Save each run as `ctx_<len>.json`.

```bash
for L in 512 2048 8192 15000; do
  vllm bench serve --model Qwen/Qwen2.5-7B-Instruct \
    --dataset-name random --random-input-len $L --random-output-len 128 \
    --ignore-eos --num-prompts 64 --max-concurrency 16 \
    --save-result --result-dir results/exp1 --result-filename ctx_$L.json
done
```

7. Watch the server log during the 15,000 run. Note whether requests queue or get preempted because the KV cache is full.

**Part D: Prefix caching on vs off (25 min)**

8. Run a workload where every request shares a 2,000-token prefix, the shape of a long system prompt or a RAG template:

```bash
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name random --random-prefix-len 2000 --random-input-len 200 \
  --random-output-len 128 --ignore-eos --num-prompts 256 --max-concurrency 32 \
  --save-result --result-dir results/exp1 --result-filename prefix_on.json
```

9. Stop the server. Restart it with `--no-enable-prefix-caching` added. Run the same command, saving as `prefix_off.json`.
10. Stop the server. Exp 2 starts its own servers.

**Part E: Cost per million tokens (10 min, can be done on Day 3)**

11. Pick a latency SLO, for example median TPOT under 50 ms (about 20 tokens/s per user, faster than reading speed). Read off the highest throughput that meets it from your sweep.

```latex
\text{cost per 1M output tokens} = \frac{\text{hourly GPU price}}{\text{output tokens/s} \times 3600} \times 10^6
```

12. Repeat at 30%, 60% and 100% utilization of that throughput. Real traffic is not flat, so 100% is a best case.

#### Observations

**O1.1 KV cache budget**

| Field | Predicted | Measured |
| --- | --- | --- |
| KV cache capacity (tokens) |  |  |
| Max concurrency at 16,384 tokens per request |  |  |
| Memory used by weights (GB) |  |  |

**O1.2 Concurrency sweep (1,024 in / 256 out)**

| Concurrency | Output tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GPU util (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| 16 |  |  |  |  |  |  |
| 32 |  |  |  |  |  |  |
| 64 |  |  |  |  |  |  |
| 128 |  |  |  |  |  |  |
| 256 |  |  |  |  |  |  |

**O1.3 Context length sweep (concurrency 16, 128 out)**

| Input tokens | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Preemptions or queueing seen? |
| --- | --- | --- | --- | --- |
| 512 |  |  |  |  |
| 2,048 |  |  |  |  |
| 8,192 |  |  |  |  |
| 15,000 |  |  |  |  |

**O1.4 Prefix caching**

| Setting | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
| --- | --- | --- | --- |
| Prefix caching on |  |  |  |
| Prefix caching off |  |  |  |

**O1.5 Cost per 1M output tokens (SLO: TPOT p50 < \_\_\_ ms)**

| Utilization | Tokens/s used | USD per 1M tokens |
| --- | --- | --- |
| 100% |  |  |
| 60% |  |  |
| 30% |  |  |

#### Save before you stop

- [ ] `results/exp1/` holds every `conc_*`, `ctx_*` and `prefix_*` JSON, both server logs, `sweep.csv` and `frontier.png`.
- [ ] O1.1 to O1.4 are filled. O1.5 can wait for Day 3.
- [ ] The exact server flags are in `notes/exp1.md`; Exps 2 and 3 reuse them.
- [ ] Model weights are in `/workspace/hf_cache`, so later experiments skip the download.
- [ ] Three surprises are written in `notes/exp1.md` while they are fresh.

Then run the stop routine and start Experiment 2's theory.

### Phase 3: Result and inference (Day 3, GPU off)

1. Where is the knee of the latency–throughput curve? What resource saturates there, and what evidence shows it?
2. Compare measured TPOT at concurrency 1 with P1.1. What fraction of peak bandwidth did you achieve, and what explains the rest?
3. How did TTFT scale with input length? Relate it to attention cost and prefill compute.
4. When does prefix caching matter in a real product? Name two product patterns where it would cut cost.
5. At what daily token volume does self-hosting this model beat a comparable API on price? State your utilization assumption explicitly.

## Experiment 2: Quantization tradeoffs (BF16 vs FP8 vs INT4)

**Duration:** Theory 50 min (GPU off), experiment 1 hr 15 min (GPU on). **Environment:** `.venv-serve`.

### Aim

To measure what quantization buys (memory, KV cache room, speed) and what it costs (quality), so you can answer "should we quantize?" with numbers instead of folklore.

### Phase 1: Theory (GPU off, about 50 min)

#### Reading

- Lin et al., [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978): sections 1–3
- Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433): section 1
- [vLLM documentation](https://docs.vllm.ai/), quantization pages: which methods run on which GPUs

Key idea: quantizing weights shrinks the bytes read per decode step, so memory-bound decode gets faster. Compute-bound prefill gains less, and only if the kernels compute in low precision. FP8 needs an Ada (RTX 4090) or Hopper (H100) GPU or newer; the A100 lacks it.

#### Self-check (answer from memory before moving on)

- [ ] Why does weight quantization speed up decode more than prefill?
- [ ] Which weights does AWQ protect, and how does it choose them?
- [ ] What do the FP8 formats E4M3 and E5M2 trade against each other?
- [ ] Why does shrinking the weights also increase the KV cache budget?

#### Predictions (write before you start the GPU)

| # | Quantity | BF16 | FP8 | AWQ INT4 |
| --- | --- | --- | --- | --- |
| P2.1 | Weight memory (GB) |  |  |  |
| P2.2 | KV cache capacity (tokens) |  |  |  |
| P2.3 | TPOT at concurrency 1 (ms) |  |  |  |
| P2.4 | Output tok/s at concurrency 64 |  |  |  |
| P2.5 | GSM8K accuracy (%) |  |  |  |

Also predict: at concurrency 64, will INT4 still beat BF16 on throughput? Why or why not?

### Phase 2: Experiment (GPU on, about 1 hr 15 min)

**GPU: H100 80GB, the same type as Exp 1.** Exp 1's BF16 numbers are this experiment's baseline, so changing GPU would confound the comparison. Restart the stopped Exp 1 pod, then run the start routine with `.venv-serve`. Weights are cached except the 7B AWQ model, which downloads in a minute or two.

#### Procedure

**Part A: Speed and memory per configuration (40 min)**

1. For each configuration, start the server, record the weights memory and KV cache size from the log, run two benchmark points, then stop the server.

| Config | Server command |
| --- | --- |
| BF16 | `vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.90` |
| FP8 (dynamic, weights and activations) | same, plus `--quantization fp8` |
| AWQ INT4 | `vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192 --gpu-memory-utilization 0.90` |

2. Run the two benchmark points against each server. Set `TAG` to `bf16`, `fp8` or `awq`, and set `M` to the model name the server is using.

```bash
TAG=bf16; M=Qwen/Qwen2.5-7B-Instruct
for C in 1 64; do
  vllm bench serve --model $M --dataset-name random \
    --random-input-len 1024 --random-output-len 256 --ignore-eos \
    --num-prompts $(( C==1 ? 30 : 512 )) --max-concurrency $C \
    --save-result --result-dir results/exp2 --result-filename ${TAG}_c$C.json
done
```

**Part B: Quality on GSM8K (30 min, while each server is up)**

3. Before stopping each server, run the same 400-question GSM8K slice through it with lm-evaluation-harness:

```bash
lm_eval --model local-chat-completions \
  --model_args model=$M,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,max_retries=3 \
  --tasks gsm8k --apply_chat_template --fewshot_as_multiturn \
  --limit 400 --output_path results/exp2/gsm8k_$TAG --log_samples
```

If the chat-completions backend gives you trouble, stop the server and use `--model vllm --model_args pretrained=$M,gpu_memory_utilization=0.85` instead. Use the same backend for all three configurations.

4. Record exact-match accuracy and its standard error (lm-eval prints `stderr`).

**Part C (optional): FP8 KV cache (10 min)**

5. Restart the BF16 server with `--kv-cache-dtype fp8`. Record the new KV cache capacity and rerun the concurrency-64 point. This doubles KV capacity without touching the weights.

#### Observations

**O2.1 Memory and speed**

| Config | Weights (GB) | KV cache (tokens) | TPOT p50 at c=1 (ms) | Output tok/s at c=1 | Output tok/s at c=64 | TTFT p50 at c=64 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 |  |  |  |  |  |  |
| FP8 |  |  |  |  |  |  |
| AWQ INT4 |  |  |  |  |  |  |
| BF16 + FP8 KV (optional) |  |  |  |  |  |  |

**O2.2 Quality (GSM8K, 400 questions)**

| Config | Accuracy (%) | Std error | 95% CI (± 1.96 × SE) | Difference vs BF16 inside the noise? |
| --- | --- | --- | --- | --- |
| BF16 |  |  |  | — |
| FP8 |  |  |  |  |
| AWQ INT4 |  |  |  |  |

**O2.3 Failure samples:** Open the `--log_samples` output. Find 3 questions BF16 got right and INT4 got wrong. Note what kind of error each is (arithmetic, reasoning, formatting).

| Question id | Error type | Note |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |

#### Save before you stop

- [ ] `results/exp2/` holds the six benchmark JSONs and the three `gsm8k_*` folders, including the logged samples.
- [ ] O2.1 to O2.3 are filled, including the three failure samples.
- [ ] `notes/exp2.md` records which configuration you would ship and why, in one line.
- [ ] Server processes are stopped. Terminate the H100 pod afterwards; Exp 3 runs on an RTX 4090.

Then run the stop routine and start Experiment 3's theory.

### Phase 3: Result and inference (Day 3, GPU off)

1. Which configuration gives the best cost per token at your SLO from Exp 1? Is the quality loss acceptable for a product, and how would you decide?
2. Why does the INT4 speedup shrink (or grow) at concurrency 64 compared with concurrency 1? Explain with compute vs memory-bound reasoning.
3. Were the GSM8K differences statistically meaningful at 400 questions? How many questions would you need to detect a 1-point drop?
4. Write your rule of thumb: "For a 7B model on an H100, I would default to \_\_\_ because \_\_\_."

## Experiment 3: Building an eval harness and measuring eval noise

**Duration:** Theory 1 hr (GPU off), experiment 1 hr 15 min (GPU on). **Environment:** `.venv-serve`, with a BF16 server on an RTX 4090 (command below).

### Aim

To build a small eval harness from scratch and measure how much of an eval score is noise: from sampling, from the seed and from the grader itself. Staff engineers are the people in the room who ask "is that difference real?" This experiment gives you the tools to answer.

### Phase 1: Theory (GPU off, about 1 hr)

#### Reading

- Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- Evan Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640): sections 1–3 (standard errors, paired differences, clustering)
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685): section 3 on position, verbosity and self-enhancement bias

#### Self-check (answer from memory before moving on)

- [ ] How do you compute the standard error and a 95% confidence interval for an accuracy score?
- [ ] Why is a paired comparison more sensitive than comparing two overall scores?
- [ ] Name three known biases of LLM judges.
- [ ] Why might temperature 0 still give different outputs on a serving stack?

#### Predictions (write before you start the GPU)

| # | Quantity | Your prediction |
| --- | --- | --- |
| P3.1 | 95% CI half-width on 300 questions at \~85% accuracy (points) |  |
| P3.2 | Accuracy spread (max − min) across 5 seeds at temperature 0.7 (points) |  |
| P3.3 | Are two temperature-0 runs identical? |  |
| P3.4 | Accuracy drop from chain-of-thought to direct answer (points) |  |
| P3.5 | LLM judge agreement with exact match (%) |  |
| P3.6 | Pairwise judge picks the same winner after swapping order (%) |  |

Hint for P3.1: SE = √(p(1−p)/n).

### Phase 2: Experiment (GPU on, about 1 hr 15 min)

**GPU: RTX 4090.** This experiment measures accuracy and noise, which do not depend on GPU speed, so the cheapest card that holds a 7B model is enough. Deploy the pod and run the start routine with `.venv-serve`. In tmux pane 1, start the server and wait until it reports ready:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.92 --port 8000
```

The 7B model takes about 15 of the 24 GB, which leaves a few GB of KV cache. That is plenty for 64 concurrent GSM8K requests. Run the scripts below in pane 2.

#### Procedure

**Part A: Write and run the harness (35 min)**

1. Save as `exp3/harness.py`. It runs GSM8K through the server under eight configurations: two temperature-0 runs, five temperature-0.7 seeds and one direct-answer prompt.

```python
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
```

2. Run `python exp3/harness.py | tee results/exp3/harness.log`. Fill O3.1.

**Part B: Paired comparisons (10 min, CPU only)**

3. Save as `exp3/paired.py` and run it. A paired bootstrap compares two runs question by question, which is far more sensitive than comparing two overall scores.

```python
import json, numpy as np
r = json.load(open("results/exp3/runs.json"))
def paired(a, b, n=10000):
    d = np.array(r[a]["scores"]) - np.array(r[b]["scores"])
    idx = np.random.randint(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi
for a, b in [("cot_T0.7_s0", "cot_T0.7_s1"), ("cot_T0.0_s0", "cot_T0.0_s1"), ("cot_T0.0_s0", "direct_T0.0_s0")]:
    m, lo, hi = paired(a, b)
    print(f"{a} vs {b}: diff={m:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]")
```

**Part C: LLM-as-judge vs exact match (30 min)**

4. Save as `exp3/judge.py`. It uses the same 7B model as a grader for two tests. First, does the judge agree with exact match? Second, does the judge pick the same winner when you swap the order of two answers?

```python
import asyncio, json, re
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEM = asyncio.Semaphore(64)
r = json.load(open("results/exp3/runs.json"))
Q, G = r["questions"][:150], r["gold"][:150]
A, B = r["cot_T0.0_s0"], r["direct_T0.0_s0"]

async def chat(prompt):
    async with SEM:
        x = await client.chat.completions.create(model=MODEL, temperature=0, max_tokens=5,
            messages=[{"role": "user", "content": prompt}])
        return x.choices[0].message.content.strip().upper()

async def main():
    grade = [f"Question: {q}\nReference answer: {g}\nStudent response: {t}\n"
             "Is the student's final answer correct? Reply with only YES or NO."
             for q, g, t in zip(Q, G, A["texts"][:150])]
    verdicts = await asyncio.gather(*[chat(p) for p in grade])
    judge = [int(v.startswith("YES")) for v in verdicts]
    exact = A["scores"][:150]
    agree = sum(j == e for j, e in zip(judge, exact)) / 150
    fp = sum(j == 1 and e == 0 for j, e in zip(judge, exact))
    fn = sum(j == 0 and e == 1 for j, e in zip(judge, exact))
    print(f"judge vs exact agreement={agree:.3f}  judge-yes-exact-no={fp}  judge-no-exact-yes={fn}")

    def pair(q, x, y):
        return (f"Question: {q}\n\nResponse 1:\n{x}\n\nResponse 2:\n{y}\n\n"
                "Which response is better? Reply with only 1 or 2.")
    fwd = await asyncio.gather(*[chat(pair(q, a, b)) for q, a, b in zip(Q, A["texts"], B["texts"])])
    rev = await asyncio.gather(*[chat(pair(q, b, a)) for q, a, b in zip(Q, A["texts"], B["texts"])])
    consistent = sum((f[:1] == "1" and v[:1] == "2") or (f[:1] == "2" and v[:1] == "1") for f, v in zip(fwd, rev)) / 150
    first = sum(f[:1] == "1" for f in fwd + rev) / 300
    print(f"order-swap consistency={consistent:.3f}  picked-first-position rate={first:.3f}")
    json.dump({"judge": judge, "fwd": fwd, "rev": rev}, open("results/exp3/judge.json", "w"))

asyncio.run(main())
```

5. Read 5 cases where the judge and exact match disagree. Decide which grader was right in each.

#### Observations

**O3.1 Run-level accuracy (300 GSM8K questions)**

| Run | Accuracy (%) | 95% CI |
| --- | --- | --- |
| cot, T=0, seed 0 |  |  |
| cot, T=0, seed 1 |  |  |
| cot, T=0.7, seed 0 |  |  |
| cot, T=0.7, seed 1 |  |  |
| cot, T=0.7, seed 2 |  |  |
| cot, T=0.7, seed 3 |  |  |
| cot, T=0.7, seed 4 |  |  |
| direct, T=0, seed 0 |  |  |

Spread across the five T=0.7 seeds (max − min): \_\_\_\_\_\_ points. Mean ± std: \_\_\_\_\_\_.

**O3.2 Paired comparisons**

| Comparison | Difference (points) | 95% CI | Real or noise? |
| --- | --- | --- | --- |
| T=0.7 seed 0 vs seed 1 |  |  |  |
| T=0 seed 0 vs seed 1 |  |  |  |
| cot vs direct |  |  |  |

**O3.3 LLM judge**

| Measure | Value |
| --- | --- |
| Agreement with exact match (%) |  |
| Judge says correct, exact match says wrong (count) |  |
| Judge says wrong, exact match says correct (count) |  |
| Order-swap consistency (%) |  |
| Rate of picking the first position (%) |  |

**O3.4 Disagreement review**

| Question index | Who was right | Why the other grader failed |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |

#### Save before you stop

- [ ] `results/exp3/` holds `runs.json`, `judge.json` and `harness.log`. Day 3 needs these for the eval policy answer.
- [ ] O3.1 to O3.4 are filled, including your verdict on 5 judge disagreements.
- [ ] `notes/exp3.md` has a first draft of your eval policy while the numbers are fresh.

This ends Day 1. Run the stop routine; overnight you pay only for storage. Start Experiment 4's theory tomorrow.

### Phase 3: Result and inference (Day 3, GPU off)

1. Was temperature 0 deterministic? If not, explain why. Batching and floating-point non-associativity are a good place to start.
2. Given the seed-to-seed spread, what is the smallest improvement you would trust on this 300-question eval? How many questions would halve that threshold?
3. When would you trust an LLM judge over exact match, and when never? What would you do to calibrate a judge before relying on it?
4. Write the eval policy you would propose to a team: sample size, number of seeds, paired testing, how judges get validated, and what gets reported alongside a score.

## Experiment 4: Prompting vs LoRA vs QLoRA fine-tuning

**Duration:** Theory 1 hr 25 min (GPU off), experiment 2 hrs (GPU on). **Environment:** `.venv-serve` for evaluation, `.venv-train` for training. Day 2 starts here.

### Aim

To answer the most common applied-AI question with your own numbers: "Should we fine-tune, or just prompt better?" You will compare zero-shot and few-shot prompting with LoRA and QLoRA on a 77-class intent classification task. You will measure accuracy, training cost, memory and inference prompt length.

The task is Banking77: customer banking queries labelled with one of 77 intents. It is hard for prompting because the labels are fine-grained and similar, and easy to score because the answer is an exact label.

### Phase 1: Theory (GPU off, about 1 hr 25 min)

#### Reading

- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685): sections 1–4
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314): sections 1–3 (NF4, double quantization, paged optimizers)
- Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): which layers to target, learning rate, when LoRA matches full fine-tuning
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer) for your pinned version

#### Self-check (answer from memory before moving on)

- [ ] What do LoRA rank and alpha control, and how does the update scale with them?
- [ ] Which layers should LoRA target, and what does LoRA Without Regret say about learning rate?
- [ ] What three tricks let QLoRA fit a large model on one GPU?
- [ ] Why can a fine-tuned model work with a much shorter prompt than a prompted one?

#### Predictions (write before you start the GPU)

| # | Quantity | Zero-shot | Few-shot (77 examples) | LoRA r=16 | QLoRA r=16 |
| --- | --- | --- | --- | --- | --- |
| P4.1 | Test accuracy (%) |  |  |  |  |
| P4.2 | Prompt tokens per request |  |  |  |  |
| P4.3 | Peak training memory (GB) | — | — |  |  |
| P4.4 | Training time for 3,000 examples × 2 epochs (min) | — | — |  |  |

Also predict: how many training examples does LoRA need to beat few-shot prompting?

### Phase 2: Experiment (GPU on, about 2 hrs)

**GPU: A100 80GB.** BF16 LoRA on a 7B model is borderline on the 4090's 24 GB: with these short sequences and gradient checkpointing it may just fit, but there is no headroom to compare LoRA and QLoRA memory fairly. The A100 is the cheapest card that does both, and FP8 is not needed here. Do Part A on your laptop during the theory phase. Then deploy the pod and run the start routine with `.venv-serve`; the procedure tells you when to switch environments.

#### Procedure

**Part A: Check the data (theory phase, on your laptop)**

1. Confirm the dataset loads and has the columns `text` and `label_text`. If `mteb/banking77` has changed, find another Banking77 mirror on Hugging Face and update the scripts.

```python
from datasets import load_dataset
ds = load_dataset("mteb/banking77")
print(ds, ds["train"][0], len(set(ds["train"]["label_text"])))
```

**Part B: Prompting baselines (25 min)**

2. Save as `exp4/eval_cls.py`. One script evaluates every variant on the same 1,000 test examples.

```python
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
```

3. Run both baselines:

```bash
source .venv-serve/bin/activate; mkdir -p results/exp4
python exp4/eval_cls.py zeroshot | tee -a results/exp4/eval.log
python exp4/eval_cls.py fewshot  | tee -a results/exp4/eval.log
```

**Part C: LoRA and QLoRA training (55 min)**

4. Save as `exp4/train_lora.py`. The fine-tuned model uses the short system prompt, with no label list. That is part of what fine-tuning buys you.

```python
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
                max_length=512, save_strategy="no", report_to="none")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)

torch.cuda.reset_peak_memory_stats(); t0 = time.time()
trainer.train()
stats = {"minutes": (time.time() - t0) / 60, "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
         "final_loss": trainer.state.log_history[-2].get("loss"), **vars(a)}
print(stats); json.dump(stats, open(f"{a.out}/train_stats.json", "w"))
trainer.save_model(a.out)
```

5. Train both variants. While the first one runs, read the loss in the log and note how fast it falls.

```bash
source .venv-train/bin/activate
python exp4/train_lora.py --out runs/lora_r16 | tee results/exp4/train_lora.log
python exp4/train_lora.py --qlora --out runs/qlora_r16 | tee results/exp4/train_qlora.log
```

6. Evaluate both adapters. The QLoRA adapter was trained against a 4-bit base but is served on the BF16 base here; that is common practice, and you should note it.

```bash
source .venv-serve/bin/activate
python exp4/eval_cls.py lora runs/lora_r16  | tee -a results/exp4/eval.log
python exp4/eval_cls.py lora runs/qlora_r16 | tee -a results/exp4/eval.log
```

**Part D: Data scaling and overfitting (30 min)**

7. Train and evaluate LoRA with 300 examples and 1,000 examples (2 epochs each). This gives you a data-scaling curve with the 3,000-example run.
8. Deliberately overfit: 200 examples for 15 epochs. Record the training loss and test accuracy. Training loss near zero with worse test accuracy is the signature you want to recognize on sight.

```bash
source .venv-train/bin/activate
python exp4/train_lora.py --n 300  --out runs/lora_n300
python exp4/train_lora.py --n 1000 --out runs/lora_n1000
python exp4/train_lora.py --n 200 --epochs 15 --out runs/lora_overfit
source .venv-serve/bin/activate
for r in lora_n300 lora_n1000 lora_overfit; do python exp4/eval_cls.py lora runs/$r | tee -a results/exp4/eval.log; done
```

#### Observations

**O4.1 Main comparison (1,000 test examples)**

| Variant | Accuracy (%) | Invalid label rate (%) | Prompt tokens per request | Train time (min) | Peak train memory (GB) | Adapter size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot |  |  |  | — | — | — |
| Few-shot (77 examples) |  |  |  | — | — | — |
| LoRA r=16, 3,000 examples |  |  |  |  |  |  |
| QLoRA r=16, 3,000 examples |  |  |  |  |  |  |

**O4.2 Data scaling and overfitting (LoRA r=16)**

| Training examples | Epochs | Final train loss | Test accuracy (%) |
| --- | --- | --- | --- |
| 200 | 15 |  |  |
| 300 | 2 |  |  |
| 1,000 | 2 |  |  |
| 3,000 | 2 |  |  |

**O4.3 Error analysis:** From the best variant's predictions, list the 3 most frequent confusions (gold → predicted). Decide whether each is a model error or a label ambiguity.

| Gold label | Predicted label | Count | Model error or label ambiguity? |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

#### Save before you stop

- [ ] Adapters are in `runs/` on `/workspace`. They are too large for git; optionally push the best one to a private Hugging Face repo with `huggingface-cli upload <you>/banking77-lora runs/lora_r16`.
- [ ] Training stats are copied next to the results: `for r in runs/*/; do cp $r/train_stats.json results/exp4/$(basename $r)_train_stats.json; done`.
- [ ] `results/exp4/` holds every `eval_*.json`, `eval.log` and both training logs.
- [ ] O4.1 to O4.3 are filled, including the confusion analysis.

Then run the stop routine and start Experiment 5's theory.

### Phase 3: Result and inference (Day 3, GPU off)

1. Where did few-shot prompting top out, and how many labelled examples did LoRA need to beat it?
2. Compare inference cost: prompt tokens per request for few-shot vs LoRA, multiplied by a realistic daily request volume. When does fine-tuning pay for itself through shorter prompts alone?
3. What did QLoRA cost in accuracy and speed in exchange for its memory savings? When would you choose it?
4. Write your decision rule: "Fine-tune when \_\_\_; prompt when \_\_\_; neither when \_\_\_." Include the operational costs: data labelling, retraining when labels change, and serving adapters.

## Experiment 5: Train a small GPT from scratch, compute MFU, and profile

**Duration:** Theory 1 hr 10 min (GPU off), experiment 1 hr 45 min (GPU on). **Environment:** `.venv-train`, run from inside the `nanoGPT` folder.

### Aim

To build first-hand intuition for training dynamics and hardware efficiency. You will see learning-rate sensitivity, measure how close you get to the GPU's peak FLOPs (MFU), learn which tricks move that number, and read a profiler trace to find a bottleneck.

### Phase 1: Theory (GPU off, about 1 hr 10 min)

#### Reading

- Karpathy, [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the [nanoGPT repo](https://github.com/karpathy/nanoGPT) README
- Chowdhery et al., [PaLM](https://arxiv.org/abs/2204.02311): appendix B, the definition of MFU
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135): sections 1–3
- Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html): the overhead section, for Part C

MFU is the fraction of the GPU's peak FLOPs spent on useful model math:

```latex
\text{MFU} = \frac{\text{tokens per second} \times 6N}{\text{peak FLOPs per second}}
```

Here N is the parameter count. Add 12 × layers × sequence length × width per token for attention when sequences are long.

#### Self-check (answer from memory before moving on)

- [ ] What is MFU, and why does 40–50% count as good?
- [ ] How does FlashAttention speed up attention without reducing its FLOPs?
- [ ] Horace He describes three regimes: compute, memory bandwidth and overhead. How do you tell which one a workload is in?
- [ ] Why does a learning rate that is too high make loss diverge instead of just training faster?

#### Predictions (write before you start the GPU)

| # | Quantity | Your prediction |
| --- | --- | --- |
| P5.1 | Best learning rate for the small Shakespeare model, out of 1e-4 to 3e-2 |  |
| P5.2 | Learning rate at which training diverges |  |
| P5.3 | MFU for the 124M-parameter configuration, BF16 + compile, batch 16 (%) |  |
| P5.4 | Speedup from BF16 over FP32 (TF32 matmuls) |  |
| P5.5 | Speedup from `torch.compile` |  |
| P5.6 | Slowdown from turning off FlashAttention |  |
| P5.7 | In a batch-2, no-compile run, what share of GPU time is idle? |  |

### Phase 2: Experiment (GPU on, about 1 hr 45 min)

**GPU: A100 80GB.** Restart the stopped Exp 4 pod. MFU is measured against each GPU's own peak, so the lessons carry over to any GPU. Run the start routine with `.venv-train`, then `cd nanoGPT` and create the results folder from there with `mkdir -p ../results/exp5`.

#### Procedure

**Part A: Learning-rate sweep (35 min)**

1. Prepare the character-level Shakespeare data:

```bash
python data/shakespeare_char/prepare.py
```

2. Run five learning rates with the small default model (6 layers, 384 width). Each run should take a few minutes.

```bash
for LR in 1e-4 1e-3 3e-3 1e-2 3e-2; do
  python train.py config/train_shakespeare_char.py --learning_rate=$LR \
    --max_iters=2000 --lr_decay_iters=2000 --eval_interval=250 \
    --out_dir=runs/lr_$LR 2>&1 | tee ../results/exp5/lr_$LR.log
done
```

3. Pull the validation losses out of the logs (`grep "val loss"`). Sample text from the best run with `python sample.py --out_dir=runs/lr_<best>` and paste a few lines into your notes.

**Part B: Throughput and MFU ablations (40 min)**

4. nanoGPT's MFU estimate assumes an A100 (312 TFLOPS), which is the GPU you are on, so no edit is needed. On any other GPU, edit `estimate_mfu` in `model.py` and change `flops_promised = 312e12` to your GPU's dense BF16 peak (about 989e12 for H100 SXM, 756e12 for H100 PCIe).
5. Use a GPT-2-small shape (12 layers, 12 heads, 768 width, 1,024 context) on the same data. The data does not matter here, only the speed.

```bash
BASE="config/train_shakespeare_char.py --n_layer=12 --n_head=12 --n_embd=768 --block_size=1024 \
  --dropout=0.0 --gradient_accumulation_steps=1 --max_iters=60 --eval_interval=10000 --eval_iters=1 \
  --always_save_checkpoint=False --log_interval=10"
python train.py $BASE --batch_size=16 --dtype=bfloat16 --compile=True  --out_dir=runs/b16_bf16_c | tee ../results/exp5/b16_bf16_c.log
python train.py $BASE --batch_size=16 --dtype=bfloat16 --compile=False --out_dir=runs/b16_bf16   | tee ../results/exp5/b16_bf16.log
python train.py $BASE --batch_size=16 --dtype=float32  --compile=True  --out_dir=runs/b16_fp32_c | tee ../results/exp5/b16_fp32_c.log
python train.py $BASE --batch_size=4  --dtype=bfloat16 --compile=True  --out_dir=runs/b4_bf16_c  | tee ../results/exp5/b4_bf16_c.log
python train.py $BASE --batch_size=32 --dtype=bfloat16 --compile=True  --out_dir=runs/b32_bf16_c | tee ../results/exp5/b32_bf16_c.log
```

6. Turn off FlashAttention: in `model.py`, find `self.flash = hasattr(...)` and change it to `self.flash = False`. Rerun the first command as `b16_bf16_c_noflash`. Then revert the change.
7. For each run, ignore the first 20 iterations (compile and warmup). Record the steady-state ms per iteration and MFU. Compute tokens per second as batch size × 1,024 × 1,000 ÷ ms per iteration.

**Part C: Profile a training step (30 min)**

8. Save as `nanoGPT/profile_step.py`:

```python
import sys, torch
from torch.profiler import profile, schedule, ProfilerActivity
from model import GPT, GPTConfig

batch, do_compile = int(sys.argv[1]), sys.argv[2] == "compile"
cfg = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50304, dropout=0.0)
model = GPT(cfg).cuda()
opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda")
if do_compile: model = torch.compile(model)
x = torch.randint(0, 50304, (batch, 1024), device="cuda")
y = torch.randint(0, 50304, (batch, 1024), device="cuda")

def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

for _ in range(5): step()          # warmup and compile
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=2, active=3)) as prof:
    for _ in range(6): step(); prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace(f"../results/exp5/trace_b{batch}_{sys.argv[2]}.json")
```

If `cuda_time_total` errors in your torch version, use `device_time_total`.

9. Profile a healthy configuration and an unhealthy one:

```bash
python profile_step.py 16 compile   | tee ../results/exp5/prof_b16_compile.txt
python profile_step.py 2  nocompile | tee ../results/exp5/prof_b2_nocompile.txt
```

10. Copy both trace files to your laptop and open them in [Perfetto](https://ui.perfetto.dev). Compare the GPU stream rows. Gaps between kernels mean the GPU is waiting on the CPU to launch work.

#### Observations

**O5.1 Learning-rate sweep (Shakespeare, 2,000 iterations)**

| Learning rate | Best val loss | Iteration of best val loss | Diverged? | Notes |
| --- | --- | --- | --- | --- |
| 1e-4 |  |  |  |  |
| 1e-3 |  |  |  |  |
| 3e-3 |  |  |  |  |
| 1e-2 |  |  |  |  |
| 3e-2 |  |  |  |  |

**O5.2 Throughput ablations (124M-shape, 1,024 context)**

| Run | Batch | Dtype | Compile | Flash | ms/iter | Tokens/s | MFU (%) | Peak memory (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b16\_bf16\_c | 16 | bf16 | yes | yes |  |  |  |  |
| b16\_bf16 | 16 | bf16 | no | yes |  |  |  |  |
| b16\_fp32\_c | 16 | fp32 (TF32) | yes | yes |  |  |  |  |
| b4\_bf16\_c | 4 | bf16 | yes | yes |  |  |  |  |
| b32\_bf16\_c | 32 | bf16 | yes | yes |  |  |  |  |
| b16\_bf16\_c\_noflash | 16 | bf16 | yes | no |  |  |  |  |

**O5.3 Profiler**

| Measure | Batch 16, compiled | Batch 2, not compiled |
| --- | --- | --- |
| Top kernel by GPU time |  |  |
| Share of GPU time in matmuls (%) |  |  |
| Visible idle gaps on GPU stream? |  |  |
| Number of distinct kernels per step (rough) |  |  |

#### Save before you stop

- [ ] Copy both `trace_*.json` files to your laptop now (for example with `scp`). You need them in Perfetto, and the volume is unreachable while the pod is stopped.
- [ ] `results/exp5/` holds every `lr_*`, `b*` and `prof_*` log.
- [ ] In `model.py`, the FlashAttention change is reverted. If you changed the MFU peak FLOPs, keep that change and commit it.
- [ ] Paste a few lines of generated Shakespeare from the best run into `notes/exp5.md`.
- [ ] O5.1 to O5.3 are filled.

Then run the stop routine and start Experiment 6's theory.

### Phase 3: Result and inference (Day 3, GPU off)

1. Describe the learning-rate curve: too low, best and diverging. How would you choose a learning rate for a 10× larger model without sweeping it at full scale?
2. Which single change moved MFU the most? Explain why in terms of compute, memory traffic and launch overhead.
3. Why did batch size change MFU? Connect it to arithmetic intensity and the ridge point from Experiment 1's Numbers to know.
4. Using your best tokens per second, estimate how long and how much it would cost to train this 124M model on 10B tokens on one GPU, and on 8 GPUs at 90% scaling efficiency.

## Experiment 6: RL post-training with GRPO

**Duration:** Theory 55 min (GPU off), experiment 2 hrs (GPU on). **Environment:** `.venv-train`, with vLLM installed in it for fast generation. If that clashes, set `use_vllm=False`; it will be slower, so halve the step counts.

### Aim

To run reinforcement learning with verifiable rewards on a small model and watch its dynamics directly: reward rising, completion length changing, format learning versus real capability gains, and reward hacking when the reward function has a bug.

### Phase 1: Theory (GPU off, about 55 min)

#### Reading

- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300): section 4, GRPO
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948): sections 1–2, RL with rule-based rewards
- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer) for your pinned version
- Optional: Nathan Lambert, [RLHF Book](https://rlhfbook.com/), the chapters on policy gradients and over-optimization

Key idea: GRPO samples a group of completions per prompt and scores each one. Each completion's advantage is its reward minus the group mean, divided by the group standard deviation. No value model is needed. If every completion in a group gets the same reward, that prompt teaches nothing.

#### Self-check (answer from memory before moving on)

- [ ] How does GRPO compute an advantage without a value model?
- [ ] Why does a prompt where every completion gets the same reward teach nothing?
- [ ] What is reward hacking? Write down one hack you expect from the lenient reward.
- [ ] Why do RL-trained reasoning models often produce longer outputs over training?

#### Predictions (write before you start the GPU)

| # | Quantity | Your prediction |
| --- | --- | --- |
| P6.1 | Base Qwen2.5-1.5B-Instruct GSM8K accuracy with this prompt (%) |  |
| P6.2 | Accuracy after 150 GRPO steps with the strict reward (%) |  |
| P6.3 | How much of the gain comes from format compliance rather than better reasoning? |  |
| P6.4 | Mean completion length: rises, falls or stays flat? |  |
| P6.5 | What the model learns to exploit under the buggy lenient reward |  |

### Phase 2: Experiment (GPU on, about 2 hrs)

**GPU: A100 80GB.** Restart the stopped Exp 5 pod and run the start routine with `.venv-train`. 80 GB fits a 1.5B policy plus colocated vLLM comfortably. Generation is slower than on an H100: if steps take more than 25 seconds, cut the strict run to 100 steps and the lenient run to 60. This is the last GPU session.

#### Procedure

**Part A: Baseline accuracy (10 min)**

1. Save as `exp6/eval_gsm8k.py`. It scores a model or checkpoint on 500 GSM8K test questions and records format compliance separately from correctness.

```python
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
```

2. Run it on the base model: `python exp6/eval_gsm8k.py Qwen/Qwen2.5-1.5B-Instruct`.

**Part B: GRPO with a strict reward (55 min)**

3. Save as `exp6/grpo.py`. The strict reward gives 1.0 only when the final `Answer:` line matches. The lenient reward contains a deliberate bug: it gives credit if the right number appears anywhere in the text.

```python
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
                 bf16=True, logging_steps=5, save_strategy="no", report_to="none",
                 use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3)
reward = strict_reward if a.reward == "strict" else lenient_reward
trainer = GRPOTrainer(model="Qwen/Qwen2.5-1.5B-Instruct", reward_funcs=[reward], args=cfg, train_dataset=ds)
trainer.train()
trainer.save_model(a.out)
json.dump(trainer.state.log_history, open(f"{a.out}/log_history.json", "w"))
```

4. Run the strict version, then evaluate the result:

```bash
mkdir -p results/exp6
python exp6/grpo.py --reward strict --out runs/grpo_strict | tee results/exp6/grpo_strict.log
python exp6/eval_gsm8k.py runs/grpo_strict
```

5. While it trains, watch the logged reward, its standard deviation and the completion length every 5 steps. Fill O6.2 from `log_history.json` afterwards. Metric names vary between TRL versions; look for keys containing `reward` and `length`.

**Part C: Reward hacking with the lenient reward (40 min)**

6. Run the buggy reward for 100 steps, then evaluate with the strict evaluator:

```bash
python exp6/grpo.py --reward lenient --steps 100 --out runs/grpo_lenient | tee results/exp6/grpo_lenient.log
python exp6/eval_gsm8k.py runs/grpo_lenient
```

7. Read 10 completions from late in the lenient run. Look for answers that list several numbers, hedge between candidates or drop the `Answer:` line. Training reward rising while strict accuracy stays flat or falls is the signature of reward hacking. It may be subtle in 100 steps; record exactly what you see.

#### Observations

**O6.1 Accuracy before and after (500 GSM8K test questions, strict scoring)**

| Model | Accuracy (%) | Format compliance (%) | Mean completion length (tokens) |
| --- | --- | --- | --- |
| Base |  |  |  |
| GRPO strict, 150 steps |  |  |  |
| GRPO lenient, 100 steps |  |  |  |

**O6.2 Training dynamics (strict run)**

| Step | Mean reward | Reward std | Mean completion length | Seconds per step |
| --- | --- | --- | --- | --- |
| 5 |  |  |  |  |
| 25 |  |  |  |  |
| 50 |  |  |  |  |
| 100 |  |  |  |  |
| 150 |  |  |  |  |

**O6.3 Reward hacking evidence (lenient run)**

| Step | Training reward | Example behaviour seen in completions |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |

#### Save before you stop

- [ ] `results/exp6/` holds both training logs and every `eval_*.json`. Copy `runs/grpo_*/log_history.json` into it too.
- [ ] Ten late completions from the lenient run are pasted into `notes/exp6.md`, with the hack (if any) described.
- [ ] O6.1 to O6.3 are filled.
- [ ] Everything you want to keep is pushed to git or copied to your laptop. Checkpoints can stay on the volume until Day 3 is done.

Run the stop routine. Day 2 is complete.

### Phase 3: Result and inference (Day 3, GPU off)

1. Split the strict run's gain into format compliance and real accuracy. What does that tell you about headline RL improvements in papers and vendor claims?
2. What happened to completion length, and why does GRPO tend to push it in that direction?
3. Describe the reward hack, or explain why it did not appear in 100 steps. How would you design a reward and a monitoring setup to catch hacks in production RL?
4. Seconds per step split into generation and training. Which dominated, and what does that imply for scaling RL infrastructure?

## Day 3: Synthesis

Day 3 turns tables into judgment. By the end you should have three artifacts: a prediction scorecard, one decision memo, and a story bank you can use in interviews and design reviews.

### Step 1: Prediction scorecard (60 min)

Copy every prediction from the theory phases next to its measured value. The "why I was wrong" column is the most important thing you produce all week.

| ID | Predicted | Measured | Off by (×) | Why I was wrong (or right) |
| --- | --- | --- | --- | --- |
| P1.1 |  |  |  |  |
| P1.4 |  |  |  |  |
| P1.5 |  |  |  |  |
| P2.3 |  |  |  |  |
| P2.5 |  |  |  |  |
| P3.2 |  |  |  |  |
| P4.1 |  |  |  |  |
| P5.3 |  |  |  |  |
| P6.3 |  |  |  |  |

Add rows for every other prediction you made. Then write three sentences: which of your mental models held up, which broke, and what you will now believe differently.

### Step 2: Write the Result and inference sections (2–3 hrs)

Answer the questions at the end of each experiment, in full sentences, citing your observation tables. Keep each answer under 150 words. If an answer needs a chart, use the CSV from `results/` and make one chart per experiment, not more.

### Step 3: Decision memo (90 min)

Write a two-page memo as if your VP asked: "We want to launch an LLM feature that classifies and answers customer banking queries at 2 million requests a day. Should we use an API, self-host, fine-tune, or quantize?" Use only numbers you measured, plus clearly labelled assumptions.

Template:

1. **Recommendation** in one sentence, with the expected cost per month and the quality bar it meets.
2. **Options considered:** API, self-hosted BF16, self-hosted FP8 or INT4, prompting vs LoRA. One row each with cost, latency, quality and operational burden.
3. **Evidence:** the 3–4 measurements that decide it (from Exps 1, 2 and 4).
4. **How we will know it works:** eval design, sample size and confidence intervals (from Exp 3).
5. **Risks and what would change the decision:** traffic shape, label drift, model upgrades, GPU price changes.
6. **What I would test next** with one more day of GPU time.

### Step 4: Story bank (60 min)

For each experiment, write one story in this shape. Staff-level interviews reward a crisp story built around a surprising number.

| Experiment | Situation and question | What I predicted | What I measured | What it changed in my thinking | How I would apply it at work |
| --- | --- | --- | --- | --- | --- |
| Exp 1 |  |  |  |  |  |
| Exp 2 |  |  |  |  |  |
| Exp 3 |  |  |  |  |  |
| Exp 4 |  |  |  |  |  |
| Exp 5 |  |  |  |  |  |
| Exp 6 |  |  |  |  |  |

### Step 5: Publish (30 min)

- [ ] Push the repo with scripts, CSVs, charts and the filled workbook.
- [ ] Turn the decision memo into a blog post or LinkedIn write-up, with your numbers and your GPU price.
- [ ] Pick the next experiment to go deeper on, and plan a second GPU day if it is worth it.

**Final cleanup:** once everything is off the volume, terminate the pod and delete the network volume. Storage keeps billing until you do.

## Appendix

### A. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No such file or directory` when saving results | Results folder missing | Run `mkdir -p results/exp{1..6}` from the repo root at setup |
| CUDA out of memory when starting a training job | A vLLM server or old process still holds GPU memory | `nvidia-smi`, then kill the process; stop the server before Exps 4–6 |
| vLLM fails to start: "not enough memory for KV cache" | `--max-model-len` too high for remaining memory | Lower `--max-model-len` or raise `--gpu-memory-utilization` to 0.92 |
| An unknown-argument error in `vllm bench serve` or TRL | Flag renamed between versions | Run the command with `--help`, or check the docs for the version in `env_versions.txt` |
| FP8 errors on startup | GPU lacks FP8 support (A100 and older; the RTX 4090 and H100 have it) | Skip FP8 and record it as a hardware limitation |
| Hugging Face download is very slow | Rate limits or no token | Set `HF_TOKEN`; try `pip install hf_transfer` and `export HF_HUB_ENABLE_HF_TRANSFER=1` |
| GRPO reward stuck at 0 | Answers never match the format, so every group has the same reward | Read 5 completions; check the regex; raise `max_completion_length` if answers are cut off |
| GRPO reward stuck at 1 | Task too easy, so no learning signal | Use harder questions or a smaller model |
| `torch.compile` takes many minutes | First compile of a new shape | Normal; exclude warmup iterations from timings |
| SSH dropped and job died | Not running inside tmux | Always run jobs inside `tmux`; reattach with `tmux attach -t lab` |

### B. Cost log

| Event | Time | Running cost (USD) | Note |
| --- | --- | --- | --- |
| Pod launched |  |  |  |
| Lunch stop |  |  |  |
| Lunch restart |  |  |  |
| Pod terminated |  |  |  |
| Total GPU hours and cost |  |  |  |

### D. Measured resources

Fill one row at the end of every GPU session. Peak GPU memory comes from the session's monitor log; disk comes from `df -h /workspace`. The rehearsal row uses a 0.5B model, so its GPU memory says nothing about the real runs, but its disk figure covers the environments and all model weights.

| Session | GPU | Peak GPU memory (GiB) | Volume used (GB) | Largest items on disk | Notes |
| --- | --- | --- | --- | --- | --- |
| Rehearsal | RTX 4090 |  |  |  |  |
| Exp 1 | H100 |  |  |  |  |
| Exp 2 | H100 |  |  |  |  |
| Exp 3 | RTX 4090 |  |  |  |  |
| Exp 4 | A100 |  |  |  |  |
| Exp 5 | A100 |  |  |  |  |
| Exp 6 | A100 |  |  |  |  |

### C. Viva questions

Practise answering these aloud in under 2 minutes each, using your own numbers. These are the questions a staff-level interviewer or design reviewer is likely to ask.

1. A product team says their LLM endpoint is "slow". What do you measure first, and how do you tell whether TTFT or TPOT is the problem?
2. Why does doubling concurrency nearly double throughput at first and then stop? What saturates?
3. Your team wants to serve 32k-context requests. How does that change GPU count and cost? Show the KV cache arithmetic.
4. Someone proposes INT4 to cut cost. What do you ask for before approving it?
5. A new prompt improved the eval by 1.5 points. Do you ship it? What would make you confident?
6. When is an LLM judge acceptable as the main metric? How do you validate one?
7. Fine-tuning or prompting for a new classification feature: walk through your decision and its ongoing costs.
8. What does MFU tell you that GPU utilization in `nvidia-smi` does not?
9. A training run diverged at step 3,000. What are your first five checks?
10. An RL run's reward keeps rising, but user satisfaction is flat. What is happening, and how would you have caught it earlier?
11. Estimate the cost to serve a 70B model at 1,000 requests per minute. Which numbers from this lab would you scale, and how?
12. What is the single most surprising thing you measured in this lab, and how did it change a belief you held?
