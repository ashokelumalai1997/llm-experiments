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
