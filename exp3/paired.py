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
