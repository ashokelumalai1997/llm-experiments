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
