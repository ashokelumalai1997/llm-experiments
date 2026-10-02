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
