"""The fp16 grid again, served by vLLM with continuous batching instead of static batches.

vLLM keeps up to `seqs` replies in flight and starts the next prompt the moment one
finishes, so a short reply never waits for a long one. If the Alpaca penalty in
bench.py comes from padding, it should mostly disappear here.

Linux only (run it in WSL). Same prompts, grader, sampling and power meter as bench.py.

Usage: python vbench.py                       run the grid, one process per run
       python vbench.py MODEL DATASET SEQS    one run (what the grid calls)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from bench import MAX_NEW_TOKENS, MODELS, N_PROMPTS, PowerMeter, correct, prompts

OUT = Path(__file__).parent / "results-vllm.jsonl"
GRID = [(m, d, s) for m in MODELS for d in ("gsm8k", "alpaca") for s in (4, 16)]


def run(model_id, dataset, seqs, n=N_PROMPTS):
    from vllm import LLM, SamplingParams

    rec = dict(model=model_id.split("/")[-1], precision="fp16", dataset=dataset, batch=seqs, n=n,
               max_new_tokens=MAX_NEW_TOKENS, engine="vllm")
    texts, gold = prompts(dataset, n)
    with PowerMeter() as idle:
        time.sleep(3)
    # ponytail: 0.93 of the card, a 1,024-token context and 1,024-token prefill steps so 3B fits
    # in 8 GB with room for its cache. Bigger cards can drop all three.
    llm = LLM(model_id, dtype="float16", max_num_seqs=seqs, gpu_memory_utilization=0.93, max_model_len=1024,
              max_num_batched_tokens=1024, seed=0,
              enforce_eager=True)  # no torch.compile: WSL has no C compiler, and HF ran uncompiled too
    tok = llm.get_tokenizer()
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
             for t in texts]
    sampling = SamplingParams(temperature=0.8, top_p=0.95, top_k=-1, repetition_penalty=1.0,
                              max_tokens=MAX_NEW_TOKENS, seed=0)
    llm.generate(chats[:1], SamplingParams(max_tokens=8), use_tqdm=False)  # warm-up
    with PowerMeter() as meter:
        outs = llm.generate(chats, sampling, use_tqdm=False)
    replies = [o.outputs[0].text for o in outs]
    lengths = [len(o.outputs[0].token_ids) for o in outs]
    tokens, words = sum(lengths), sum(len(r.split()) for r in replies)
    rec.update(gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip(),
               seconds=meter.seconds, joules=meter.joules, watts=meter.watts, idle_watts=idle.watts,
               power_samples=meter.n, max_temp_c=meter.max_temp, tokens=tokens, words=words,
               tokens_per_s=tokens / meter.seconds, words_per_s=words / meter.seconds,
               responses_per_s=n / meter.seconds, j_per_token=meter.joules / tokens,
               j_per_response=meter.joules / n, reply_tokens=lengths,
               accuracy=sum(map(correct, replies, gold)) / n if gold else None)
    return rec


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(json.dumps(run(sys.argv[1], sys.argv[2], int(sys.argv[3]))))
        sys.exit()
    done = set()
    if OUT.exists():
        done = {(r["model"], r["dataset"], r["batch"]) for r in map(json.loads, OUT.open())}
    for m, d, s in GRID:
        if (m.split("/")[-1], d, s) in done:
            continue
        # A fresh process per run: vLLM does not hand GPU memory back cleanly within one.
        child = subprocess.run([sys.executable, __file__, m, d, str(s)], capture_output=True, text=True)
        lines = child.stdout.strip().splitlines()
        rec = json.loads(lines[-1]) if child.returncode == 0 and lines else dict(
            model=m.split("/")[-1], precision="fp16", dataset=d, batch=s, engine="vllm",
            error=child.stderr.strip().splitlines()[-1] if child.stderr.strip() else f"exit {child.returncode}")
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()
               if k in ("model", "dataset", "batch", "tokens_per_s", "watts", "j_per_token", "accuracy", "error")},
              flush=True)
