"""Measure the speed and GPU energy of LLM inference.

Follows the method of Samsi et al. (2023), "From Words to Watts": sample GPU power
with nvidia-smi every 100 ms while a model answers a fixed set of prompts, then
report words, tokens and responses per second and energy per token and per response.

One run = one model at one precision answering the prompts at one batch size.
Each run appends a line to results.jsonl. Rerunning skips runs already there.

Usage: python bench.py            run the whole grid
       python bench.py --smoke    check the grader, then do one tiny run
"""
import gc
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUT = Path(__file__).parent / "results.jsonl"
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
N_PROMPTS, MAX_NEW_TOKENS = 128, 256
# (model, precision, dataset, batch size)
GRID = ([(m, "fp16", d, b) for m in MODELS for d in ("gsm8k", "alpaca") for b in (1, 4, 16)]
        + [(m, p, "gsm8k", 4) for m in MODELS for p in ("int8", "int4")])


class PowerMeter:
    """Records GPU power draw (watts) and temperature every 100 ms while the block runs."""

    def __enter__(self):
        self.samples = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        time.sleep(0.5)  # let the first samples arrive
        self.start = time.perf_counter()
        return self

    def _read(self):
        for line in self.proc.stdout:
            watts, temp = line.split(",")
            self.samples.append((time.perf_counter(), float(watts), float(temp)))

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        self.proc.terminate()
        inside = [(w, t) for at, w, t in self.samples if at >= self.start]
        self.n = len(inside)
        self.watts = sum(w for w, _ in inside) / self.n
        self.max_temp = max(t for _, t in inside)
        # ponytail: mean power times duration. Fine for runs of a minute or more;
        # integrate sample by sample if runs get short enough for the edges to matter.
        self.joules = self.watts * self.seconds


def prompts(dataset, n):
    """Returns (prompts, gold answers). Gold is None for Alpaca, which has nothing to grade."""
    if dataset == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(n))
        return ([r["question"] + '\nShow your working, then end with "Answer: <number>".' for r in rows],
                [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows])
    rows = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0).select(range(n))
    return [r["instruction"] + ("\n" + r["input"] if r["input"] else "") for r in rows], None


def correct(text, gold):
    """The first number after the last "Answer:", or failing that the last number in the reply."""
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.split("Answer:")[-1])
    if not numbers:
        return False
    picked = numbers[0 if "Answer:" in text else -1].replace(",", "")
    return abs(float(picked) - float(gold)) < 1e-6


def run(model_id, precision, dataset, batch, n=N_PROMPTS):
    rec = dict(model=model_id.split("/")[-1], precision=precision, dataset=dataset, batch=batch, n=n,
               max_new_tokens=MAX_NEW_TOKENS, gpu=torch.cuda.get_device_name(0))
    texts, gold = prompts(dataset, n)
    with PowerMeter() as idle:
        time.sleep(3)
    quant = {"int8": BitsAndBytesConfig(load_in_8bit=True),
             "int4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)}.get(precision)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                 quantization_config=quant, device_map={"": 0}).eval()
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
             for t in texts]
    # The paper's decoding settings (temperature 0.8, top-p 0.95), with the model's own extras switched off.
    sampling = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.0)
    replies, tokens = [], 0
    try:
        with torch.no_grad():
            model.generate(**tok(chats[:1], return_tensors="pt").to(0), max_new_tokens=8, **sampling)  # warm-up
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            with PowerMeter() as meter:
                for i in range(0, n, batch):
                    enc = tok(chats[i:i + batch], return_tensors="pt", padding=True).to(0)
                    new = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, **sampling)[:, enc.input_ids.shape[1]:]
                    tokens += int((new != tok.pad_token_id).sum())
                    replies += tok.batch_decode(new, skip_special_tokens=True)
                torch.cuda.synchronize()
        words = sum(len(r.split()) for r in replies)
        rec.update(seconds=meter.seconds, joules=meter.joules, watts=meter.watts, idle_watts=idle.watts,
                   power_samples=meter.n, max_temp_c=meter.max_temp, tokens=tokens, words=words,
                   tokens_per_s=tokens / meter.seconds, words_per_s=words / meter.seconds,
                   responses_per_s=n / meter.seconds, j_per_token=meter.joules / tokens,
                   j_per_response=meter.joules / n, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                   accuracy=sum(map(correct, replies, gold)) / n if gold else None)
    except torch.OutOfMemoryError:
        rec["error"] = "out of memory"  # recorded so the grid shows which settings do not fit
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    # On Windows a full GPU spills into system RAM and slows to a crawl instead of failing.
    # Capping the process below the card's free memory turns that into a clean out-of-memory.
    torch.cuda.set_per_process_memory_fraction(0.85)
    if "--smoke" in sys.argv:
        assert correct("3 + 4 = 7\nAnswer: 7", "7") and correct("Answer: $1,250.", "1250")
        assert correct("so she has 18 left", "18") and not correct("Answer: 7", "8") and not correct("no idea", "3")
        print(json.dumps(run(MODELS[0], "fp16", "gsm8k", 4, n=8), indent=1))
        sys.exit()
    done = set()
    if OUT.exists():
        done = {(r["model"], r["precision"], r["dataset"], r["batch"]) for r in map(json.loads, OUT.open())}
    for cfg in GRID:
        if (cfg[0].split("/")[-1], *cfg[1:]) in done:
            continue
        rec = run(*cfg)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()
               if k in ("model", "precision", "dataset", "batch", "tokens_per_s", "watts", "j_per_token", "accuracy", "error")})
