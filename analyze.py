"""Turn results.jsonl into tables and charts. Runs on the saved results, no GPU needed.

Usage: python analyze.py
Prints the tables and writes energy.png next to the results.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = [r for r in map(json.loads, (HERE / "results.jsonl").open()) if "error" not in r]
VLLM = [r for r in map(json.loads, (HERE / "results-vllm.jsonl").open()) if "error" not in r] \
    if (HERE / "results-vllm.jsonl").exists() else []
MODELS = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, one per model


def short(model):
    return model.split("-")[1]


def j_per_correct(r):
    return r["joules"] / (r["accuracy"] * r["n"]) if r.get("accuracy") else None


def interval(r):
    """90% normal interval on accuracy, in points. Tells you which accuracy gaps are noise."""
    p = r["accuracy"]
    return 100 * 1.645 * (p * (1 - p) / r["n"]) ** 0.5


def table(title, rows, cols):
    print(f"\n{title}")
    print("".join(f"{c:>{w}}" for c, w, _ in cols))
    for r in rows:
        print("".join(f"{f(r):>{w}}" for _, w, f in cols))


def num(key, fmt):
    return lambda r: format(r[key], fmt) if r.get(key) is not None else "-"


def main():
    find = lambda **k: [r for r in RUNS if all(r[a] == b for a, b in k.items())]
    base = [("model", 7, lambda r: short(r["model"])), ("batch", 7, num("batch", "d"))]
    speed = [("tok/s", 9, num("tokens_per_s", ".1f")), ("watts", 8, num("watts", ".1f")),
             ("J/token", 10, num("j_per_token", ".3f")), ("tok/reply", 11, lambda r: f"{r['tokens'] / r['n']:.0f}"),
             ("max C", 7, num("max_temp_c", ".0f")), ("VRAM GB", 9, num("peak_vram_gb", ".1f"))]
    graded = [("accuracy", 10, lambda r: f"{r['accuracy']:.0%}"), ("+/-", 6, lambda r: f"{interval(r):.0f}"),
              ("J/correct", 11, lambda r: f"{j_per_correct(r):.0f}")]

    gpu = RUNS[0]["gpu"]
    print(f"{len(RUNS)} runs on {gpu}, {RUNS[0]['n']} prompts each, up to {RUNS[0]['max_new_tokens']} new tokens")
    table("GSM8K, fp16", find(dataset="gsm8k", precision="fp16"), base + speed + graded)
    table("Alpaca, fp16", find(dataset="alpaca", precision="fp16"), base + speed)
    table("GSM8K, batch 4, by precision", [r for r in RUNS if r["dataset"] == "gsm8k" and r["batch"] == 4],
          base[:1] + [("prec", 6, lambda r: r["precision"])] + speed + graded)

    print("\nAlpaca vs GSM8K energy per token, fp16")
    for m in MODELS:
        pairs = [(a, g) for a in find(model=m, dataset="alpaca", precision="fp16")
                 for g in find(model=m, dataset="gsm8k", precision="fp16", batch=a["batch"])]
        print(f"  {short(m):>5}  " + "   ".join(f"batch {a['batch']}: {a['j_per_token'] / g['j_per_token'] - 1:+.0%}"
                                             for a, g in pairs))
    if VLLM:
        vllm_vs_static(find)
    plot(find)


def vllm_vs_static(find):
    """Same prompts at the same concurrency, static batches (HF) against continuous batching (vLLM)."""
    print("\nStatic (HF generate) vs continuous (vLLM) batching, fp16. batch = replies in flight")
    print(f"{'model':>7}{'data':>8}{'batch':>7}{'HF J/tok':>10}{'vLLM J/tok':>12}{'saving':>8}"
          f"{'HF acc':>8}{'vLLM acc':>10}{'reply sd':>10}{'at cap':>8}")
    for v in VLLM:
        h = find(model=v["model"], dataset=v["dataset"], precision="fp16", batch=v["batch"])[0]
        lengths = v["reply_tokens"]
        mean = sum(lengths) / len(lengths)
        sd = (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
        acc = lambda r: f"{r['accuracy']:.0%}" if r.get("accuracy") is not None else "-"
        print(f"{short(v['model']):>7}{v['dataset']:>8}{v['batch']:>7}{h['j_per_token']:>10.3f}{v['j_per_token']:>12.3f}"
              f"{1 - v['j_per_token'] / h['j_per_token']:>8.0%}{acc(h):>8}{acc(v):>10}{sd:>10.0f}"
              f"{sum(x >= v['max_new_tokens'] for x in lengths) / len(lengths):>8.0%}")
    print("\nAlpaca vs GSM8K energy per token under vLLM")
    for m in MODELS:
        for b in (4, 16):
            a = [r for r in VLLM if r["model"] == m and r["dataset"] == "alpaca" and r["batch"] == b]
            g = [r for r in VLLM if r["model"] == m and r["dataset"] == "gsm8k" and r["batch"] == b]
            if a and g:
                print(f"  {short(m):>5}  batch {b}: {a[0]['j_per_token'] / g[0]['j_per_token'] - 1:+.0%}")


def plot(find):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=surface, sharey=True,
                                      gridspec_kw=dict(width_ratios=[1.15, 1]))

    # Left: batching. One line per model, labelled at batch 1, where the lines are furthest apart.
    for m, color in zip(MODELS, COLORS):
        pts = sorted(find(model=m, dataset="gsm8k", precision="fp16"), key=lambda r: r["batch"])
        x, y = [r["batch"] for r in pts], [j_per_correct(r) for r in pts]
        left.plot(x, y, color=color, lw=2, marker="o", ms=8, mec=surface, mew=2, label=short(m))
        left.annotate(short(m), (x[0], y[0]), xytext=(-9, 0), textcoords="offset points",
                      ha="right", va="center", fontsize=8.5, color=ink)
    left.set_xscale("log", base=2)
    left.set_xticks([1, 4, 16], ["1", "4", "16"])
    left.set_xlim(0.6, 20)
    left.set_xlabel("Batch size (fp16)", color=muted, fontsize=9)
    left.set_ylabel("GPU energy per correct GSM8K answer (J)", color=muted, fontsize=9)
    left.set_title("Batching", loc="left", color=ink, fontsize=10.5)

    # Right: precision at batch 4. Grouped bars, 2px surface gap between neighbours.
    precs, width = ["fp16", "int8", "int4"], 0.26
    for i, (m, color) in enumerate(zip(MODELS, COLORS)):
        for j, p in enumerate(precs):
            r = find(model=m, dataset="gsm8k", precision=p, batch=4)
            if r:
                right.bar(j + (i - 1) * width, j_per_correct(r[0]), width, color=color,
                          edgecolor=surface, linewidth=2, label=short(m) if j == 0 else None)
    right.set_xticks(range(len(precs)), precs)
    right.set_xlabel("Weight precision (batch 4)", color=muted, fontsize=9)
    right.set_title("Quantisation", loc="left", color=ink, fontsize=10.5)
    right.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor="#52514e")

    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=8.5, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    left.set_ylim(bottom=0)
    fig.suptitle(f"Energy per correct answer, Qwen2.5 on an {RUNS[0]['gpu'].replace('NVIDIA GeForce ', '')}",
                 x=0.01, ha="left", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "energy.png", dpi=160, facecolor=surface)


if __name__ == "__main__":
    main()
