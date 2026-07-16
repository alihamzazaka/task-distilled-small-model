"""
Generate the four paper figures for Project 02 (Task-Distilled Small Model).
Every number is a REAL measured value pulled from the repo's reports/*.json.
Run:  python make_figures.py   (writes PNGs into figures/, dpi=150)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- house style
BLUE_D = "#0b5394"   # dark
BLUE_N = "#274257"   # navy
BLUE_M = "#5a8bbd"   # mid
BLUE_L = "#9db4c9"   # light
GREY   = "#c9c9c9"
GOOD   = "#2e7d5b"
BAD    = "#b3401f"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figures")
os.makedirs(FIGD, exist_ok=True)


def _hide_spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def save(fig, name):
    p = os.path.join(FIGD, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# =================================================================== FIGURE 1
# Pipeline / architecture: teacher -> filter -> student -> eval loop + gold lane
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.5,rounding_size=2.0",
                     linewidth=0, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, weight="bold", zorder=5)

    def arrow(x1, y1, x2, y2, color=BLUE_N, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle=style, mutation_scale=11, lw=1.5,
                     color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))

    W, H, Y = 19, 13, 33
    xs = [1, 25.5, 50, 74.5, 99]
    box(xs[0], Y, W, H, "Teacher\nqwen3:14b\n(800 seeds)", BLUE_N, fs=7.4)
    box(xs[1], Y, W, H, "Quality filter\nschema, 2-pass\nconsist., dedup", BLUE_D, fs=7.4)
    box(xs[2], Y, W, H, "Distil data\ntrain 568/dev 78\ntest 53", BLUE_M, fs=7.4)
    box(xs[3], Y, W, H, "Student SFT\nQwen2.5-0.5B\nfull FT / 5080", BLUE_D, fs=7.4)
    box(xs[4], Y, W, H, "Eval harness\nF1, agreement\ncost, latency", BLUE_N, fs=7.4)

    yc = Y + H / 2
    for i in range(4):
        arrow(xs[i] + W - 0.5, yc, xs[i + 1] + 0.5, yc)

    ecx = xs[4] + W / 2   # eval-box centre x
    # gold lane (firewall) into eval
    box(58, 49, 57, 7, "Silver gold set (37 items, cross-model) -> eval only, never trained on",
        GREY, tc="#333", fs=7.4)
    arrow(ecx, 49, ecx, Y + H, color="#777")

    # quality gate + loop back
    box(xs[4], 12, W, 13, "Quality gate\nF1 >= 0.95 x\nteacher?", GOOD, fs=7.6)
    arrow(ecx, Y, ecx, 25)
    ax.text(ecx + 2.6, 29, "meets bar\n(0.965)", fontsize=6.8, color=GOOD, weight="bold", va="center")
    # loop back on fail
    arrow(xs[4], 18.5, 10, 18.5, color=BAD, rad=-0.10)
    ax.text(54, 14.0, "below bar -> regenerate / more data", fontsize=7.0,
            color=BAD, ha="center", style="italic")
    arrow(10, 18.5, 10, Y, color=BAD)

    # package
    box(1, 4, 40, 8, "Package: merged 0.5B + Modelfile\n+ serve/infer.py (schema-enforced)",
        BLUE_L, tc="#1a2b3a", fs=7.4)
    arrow(ecx - 6, 12, 33, 8.5, color=GOOD, rad=0.10)

    ax.text(58, 57, "Teacher -> Filter -> Student -> Measured Eval (gated loop)",
            ha="center", fontsize=10.5, weight="bold", color=BLUE_N)
    save(fig, "fig1_pipeline.png")


# =================================================================== FIGURE 2
# Headline: student field-F1 0.9647 = 96.5% of teacher vs 95% bar
def fig2():
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    labels = ["Teacher\nqwen3:14b\n(reference)", "95% quality\nbar", "Student\nQwen2.5-0.5B"]
    vals = [1.0000, 0.95, 0.9647]           # real: eval_report.json
    colors = [BLUE_N, GREY, BLUE_D]
    x = range(len(vals))
    bars = ax.bar(x, vals, width=0.6, color=colors, zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006,
                f"{v:.4f}".rstrip("0").rstrip(".") if i != 1 else "0.95",
                ha="center", va="bottom", fontsize=9, weight="bold",
                color=colors[i] if i != 1 else "#555")
    # bar line across
    ax.axhline(0.95, color=GREY, ls="--", lw=1.1, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("field-F1 on silver gold set  (higher better ^)")
    ax.set_title("Student reaches 96.5% of teacher field-F1 — clears the 95% bar")
    ax.annotate("96.5% of teacher\nmeets_bar = true", xy=(2, 0.9647), xytext=(1.35, 0.925),
                fontsize=7.8, color=GOOD, weight="bold",
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.1))
    _hide_spines(ax)
    ax.grid(axis="y", color="#eee", zorder=0)
    save(fig, "fig2_headline_f1.png")


# =================================================================== FIGURE 3
# Failure taxonomy: category shares (failure_analysis.md)
def fig3():
    cats = ["wrong_text", "hallucinated\nline_item", "wrong_money",
            "hallucinated\nfield", "missing_field", "wrong_date"]
    errs = [11, 4, 4, 2, 2, 2]                 # real: failure_analysis.md
    share = [44, 16, 16, 8, 8, 8]
    order = range(len(cats))
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ypos = list(order)[::-1]
    colors = [BLUE_D, BLUE_M, BLUE_M, BLUE_L, BLUE_L, BLUE_L]
    bars = ax.barh(ypos, errs, color=colors, zorder=3, height=0.66)
    for b, e, s, y in zip(bars, errs, share, ypos):
        ax.text(e + 0.15, y, f"{e}  ({s}%)", va="center", fontsize=8.2, weight="bold",
                color=BLUE_N)
    ax.set_yticks(ypos)
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_xlim(0, 13)
    ax.set_xlabel("error count  (25 field errors over 711 fields = 3.5%)")
    ax.set_title("Residual-gap taxonomy: text normalization dominates")
    _hide_spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eee", zorder=0)
    save(fig, "fig3_failure_taxonomy.png")


# =================================================================== FIGURE 4
# Footprint win (28x) + honest cost caveat
def fig4():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.6, 3.3))

    # -- left: footprint, log scale (14B vs 0.5B ~= 28x)
    m = ["Teacher\nqwen3:14b", "Student\nQwen2.5-0.5B"]
    params = [14.0, 0.5]                        # money_table.md: ~28x smaller
    bars = axl.bar(m, params, color=[BLUE_N, BLUE_D], width=0.6, zorder=3, log=True)
    for b, v in zip(bars, params):
        axl.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:g}B",
                 ha="center", va="bottom", fontsize=9, weight="bold")
    axl.set_ylabel("parameters (billions, log)  (smaller better v)")
    axl.set_ylim(0.2, 40)
    axl.set_title("~28x smaller footprint")
    axl.annotate("28x", xy=(0.5, 3.0), fontsize=13, weight="bold", color=GOOD, ha="center")
    _hide_spines(axl)
    axl.grid(axis="y", color="#eee", zorder=0)

    # -- right: $/1k requests, honest
    lab = ["Teacher\n(free-local)", "Student\n(GPU amortized)"]
    cost = [0.0, 0.1178]                        # real: eval_report.json
    bars = axr.bar(lab, cost, color=[GREY, BLUE_M], width=0.6, zorder=3)
    for b, v in zip(bars, cost):
        axr.text(b.get_x() + b.get_width() / 2, v + 0.004, f"${v:.4f}",
                 ha="center", va="bottom", fontsize=8.6, weight="bold")
    axr.set_ylabel("USD / 1k requests")
    axr.set_ylim(0, 0.16)
    axr.set_title("Cost: dollar win UNPROVEN")
    axr.text(0.5, 0.135, "free-local teacher =>\nno $ arbitrage yet\n(gated on paid API)",
             ha="center", fontsize=6.8, color=BAD, style="italic")
    _hide_spines(axr)
    axr.grid(axis="y", color="#eee", zorder=0)

    fig.suptitle("Real win is footprint + privacy — cost thesis honestly gated",
                 fontsize=10, weight="bold", color=BLUE_N, y=1.02)
    save(fig, "fig4_footprint_cost.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("done")
