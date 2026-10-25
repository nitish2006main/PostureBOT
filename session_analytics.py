"""
session_analytics.py  —  PostureBOT session browser, trend analysis, and overlay comparison
======================================================================================
Extends the original single-session replay script with two new views:
  - Trend: plots a stability score across every saved session and flags
    whether each one improved or regressed vs. the previous one.
  - Overlay: lets you pick 2+ sessions and plots their head-movement paths
    (top-down X/Y) on the same chart for direct visual comparison.
"""

import os
import json
import numpy as np

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

# Sessions known to have bad/invalid data, excluded from every view below.
# session_2026-05-04_17-55-44.json: max_z=301.6cm, max_x=209.7cm — an early
# build likely had REAL_FACE_WIDTH_CM set in mm instead of cm, inflating
# every distance calculation ~10x for that entire session. Confirmed by
# dividing its values by 10, which brings it in line with every other
# session's normal range.
EXCLUDED_SESSIONS = {
    "session_2026-05-04_17-55-44.json",
}


# ══════════════════════════════════════════════════════════
#  LOAD SESSION LIST + STATS
# ══════════════════════════════════════════════════════════

def list_session_files():
    if not os.path.isdir(SESSIONS_DIR):
        return []
    all_files = sorted(
        f for f in os.listdir(SESSIONS_DIR)
        if f.startswith("session_") and f.endswith(".json")
    )
    kept = [f for f in all_files if f not in EXCLUDED_SESSIONS]
    skipped = len(all_files) - len(kept)
    if skipped:
        print(f"[Sessions] Skipping {skipped} excluded session(s) with known bad data.")
    return kept


def load_session(name):
    with open(os.path.join(SESSIONS_DIR, name)) as f:
        return json.load(f)


def load_all_stats(files):
    """Pulls the summary stats (not the full data array) from every session."""
    out = []
    for name in files:
        try:
            s = load_session(name)
            stats = s.get("stats", {})
            data = [p for p in s.get("data", []) if p.get("face", True)]
            rms_xy = (np.sqrt(np.mean([p["x"]**2 + p["y"]**2 for p in data]))
            if data else None)
            out.append({
                "name":      name,
                "timestamp": s.get("timestamp", name),
                "duration":  stats.get("duration", 0),
                "max_x":     stats.get("max_x", 0),
                "max_y":     stats.get("max_y", 0),
                "max_z":     stats.get("max_z", 0),
                "min_z":     stats.get("min_z"),
                "avg_z":     stats.get("avg_z", 0),
                "pts":       len(s.get("data", [])),
                "rms_xy":    rms_xy,
            })
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return out

def z_deviation(s):
    if s["min_z"] is None:        # older sessions may not have min_z saved
        return None
    return max(abs(s["max_z"] or 0), abs(s["min_z"]))

def print_index(stats_list):
    print(f"\n{'#':>3}  {'Timestamp':<22}  {'Duration':>8}  {'Pts':>5}  {'Score':>6}")
    print("-" * 56)
    for i, s in enumerate(stats_list, 1):
        score = stability_score(s["max_x"], s["max_y"])
        print(f"{i:>3}  {s['timestamp']:<22}  {str(s['duration']):>7}s  "
              f"{s['pts']:>5}  {score:>5.0f}%")


# ══════════════════════════════════════════════════════════
#  STABILITY SCORE
#  Same spirit as the ESP32 OLED's on-device score, adapted for the
#  cm-scale X/Y deviation logged in each session's stats instead of
#  the servo's pan/tilt degrees.  100% = no lateral/vertical movement
#  from the calibration origin; score drops as deviation grows.
# ══════════════════════════════════════════════════════════

STABILITY_CM_PER_POINT = 1.0   # calibrated against ~57 real sessions: median deviation
                                # ~36cm, 75th pct ~55cm — this spreads real sessions across
                                # the 0-100 range instead of clipping half of them to 0


def stability_score(max_x, max_y):
    deviation = (max_x or 0) + (max_y or 0)
    return max(0.0, min(100.0, 100.0 - deviation * STABILITY_CM_PER_POINT))


# ══════════════════════════════════════════════════════════
#  TREND VIEW
#  Plots stability score across every session in chronological order
#  and prints an improved/regressed flag comparing each session to
#  the one immediately before it.
# ══════════════════════════════════════════════════════════

def show_trend(stats_list):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if len(stats_list) < 2:
        print("[Trend] Need at least 2 sessions to show a trend.")
        return

    scores = [stability_score(s["max_x"], s["max_y"]) for s in stats_list]
    labels = [s["timestamp"].split(" ")[0] for s in stats_list]   # date only

    # With many sessions (dense testing days repeat the same date many times),
    # showing every label overlaps into an unreadable smear — thin them out.
    max_labels_shown = 20
    label_stride = max(1, len(labels) // max_labels_shown)

    # ── console flags: each session vs. the one before it ──
    print("\n[Trend] Session-over-session change:")
    for i in range(1, len(scores)):
        delta = scores[i] - scores[i - 1]
        if delta > 0.5:
            flag = f"↑ improved {delta:.1f} pts"
        elif delta < -0.5:
            flag = f"↓ regressed {abs(delta):.1f} pts"
        else:
            flag = "— unchanged"
        print(f"  Session {i + 1} ({labels[i]}): {flag}")

    # ── plot ──
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#040810")
    ax.set_facecolor("#08111e")

    xs = list(range(1, len(scores) + 1))
    colours = ["#4ade80" if s >= 80 else "#facc15" if s >= 50 else "#f87171"
               for s in scores]

    ax.plot(xs, scores, color="#2a4a6a", linewidth=1.2, zorder=1)
    ax.scatter(xs, scores, c=colours, s=90, zorder=2, edgecolors="white", linewidths=0.5)

    for x, y in zip(xs, scores):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#b8d8e8", fontsize=8)

    ax.axhline(80, color="#4ade80", alpha=0.25, linewidth=0.8, linestyle="--")
    ax.axhline(50, color="#facc15", alpha=0.25, linewidth=0.8, linestyle="--")

        # ── linear trend ──
    import numpy as np
    if len(scores) >= 3:
        m, b = np.polyfit(xs, scores, 1)
        fitted = np.polyval([m, b], xs)
        ss_res = np.sum((np.array(scores) - fitted) ** 2)
        ss_tot = np.sum((np.array(scores) - np.mean(scores)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        sign = "+" if b >= 0 else "-"
        ax.plot(xs, fitted, "--", color="#b8d8e8", linewidth=1.4, zorder=1,
                label=f"y = {m:.2f}x {sign} {abs(b):.2f}   (R² = {r2:.2f})")
        ax.legend(loc="lower right", fontsize=8, facecolor="#08111e",
                  edgecolor="#0e2236", labelcolor="white")
        print(f"\n[Trend] y = {m:.2f}x {sign} {abs(b):.2f}, R² = {r2:.2f}")

        # ── duration check: is the score really just session length? ──
        durs = [float(s["duration"] or 0) for s in stats_list]
        if np.std(durs) > 0:
            print(f"[Trend] corr(score, duration) = "
                  f"{np.corrcoef(durs, scores)[0, 1]:+.2f}")
                # ── max vs RMS: do they rank sessions the same way? ──
        pairs = [(sc, s["rms_xy"], (s["max_x"] or 0) + (s["max_y"] or 0), i)
            for i, (sc, s) in enumerate(zip(scores, stats_list), 1)
            if s["rms_xy"]]
        if len(pairs) >= 3:
            sc_, rms_, dev_, idx_ = map(np.array, zip(*pairs))
            rank = lambda a: np.argsort(np.argsort(a))
            print(f"[Trend] rank corr(score, RMS) = "
                  f"{np.corrcoef(rank(sc_), rank(rms_))[0, 1]:+.2f}  "
                  f"(near -1 = same ranking; high score should mean low RMS)")
            spiky = sorted(zip(dev_ / rms_, idx_), reverse=True)[:5]
            print("[Trend] Most spike-driven sessions (max deviation ÷ RMS):")
            for ratio, i in spiky:
                print(f"  session {i}: {ratio:.1f}x")

        print("[Trend] Z deviation per session (cm):")
        for i, s in enumerate(stats_list, 1):
            zd = z_deviation(s)
            print(f"  {i:>3}: {'n/a' if zd is None else f'{zd:.1f}'}")

    shown_xs     = xs[::label_stride]
    shown_labels = labels[::label_stride]
    ax.set_xticks(shown_xs)
    ax.set_xticklabels(shown_labels, rotation=30, ha="right", color="#b8d8e8", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Stability score (%)", color="#b8d8e8")
    ax.tick_params(colors="#2a4a6a")
    for spine in ax.spines.values():
        spine.set_color("#0e2236")
    ax.set_title("Posture stability across sessions", color="#b8d8e8", fontsize=11)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════
#  OVERLAY VIEW
#  Plots the top-down (X vs Y) head path of 2+ chosen sessions on the
#  same chart, colour-coded, sharing one calibration-origin marker.
#  Deliberately 2-D (not the full 3-D globe) so multiple sessions
#  stay readable together.
# ══════════════════════════════════════════════════════════

def show_overlay(files, indices):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#040810")
    ax.set_facecolor("#08111e")

    cmap = plt.cm.plasma(np.linspace(0.15, 0.9, len(indices)))

    for colour, idx in zip(cmap, indices):
        name = files[idx - 1]
        s = load_session(name)
        data = [p for p in s.get("data", []) if p.get("face", True)]
        if len(data) < 2:
            print(f"  [skip] {name}: not enough points")
            continue
        xs = [p["x"] for p in data]
        ys = [p["y"] for p in data]
        score = stability_score(
            s.get("stats", {}).get("max_x", 0),
            s.get("stats", {}).get("max_y", 0),
        )
        label = f"{s.get('timestamp', name)}  ({score:.0f}%)"
        ax.plot(xs, ys, color=colour, linewidth=1.6, alpha=0.9, label=label)

    ax.scatter(0, 0, color="white", marker="*", s=160, zorder=5,
               label="Calibration origin")

    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="#1a3550", linewidth=0.6)
    ax.axvline(0, color="#1a3550", linewidth=0.6)
    ax.set_xlabel("X — lateral (cm)", color="#b8d8e8")
    ax.set_ylabel("Y — vertical (cm)", color="#b8d8e8")
    ax.tick_params(colors="#2a4a6a")
    for spine in ax.spines.values():
        spine.set_color("#0e2236")
    ax.set_title("Session overlay — head movement (Front View)",
                 color="#b8d8e8", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, facecolor="#08111e",
              edgecolor="#0e2236", labelcolor="white")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════
#  SESSION SELECTION HELPERS
# ══════════════════════════════════════════════════════════

def prompt_single(n):
    while True:
        try:
            choice = int(input(f"Session number [1-{n}]: "))
            if 1 <= choice <= n:
                return choice
            print(f"  Enter a number between 1 and {n}")
        except ValueError:
            print("  Please enter a number.")


def prompt_multi(n):
    while True:
        raw = input(f"Session numbers to overlay, comma-separated [1-{n}]: ")
        try:
            picks = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if len(picks) >= 2 and all(1 <= p <= n for p in picks):
                return picks
            print(f"  Enter at least 2 numbers between 1 and {n}, comma-separated.")
        except ValueError:
            print("  Please enter numbers separated by commas.")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    files = list_session_files()
    if not files:
        print("No sessions found in", SESSIONS_DIR)
        return

    stats_list = load_all_stats(files)
    print_index(stats_list)

    print("\nWhat would you like to do?")
    print("  [1] Replay a single session (3D path)")
    print("  [2] View stability trend across all sessions")
    print("  [3] Overlay compare 2+ sessions (Front View)")
    mode = input("Choice [1-3]: ").strip()

    if mode == "1":
        choice = prompt_single(len(files))
        session = load_session(files[choice - 1])
        from face_tracker_v4 import show_session_graph
        show_session_graph(session)

    elif mode == "2":
        show_trend(stats_list)

    elif mode == "3":
        picks = prompt_multi(len(files))
        show_overlay(files, picks)

    else:
        print("Not a valid option.")


if __name__ == "__main__":
    main()
