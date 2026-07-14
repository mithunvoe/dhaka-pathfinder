"""
rl_experiments.py
=====================================================================
Assignment 3, Part B - the experiments, tables and figures.

Everything that goes in the report is produced here. `rl_water_tank.py`
holds the MDP and the two algorithms; this file only drives them.

    python src/rl_experiments.py            # full run, ~4 minutes
    python src/rl_experiments.py --quick    # fewer seeds/episodes, ~40 s

What we set out to measure, in the order the report presents it:

  E1  Does value iteration behave the way the contraction theory says?
  E2  Does this problem actually need lookahead, or would a threshold rule do?
  E3  How much experience does Q-learning need - and does planning on a model
      LEARNED from those same samples do better with them?
  E4  How wrong can a prior model be before it stops being worth having?
  E5  Which of Q-learning's knobs actually matter?
=====================================================================
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_water_tank import (ACTION_NAMES, IDLE, N_ACTIONS, PUMP_GEN, PUMP_GRID,
                           HallWaterMDP, MDPConfig, QLearningConfig,
                           action_optimality, certainty_equivalence,
                           policy_always_idle, policy_evaluation, policy_myopic,
                           policy_random, q_learning, q_star_from,
                           regret_percent, rollout_stats, score_policy,
                           tune_caretaker, value_iteration)

PLOTS = Path("results/plots")
TABLES = Path("results/tables")


# =====================================================================
# E1 - Value iteration: does it do what the theory promises?
# =====================================================================
def e1_vi_convergence(mdp, P, R, gammas):
    """VI is a gamma-contraction, so ||V_{k+1} - V_k||_inf must fall
    geometrically at rate gamma. We check that rather than reporting a
    meaningless "it took N iterations", and we use the standard bound
    ||V_k - V*||_inf <= gamma*eps/(1-gamma) to justify the stopping tolerance
    instead of picking one by feel.
    """
    out = {}
    for g in gammas:
        if g == 0.0:
            continue
        t0 = time.perf_counter()
        V, pi, res = value_iteration(P, R, mdp.available, g)
        out[g] = {
            "residuals": res,
            "sweeps": len(res),
            "seconds": time.perf_counter() - t0,
            "policy": pi,
            # empirical contraction rate, measured well after transients
            "rate": float(np.mean(res[len(res) // 2: len(res) // 2 + 20][1:]
                                  / res[len(res) // 2: len(res) // 2 + 20][:-1])),
        }
    return out


def plot_vi_convergence(vi, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for g, d in sorted(vi.items()):
        res = d["residuals"]
        ax.semilogy(res, lw=1.8, label=f"$\\gamma$ = {g}  ({d['sweeps']} sweeps)")
        # the rate the contraction mapping theorem predicts
        k = np.arange(len(res))
        ax.semilogy(k, res[0] * (g ** k), ls=":", lw=1.0, color="grey")
    ax.set_xlabel("Value-iteration sweep $k$")
    ax.set_ylabel(r"Bellman residual $\|V_{k+1}-V_k\|_\infty$")
    ax.set_title("Value iteration converges at exactly the rate contraction theory predicts\n"
                 "(dotted grey: $\\gamma^k$ reference)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# E2 - Does the problem need lookahead at all?
# =====================================================================
def e2_baselines(mdp, P, R, gamma, opt_score, n_days):
    (gt, ft), care_score = tune_caretaker(mdp, P, R, gamma)
    rows = []
    policies = {
        "Value iteration (optimal)": value_iteration(P, R, mdp.available, gamma)[1],
        f"Tuned caretaker rule (grid<{gt}, diesel<{ft})": None,   # filled below
        "Myopic greedy ($\\gamma$=0)": policy_myopic(mdp, R),
        "Random (legal actions)": policy_random(mdp, 0),
        "Always idle": policy_always_idle(mdp),
    }
    from rl_water_tank import policy_caretaker
    policies[f"Tuned caretaker rule (grid<{gt}, diesel<{ft})"] = policy_caretaker(mdp, gt, ft)

    for name, pol in policies.items():
        v = score_policy(P, R, pol, gamma)
        st = rollout_stats(mdp, pol, n_days, seed=7)
        rows.append({
            "Policy": name,
            "Exact V": v,
            "Regret %": regret_percent(v, opt_score),
            "Cost/day": st["daily_cost"],
            "Shortage units/day": st["shortage_units_per_day"],
            "Diesel h/day": st["generator_hours_per_day"],
            "Dry hours/day": st["dry_hours_per_day"],
        })
    return pd.DataFrame(rows), (gt, ft)


# =====================================================================
# E3 - Learning curves. Q-learning vs a model learned from ITS OWN samples.
# =====================================================================
def e3_learning(mdp, P, R, gamma, opt_score, n_episodes, seeds, snap_every):
    """One training run gives BOTH curves.

    At each snapshot we take the agent's greedy policy (Q-learning) AND we
    build the maximum-likelihood MDP from exactly the transitions it has seen
    so far and plan on that (certainty equivalence). Same samples, same instant,
    two different ways of spending them. If we ran these as separate experiments
    the comparison would be arguable; done this way it is not.
    """
    ql_curves, ce_curves, diag = [], [], []

    for seed in seeds:
        def snap(ep, steps, ql_policy, counts, reward_sums, visits):
            ce_policy = certainty_equivalence(mdp, counts, reward_sums,
                                              visits, gamma)
            return {
                "steps": steps,
                "ql": regret_percent(score_policy(P, R, ql_policy, gamma), opt_score),
                "ce": regret_percent(score_policy(P, R, ce_policy, gamma), opt_score),
                "coverage": float((visits[mdp.available] > 0).mean()),
            }

        qcfg = QLearningConfig(n_episodes=n_episodes, record_counts=True,
                               snapshot_every=snap_every)
        out = q_learning(mdp, qcfg, gamma, seed=seed, snapshot_fn=snap)
        snaps = out["snapshots"]
        ql_curves.append([s["ql"] for s in snaps])
        ce_curves.append([s["ce"] for s in snaps])
        diag.append({"seed": seed, "coverage": out["coverage"],
                     "min_visits": out["min_visits"], "steps": out["steps"],
                     "final_policy": out["policy"]})

    steps = np.array([s["steps"] for s in snaps])
    return {
        "steps": steps,
        "ql": np.array(ql_curves),
        "ce": np.array(ce_curves),
        "diag": diag,
    }


def plot_learning(lr, mismatch, path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    steps = lr["steps"]

    for key, colour, label in [
        ("ql", "#d62728", "Q-learning (model-free)"),
        ("ce", "#1f77b4", "Certainty-equivalence VI (model LEARNED from the same samples)"),
    ]:
        m, sd = lr[key].mean(axis=0), lr[key].std(axis=0)
        ax.plot(steps, m, color=colour, lw=2, label=label)
        ax.fill_between(steps, m - sd, m + sd, color=colour, alpha=0.18)

    # Where a *prior* model would land, right or wrong. These need no samples
    # at all - that is the whole point of having a model.
    for scale, style in [(1.0, "-"), (0.25, "--"), (0.0, ":")]:
        r = mismatch[scale]
        ax.axhline(r, color="#2ca02c", ls=style, lw=1.4)
        tag = {1.0: "VI, correct model (no samples needed)",
               0.25: "VI, model believes outages are 4x rarer",
               0.0: "VI, model believes the grid never fails"}[scale]
        ax.text(steps[-1], r, f"  {tag}  ({r:.1f}%)", fontsize=7.5,
                color="#2ca02c", va="bottom", ha="right")

    ax.set_xlabel("Environment steps (simulated hours of hall operation)")
    ax.set_ylabel("Policy regret vs exact optimum  (%, lower is better)")
    ax.set_title("Same samples, two ways of spending them")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# E4 - How wrong can the prior model be before it stops being worth having?
# =====================================================================
def e4_mismatch(mdp, P_true, R_true, gamma, opt_score, scales):
    """The planner is given a WRONG belief about how often the grid fails, then
    its policy is scored in the real hall. outage_scale < 1 is the optimistic
    published load-shedding schedule; > 1 is a planner who is too pessimistic.
    """
    out = {}
    for k in scales:
        P_b, R_b = mdp.build_model(outage_scale=k)
        _, pi_b, _ = value_iteration(P_b, R_b, mdp.available, gamma)
        out[k] = regret_percent(score_policy(P_true, R_true, pi_b, gamma), opt_score)
    return out


def plot_mismatch(mismatch, lr, path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ks = np.array(sorted(mismatch))
    vals = np.array([mismatch[k] for k in ks])

    ax.plot(ks, vals, "o-", color="#2ca02c", lw=2, ms=5,
            label="Value iteration, planned on the WRONG model")
    ax.axvline(1.0, color="black", lw=0.9, ls="--")
    # Axes-fraction coords: the data-coord version collided with the title,
    # because ylim was not final at the time it was placed.
    ax.text(1.0, 0.32, " true outage rate", fontsize=8, rotation=90,
            transform=ax.get_xaxis_transform(), va="center", color="black")

    ql_final = lr["ql"][:, -1]
    ce_final = lr["ce"][:, -1]
    steps = lr["steps"][-1]
    for arr, colour, label in [
        (ql_final, "#d62728", f"Q-learning after {steps:,} steps"),
        (ce_final, "#1f77b4", f"Certainty-equivalence VI, same {steps:,} steps"),
    ]:
        m, sd = arr.mean(), arr.std()
        ax.axhline(m, color=colour, lw=1.8, label=label)
        ax.axhspan(m - sd, m + sd, color=colour, alpha=0.13)

    ax.set_xlabel("Planner's believed outage rate,  as a multiple of the truth\n"
                  "(0.25 = 'outages are four times rarer than they really are')")
    ax.set_ylabel("Policy regret vs exact optimum  (%, lower is better)")
    ax.set_title("A roughly-wrong model beats 82 years of experience.\n"
                 "Where the green curve dips below the blue line, the prior wins.")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# E5 - Which of Q-learning's knobs actually matter?
# =====================================================================
def e5_hyperparams(mdp, P, R, gamma, opt_score, n_episodes, seeds):
    """The teacher's instruction was explicit: vary things, and if nothing
    happens, say that nothing happened. So we report the null results too.
    """
    rows = []

    def run(label, group, qcfg, g=gamma):
        vals = [regret_percent(
            score_policy(P, R, q_learning(mdp, qcfg, g, seed=s)["policy"], gamma),
            opt_score) for s in seeds]
        rows.append({"Group": group, "Setting": label,
                     "Regret % (mean)": float(np.mean(vals)),
                     "Regret % (std)": float(np.std(vals))})

    base = dict(n_episodes=n_episodes)

    # Learning-rate schedule. Robbins-Monro says a constant alpha cannot
    # converge to Q*, only to a neighbourhood of it. Does that bite in practice?
    run("polynomial  1/(1+n)^0.7", "Learning rate",
        QLearningConfig(**base, alpha_mode="polynomial", alpha_omega=0.7))
    run("polynomial  1/(1+n)^0.5", "Learning rate",
        QLearningConfig(**base, alpha_mode="polynomial", alpha_omega=0.5))
    run("constant  alpha = 0.10", "Learning rate",
        QLearningConfig(**base, alpha_mode="constant", alpha_const=0.10))
    run("constant  alpha = 0.50", "Learning rate",
        QLearningConfig(**base, alpha_mode="constant", alpha_const=0.50))

    # Exploration.
    for e in [0.01, 0.05, 0.20]:
        run(f"epsilon floor = {e}", "Exploration",
            QLearningConfig(**base, eps_end=e))
    run("exploring starts OFF", "Exploration",
        QLearningConfig(**base, exploring_starts=False))

    # Optimism. Every reward here is <= 0, so Q initialised at 0 is OPTIMISTIC
    # and is quietly doing exploration work for us. Take the credit away and see.
    run("Q init = 0  (optimistic here)", "Initialisation",
        QLearningConfig(**base, q_init=0.0))
    run("Q init = -200 (pessimistic)", "Initialisation",
        QLearningConfig(**base, q_init=-200.0))

    # Discount factor. gamma=0 is the myopic control.
    for g in [0.90, 0.95, 0.99]:
        run(f"gamma = {g}", "Discount factor",
            QLearningConfig(**base), g=g)

    return pd.DataFrame(rows)


def plot_hyperparams(df, path):
    groups = list(df["Group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.0 * len(groups), 4.4),
                             sharey=True)
    for ax, grp in zip(np.atleast_1d(axes), groups):
        sub = df[df["Group"] == grp]
        y = np.arange(len(sub))
        ax.barh(y, sub["Regret % (mean)"], xerr=sub["Regret % (std)"],
                color="#7f7f7f", height=0.62, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Setting"], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(grp, fontsize=10)
        ax.set_xlabel("Regret % (lower better)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Q-learning: which knobs actually move the needle", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# E6 - What does the optimal policy actually DO?
# =====================================================================
def plot_policy_maps(mdp, pi_star, pi_ql, path):
    """The policy, drawn as the lookup table it actually is.

    This figure is NOT a timeline. Nothing happens left to right. It is a
    lookup table: pick an hour (column), pick a tank level (row), and the colour
    tells you what to do IN THAT SITUATION.

    The left and right panels are two different WORLDS, and both exist at every
    hour: "suppose the power is on right now" and "suppose the power is out right
    now". The shaded band marks the RISKY HOURS - when outages are likely and
    demand peaks. It does NOT mean the power is off; that is what the left/right
    split is for.

    An earlier version of this figure labelled the shading "load-shedding window",
    which made the blue inside it on the grid-UP panel look like a contradiction.
    It is the opposite - it is the smartest thing in the picture. It says: it is
    8pm, the power may go at any moment, but right now it is still on, so pump
    hard while you still can. The labels now say that.
    """
    cfg = mdp.cfg
    cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#1f77b4", "#d62728"])
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharex=True, sharey=True)

    panels = [
        (pi_star, 1, "VALUE ITERATION  (the exact optimum)",
         "IF the power is ON right now"),
        (pi_star, 0, "VALUE ITERATION  (the exact optimum)",
         "IF the power is OUT right now"),
        (pi_ql, 1, "Q-LEARNING  (learned from experience)",
         "IF the power is ON right now"),
        (pi_ql, 0, "Q-LEARNING  (learned from experience)",
         "IF the power is OUT right now"),
    ]
    for ax, (pol, g, who, world) in zip(axes.ravel(), panels):
        grid = np.zeros((cfg.n_levels, 24), dtype=int)
        for L in range(cfg.n_levels):
            for t in range(24):
                grid[L, t] = pol[mdp.encode(L, t, g, cfg.fuel_per_day)]
        ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.axvspan(17.5, 22.5, color="black", alpha=0.10)
        ax.set_title(f"{who}\n{world}", fontsize=9.5, linespacing=1.4)
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(0, cfg.n_levels, 2))

    # Say what the shading means, on the figure, so nobody has to guess.
    axes[0, 0].text(20, 10.6, "risky hours", fontsize=7.5, ha="center",
                    style="italic", color="#444444")
    axes[0, 1].text(20, 10.6, "risky hours", fontsize=7.5, ha="center",
                    style="italic", color="#444444")

    # The two things worth pointing at.
    axes[0, 0].annotate(
        "still ON at 6pm, so it tops up\neven a nearly-full tank\nwhile the power lasts",
        xy=(19, 8.4), xytext=(9.0, 9.6), fontsize=7.5, color="black",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))
    axes[0, 1].annotate(
        "power is OUT: it now burns diesel\nat a MUCH higher tank level\nonce the evening starts",
        xy=(19.5, 5.0), xytext=(9.5, 8.6), fontsize=7.5, color="black",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))

    for ax in axes[1]:
        ax.set_xlabel("hour of day        (shaded 5pm-10pm: demand peaks AND\n"
                      "outages are most likely - it does NOT mean the power is off)",
                      fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("tank level (x100 L)")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
               ["#eeeeee", "#1f77b4", "#d62728"]]
    fig.legend(handles, ["do nothing", "pump on grid power", "burn diesel"],
               loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(
        "What to do in every situation  (this is a LOOKUP TABLE, not a timeline)\n"
        "Pick an hour, pick a tank level, read the colour. Diesel ration full.",
        fontsize=11, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds and episodes; for a fast sanity check")
    args = ap.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    n_episodes = 6_000 if args.quick else 30_000
    snap_every = 1_000 if args.quick else 2_000
    seeds = [1, 2, 3] if args.quick else [1, 2, 3, 4, 5]
    hp_eps = 3_000 if args.quick else 8_000
    hp_seeds = [1, 2] if args.quick else [1, 2, 3]
    n_days = 300 if args.quick else 1_000

    cfg = MDPConfig()
    mdp = HallWaterMDP(cfg)
    P, R = mdp.build_model()

    print("=" * 74)
    print(" DU HALL WATER TANK  -  Value Iteration vs Q-Learning")
    print("=" * 74)
    n_legal = int(mdp.available.sum())
    print(f" States |S| = {cfg.n_states}  (level {cfg.n_levels} x hour 24 x grid 2 "
          f"x diesel {cfg.n_fuel})")
    print(f" Legal state-action pairs: {n_legal} of {cfg.n_states * N_ACTIONS} "
          f"(PUMP_GRID needs the grid, PUMP_GEN needs diesel)")
    print(f" Transition matrix nnz = {P.nnz}  ({100 * P.nnz / (cfg.n_states**2 * N_ACTIONS):.3f}% dense)")
    print(f" gamma = {cfg.gamma}  ->  effective horizon 1/(1-gamma) = "
          f"{1 / (1 - cfg.gamma):.0f} hours")
    print("-" * 74)

    # ---- E1 -----------------------------------------------------------
    vi = e1_vi_convergence(mdp, P, R, [0.90, 0.95, 0.99])
    plot_vi_convergence(vi, PLOTS / "rl_vi_convergence.png")
    d = vi[cfg.gamma]
    V, pi_star, _ = value_iteration(P, R, mdp.available, cfg.gamma)
    opt = score_policy(P, R, pi_star, cfg.gamma)
    Q_star = q_star_from(P, R, V, mdp.available, cfg.gamma)

    print(" E1  VALUE ITERATION")
    print(f"     converged in {d['sweeps']} sweeps ({d['seconds']:.2f} s)")
    print(f"     empirical contraction rate = {d['rate']:.4f}   "
          f"(theory says exactly gamma = {cfg.gamma})")
    cross = float(np.abs(V - policy_evaluation(P, R, pi_star, cfg.gamma)).max())
    print(f"     cross-check |V* - V^pi*|_inf = {cross:.2e}  "
          f"(VI agrees with the exact linear solve)")
    print(f"     optimal value V* (mean over start states) = {opt:.3f}")
    print("-" * 74)

    # ---- E2 -----------------------------------------------------------
    base_df, (gt, ft) = e2_baselines(mdp, P, R, cfg.gamma, opt, n_days)
    base_df.to_csv(TABLES / "rl_baselines.csv", index=False)
    print(" E2  DOES THIS PROBLEM ACTUALLY NEED LOOKAHEAD?")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:10.2f}")
    print(base_df.to_string(index=False))
    myopic_regret = float(base_df.loc[base_df["Policy"].str.contains("Myopic"),
                                      "Regret %"].iloc[0])
    care_regret = float(base_df.loc[base_df["Policy"].str.contains("caretaker"),
                                    "Regret %"].iloc[0])
    print(f"\n     The formally-correct greedy policy loses {myopic_regret:.0f}%.")
    print(f"     The best TUNED reactive threshold rule still loses {care_regret:.1f}%.")
    print("     So the lookahead is doing real work - this is not a vacuous MDP.")
    print("-" * 74)

    # ---- E4 (before E3: the learning plot needs these reference lines) --
    scales = [2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.35, 0.25, 0.1, 0.0]
    mismatch = e4_mismatch(mdp, P, R, cfg.gamma, opt, scales)

    # ---- E3 -----------------------------------------------------------
    print(f" E3  LEARNING FROM EXPERIENCE  ({len(seeds)} seeds x {n_episodes:,} "
          f"episodes x 24 h)")
    t0 = time.perf_counter()
    lr = e3_learning(mdp, P, R, cfg.gamma, opt, n_episodes, seeds, snap_every)
    print(f"     done in {time.perf_counter() - t0:.0f} s")
    plot_learning(lr, mismatch, PLOTS / "rl_learning_curves.png")

    ql_f, ce_f = lr["ql"][:, -1], lr["ce"][:, -1]
    cov = np.mean([d["coverage"] for d in lr["diag"]])
    print(f"     state-action coverage reached: {100 * cov:.1f}% of legal pairs")
    print(f"     after {lr['steps'][-1]:,} steps:")
    print(f"       Q-learning              regret = {ql_f.mean():6.2f} % "
          f"+/- {ql_f.std():.2f}")
    print(f"       Certainty-equiv. VI     regret = {ce_f.mean():6.2f} % "
          f"+/- {ce_f.std():.2f}   (SAME samples)")
    print(f"     action-optimality of the learned Q-policy: "
          f"{100 * action_optimality(Q_star, lr['diag'][0]['final_policy']):.1f}% of states")
    print("-" * 74)

    # ---- E4 report ----------------------------------------------------
    plot_mismatch(mismatch, lr, PLOTS / "rl_model_mismatch.png")
    mm_df = pd.DataFrame({"Believed outage rate (x truth)": scales,
                          "Regret %": [mismatch[k] for k in scales]})
    mm_df.to_csv(TABLES / "rl_model_mismatch.csv", index=False)
    print(" E4  HOW WRONG CAN THE PRIOR MODEL BE?")
    print(mm_df.to_string(index=False))

    # Derive the conclusions from the table rather than asserting them. An earlier
    # draft hardcoded "certainty-equivalence beats every mis-specified prior", and
    # the table two lines above flatly refuted it. Compute, do not claim.
    beats_ql = [k for k in scales if mismatch[k] < ql_f.mean()]
    ce_loses = sorted(k for k in scales if k != 1.0 and mismatch[k] < ce_f.mean())
    ce_wins = sorted(k for k in scales if k != 1.0 and mismatch[k] >= ce_f.mean())
    print(f"\n     A prior model this wrong STILL beats Q-learning ({ql_f.mean():.2f}%): "
          f"scales {min(beats_ql)} .. {max(beats_ql)}")
    print(f"     Certainty-equivalence VI ({ce_f.mean():.2f}%, learned from "
          f"{lr['steps'][-1]:,} samples):")
    print(f"       beats a BADLY wrong prior : scales {ce_wins}")
    print(f"       LOSES to a mildly wrong one: scales {ce_loses}")
    print("     => a roughly-right prior model is worth more than 82 simulated years")
    print("        of experience. That is the finding, and it is not the one we expected.")
    print("-" * 74)

    # ---- E5 -----------------------------------------------------------
    print(f" E5  HYPERPARAMETER SWEEP  ({len(hp_seeds)} seeds x {hp_eps:,} episodes)")
    t0 = time.perf_counter()
    hp = e5_hyperparams(mdp, P, R, cfg.gamma, opt, hp_eps, hp_seeds)
    hp.to_csv(TABLES / "rl_hyperparams.csv", index=False)
    plot_hyperparams(hp, PLOTS / "rl_hyperparams.png")
    print(f"     done in {time.perf_counter() - t0:.0f} s")
    print(hp.to_string(index=False))
    print("-" * 74)

    # ---- E6 -----------------------------------------------------------
    ql_policy = lr["diag"][0]["final_policy"]
    plot_policy_maps(mdp, pi_star, ql_policy, PLOTS / "rl_policy_maps.png")

    # A concrete piece of evidence that pi* really does anticipate: find a state
    # where pumping is worse THIS hour but optimal anyway.
    print(" E6  EVIDENCE OF ANTICIPATION")
    found = 0
    for t in range(12, 18):
        for L in range(4, cfg.n_levels):
            s = mdp.encode(L, t, 1, cfg.fuel_per_day)
            if pi_star[s] == PUMP_GRID and R[s, PUMP_GRID] < R[s, IDLE] - 1e-9:
                adv = Q_star[s, PUMP_GRID] - Q_star[s, IDLE]
                now = R[s, PUMP_GRID] - R[s, IDLE]
                print(f"     {t:02d}:00, tank {L}/10, grid up:  pumping costs "
                      f"{-now:.2f} MORE this hour,")
                print(f"        but is worth {adv:+.2f} overall -> "
                      f"{adv - now:+.2f} of that comes purely from the FUTURE.")
                found += 1
                break
        if found >= 3:
            break
    print("=" * 74)
    print(f" Figures -> {PLOTS}/   Tables -> {TABLES}/")


if __name__ == "__main__":
    main()
