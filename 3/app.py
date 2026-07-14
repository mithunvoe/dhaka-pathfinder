"""
app.py - Assignment 3, interactive demo.
=====================================================================
    ./run.sh ui        (or:  streamlit run app.py)

Two tabs, one per part of the assignment.

  Part A  Watch a swarm place Wi-Fi access points, and - the point of the
          whole thing - untick "particles share gbest" and watch the swarm
          collapse to the level of random search without losing a single
          particle or a single evaluation.

  Part B  Watch what the pump controller learned, step through a simulated
          day, and drag the planner's belief about load-shedding away from
          the truth to see how much a wrong model actually costs.

Nothing here re-implements anything. It imports the same code the batch
experiments and the report use, so anything you can do in this UI, you can
defend from the source.
=====================================================================
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pso_wifi_placement import (Config, ParticleSwarmOptimizer, WifiFloorProblem,
                                grid_search, random_search)
from rl_water_tank import (ACTION_NAMES, IDLE, PUMP_GEN, PUMP_GRID, HallWaterMDP,
                           MDPConfig, QLearningConfig, certainty_equivalence,
                           policy_caretaker, policy_myopic, q_learning,
                           regret_percent, score_policy, tune_caretaker,
                           value_iteration)

st.set_page_config(page_title="AI Lab - Assignment 3", layout="wide")

# One fixed seed for the whole demo.
#
# There used to be a "random seed" box here. It was a liability: it let the viewer
# change the seed without re-running, which put the stored result and the widgets
# out of step. It also invited the wrong idea - that a single run means anything.
# It does not. The scientific claims in the report come from 15 or 30 SEEDED runs
# with a Wilcoxon test, not from whatever one seed happened to do. The UI is here
# to show the MECHANISM (turn communication off and watch the swarm collapse); the
# statistics live in `./run.sh swarm`.
DEMO_SEED = 1042


# =====================================================================
# Cached heavy lifting
# =====================================================================
@st.cache_resource
def load_wifi():
    """The floor GEOMETRY, for drawing: rooms, walls, weights.

    Read-only. Never run an optimiser against this instance - see below.
    """
    cfg = Config()
    return cfg, WifiFloorProblem(cfg)


def fresh_problem(cfg: Config) -> WifiFloorProblem:
    """A private problem instance, for one run only.

    This exists because of a real bug. `WifiFloorProblem` carries a mutable
    counter, `n_fitness_calls`, which is what lets us ASSERT that every method
    gets exactly the same evaluation budget. That assertion is the backbone of
    the whole Part A comparison, so it stays.

    But Streamlit re-runs the whole script on every widget change, and a cached
    (@st.cache_resource) problem is a singleton shared across those runs. Click
    two controls faster than a run completes and two script runs increment the
    SAME counter, so the assertion fires - not because the budget was violated,
    but because two runs were counting into one box.

    The instance is cheap to build (a seeded RNG and a meshgrid), and it is
    deterministic, so every run gets an identical floor and its own private
    counter. The fix is to not share mutable state, not to weaken the assert.
    """
    return WifiFloorProblem(cfg)


@st.cache_data(show_spinner=False)
def wifi_baselines(seed: int):
    """Random and Grid search at the same budget. Cached: they never change."""
    cfg = Config()
    problem = fresh_problem(cfg)          # its own counter, not the shared one
    rnd = random_search(problem, cfg, seed)
    grd = grid_search(problem, cfg)
    return rnd["best_fitness"], grd["best_fitness"]


@st.cache_resource
def load_mdp():
    cfg = MDPConfig()
    mdp = HallWaterMDP(cfg)
    P, R = mdp.build_model()
    V, pi, res = value_iteration(P, R, mdp.available, cfg.gamma)
    opt = score_policy(P, R, pi, cfg.gamma)
    return cfg, mdp, P, R, V, pi, opt


@st.cache_data(show_spinner=False)
def vi_on_believed_model(outage_scale: float):
    """Plan on a WRONG model, then score the resulting policy in the REAL hall."""
    cfg, mdp, P_true, R_true, _, _, opt = load_mdp()
    P_b, R_b = mdp.build_model(outage_scale=outage_scale)
    _, pi_b, _ = value_iteration(P_b, R_b, mdp.available, cfg.gamma)
    return pi_b, regret_percent(score_policy(P_true, R_true, pi_b, cfg.gamma), opt)


@st.cache_resource(show_spinner="Training Q-learning (about 15 s, once)...")
def train_q_learning(n_episodes: int = 30_000, seed: int = 1):
    """Q-learning + the certainty-equivalence planner built from its OWN samples."""
    cfg, mdp, P, R, _, _, opt = load_mdp()
    out = q_learning(mdp, QLearningConfig(n_episodes=n_episodes, record_counts=True),
                     cfg.gamma, seed=seed)
    ce = certainty_equivalence(mdp, out["counts"], out["reward_sums"],
                               out["visits"], cfg.gamma)
    return (out["policy"], regret_percent(score_policy(P, R, out["policy"], cfg.gamma), opt),
            ce, regret_percent(score_policy(P, R, ce, cfg.gamma), opt),
            out["coverage"], out["steps"])


# =====================================================================
# PART A
# =====================================================================
def draw_floor(problem, cfg, aps, swarm=None, trail=None):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    # signal heatmap under the current AP layout
    gx = np.linspace(0, cfg.floor_w, 200)
    gy = np.linspace(0, cfg.floor_h, 140)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    d = np.maximum(np.sqrt(((pts[:, None, :] - aps[None, :, :]) ** 2).sum(2)), 1.0)
    pl = cfg.path_loss_d0_db + 10 * cfg.path_loss_exp * np.log10(d)
    rssi = (cfg.tx_power_dbm - pl).max(axis=1).reshape(GX.shape)
    hm = ax.contourf(GX, GY, rssi, levels=18, cmap="viridis")
    fig.colorbar(hm, ax=ax, pad=0.01).set_label("best-AP signal (dBm)")

    for wx1, wy1, wx2, wy2 in problem.walls:
        ax.plot([wx1, wx2], [wy1, wy2], color="firebrick", lw=5, zorder=3)

    ax.scatter(problem.rooms[:, 0], problem.rooms[:, 1], s=16 * problem.weights,
               c="white", edgecolors="black", linewidths=0.6, zorder=4,
               label="rooms (size = students)")

    if swarm is not None:                     # every particle's AP guesses
        pts_ = swarm.reshape(-1, 2)
        ax.scatter(pts_[:, 0], pts_[:, 1], s=9, c="orange", alpha=0.45, zorder=5,
                   label="particles (candidate layouts)")
    if trail is not None:
        for j in range(cfg.n_aps):
            ax.plot(trail[:, 2 * j], trail[:, 2 * j + 1], color="orange", lw=1.0,
                    alpha=0.8, zorder=5)

    ax.scatter(aps[:, 0], aps[:, 1], marker="*", s=460, c="red",
               edgecolors="black", linewidths=1.2, zorder=6, label="access points")
    ax.set_xlim(0, cfg.floor_w); ax.set_ylim(0, cfg.floor_h)
    ax.set_xlabel("metres"); ax.set_ylabel("metres")
    ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def tab_swarm():
    cfg, problem = load_wifi()

    st.markdown(
        "#### Where do you bolt 3 Wi-Fi access points to a hall ceiling?\n"
        "40 rooms, 3 concrete walls, and only 3 APs. The signal drops 8 dB every time it "
        "crosses a wall, in a **step**, so the objective is piecewise-constant: there is no "
        "gradient to descend. So we throw a swarm at it."
    )

    left, right = st.columns([1, 2.1])

    with left:
        st.markdown("**Swarm settings**")
        size = st.select_slider("Particles in the swarm", [1, 2, 5, 10, 15, 30, 101],
                                value=30)
        share = st.checkbox("Particles share their global best", value=True,
                            help="Untick this to set c2 = 0. The particles still search "
                                 "and still remember their own best. They simply stop "
                                 "telling each other anything.")

        iters = cfg.n_evals // size - 1
        st.caption(
            f"**Budget is held fixed at {cfg.n_evals} evaluations.** "
            f"With {size} particle{'s' if size > 1 else ''} that buys {iters + 1} "
            f"iterations. Fewer particles do not get less compute; they get more turns."
        )
        run = st.button("Run the swarm", type="primary", width="stretch")

    # Recompute whenever the SETTINGS change, not only when the button is pressed.
    #
    # The previous version only recomputed on `run`, so changing a control without
    # pressing the button left st.session_state holding a result from the OLD
    # settings while the widgets showed the new ones. The iteration slider below is
    # sized from that stored result, so a stale result with a different history
    # length gave the slider a value outside its own range, and Streamlit threw.
    # Keying the cache on the settings themselves removes the whole class of bug.
    sig = (size, share, DEMO_SEED)
    if run or st.session_state.get("pso_sig") != sig:
        variant = replace(cfg, swarm_size=size, n_iters=iters,
                          c2=cfg.c2 if share else 0.0)
        # A PRIVATE problem for this run, so the evaluation counter cannot be
        # clobbered by a concurrent re-run of the script. See fresh_problem().
        run_problem = fresh_problem(cfg)
        res = ParticleSwarmOptimizer(run_problem, variant, DEMO_SEED).optimize(
            record_trace=True, record_swarm=True)
        # The budget really is matched, and we check it rather than claim it.
        assert run_problem.n_fitness_calls == variant.n_evals, (
            f"budget violated: {run_problem.n_fitness_calls} != {variant.n_evals}")
        st.session_state.pso = (res, variant, share, size)
        st.session_state.pso_sig = sig

    res, variant, share, size = st.session_state.pso
    rnd_fit, grid_fit = wifi_baselines(DEMO_SEED)

    with right:
        n_iter = len(res["history"]) - 1
        # The key ties the slider to THIS run. Without it the slider keeps its old
        # value across a change of swarm size, and 3029 is not a legal position on a
        # 100-step run.
        t = st.slider("Iteration", 0, n_iter, n_iter,
                      key=f"iter_{size}_{share}",
                      help="Drag back to zero and watch the swarm converge.")
        t = min(t, n_iter)                       # belt and braces
        best_at_t = res["gbest_trace"][t].reshape(variant.n_aps, 2)
        st.pyplot(draw_floor(problem, variant, best_at_t,
                             swarm=res["swarm_trace"][t],
                             trail=res["gbest_trace"][:t + 1]),
                  width="stretch")

    c1, c2_, c3, c4 = st.columns(4)
    final = res["best_fitness"]
    c1.metric("PSO score", f"{final:.2f}", f"{final - rnd_fit:+.2f} vs random")
    c2_.metric("Random search", f"{rnd_fit:.2f}", "same 3030 evals", delta_color="off")
    c3.metric("Grid search", f"{grid_fit:.2f}", "2300 evals", delta_color="off")
    c4.metric("Rooms connected", f"{problem.coverage_percent(res['best_x']):.0f}%")

    if not share:
        st.error(
            f"**Communication is off (c2 = 0).** Thirty particles are still searching, and "
            f"they still spend all {variant.n_evals} evaluations — they just never tell each "
            f"other anything. The swarm scores **{final:.2f}**. Random search scores "
            f"**{rnd_fit:.2f}**.\n\n"
            "They are the same. The population is not what helps. The communication is. "
            "This is the ant-in-a-bowl point, and it is the whole argument for Part A."
        )
    elif size == 1:
        verdict = ("**worse than random guessing**" if final < rnd_fit
                   else "barely better than random guessing")
        st.warning(
            f"**One particle.** With a swarm of one, `pbest` and `gbest` are the same point, so "
            f"the social term vanishes identically and PSO degenerates into a single agent with "
            f"momentum and a memory. It gets all {variant.n_evals} evaluations to itself and "
            f"scores **{final:.2f}** — {verdict} ({rnd_fit:.2f}).\n\n"
            f"Across 15 seeds it averages 76.8 with a standard deviation of 6.7: not just worse, "
            f"but wildly unreliable. This is the single ant in the bowl."
        )
    else:
        st.success(
            f"**{size} particles, sharing one number.** Score **{final:.2f}** against random "
            f"search's **{rnd_fit:.2f}**, on an identical budget. Now untick the sharing box "
            f"and run it again."
        )


# =====================================================================
# PART B
# =====================================================================
def policy_grid(mdp, pol, grid_up: int, fuel: int):
    cfg = mdp.cfg
    return np.array([[pol[mdp.encode(L, t, grid_up, fuel)] for t in range(24)]
                     for L in range(cfg.n_levels)])


def draw_policy(mdp, pol, fuel, title):
    cmap = matplotlib.colors.ListedColormap(["#e8e8e8", "#1f77b4", "#d62728"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, g, lab in [(axes[0], 1, "grid UP"), (axes[1], 0, "grid DOWN (load-shedding)")]:
        ax.imshow(policy_grid(mdp, pol, g, fuel), origin="lower", aspect="auto",
                  cmap=cmap, vmin=0, vmax=2)
        ax.axvspan(17.5, 22.5, color="black", alpha=0.12)
        ax.set_title(f"{title} - {lab}", fontsize=10)
        ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("hour")
    axes[0].set_ylabel("tank level")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c)
               for c in ["#e8e8e8", "#1f77b4", "#d62728"]]
    fig.legend(handles, ACTION_NAMES, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def tab_decision():
    cfg, mdp, P, R, V, pi_star, opt = load_mdp()

    st.markdown(
        "#### When do you switch on the water pump?\n"
        "The tank on the hall roof is filled by an electric pump. Load-shedding kills the "
        "power **worst in the evening**, exactly when everyone wants a shower, and the hall "
        "buys only **2 hours of diesel a day**. So you have to fill the tank *before* the "
        "power goes, and not waste the diesel doing it."
    )

    sub = st.radio("View", ["What the agent learned", "Live a day",
                            "How wrong can your model be?"],
                   horizontal=True, label_visibility="collapsed")

    # ---------------------------------------------------------------
    if sub == "What the agent learned":
        fuel = st.select_slider("Diesel left today (generator-hours)",
                                list(range(cfg.n_fuel)), value=cfg.fuel_per_day)
        st.pyplot(draw_policy(mdp, pi_star, fuel, "Value Iteration (exact optimum)"),
                  width="stretch")
        st.info(
            "Look at the right-hand panel, where the power is out. The **red** region (burn "
            "diesel) *grows* as you enter the shaded evening window: at 7am with a low tank it "
            "idles, at 7pm with the same tank it burns diesel — because it knows the outage "
            "will be long. **That is the agent anticipating.** No clock-blind threshold rule "
            "can express that shape, which is why the best tuned one still loses 14.5%."
        )

        with st.expander("Show the Q-learning policy for comparison"):
            ql_pol, ql_reg, ce_pol, ce_reg, cov, steps = train_q_learning()
            st.pyplot(draw_policy(mdp, ql_pol, fuel,
                                  f"Q-learning ({steps:,} steps)"),
                      width="stretch")
            st.caption(
                f"Same overall structure, but speckled. That is what 'has not seen enough data "
                f"yet' looks like — {ql_reg:.1f}% regret, having visited {100 * cov:.0f}% of "
                f"the legal state-action pairs."
            )

    # ---------------------------------------------------------------
    elif sub == "Live a day":
        pol_name = st.selectbox(
            "Whose policy should run the pump today?",
            ["Value Iteration (optimal)", "Q-learning", "Myopic greedy (gamma = 0)",
             "Tuned caretaker rule"])
        seed = st.number_input("Day seed", 0, 9999, 7, step=1)

        if pol_name.startswith("Value"):
            pol = pi_star
        elif pol_name.startswith("Q-learning"):
            pol = train_q_learning()[0]
        elif pol_name.startswith("Myopic"):
            pol = policy_myopic(mdp, R)
        else:
            (gt, ft), _ = tune_caretaker(mdp, P, R, cfg.gamma)
            pol = policy_caretaker(mdp, gt, ft)

        rng = np.random.default_rng(int(seed))
        s = mdp.encode(cfg.tank_capacity // 2, 0, 1, cfg.fuel_per_day)
        rows, cost, short = [], 0.0, 0.0
        for _ in range(24):
            L, t, g, F = mdp.decode(s)
            a = int(pol[s])
            s, r, info = mdp.simulate_hour(s, a, rng)
            cost += -r
            short += info["unmet"]
            rows.append({"hour": t, "tank": L, "grid": "up" if g else "OUT",
                         "diesel left": F, "action": ACTION_NAMES[a],
                         "students unserved": info["unmet"]})

        m1, m2, m3 = st.columns(3)
        m1.metric("Cost for the day", f"{cost:.1f}")
        m2.metric("Water students wanted and did not get", f"{short:.0f} units")
        m3.metric("Dry hours", f"{sum(1 for r_ in rows if r_['students unserved'] > 0)}")

        fig, ax = plt.subplots(figsize=(11, 3.0))
        hrs = [r_["hour"] for r_ in rows]
        ax.step(hrs, [r_["tank"] for r_ in rows], where="mid", lw=2, color="#1f77b4",
                label="tank level")
        for i, r_ in enumerate(rows):
            if r_["grid"] == "OUT":
                ax.axvspan(hrs[i] - 0.5, hrs[i] + 0.5, color="grey", alpha=0.25)
            if r_["action"] == "pump/diesel":
                ax.plot(hrs[i], 0.3, marker="^", color="#d62728", ms=9)
            if r_["students unserved"] > 0:
                ax.plot(hrs[i], 10.3, marker="x", color="black", ms=10, mew=2)
        ax.set_ylim(0, 11); ax.set_xlim(-0.5, 23.5)
        ax.set_xlabel("hour  (grey = no grid,  red triangle = diesel,  black x = students went dry)")
        ax.set_ylabel("tank level")
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        st.dataframe(rows, width="stretch", height=210)
        st.caption(
            "Run the optimal policy and the myopic one on the same seed. The myopic agent "
            "waits until the tank is low before it does anything — and by then the power is "
            "already gone."
        )

    # ---------------------------------------------------------------
    else:
        st.markdown(
            "Dhaka publishes a load-shedding schedule, and it is famously optimistic. So: give "
            "the planner a **wrong** belief about how often the grid fails, then score the "
            "policy it produces **in the real hall**."
        )
        scale = st.slider(
            "What the planner BELIEVES the outage rate is (x the truth)",
            0.0, 2.0, 1.0, 0.05,
            help="1.0 = the planner has the correct model. 0.25 = it thinks outages are four "
                 "times rarer than they really are. 0.0 = it thinks the grid never fails.")

        pi_b, reg_b = vi_on_believed_model(round(scale, 2))
        ql_pol, ql_reg, ce_pol, ce_reg, _, steps = train_q_learning()

        c1, c2_, c3 = st.columns(3)
        c1.metric("Value Iteration on this wrong model", f"{reg_b:.2f}% regret",
                  "0% is perfect", delta_color="off")
        c2_.metric(f"Q-learning ({steps:,} steps, no model)", f"{ql_reg:.2f}% regret",
                   "82 simulated years", delta_color="off")
        c3.metric("VI on a model LEARNED from those same samples", f"{ce_reg:.2f}% regret",
                  "same data as Q-learning", delta_color="off")

        ks = np.linspace(0, 2, 21)
        curve = [vi_on_believed_model(round(float(k), 2))[1] for k in ks]
        fig, ax = plt.subplots(figsize=(10, 4.0))
        ax.plot(ks, curve, "-", color="#2ca02c", lw=2,
                label="Value Iteration, planned on the wrong model")
        ax.axhline(ql_reg, color="#d62728", lw=1.8, label=f"Q-learning ({steps:,} steps)")
        ax.axhline(ce_reg, color="#1f77b4", lw=1.8,
                   label="VI on a model learned from the same samples")
        ax.plot(scale, reg_b, "o", ms=13, color="black", zorder=5, label="you are here")
        ax.axvline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("planner's believed outage rate (x truth)")
        ax.set_ylabel("regret % (lower is better)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

        if reg_b < ql_reg:
            st.success(
                f"**The wrong model still wins.** Even believing the outage rate is "
                f"{scale:.2f}x what it really is, planning beats {steps:,} steps of "
                f"Q-learning ({reg_b:.2f}% against {ql_reg:.2f}%). Drag the slider all the way "
                f"to 0 — the planner has to believe the grid *never fails* before experience "
                f"finally wins."
            )
        else:
            st.error(
                f"**Now the model is too broken to be worth having.** At {scale:.2f}x the true "
                f"outage rate the planner scores {reg_b:.2f}%, worse than Q-learning's "
                f"{ql_reg:.2f}%. This only happens at the very edge."
            )
        st.info(
            "**The finding.** We expected Q-learning to win here, and it does not. A model that "
            "is wrong by a factor of four still beats it. And a model *estimated from "
            "Q-learning's own samples* (the blue line) beats Q-learning by roughly ten times on "
            "identical data. So the argument for model-free control is not that models are "
            "often wrong — they can be quite wrong and still win. It is that sometimes you "
            "**cannot write one down at all**."
        )


# =====================================================================
st.title("AI Lab - Assignment 3")
st.caption("Two parts, one University of Dhaka hall. Part A places its Wi-Fi with a swarm. "
           "Part B runs its water pump with a Markov decision process.")

tab_a, tab_b = st.tabs(["Part A - Population-based search",
                        "Part B - Decision making"])
with tab_a:
    tab_swarm()
with tab_b:
    tab_decision()
