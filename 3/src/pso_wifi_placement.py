"""
pso_wifi_placement.py
=====================================================================
Lab 3 - Population-Based / Swarm Intelligence (CSEDU AI Lab)

ORIGINAL PROBLEM
    Optimal placement of a small number of Wi-Fi access points (APs)
    on one floor of a University of Dhaka residential hall so that the
    weighted coverage of student rooms is maximised, subject to a
    minimum-separation (co-channel interference) planning rule and the
    physical boundary of the floor.

WHY SWARM / PSO
    The decision variable is a continuous vector of AP coordinates.
    The objective is:
        - non-convex          (many good placements => multimodal),
        - non-differentiable  (max-over-APs, a soft threshold, and
                               concrete-wall attenuation are all kinks),
        - black-box-like      (no useful analytic gradient).
    Gradient descent has no reliable gradient and gets trapped; exact
    solvers explode because the search space is continuous R^(2K).
    Particle Swarm Optimization (PSO) is a natural fit: each particle
    IS a candidate placement (a point in R^(2K)); the swarm explores the
    floor collectively and converges on a strong layout.

CONTRACT WITH plan.md (Part 5)
    - Only numpy / matplotlib / scipy are used.
    - The optimizer is written FROM SCRATCH (no DEAP / pyswarm / ...).
    - Config dict-class on top, Optimizer class, standalone Fitness
      evaluator, clean execution block.
    - Two matched-budget baselines (Random Search + Grid Search).
    - Two required plots: (1) convergence curve, (2) spatial swarm map.
    - Wilcoxon signed-rank test for statistical significance.

Run:  python src/pso_wifi_placement.py
=====================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =====================================================================
# 1. CONFIGURATION  (every tunable in one place - defend each choice)
# =====================================================================
@dataclass
class Config:
    # ---- Floor geometry (metres) : a 2-D hall BLOCK (multi-corridor wing),
    #      rooms on a grid. Large relative to radio reach so K APs cannot
    #      trivially blanket it -> spatial placement genuinely matters and
    #      the search is a true 6-D continuous facility-location problem
    #      (K = 3 access points, two coordinates each).
    floor_w: float = 60.0
    floor_h: float = 40.0
    n_room_cols: int = 10              # grid of rooms ...
    n_room_rows: int = 4              # ... 10 x 4 = 40 demand points

    # ---- Radio model (2.4 GHz indoor, log-distance path loss)
    tx_power_dbm: float = 20.0          # AP transmit power (100 mW EIRP)
    path_loss_d0_db: float = 40.0       # reference loss at d0 = 1 m (2.4 GHz)
    path_loss_exp: float = 3.3          # indoor path-loss exponent n (dense hall)
    wall_loss_db: float = 8.0           # extra loss per concrete partition
    rssi_threshold_dbm: float = -66.0   # "usable link" threshold (strict)
    rssi_softness_db: float = 4.0       # logistic softness s around threshold

    # ---- Problem sizing / constraints
    n_aps: int = 3                      # K access points to place (tight budget)
    min_separation_m: float = 10.0      # co-channel interference rule d_min
    lambda_sep: float = 0.30            # penalty weight for separation breach

    # ---- PSO hyper-parameters (inertia form, matches the course slide)
    swarm_size: int = 30                # P particles
    n_iters: int = 100                  # T iterations
    w_max: float = 0.9                  # inertia start (exploration)
    w_min: float = 0.4                  # inertia end   (exploitation)
    c1: float = 1.49445                 # cognitive (personal) coefficient
    c2: float = 1.49445                 # social   (global)   coefficient
    v_max_frac: float = 0.20            # velocity clamp = frac x axis range

    # ---- Experiment / reproducibility
    n_runs: int = 15                    # independent runs for statistics
    base_seed: int = 20250707
    stagnation_eps: float = 1e-4        # "no meaningful improvement" delta

    # ---- Derived (filled in __post_init__)
    n_rooms: int = field(init=False)
    n_dims: int = field(init=False)     # 2 * n_aps
    n_evals: int = field(init=False)    # shared budget = P * T

    def __post_init__(self) -> None:
        self.n_rooms = self.n_room_rows * self.n_room_cols
        self.n_dims = 2 * self.n_aps
        # TRUE total evaluations of one PSO run = P initial + P*T in the loop
        # = P*(T+1). The baselines are handed this EXACT number for a fair fight.
        self.n_evals = self.swarm_size * (self.n_iters + 1)


# =====================================================================
# 2. PROBLEM INSTANCE + FITNESS EVALUATOR  (the "environment")
# =====================================================================
class WifiFloorProblem:
    """Builds a deterministic hall-floor instance and scores placements.

    A *placement* (one particle / chromosome / agent) is a flat vector
        x = [ax_0, ay_0, ax_1, ay_1, ..., ax_{K-1}, ay_{K-1}]  in R^(2K)
    i.e. the (x, y) metre-coordinates of the K access points.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.base_seed)

        # ---- Demand points: student rooms on a (cols x rows) grid across the
        #      hall block (a multi-corridor wing seen from above).
        xs = np.linspace(4.0, cfg.floor_w - 4.0, cfg.n_room_cols)
        ys = np.linspace(4.0, cfg.floor_h - 4.0, cfg.n_room_rows)
        gx, gy = np.meshgrid(xs, ys)
        rooms = np.column_stack([gx.ravel(), gy.ravel()])       # (N, 2)
        # Real rooms are NOT on a perfect grid: jitter them so the optimum
        # sits OFF any fixed grid. This is what gives a continuous optimizer
        # (PSO) a genuine edge over a discrete Grid Search.
        rooms += rng.normal(0.0, 1.5, size=rooms.shape)
        self.rooms = np.clip(rooms, [1.0, 1.0], [cfg.floor_w - 1, cfg.floor_h - 1])

        # ---- Room weights: occupancy (students per room). Busier rooms
        #      matter more, so coverage is weighted, not just counted.
        self.weights = rng.integers(1, 5, size=self.rooms.shape[0]).astype(float)

        # ---- Interior concrete partition walls (endpoints as segments).
        #      Signal crossing a wall loses `wall_loss_db`. These kinks make
        #      the landscape non-differentiable and cast RF "shadows" that a
        #      single AP cannot cover, so APs must be spread intelligently.
        self.walls = np.array(
            [
                [20.0, 0.0, 20.0, 24.0],   # long vertical partition (lower)
                [40.0, 16.0, 40.0, 40.0],  # long vertical partition (upper)
                [12.0, 20.0, 48.0, 20.0],  # horizontal spine wall
            ]
        )  # each row: [x1, y1, x2, y2]

        # ---- Search-space bounds for every dimension (per-AP x then y).
        self.lower = np.tile([0.0, 0.0], cfg.n_aps)
        self.upper = np.tile([cfg.floor_w, cfg.floor_h], cfg.n_aps)

        # Evaluation counter - used to PROVE baselines share the budget.
        self.n_fitness_calls = 0

    # ---------- geometry helpers ----------
    @staticmethod
    def _ccw(ax, ay, bx, by, cx, cy):
        """Vectorised orientation test (>0 => counter-clockwise turn)."""
        return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)

    def _wall_crossings(self, aps: np.ndarray) -> np.ndarray:
        """Count wall intersections for every (room, AP) line of sight.

        Returns integer matrix C of shape (N_rooms, K). C[i, j] = number
        of partition walls between room i and AP j. Fully vectorised over
        rooms x APs for each wall segment (walls are few).
        """
        R = self.rooms                                    # (N, 2)
        counts = np.zeros((R.shape[0], aps.shape[0]), dtype=int)
        rx, ry = R[:, 0][:, None], R[:, 1][:, None]       # (N, 1)
        px, py = aps[:, 0][None, :], aps[:, 1][None, :]    # (1, K)
        for wx1, wy1, wx2, wy2 in self.walls:
            # Segment room-AP (r->p) vs wall (w1->w2): proper intersection
            # iff the endpoints straddle each other on both segments.
            d1 = self._ccw(rx, ry, px, py, wx1, wy1)
            d2 = self._ccw(rx, ry, px, py, wx2, wy2)
            d3 = self._ccw(wx1, wy1, wx2, wy2, rx, ry)
            d4 = self._ccw(wx1, wy1, wx2, wy2, px, py)
            hit = ((d1 * d2) < 0) & ((d3 * d4) < 0)
            counts += hit.astype(int)
        return counts

    # ---------- the fitness function ----------
    def fitness(self, x: np.ndarray) -> float:
        """Score ONE placement. Higher is better.

            F(x) = WeightedCoverage%(x)  -  lambda_sep * SeparationBreach(x)

        WeightedCoverage% in [0, 100] is the occupancy-weighted fraction of
        rooms with a usable link to their BEST AP. The separation penalty
        discourages APs sitting on top of each other (co-channel interference).
        """
        cfg = self.cfg
        self.n_fitness_calls += 1
        aps = x.reshape(cfg.n_aps, 2)                     # (K, 2)

        # --- distances room -> AP (metres), floored at 1 m to avoid log(0).
        diff = self.rooms[:, None, :] - aps[None, :, :]   # (N, K, 2)
        dist = np.sqrt((diff ** 2).sum(axis=2))           # (N, K)
        dist = np.maximum(dist, 1.0)

        # --- log-distance path loss + concrete-wall attenuation -> RSSI (dBm)
        path_loss = cfg.path_loss_d0_db + 10.0 * cfg.path_loss_exp * np.log10(dist)
        rssi = cfg.tx_power_dbm - path_loss - cfg.wall_loss_db * self._wall_crossings(aps)

        # --- each room hears its strongest AP; soft-threshold => coverage in [0,1]
        best_rssi = rssi.max(axis=1)                      # (N,)
        coverage = 1.0 / (1.0 + np.exp(-(best_rssi - cfg.rssi_threshold_dbm) / cfg.rssi_softness_db))

        weighted_cov = 100.0 * np.sum(self.weights * coverage) / np.sum(self.weights)

        # --- soft separation constraint: quadratic breach of d_min (metres).
        breach = 0.0
        for j in range(cfg.n_aps):
            for k in range(j + 1, cfg.n_aps):
                gap = np.linalg.norm(aps[j] - aps[k])
                breach += max(0.0, cfg.min_separation_m - gap) ** 2

        return float(weighted_cov - cfg.lambda_sep * breach)

    def coverage_percent(self, x: np.ndarray) -> float:
        """Reporting helper: HARD coverage % (link >= threshold), unweighted."""
        cfg = self.cfg
        aps = x.reshape(cfg.n_aps, 2)
        diff = self.rooms[:, None, :] - aps[None, :, :]
        dist = np.maximum(np.sqrt((diff ** 2).sum(axis=2)), 1.0)
        path_loss = cfg.path_loss_d0_db + 10.0 * cfg.path_loss_exp * np.log10(dist)
        rssi = cfg.tx_power_dbm - path_loss - cfg.wall_loss_db * self._wall_crossings(aps)
        return float(100.0 * np.mean(rssi.max(axis=1) >= cfg.rssi_threshold_dbm))


# =====================================================================
# 3. THE OPTIMIZER  -  Particle Swarm Optimization, FROM SCRATCH
# =====================================================================
class ParticleSwarmOptimizer:
    """Canonical inertia-weight PSO (Shi & Eberhart 1998), matching the
    course slide's update rule:

        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)      # velocity
        x <- x + v                                            # position

    BIOLOGICAL ANALOGY
        Each particle is a bird in a flock hunting for the best feeding
        spot. `pbest` is the best spot IT has personally found (memory /
        nostalgia); `gbest` is the best spot ANY bird has found (social
        signalling). Inertia `w` is the bird's momentum. r1, r2 are the
        random "wing flutter" that keeps the search alive.
    """

    def __init__(self, problem: WifiFloorProblem, cfg: Config, seed: int) -> None:
        self.p = problem
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # Per-dimension velocity clamp: a fraction of each axis' range so a
        # particle cannot teleport across the whole floor in one step.
        self.v_max = cfg.v_max_frac * (problem.upper - problem.lower)

    def optimize(self, record_trace: bool = False, record_swarm: bool = False):
        cfg, p, rng = self.cfg, self.p, self.rng

        # ---- INITIALISATION : scatter the swarm uniformly over the floor.
        X = rng.uniform(p.lower, p.upper, size=(cfg.swarm_size, cfg.n_dims))
        V = rng.uniform(-self.v_max, self.v_max, size=(cfg.swarm_size, cfg.n_dims))

        fitness = np.array([p.fitness(x) for x in X])     # evaluate the flock
        pbest = X.copy()                                  # personal-best positions
        pbest_fit = fitness.copy()                        # personal-best scores
        g_idx = int(np.argmax(pbest_fit))
        gbest = pbest[g_idx].copy()                       # global-best position
        gbest_fit = float(pbest_fit[g_idx])               # global-best score

        history = [gbest_fit]                             # convergence curve
        diversity = [self._diversity(X)]                 # swarm spread / iter
        gbest_trace = [gbest.copy()] if record_trace else None
        # Every particle's position at every iteration. Only the UI asks for this
        # (it lets you scrub the slider and watch the flock actually converge);
        # the batch experiments never do, because it is P*T*2K floats of nothing.
        swarm_trace = [X.copy()] if record_swarm else None

        # ---- MAIN LOOP
        for t in range(cfg.n_iters):
            # Linearly decay inertia: broad exploration -> fine exploitation.
            w = cfg.w_max - (cfg.w_max - cfg.w_min) * (t / max(1, cfg.n_iters - 1))

            # Fresh stochasticity every step (r1, r2 in [0,1]^{PxD}).
            r1 = rng.random((cfg.swarm_size, cfg.n_dims))
            r2 = rng.random((cfg.swarm_size, cfg.n_dims))

            # --- VELOCITY UPDATE (the heart of PSO) ---------------------
            cognitive = cfg.c1 * r1 * (pbest - X)         # pull to own best
            social = cfg.c2 * r2 * (gbest - X)            # pull to swarm best
            V = w * V + cognitive + social
            V = np.clip(V, -self.v_max, self.v_max)       # anti-explosion clamp

            # --- POSITION UPDATE ---------------------------------------
            X = X + V

            # --- BOUNDARY HANDLING (absorbing walls): clamp into the floor
            #     and kill the offending velocity component so particles
            #     don't grind along the wall. This is constraint REPAIR.
            below, above = X < p.lower, X > p.upper
            X = np.clip(X, p.lower, p.upper)
            V[below | above] = 0.0

            # --- EVALUATE + UPDATE MEMORIES ----------------------------
            fitness = np.array([p.fitness(x) for x in X])
            improved = fitness > pbest_fit
            pbest[improved] = X[improved]
            pbest_fit[improved] = fitness[improved]

            step_best = int(np.argmax(pbest_fit))
            if pbest_fit[step_best] > gbest_fit:
                gbest = pbest[step_best].copy()
                gbest_fit = float(pbest_fit[step_best])

            history.append(gbest_fit)
            diversity.append(self._diversity(X))
            if record_trace:
                gbest_trace.append(gbest.copy())
            if record_swarm:
                swarm_trace.append(X.copy())

        result = {
            "best_x": gbest,
            "best_fitness": gbest_fit,
            "history": np.array(history),
            "diversity": np.array(diversity),
            "stagnation_iter": self._stagnation_iter(np.array(history)),
        }
        if record_trace:
            result["gbest_trace"] = np.array(gbest_trace)
        if record_swarm:
            result["swarm_trace"] = np.array(swarm_trace)   # (T+1, P, 2K)
        return result

    @staticmethod
    def _diversity(X: np.ndarray) -> float:
        """Mean distance of particles to the swarm centroid: a proxy for
        exploration. High early (spread out) -> low late (converged)."""
        centroid = X.mean(axis=0)
        return float(np.mean(np.linalg.norm(X - centroid, axis=1)))

    def _stagnation_iter(self, history: np.ndarray) -> int:
        """Iteration index after which gbest never improves by > eps again."""
        last = 0
        for t in range(1, len(history)):
            if history[t] - history[t - 1] > self.cfg.stagnation_eps:
                last = t
        return last


# =====================================================================
# 4. BASELINES  (identical evaluation budget => fair comparison)
# =====================================================================
def random_search(problem: WifiFloorProblem, cfg: Config, seed: int):
    """Pure Random Search: sample `n_evals` placements, keep the best.
    Uses EXACTLY the same number of fitness evaluations as one PSO run."""
    rng = np.random.default_rng(seed)
    best_x, best_fit = None, -np.inf
    curve = []
    for _ in range(cfg.n_evals):
        x = rng.uniform(problem.lower, problem.upper)
        f = problem.fitness(x)
        if f > best_fit:
            best_fit, best_x = f, x
        curve.append(best_fit)                            # best-so-far
    return {"best_x": best_x, "best_fitness": float(best_fit),
            "history": np.array(curve)}


def grid_search(problem: WifiFloorProblem, cfg: Config):
    """Structured Grid Search: lay a coarse grid of candidate AP sites over
    the floor, then evaluate combinations of K sites until the shared
    evaluation budget is spent. Deterministic given the grid."""
    # Grid resolution chosen so ALL C(G, K) combinations fit INSIDE the shared
    # budget -> no truncation bias. With G = 25 sites and K = 3 that is
    # C(25,3) = 2300 evaluations, comfortably under the 3030 budget.
    # The point this makes: refining the grid is not cheap. Going to G = 36
    # sites costs C(36,3) = 7140 evaluations, well over budget, and even that
    # grid is 10 m coarse. Enumeration cannot chase a continuous optimum -
    # and because the rooms are jittered, the optimum is not on any lattice.
    gx = np.linspace(8.0, cfg.floor_w - 8.0, 5)           # 5 x-positions
    gy = np.linspace(6.0, cfg.floor_h - 6.0, 5)           # 5 y-positions
    sites = np.array([[x, y] for x in gx for y in gy])    # G = 25 sites
    from itertools import combinations

    best_x, best_fit, used = None, -np.inf, 0
    for combo in combinations(range(len(sites)), cfg.n_aps):
        if used >= cfg.n_evals:
            break
        x = sites[list(combo)].flatten()
        f = problem.fitness(x)
        used += 1
        if f > best_fit:
            best_fit, best_x = f, x
    return {"best_x": best_x, "best_fitness": float(best_fit), "evals_used": used}


# =====================================================================
# 4b. IS THE COLLECTIVE ACTUALLY DOING ANYTHING?
#
# The whole premise of a population method is that individually weak agents
# solve a problem together that none of them could solve alone. That is a
# CLAIM, and it is cheap to test, so we test it instead of asserting it.
#
# Two ablations, both at the identical fitness-evaluation budget, so nobody
# can say the swarm simply got more compute:
#
#   (a) SWARM SIZE. Shrink the flock to a single particle and give it all
#       3030 evaluations to itself. With one particle, pbest and gbest are
#       the same point, the social term vanishes identically, and PSO
#       degenerates into one lone agent with momentum and a memory. This is
#       the single ant in the bowl.
#
#   (b) SILENCE THE SWARM. Keep all 30 particles, keep the budget, and set
#       c2 = 0. The particles still search, still remember their own best,
#       and still cover the floor - they simply never tell each other
#       anything. Thirty ants in thirty separate bowls.
#
# (b) is the sharper experiment: same number of agents, same effort, same
# everything, and the ONLY thing removed is communication.
# =====================================================================
def collective_behaviour_study(problem: WifiFloorProblem, cfg: Config,
                               swarm_sizes, n_runs: int):
    """Run PSO at several swarm sizes, and with the social term switched off,
    all under the SAME total number of fitness evaluations."""
    budget = cfg.n_evals
    rows = []

    for size in swarm_sizes:
        # Hold the budget fixed: fewer particles simply get more iterations.
        iters = budget // size - 1
        variant = replace(cfg, swarm_size=size, n_iters=iters)
        scores = []
        for r in range(n_runs):
            problem.n_fitness_calls = 0
            res = ParticleSwarmOptimizer(problem, variant, cfg.base_seed + 1000 + r).optimize()
            assert problem.n_fitness_calls == variant.n_evals, problem.n_fitness_calls
            scores.append(res["best_fitness"])
        rows.append({"Variant": f"swarm = {size:>3}",
                     "Particles": size, "Iters": iters + 1,
                     "Evals": variant.n_evals,
                     "Mean fitness": float(np.mean(scores)),
                     "Std": float(np.std(scores))})

    # The communication ablation: 30 particles, no social term.
    mute = replace(cfg, c2=0.0)
    scores = []
    for r in range(n_runs):
        problem.n_fitness_calls = 0
        res = ParticleSwarmOptimizer(problem, mute, cfg.base_seed + 1000 + r).optimize()
        scores.append(res["best_fitness"])
    rows.append({"Variant": "swarm =  30, c2 = 0 (no sharing)",
                 "Particles": cfg.swarm_size, "Iters": cfg.n_iters + 1,
                 "Evals": mute.n_evals,
                 "Mean fitness": float(np.mean(scores)),
                 "Std": float(np.std(scores))})
    return pd.DataFrame(rows)


def plot_collective(df, cfg, path):
    """The figure that answers 'why a population at all?'."""
    sized = df[df["Variant"].str.startswith("swarm =") &
               ~df["Variant"].str.contains("c2")]
    mute = df[df["Variant"].str.contains("c2")].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = sized["Particles"].to_numpy()
    y = sized["Mean fitness"].to_numpy()
    e = sized["Std"].to_numpy()

    ax.errorbar(x, y, yerr=e, fmt="o-", color="#1f77b4", lw=2, ms=6, capsize=4,
                label="PSO, information shared (varying swarm size)")
    ax.set_xscale("log")
    ax.axhline(mute["Mean fitness"], color="#d62728", ls="--", lw=1.6,
               label="30 particles that never share (c2 = 0)")
    ax.axhspan(mute["Mean fitness"] - mute["Std"], mute["Mean fitness"] + mute["Std"],
               color="#d62728", alpha=0.12)

    lone = sized.iloc[0]
    ax.annotate("a single particle:\nno swarm to learn from",
                xy=(lone["Particles"], lone["Mean fitness"]),
                xytext=(1.6, lone["Mean fitness"] - 0.15), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.9))

    ax.set_xlabel("Particles in the swarm  (log scale) - total evaluations held fixed at "
                  f"{cfg.n_evals}")
    ax.set_ylabel("Best fitness reached (mean of runs)")
    ax.set_title("The collective is doing the work, not the compute")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# 5. EXPERIMENT DRIVER  (multi-run, with Wilcoxon significance test)
# =====================================================================
def run_experiments(problem: WifiFloorProblem, cfg: Config):
    pso_best, rand_best = [], []
    pso_curves, div_curves = [], []
    t_pso = t_rand = 0.0
    best_rand_x, best_rand_fit = None, -np.inf   # best random layout across runs

    for r in range(cfg.n_runs):
        seed = cfg.base_seed + 1000 + r

        problem.n_fitness_calls = 0
        t0 = time.perf_counter()
        pso_res = ParticleSwarmOptimizer(problem, cfg, seed).optimize()
        t_pso += time.perf_counter() - t0
        assert problem.n_fitness_calls == cfg.n_evals, (
            problem.n_fitness_calls)  # P*(T+1): P init + P*T in the loop

        problem.n_fitness_calls = 0
        t0 = time.perf_counter()
        rand_res = random_search(problem, cfg, seed)
        t_rand += time.perf_counter() - t0
        if rand_res["best_fitness"] > best_rand_fit:
            best_rand_fit, best_rand_x = rand_res["best_fitness"], rand_res["best_x"]

        pso_best.append(pso_res["best_fitness"])
        rand_best.append(rand_res["best_fitness"])
        pso_curves.append(pso_res["history"])
        div_curves.append(pso_res["diversity"])

    # Grid search is deterministic -> a single run.
    grid_res = grid_search(problem, cfg)

    # Wilcoxon signed-rank test (paired by run): is PSO > Random for real?
    w_stat, p_val = stats.wilcoxon(pso_best, rand_best, alternative="greater")

    return {
        "pso_best": np.array(pso_best),
        "rand_best": np.array(rand_best),
        "rand_best_x": best_rand_x,
        "grid_best": grid_res["best_fitness"],
        "grid_best_x": grid_res["best_x"],
        "grid_evals": grid_res["evals_used"],
        "pso_curves": np.array(pso_curves),
        "div_curves": np.array(div_curves),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(p_val),
        "time_pso": t_pso,
        "time_rand": t_rand,
    }


# =====================================================================
# 6. VISUALISATION  (the two required plots + a diversity diagnostic)
# =====================================================================
def plot_convergence(exp, cfg, path):
    """Plot 1: Objective fitness vs iteration (PSO mean +/- std over runs),
    with the matched-budget Random-Search final level as a reference."""
    curves = exp["pso_curves"]
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    iters = np.arange(curves.shape[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, mean, color="#1f77b4", lw=2, label="PSO gbest (mean of runs)")
    ax.fill_between(iters, mean - std, mean + std, color="#1f77b4", alpha=0.2,
                    label="+/- 1 std")
    ax.axhline(exp["rand_best"].mean(), color="#d62728", ls="--", lw=1.5,
               label="Random Search (mean, same budget)")
    ax.axhline(exp["grid_best"], color="#2ca02c", ls=":", lw=1.5,
               label="Grid Search (same budget)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness  (weighted coverage %  -  penalty)")
    ax.set_title("Convergence: PSO vs matched-budget baselines")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spatial(problem, cfg, path):
    """Plot 2: The showpiece. Coverage heatmap of the floor under the best
    placement, the room demand points, the interior walls, and the swarm
    global-best TRAJECTORY (how the APs migrated across iterations)."""
    # One traced run (fixed seed) so the trajectory is reproducible.
    res = ParticleSwarmOptimizer(problem, cfg, cfg.base_seed + 7).optimize(
        record_trace=True)
    best = res["best_x"].reshape(cfg.n_aps, 2)
    trace = res["gbest_trace"]                            # (T+1, 2K)

    # Build a coverage heatmap over a fine grid of the floor.
    gx = np.linspace(0, cfg.floor_w, 180)
    gy = np.linspace(0, cfg.floor_h, 120)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    diff = pts[:, None, :] - best[None, :, :]
    dist = np.maximum(np.sqrt((diff ** 2).sum(axis=2)), 1.0)
    pl = cfg.path_loss_d0_db + 10 * cfg.path_loss_exp * np.log10(dist)
    rssi = (cfg.tx_power_dbm - pl).max(axis=1).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    hm = ax.contourf(GX, GY, rssi, levels=20, cmap="viridis")
    cbar = fig.colorbar(hm, ax=ax, pad=0.01)
    cbar.set_label("Best-AP RSSI (dBm)")

    # rooms sized by occupancy weight
    ax.scatter(problem.rooms[:, 0], problem.rooms[:, 1],
               s=18 * problem.weights, c="white", edgecolors="black",
               linewidths=0.6, label="rooms (size = occupancy)", zorder=3)

    # interior walls
    for wx1, wy1, wx2, wy2 in problem.walls:
        ax.plot([wx1, wx2], [wy1, wy2], color="firebrick", lw=4, zorder=2)

    # global-best trajectory per AP (faint) + final positions (stars)
    for j in range(cfg.n_aps):
        tx, ty = trace[:, 2 * j], trace[:, 2 * j + 1]
        ax.plot(tx, ty, color="orange", lw=1.2, alpha=0.8, zorder=4)
        ax.scatter(tx[0], ty[0], marker="o", c="orange", s=40,
                   edgecolors="black", zorder=5)
    ax.scatter(best[:, 0], best[:, 1], marker="*", c="red", s=420,
               edgecolors="black", linewidths=1.2,
               label="final AP placement", zorder=6)

    ax.set_xlim(0, cfg.floor_w)
    ax.set_ylim(0, cfg.floor_h)
    ax.set_xlabel("metres"); ax.set_ylabel("metres")
    ax.set_title("DU hall-floor Wi-Fi: coverage + swarm global-best trajectory")
    ax.legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return res


def plot_diversity(exp, cfg, path):
    """Diagnostic: swarm diversity vs iteration -> visual proof that the
    swarm shifts from EXPLORATION (spread) to EXPLOITATION (converged)."""
    div = exp["div_curves"]
    mean, std = div.mean(axis=0), div.std(axis=0)
    iters = np.arange(div.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, mean, color="#9467bd", lw=2, label="mean swarm spread")
    ax.fill_between(iters, mean - std, mean + std, color="#9467bd", alpha=0.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean particle distance to centroid")
    ax.set_title("Exploration -> Exploitation: swarm diversity collapse")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =====================================================================
# 7. EXECUTION BLOCK
# =====================================================================
def main() -> None:
    cfg = Config()
    problem = WifiFloorProblem(cfg)

    print("=" * 68)
    print(" DU HALL-FLOOR Wi-Fi ACCESS-POINT PLACEMENT  -  PSO from scratch")
    print("=" * 68)
    print(f" Floor: {cfg.floor_w:.0f}m x {cfg.floor_h:.0f}m | rooms: {cfg.n_rooms} "
          f"| APs (K): {cfg.n_aps} | dims: {cfg.n_dims}")
    print(f" Shared budget/run: {cfg.n_evals} fitness evals "
          f"(PSO {cfg.swarm_size}x{cfg.n_iters}+init) | runs: {cfg.n_runs}")
    print("-" * 68)

    exp = run_experiments(problem, cfg)

    # ---- Best PSO solution (report the winner of the traced representative run)
    best_run = plot_spatial(problem, cfg,
                            "results/plots/spatial_swarm.png")
    plot_convergence(exp, cfg, "results/plots/convergence.png")
    plot_diversity(exp, cfg, "results/plots/diversity.png")

    best_x = best_run["best_x"]
    aps = best_x.reshape(cfg.n_aps, 2)
    print(" BEST SOLUTION (representative traced run)")
    for j, (ax_, ay_) in enumerate(aps):
        print(f"   AP{j}:  x = {ax_:6.2f} m ,  y = {ay_:5.2f} m")
    print(f" Best fitness (objective) : {best_run['best_fitness']:.3f}")
    print(f" Hard coverage of rooms   : {problem.coverage_percent(best_x):.1f} %")
    print(f" Convergence stagnated at : iteration {best_run['stagnation_iter']} "
          f"/ {cfg.n_iters}")
    print("-" * 68)

    # ---- Side-by-side comparison matrix (mean +/- std over runs)
    # Hard room coverage (% of rooms >= threshold) for each method's best layout
    # -> shows PSO connects the last stranded rooms, not just adds soft margin.
    hard = [
        problem.coverage_percent(best_x),
        problem.coverage_percent(exp["rand_best_x"]),
        problem.coverage_percent(exp["grid_best_x"]),
    ]
    table = pd.DataFrame(
        {
            "Method": ["PSO (ours)", "Random Search", "Grid Search"],
            "Evals/run": [cfg.n_evals, cfg.n_evals, exp["grid_evals"]],
            "Best fitness": [exp["pso_best"].max(), exp["rand_best"].max(),
                             exp["grid_best"]],
            "Mean fitness": [exp["pso_best"].mean(), exp["rand_best"].mean(),
                             exp["grid_best"]],
            "Std": [exp["pso_best"].std(), exp["rand_best"].std(), 0.0],
            "Hard cover %": hard,
        }
    )
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print(" PERFORMANCE COMPARISON  (identical fitness-evaluation budget)")
    print(table.to_string(index=False))
    print("-" * 68)

    # ---- Statistical significance (the viva-proof)
    improve = 100.0 * (exp["pso_best"].mean() - exp["rand_best"].mean()) / abs(
        exp["rand_best"].mean())
    print(f" PSO mean beats Random by  : {improve:+.2f} %")
    print(f" Wilcoxon signed-rank test : W = {exp['wilcoxon_stat']:.1f}, "
          f"p = {exp['wilcoxon_p']:.2e}  (H1: PSO > Random)")
    verdict = "SIGNIFICANT (p < 0.05)" if exp["wilcoxon_p"] < 0.05 else "not significant"
    print(f" Verdict                   : {verdict}")
    print(f" Wall-clock: PSO {exp['time_pso']:.2f}s | Random {exp['time_rand']:.2f}s")
    print("-" * 68)

    # ---- Is the SWARM earning its keep, or is it just the compute budget?
    coll = collective_behaviour_study(problem, cfg, [1, 2, 5, 10, 15, 30, 101, 303],
                                      cfg.n_runs)
    plot_collective(coll, cfg, "results/plots/collective_behaviour.png")
    coll.to_csv("results/tables/pso_collective.csv", index=False)
    print(" COLLECTIVE BEHAVIOUR ABLATION  (identical budget of "
          f"{cfg.n_evals} evaluations throughout)")
    print(coll.to_string(index=False))

    lone = float(coll.loc[coll["Particles"] == 1, "Mean fitness"].iloc[0])
    full = float(coll.loc[coll["Variant"] == "swarm =  30", "Mean fitness"].iloc[0])
    mute = float(coll.loc[coll["Variant"].str.contains("c2"), "Mean fitness"].iloc[0])
    print(f"\n One particle, all {cfg.n_evals} evals to itself : {lone:.3f}")
    print(f" Thirty particles that never communicate  : {mute:.3f}")
    print(f" Thirty particles that share gbest        : {full:.3f}")
    print(f" -> sharing is worth {full - mute:+.3f} fitness at ZERO extra cost;")
    print(f"    the swarm beats the lone searcher by {full - lone:+.3f}.")
    print("=" * 68)
    print(" Plots written to results/plots/ : convergence.png, spatial_swarm.png,")
    print("                                   diversity.png, collective_behaviour.png")


if __name__ == "__main__":
    main()
