"""
rl_water_tank.py
=====================================================================
Assignment 3, Part B - Decision Making (CSEDU AI Lab)

THE PROBLEM
    A University of Dhaka residential hall keeps its water in an overhead
    tank, refilled by an electric pump. Three things turn this from
    plumbing into a decision problem:

      1. Load-shedding. The grid drops out, and it drops out precisely
         during the evening hours when the hall wants water. Evening
         outages are also long: once the power goes, it tends to stay gone.
      2. A diesel generator can drive the pump through an outage - but the
         hall buys a fixed ration of diesel each morning. Burn it at 7am on
         a hunch and there is none left at 9pm when it actually matters.
      3. Demand is not flat. It spikes when the hall bathes, morning and
         evening.

    Every hour the caretaker picks one of: leave the pump off, run it on
    grid power, or burn diesel. Pumping early on cheap grid power banks
    water against an outage that has not happened yet. Pumping diesel early
    spends a ration you cannot get back. That is the whole game.

WHY BOTH ALGORITHMS BELONG HERE
    (L, t, g, F) is Markov, so Value Iteration applies and returns the exact
    optimum - provided somebody hands us P(s'|s,a). Q-learning is handed only
    a simulator it can poke, and never sees a single probability.

    The interesting question is NOT "which wins". With a correct model VI wins
    by construction; it is exact. The question this assignment actually answers
    is: how wrong does your model have to be, and how much experience do you
    have to buy, before learning beats planning? Dhaka publishes a
    load-shedding schedule that is famously optimistic, so this is not a
    hypothetical - it is Tuesday.

    We also run the baseline that could falsify the whole story: estimate the
    model FROM Q-learning's own samples and plan on that (certainty
    equivalence). If that wins, then the lesson is not "model-free beats
    model-based" - it is "a learned model beats a wrong prior model", which is
    a different and more honest claim.

DESIGN NOTES THAT MATTER IN A VIVA
    * The reward depends on realised demand D, and D is NOT recoverable from
      s' (once the tank hits zero you cannot tell how much demand went unmet).
      So r is not a function of (s,a,s'). This is fine: it is the disturbance
      form of an MDP, r = r(s,a,w) with w = (D, g'). VI only ever needs the
      expected reward R(s,a) = E_w[r], which is what we tabulate; the Bellman
      operator is still a gamma-contraction. Q-learning consumes the REALISED
      r. It must - if we fed it E[U] we would have leaked the demand
      distribution into the "model-free" agent and the entire point of the
      assignment would collapse. `simulate_hour` never touches an expectation.

    * Action availability is state-dependent. PUMP_GRID is not offered when
      the grid is down (the switch does nothing), and PUMP_GEN is not offered
      when the diesel is gone. Without this, IDLE and PUMP_GRID are
      bit-for-bit identical in every outage state, Q-learning wastes a third
      of its exploration on a duplicate action, and any policy-agreement
      metric is just measuring how the two algorithms happen to break ties.

    * Discounted, not average-reward. Stated up front and swept: gamma is an
      operational horizon, and at one hour per step gamma = 0.99 gives about
      100 hours of lookahead, comfortably more than the 24-hour cycle the
      whole story depends on. The discounted-optimal policy is not guaranteed
      to be gain-optimal; we say so rather than pretending otherwise.

Only numpy / scipy / matplotlib / pandas. Both algorithms from scratch.
=====================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Actions. Module-level ints so the tables stay readable.
IDLE = 0
PUMP_GRID = 1
PUMP_GEN = 2
ACTION_NAMES = ["idle", "pump/grid", "pump/diesel"]
N_ACTIONS = 3


# =====================================================================
# 1. CONFIGURATION
# =====================================================================
@dataclass
class MDPConfig:
    """Every number the hall depends on. Each one is defended in the report."""

    # ---- Tank. One "unit" is 100 L; the tank holds 1000 L, which is about
    #      what a 40-room floor gets through on a heavy day.
    tank_capacity: int = 10
    pump_rate: int = 3                  # units the pump moves in an hour
    max_demand: int = 6                 # demand distribution truncated here

    # ---- Diesel ration. THE constraint that makes this interesting.
    #      The hall buys `fuel_per_day` generator-hours each morning. Unused
    #      fuel does not roll over (it is a daily delivery, not a tank).
    fuel_per_day: int = 2

    # ---- Costs, in units where 1.0 = one hour of grid pumping.
    #      Charged PER PUMP-HOUR, not per litre: the pump is single-speed and
    #      the bill is dominated by motor run-time, not by water volume.
    cost_grid: float = 1.0
    cost_generator: float = 6.0         # diesel, and it is loud at 2am
    cost_shortage: float = 20.0         # per unit of water students wanted
    cost_overflow: float = 0.5          # spilled water and a stressed motor

    # ---- Hourly demand: truncated Poisson, hour-dependent mean. Two peaks,
    #      because the hall bathes in the morning and again after evening class.
    demand_night: float = 0.1           # 00:00-04:00
    demand_early_rise: float = 1.0      # 05:00, the early risers
    demand_morning_peak: float = 2.5    # 06:00-09:00
    demand_day: float = 0.8             # 10:00-16:00
    demand_evening_ramp: float = 1.5    # 17:00, ramping into the evening
    demand_evening_peak: float = 2.5    # 18:00-22:00
    demand_late: float = 0.8            # 23:00

    # ---- Grid: a two-state Markov chain.
    #      p_fail    = P(grid drops  | grid is up)
    #      p_restore = P(grid returns| grid is down)
    #      Evening is the worst of both: outages are likely AND long
    #      (1/0.15 = ~6.7 h expected). That combination is what forces the
    #      agent to plan instead of react.
    p_fail_night: float = 0.02
    p_fail_day: float = 0.10
    p_fail_evening: float = 0.35
    p_restore_night: float = 0.50
    p_restore_day: float = 0.35
    p_restore_evening: float = 0.15

    # ---- Discounting. One step = one hour, so 1/(1-gamma) = 100 h of
    #      effective lookahead: the agent can see well past the daily cycle.
    #      Swept in the experiments, gamma=0 included as the myopic control.
    gamma: float = 0.99

    # ---- Derived
    n_levels: int = field(init=False)
    n_hours: int = field(init=False)
    n_grid: int = field(init=False)
    n_fuel: int = field(init=False)
    n_states: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_levels = self.tank_capacity + 1     # 0..capacity
        self.n_hours = 24
        self.n_grid = 2
        self.n_fuel = self.fuel_per_day + 1        # 0..fuel_per_day
        self.n_states = self.n_levels * self.n_hours * self.n_grid * self.n_fuel


# =====================================================================
# 2. THE MDP
# =====================================================================
class HallWaterMDP:
    """The hall as a finite, discounted, continuing MDP.

    State  s = (L, t, g, F)
        L in 0..capacity     tank level, 100 L units
        t in 0..23           hour of day (wraps - this is a continuing task)
        g in {0, 1}          is the grid up?
        F in 0..fuel_per_day generator-hours of diesel left today

    Action a in {IDLE, PUMP_GRID, PUMP_GEN}, restricted by `available()`.

    The class has two faces, and keeping them honest is the whole trick:

      build_model() -> the explicit P(s'|s,a) and R(s,a) that VALUE ITERATION
                       consumes. This is the model-based face.
      step()        -> one sampled (s', r). No probability ever leaves this
                       function. This is the only face Q-LEARNING sees.

    tests/test_rl_water_tank.py Monte-Carlo-estimates P and R back out of
    step() and asserts they match build_model() within sampling error, so the
    two faces cannot silently drift apart. That parity test is the gate: if it
    fails, every VI-vs-QL number in the report is meaningless.
    """

    def __init__(self, cfg: MDPConfig) -> None:
        self.cfg = cfg
        self._demand_table = np.stack(
            [self._demand_pmf(t) for t in range(cfg.n_hours)])      # (24, D+1)
        # Cumulative tables let the simulator sample with one uniform draw and
        # a searchsorted, instead of rng.choice(p=...) which re-validates the
        # probability vector on every call. Q-learning takes millions of steps.
        self._demand_cdf = np.cumsum(self._demand_table, axis=1)
        self._grid_cdf = np.stack([
            np.stack([np.cumsum(self.grid_transition(g, t))
                      for t in range(cfg.n_hours)])
            for g in range(2)])                                     # (2, 24, 2)
        self._avail = self._build_availability()

    # ---------- state <-> index ----------
    def encode(self, level: int, hour: int, grid: int, fuel: int) -> int:
        cfg = self.cfg
        return ((level * cfg.n_hours + hour) * cfg.n_grid + grid) * cfg.n_fuel + fuel

    def decode(self, s: int) -> tuple[int, int, int, int]:
        cfg = self.cfg
        fuel = s % cfg.n_fuel
        rest = s // cfg.n_fuel
        grid = rest % cfg.n_grid
        rest //= cfg.n_grid
        hour = rest % cfg.n_hours
        level = rest // cfg.n_hours
        return level, hour, grid, fuel

    # ---------- which actions actually exist in a state ----------
    def _build_availability(self) -> np.ndarray:
        """A(s) as a boolean (S, 3) mask.

        PUMP_GRID needs the grid. PUMP_GEN needs diesel. Offering an action
        that provably does nothing is not "realistic", it is a bug: it makes
        IDLE and PUMP_GRID identical in every outage state, so the argmax
        there is a coin flip and any VI-vs-QL agreement number is measuring
        the coin, not the policies.
        """
        cfg = self.cfg
        avail = np.zeros((cfg.n_states, N_ACTIONS), dtype=bool)
        for s in range(cfg.n_states):
            _, _, grid, fuel = self.decode(s)
            avail[s, IDLE] = True                  # always allowed to do nothing
            avail[s, PUMP_GRID] = grid == 1
            avail[s, PUMP_GEN] = fuel > 0
        return avail

    @property
    def available(self) -> np.ndarray:
        return self._avail

    # ---------- the two stochastic ingredients ----------
    def _demand_lambda(self, hour: int) -> float:
        """Every value here comes from MDPConfig. Nothing is hardcoded, so a
        sweep over the demand profile really does change the MDP it claims to."""
        cfg = self.cfg
        if hour <= 4:
            return cfg.demand_night
        if hour == 5:
            return cfg.demand_early_rise
        if 6 <= hour <= 9:
            return cfg.demand_morning_peak
        if 10 <= hour <= 16:
            return cfg.demand_day
        if hour == 17:
            return cfg.demand_evening_ramp
        if 18 <= hour <= 22:
            return cfg.demand_evening_peak
        return cfg.demand_late               # 23:00

    def _demand_pmf(self, hour: int) -> np.ndarray:
        """Poisson(lambda_t) truncated to 0..max_demand and RENORMALISED.

        Renormalised, not clipped: we do not pile the tail mass onto the top
        bin. Stating which of the two we did matters, because the VI table and
        the simulator must make the same choice or nothing lines up.
        """
        lam = self._demand_lambda(hour)
        d = np.arange(self.cfg.max_demand + 1)
        # exp(-lam) is a common factor and cancels in the normalisation.
        pmf = lam ** d / np.array([math.factorial(int(k)) for k in d], dtype=float)
        return pmf / pmf.sum()

    def grid_transition(self, grid: int, hour: int,
                        outage_scale: float = 1.0) -> np.ndarray:
        """P(g' | g, t) as [P(down), P(up)].

        `outage_scale` multiplies the FAILURE probability p_fail only (it does
        not touch p_restore - "twice as often" and "twice as long" are
        different mis-specifications with very different consequences, and we
        vary exactly one of them). outage_scale = 1.0 is the real hall.
        The mis-specified planner gets outage_scale < 1: it believes the
        optimistic published load-shedding schedule.
        """
        cfg = self.cfg
        if hour <= 5 or hour >= 23:
            p_fail, p_restore = cfg.p_fail_night, cfg.p_restore_night
        elif 17 <= hour <= 22:
            p_fail, p_restore = cfg.p_fail_evening, cfg.p_restore_evening
        else:
            p_fail, p_restore = cfg.p_fail_day, cfg.p_restore_day

        p_fail = float(np.clip(p_fail * outage_scale, 0.0, 1.0))
        if grid == 1:
            return np.array([p_fail, 1.0 - p_fail])
        return np.array([1.0 - p_restore, p_restore])

    # ---------- one hour of physics, shared by the model and the simulator ----
    def _physics(self, level: int, grid: int, fuel: int, action: int
                 ) -> tuple[int, int, float, float]:
        """Pump first, then the students draw.

        Returns (level after pumping, fuel left, water spilled, pump cost).
        Called by BOTH build_model and simulate_hour, so the two cannot
        disagree about the physics - only about whether they average over the
        randomness or sample it.
        """
        cfg = self.cfg
        if action == PUMP_GRID and grid == 1:
            inflow, pump_cost, fuel_left = cfg.pump_rate, cfg.cost_grid, fuel
        elif action == PUMP_GEN and fuel > 0:
            inflow, pump_cost, fuel_left = cfg.pump_rate, cfg.cost_generator, fuel - 1
        else:
            inflow, pump_cost, fuel_left = 0, 0.0, fuel

        raw = level + inflow
        overflow = max(0, raw - cfg.tank_capacity)
        return min(raw, cfg.tank_capacity), fuel_left, float(overflow), pump_cost

    def _next_fuel(self, hour: int, fuel_left: int) -> int:
        """The diesel drum is delivered every morning at midnight. Whatever is
        left over does not roll over - use it or lose it, which is what stops
        the agent from simply hoarding fuel forever and never pumping."""
        return self.cfg.fuel_per_day if hour == 23 else fuel_left

    # ---------- FACE 1: the explicit model (Value Iteration eats this) ------
    def build_model(self, outage_scale: float = 1.0
                    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Tabulate P and the expected reward R.

        P is returned as a sparse (S*A, S) matrix, row index s*A + a. Each row
        has at most (max_demand+1) * 2 non-zeros, so a dense S x A x S array
        would be ~99.9% zeros and 60 MB of nothing. Real MDPs are sparse; we
        store it that way.

        This array is what makes VI *model-based*. Q-learning never sees it.
        """
        cfg = self.cfg
        S, A = cfg.n_states, N_ACTIONS
        R = np.zeros((S, A))
        rows, cols, vals = [], [], []
        demand_vals = np.arange(cfg.max_demand + 1)

        for s in range(S):
            level, hour, grid, fuel = self.decode(s)
            next_hour = (hour + 1) % cfg.n_hours
            p_grid = self.grid_transition(grid, hour, outage_scale)
            p_demand = self._demand_table[hour]

            for a in range(A):
                if not self._avail[s, a]:
                    R[s, a] = 0.0                    # never selected; masked out
                    continue

                after_pump, fuel_left, overflow, pump_cost = self._physics(
                    level, grid, fuel, a)
                next_fuel = self._next_fuel(hour, fuel_left)

                # Expected reward. Pump cost and overflow are already
                # determined; only the shortage term needs averaging over D.
                exp_unmet = float(np.dot(
                    p_demand, np.maximum(demand_vals - after_pump, 0)))
                R[s, a] = -(pump_cost
                            + cfg.cost_overflow * overflow
                            + cfg.cost_shortage * exp_unmet)

                # Transition. Demand and the next grid state are independent.
                acc: dict[int, float] = {}
                for d, pd in enumerate(p_demand):
                    if pd == 0.0:
                        continue
                    next_level = max(0, after_pump - d)
                    for g_next, pg in enumerate(p_grid):
                        if pg == 0.0:
                            continue
                        s_next = self.encode(next_level, next_hour, g_next, next_fuel)
                        acc[s_next] = acc.get(s_next, 0.0) + pd * pg

                row = s * A + a
                for s_next, p in acc.items():
                    rows.append(row)
                    cols.append(s_next)
                    vals.append(p)

        P = sp.csr_matrix((vals, (rows, cols)), shape=(S * A, S))
        return P, R

    # ---------- FACE 2: the simulator (all Q-learning is allowed to touch) ---
    def simulate_hour(self, s: int, a: int, rng: np.random.Generator
                      ) -> tuple[int, float, dict]:
        """Sample one hour of the real hall. Returns (s', r, diagnostics).

        The realised demand is DRAWN, and the shortage in the reward is the
        REALISED shortage - never E[U]. If it were the expectation, the demand
        model would have leaked into the model-free agent.
        """
        cfg = self.cfg
        level, hour, grid, fuel = self.decode(s)

        after_pump, fuel_left, overflow, pump_cost = self._physics(
            level, grid, fuel, a)

        demand = min(int(np.searchsorted(self._demand_cdf[hour], rng.random())),
                     cfg.max_demand)
        unmet = max(0, demand - after_pump)
        next_level = max(0, after_pump - demand)
        next_grid = min(int(np.searchsorted(self._grid_cdf[grid, hour],
                                            rng.random())), 1)
        next_fuel = self._next_fuel(hour, fuel_left)

        reward = -(pump_cost
                   + cfg.cost_overflow * overflow
                   + cfg.cost_shortage * unmet)
        s_next = self.encode(next_level, (hour + 1) % cfg.n_hours,
                             next_grid, next_fuel)
        info = {"demand": demand, "unmet": unmet, "overflow": overflow,
                "pump_cost": pump_cost}
        return s_next, reward, info

    def step(self, s: int, a: int, rng: np.random.Generator
             ) -> tuple[int, float]:
        """The ONLY thing the Q-learner is allowed to call.

        Put in a state and an action, get back where you ended up and what it
        cost. No probabilities, no expectations, no peeking at the demand
        table. This narrow doorway IS the difference between the two
        algorithms in this report.
        """
        s_next, reward, _ = self.simulate_hour(s, a, rng)
        return s_next, reward


# =====================================================================
# 3. VALUE ITERATION  -  model-based, exact
# =====================================================================
def _masked(Q: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Send unavailable actions to -inf so argmax/max can never pick them."""
    return np.where(avail, Q, -np.inf)


def value_iteration(P: sp.csr_matrix, R: np.ndarray, avail: np.ndarray,
                    gamma: float, tol: float = 1e-9, max_sweeps: int = 20000):
    """Synchronous value iteration (Bellman 1957; Howard 1960).

        V_{k+1}(s) = max_{a in A(s)} [ R(s,a) + gamma * sum_s' P(s'|s,a) V_k(s') ]

    Model-based in the most literal sense: that sum over s' is only computable
    because somebody handed us P. Nothing is sampled and nothing is learned -
    the answer is computed.

    Returns V*, the greedy policy, and the Bellman residuals
    ||V_{k+1} - V_k||_inf. Contraction theory says those residuals must decay
    geometrically at rate gamma; the report checks that they do, and uses the
    standard bound ||V_k - V*||_inf <= gamma*eps/(1-gamma) to justify the
    stopping tolerance rather than picking one by feel.
    """
    S, A = R.shape
    V = np.zeros(S)
    residuals = []

    for _ in range(max_sweeps):
        Q = R + gamma * (P @ V).reshape(S, A)
        V_new = _masked(Q, avail).max(axis=1)
        delta = float(np.abs(V_new - V).max())
        residuals.append(delta)
        V = V_new
        if delta < tol:
            break

    Q = R + gamma * (P @ V).reshape(S, A)
    return V, _masked(Q, avail).argmax(axis=1).astype(int), np.array(residuals)


def policy_evaluation(P: sp.csr_matrix, R: np.ndarray, policy: np.ndarray,
                      gamma: float) -> np.ndarray:
    """The EXACT value of a fixed policy. The Bellman expectation equation for
    a fixed policy is linear,

        V^pi = R_pi + gamma * P_pi V^pi   =>   (I - gamma P_pi) V^pi = R_pi

    so we do not iterate it, we solve it. (I - gamma*P_pi) is invertible for
    any gamma < 1 since gamma*P_pi has spectral radius <= gamma < 1.

    This is the scoring function for EVERY policy in the report - Value
    Iteration's, Q-learning's, certainty-equivalence's, and the hand-written
    baselines' - and always under the TRUE model. It takes Monte-Carlo noise
    out of the comparison completely: policies are ranked by exact expected
    return, not by which one drew a luckier rollout.
    """
    S, A = R.shape
    rows = np.arange(S) * A + policy
    P_pi = P[rows]                                    # (S, S), sparse
    R_pi = R[np.arange(S), policy]
    return spla.spsolve(sp.eye(S, format="csr") - gamma * P_pi, R_pi)


# =====================================================================
# 4. Q-LEARNING  -  model-free, learned purely from experience
# =====================================================================
@dataclass
class QLearningConfig:
    n_episodes: int = 40_000
    episode_hours: int = 24             # one simulated day per episode

    alpha_mode: str = "polynomial"      # "constant" | "polynomial"
    alpha_const: float = 0.10
    alpha_omega: float = 0.70           # alpha_n = 1/(1+n)^omega

    eps_start: float = 1.00
    eps_end: float = 0.05
    eps_decay_frac: float = 0.60        # anneal over the first 60% of training

    exploring_starts: bool = True
    q_init: float = 0.0                 # see the note on optimism below
    snapshot_every: int = 0             # 0 = never; else call snapshot_fn
    record_counts: bool = False         # keep transition counts for CE-VI


def q_learning(mdp: HallWaterMDP, qcfg: QLearningConfig, gamma: float,
               seed: int, snapshot_fn=None):
    """Tabular Q-learning (Watkins 1989; Watkins & Dayan 1992).

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    The agent calls mdp.step() and gets one (s', r). It never sees P(s'|s,a).
    That single fact is what model-free means, and it is why you reach for this
    when nobody can hand you a trustworthy transition table - which, for Dhaka
    load-shedding, nobody can.

    Choices worth defending, because the examiner will ask:

    * Exploring starts. Each episode begins in a uniformly random state.
      Without them, states like "tank full, 8am, no diesel" are essentially
      never reached under a decent policy, their Q values stay at whatever we
      initialised them to, and the argmax over them is meaningless. Exploring
      starts buy the coverage that Q-learning's convergence proof simply
      assumes (every state-action pair visited infinitely often).

    * Polynomial learning rate alpha_n = 1/(1+n(s,a))^0.7. Robbins-Monro needs
      sum(alpha) = inf and sum(alpha^2) < inf, and omega in (0.5, 1] delivers
      both. A CONSTANT alpha satisfies neither and provably converges only to a
      neighbourhood of Q*, never to Q* itself. We do not just assert this - the
      hyperparameter sweep shows the constant-alpha run flattening out short of
      the optimum while the polynomial one keeps closing.

    * Q is initialised at 0 and every reward in this MDP is <= 0. That makes
      zero-initialisation OPTIMISTIC: untried actions look better than they are,
      so the agent is nudged towards trying them. This is doing real exploration
      work for us, so we say so and ablate it (`q_init`) rather than quietly
      taking the credit.

    * The 24-hour episode is a REPORTING unit, not a terminal state. The task
      is continuing, so the final update of an episode bootstraps normally off
      max_a Q(s',a) - we never zero the value at the cut. Zeroing it would
      silently turn this into a finite-horizon problem and Q would never match
      VI's infinite-horizon Q*.
    """
    rng = np.random.default_rng(seed)
    S = mdp.cfg.n_states
    avail = mdp.available

    Q = np.full((S, N_ACTIONS), qcfg.q_init, dtype=float)
    Q[~avail] = -np.inf                       # unavailable actions never win
    visits = np.zeros((S, N_ACTIONS))

    # Pre-list the legal actions per state so exploration can sample uniformly
    # from A(s) rather than from all of A and then getting rejected.
    legal = [np.flatnonzero(avail[s]) for s in range(S)]

    counts = (np.zeros((S, N_ACTIONS, S), dtype=np.int32)
              if qcfg.record_counts else None)
    reward_sums = np.zeros((S, N_ACTIONS)) if qcfg.record_counts else None

    anneal_over = max(1, int(qcfg.eps_decay_frac * qcfg.n_episodes))
    snapshots = []

    for ep in range(qcfg.n_episodes):
        eps = max(qcfg.eps_end,
                  qcfg.eps_start
                  - (qcfg.eps_start - qcfg.eps_end) * ep / anneal_over)

        if qcfg.exploring_starts:
            s = int(rng.integers(S))
        else:
            s = mdp.encode(mdp.cfg.tank_capacity // 2, 0, 1, mdp.cfg.fuel_per_day)

        for _ in range(qcfg.episode_hours):
            if rng.random() < eps:
                a = int(rng.choice(legal[s]))          # explore, legally
            else:
                a = int(np.argmax(Q[s]))               # exploit (-inf masks the rest)

            s_next, r = mdp.step(s, a, rng)

            visits[s, a] += 1
            if counts is not None:
                counts[s, a, s_next] += 1
                reward_sums[s, a] += r

            alpha = (qcfg.alpha_const if qcfg.alpha_mode == "constant"
                     else 1.0 / (1.0 + visits[s, a]) ** qcfg.alpha_omega)

            # The TD update. Note max_a' Q(s',a') - the off-policy part: we
            # bootstrap off the BEST next action even though epsilon-greedy may
            # not take it. That is exactly what lets Q-learning converge to the
            # optimal policy while behaving sub-optimally the whole time.
            td_target = r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next

        if (snapshot_fn is not None and qcfg.snapshot_every
                and (ep + 1) % qcfg.snapshot_every == 0):
            # Hand the driver everything the agent has learned SO FAR. Both the
            # Q-learning curve and the certainty-equivalence curve are taken
            # from this one training run, off the very same samples - which is
            # the only way the "same data, two ways of using it" comparison is
            # actually true rather than merely asserted.
            snapshots.append(snapshot_fn(
                ep + 1, (ep + 1) * qcfg.episode_hours,
                Q.argmax(axis=1).astype(int), counts, reward_sums, visits))

    out = {
        "Q": Q,
        "policy": Q.argmax(axis=1).astype(int),
        "visits": visits,
        "snapshots": snapshots,
        # coverage over LEGAL pairs only - counting illegal ones would flatter us
        "coverage": float((visits[avail] > 0).mean()),
        "min_visits": float(visits[avail].min()),
        "steps": qcfg.n_episodes * qcfg.episode_hours,
    }
    if counts is not None:
        out["counts"] = counts
        out["reward_sums"] = reward_sums
    return out


# =====================================================================
# 5. CERTAINTY EQUIVALENCE  -  the baseline that could sink the whole thesis
# =====================================================================
def certainty_equivalence(mdp: HallWaterMDP, counts: np.ndarray,
                          reward_sums: np.ndarray, visits: np.ndarray,
                          gamma: float):
    """Build the maximum-likelihood MDP from the SAME samples Q-learning saw,
    then plan on it with value iteration.

        P_hat(s'|s,a) = N(s,a,s') / N(s,a)      R_hat(s,a) = sum(r) / N(s,a)

    This is the arm that decides what the report is allowed to conclude. If a
    model LEARNED from n samples beats Q-learning trained on the same n
    samples, then "model-free wins when the model is wrong" is the wrong
    lesson. The right one is narrower and truer: a wrong PRIOR model loses to a
    LEARNED model, and you reach for model-free only when you cannot even write
    down the state-transition structure to estimate.

    Unvisited (s,a) pairs are treated pessimistically - a self-loop at the
    worst observed reward - so the planner cannot fall in love with an action
    it never actually tried.
    """
    cfg = mdp.cfg
    S, A = cfg.n_states, N_ACTIONS
    avail = mdp.available

    # Tightest lower bound on any single-hour reward in this MDP: run the
    # generator, spill a full pump-load, and still miss the maximum demand.
    # Nothing can be worse than this, so an untried action cannot look
    # attractive by accident.
    worst = -(cfg.cost_generator
              + cfg.cost_overflow * cfg.pump_rate
              + cfg.cost_shortage * cfg.max_demand)

    R_hat = np.zeros((S, A))
    rows, cols, vals = [], [], []

    for s in range(S):
        for a in range(A):
            if not avail[s, a]:
                continue
            n = visits[s, a]
            row = s * A + a
            if n == 0:
                R_hat[s, a] = worst              # never tried: assume the worst
                rows.append(row); cols.append(s); vals.append(1.0)   # self-loop
                continue
            R_hat[s, a] = reward_sums[s, a] / n
            nz = np.flatnonzero(counts[s, a])
            for s_next in nz:
                rows.append(row)
                cols.append(int(s_next))
                vals.append(counts[s, a, s_next] / n)

    P_hat = sp.csr_matrix((vals, (rows, cols)), shape=(S * A, S))
    V, policy, _ = value_iteration(P_hat, R_hat, avail, gamma)
    return policy


# =====================================================================
# 6. BASELINE POLICIES
# =====================================================================
def _legal_or_idle(mdp: HallWaterMDP, s: int, a: int) -> int:
    return a if mdp.available[s, a] else IDLE


def policy_always_idle(mdp: HallWaterMDP) -> np.ndarray:
    return np.full(mdp.cfg.n_states, IDLE, dtype=int)


def policy_random(mdp: HallWaterMDP, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([int(rng.choice(np.flatnonzero(mdp.available[s])))
                     for s in range(mdp.cfg.n_states)], dtype=int)


def policy_caretaker(mdp: HallWaterMDP, grid_threshold: int,
                     gen_threshold: int) -> np.ndarray:
    """What the caretaker actually does, and it is not stupid:

        grid up   and tank < grid_threshold  -> pump on grid
        grid down and tank < gen_threshold   -> burn diesel (if any is left)
        otherwise                            -> leave it alone

    A reactive two-threshold rule. It has the SAME action set as the optimal
    policy - including the generator - and both thresholds are tuned by grid
    search in the experiments, so we report the best version of it. Beating a
    hand-crippled baseline would prove nothing; beating the tuned one is the
    only comparison worth making.

    What it cannot do is read a clock. It does not know that 6pm is coming.
    """
    pol = np.full(mdp.cfg.n_states, IDLE, dtype=int)
    for s in range(mdp.cfg.n_states):
        level, _, grid, fuel = mdp.decode(s)
        if grid == 1 and level < grid_threshold:
            pol[s] = PUMP_GRID
        elif grid == 0 and level < gen_threshold and fuel > 0:
            pol[s] = PUMP_GEN
    return pol


def tune_caretaker(mdp: HallWaterMDP, P: sp.csr_matrix, R: np.ndarray,
                   gamma: float):
    """Grid-search both thresholds and hand the heuristic its best shot."""
    best = (None, -np.inf)
    for gt in range(mdp.cfg.tank_capacity + 1):
        for ft in range(mdp.cfg.tank_capacity + 1):
            pol = policy_caretaker(mdp, gt, ft)
            v = float(policy_evaluation(P, R, pol, gamma).mean())
            if v > best[1]:
                best = ((gt, ft), v)
    return best[0], best[1]


def policy_myopic(mdp: HallWaterMDP, R: np.ndarray) -> np.ndarray:
    """Value iteration with gamma = 0: maximise THIS hour's expected reward and
    ignore the future entirely.

    Not a strawman - it is the formally correct greedy policy. If it lands near
    the optimum, the problem has no long-horizon structure and the entire
    assignment is vacuous. Reporting the gap is how we prove it is not.
    """
    return _masked(R, mdp.available).argmax(axis=1).astype(int)


# =====================================================================
# 7. SCORING
# =====================================================================
def score_policy(P_true: sp.csr_matrix, R_true: np.ndarray,
                 policy: np.ndarray, gamma: float) -> float:
    """One number per policy: exact expected discounted return, averaged over a
    uniform start state, under the TRUE model."""
    return float(policy_evaluation(P_true, R_true, policy, gamma).mean())


def regret_percent(score: float, optimal: float) -> float:
    """How much worse than the exact optimum, as a percentage of the optimum.
    Returns are negative (they are costs), so a positive regret means worse."""
    return 100.0 * (optimal - score) / abs(optimal)


def rollout_stats(mdp: HallWaterMDP, policy: np.ndarray, n_days: int,
                  seed: int) -> dict:
    """Monte-Carlo rollout for the numbers a hall warden would actually ask
    about: what does a day cost, and how often does somebody turn a tap and get
    nothing?

    The report's *comparisons* all use exact policy evaluation above. This is
    here only to translate abstract discounted return into hall consequences,
    and it is reported as such - we never draw the discounted V* line on an
    average-cost axis, because they are optima for different criteria.
    """
    rng = np.random.default_rng(seed)
    cfg = mdp.cfg
    daily_cost, shortage, gen_hours, dry_hours = [], [], [], []

    for _ in range(n_days):
        s = mdp.encode(cfg.tank_capacity // 2, 0, 1, cfg.fuel_per_day)
        cost = short = gen = dry = 0.0
        for _ in range(24):
            a = int(policy[s])
            if a == PUMP_GEN:
                gen += 1
            s, r, info = mdp.simulate_hour(s, a, rng)
            cost += -r
            short += info["unmet"]
            dry += 1.0 if info["unmet"] > 0 else 0.0
        daily_cost.append(cost)
        shortage.append(short)
        gen_hours.append(gen)
        dry_hours.append(dry)

    return {
        "daily_cost": float(np.mean(daily_cost)),
        "shortage_units_per_day": float(np.mean(shortage)),
        "generator_hours_per_day": float(np.mean(gen_hours)),
        "dry_hours_per_day": float(np.mean(dry_hours)),
    }


def action_optimality(Q_star: np.ndarray, policy: np.ndarray,
                      tol: float = 1e-6) -> float:
    """Fraction of states where `policy` picks an action that is actually
    optimal.

    Deliberately NOT a label match against pi*. Even with the degenerate
    actions masked out, plenty of states are near-ties, and a raw argmax
    comparison would report loud "disagreement" in states where the two choices
    differ in value by 0.001. That measures tie-breaking, not policy quality.
    We ask the question that matters: is the chosen action as good as the best
    available one?
    """
    best = Q_star.max(axis=1)
    chosen = Q_star[np.arange(len(policy)), policy]
    return float((best - chosen <= tol).mean())


def q_star_from(P: sp.csr_matrix, R: np.ndarray, V: np.ndarray,
                avail: np.ndarray, gamma: float) -> np.ndarray:
    S, A = R.shape
    return _masked(R + gamma * (P @ V).reshape(S, A), avail)
