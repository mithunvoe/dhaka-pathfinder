"""Tests for the water-tank MDP, value iteration and Q-learning.

The important one is test_model_simulator_parity. Value iteration reads the
transition table; Q-learning samples the simulator. If those two ever disagree
about the same MDP, every VI-vs-QL number in the report is comparing two
different worlds and means nothing. That test is the gate everything else
stands on, so it estimates P and R back out of the simulator and checks them
against the table.
"""

import numpy as np
import pytest

from rl_water_tank import (IDLE, N_ACTIONS, PUMP_GEN, PUMP_GRID, HallWaterMDP,
                           MDPConfig, QLearningConfig, certainty_equivalence,
                           policy_always_idle, policy_caretaker,
                           policy_evaluation, policy_myopic, policy_random,
                           q_learning, regret_percent, score_policy,
                           tune_caretaker, value_iteration)


@pytest.fixture(scope="module")
def mdp():
    return HallWaterMDP(MDPConfig())


@pytest.fixture(scope="module")
def model(mdp):
    return mdp.build_model()


@pytest.fixture(scope="module")
def solved(mdp, model):
    P, R = model
    V, pi, res = value_iteration(P, R, mdp.available, mdp.cfg.gamma)
    return V, pi, res


# ---------------------------------------------------------------- state space
def test_encode_decode_roundtrip(mdp):
    for s in range(mdp.cfg.n_states):
        assert mdp.encode(*mdp.decode(s)) == s


def test_demand_pmf_is_a_distribution(mdp):
    for t in range(24):
        pmf = mdp._demand_pmf(t)
        assert pmf.shape == (mdp.cfg.max_demand + 1,)
        assert np.isclose(pmf.sum(), 1.0)
        assert (pmf >= 0).all()


def test_grid_transition_is_a_distribution(mdp):
    for g in (0, 1):
        for t in range(24):
            assert np.isclose(mdp.grid_transition(g, t).sum(), 1.0)


def test_transition_rows_sum_to_one_for_legal_actions(mdp, model):
    P, _ = model
    sums = np.asarray(P.sum(axis=1)).ravel().reshape(mdp.cfg.n_states, N_ACTIONS)
    assert np.allclose(sums[mdp.available], 1.0)


# ---------------------------------------------------------------- availability
def test_pump_grid_unavailable_during_outage(mdp):
    for s in range(mdp.cfg.n_states):
        _, _, grid, _ = mdp.decode(s)
        assert mdp.available[s, PUMP_GRID] == (grid == 1)


def test_generator_unavailable_without_diesel(mdp):
    for s in range(mdp.cfg.n_states):
        _, _, _, fuel = mdp.decode(s)
        assert mdp.available[s, PUMP_GEN] == (fuel > 0)


def test_idle_is_always_available(mdp):
    assert mdp.available[:, IDLE].all()


# ---------------------------------------------------------------- physics
def test_generator_burns_exactly_one_ration(mdp):
    rng = np.random.default_rng(0)
    s = mdp.encode(5, 10, 1, 2)               # 10:00, not a replenishment hour
    s_next, _, _ = mdp.simulate_hour(s, PUMP_GEN, rng)
    assert mdp.decode(s_next)[3] == 1


def test_diesel_ration_is_redelivered_at_midnight(mdp):
    rng = np.random.default_rng(0)
    s = mdp.encode(5, 23, 1, 0)               # 23:00 with an empty drum
    s_next, _, _ = mdp.simulate_hour(s, IDLE, rng)
    level, hour, _, fuel = mdp.decode(s_next)
    assert hour == 0
    assert fuel == mdp.cfg.fuel_per_day       # the morning delivery


def test_pump_cannot_overfill_the_tank(mdp):
    rng = np.random.default_rng(0)
    cap = mdp.cfg.tank_capacity
    s = mdp.encode(cap, 2, 1, 2)              # already full, night, no demand
    for _ in range(20):
        s_next, _, info = mdp.simulate_hour(s, PUMP_GRID, rng)
        assert mdp.decode(s_next)[0] <= cap
        assert info["overflow"] >= 0


def test_pumping_on_grid_does_nothing_during_an_outage(mdp):
    """The switch is dead. This is precisely why PUMP_GRID is masked out of
    A(s) when g=0 - otherwise it would be a free duplicate of IDLE."""
    rng = np.random.default_rng(1)
    s = mdp.encode(4, 20, 0, 2)               # grid down
    _, _, info = mdp.simulate_hour(s, PUMP_GRID, rng)
    assert info["pump_cost"] == 0.0


# ---------------------------------------------------------------- THE GATE
def test_model_simulator_parity(mdp, model):
    """Estimate P and R back out of the simulator and check them against the
    table value iteration reads.

    Without this, the classic failure mode is silent: the VI table pumps before
    demand and the simulator pumps after it (or one renormalises the truncated
    Poisson and the other clips it), Q-learning quietly converges to a different
    optimum, and you spend a week blaming the learning rate.
    """
    P, R = model
    rng = np.random.default_rng(1234)
    n = 8000                                  # samples per (s, a)

    # A spread of states rather than all 1584 x 3, so the test stays quick:
    # empty/half/full tank, night and both demand peaks, grid up and down, with
    # and without diesel.
    #
    # Hour 23 is in the list for a specific reason. It is the ONLY hour where
    # build_model and simulate_hour take a different branch (_next_fuel refills
    # the drum instead of carrying the remainder forward), so it is the single
    # most likely place for the two faces to drift apart. A gate that skipped it
    # would be checking everything except the thing worth checking. Hour 22 is
    # there as the control: same evening dynamics, no reset.
    probes = [(L, t, g, f)
              for L in (0, 5, 10) for t in (3, 8, 20, 22, 23)
              for g in (0, 1) for f in (0, 2)]

    for level, hour, grid, fuel in probes:
        s = mdp.encode(level, hour, grid, fuel)
        for a in range(N_ACTIONS):
            if not mdp.available[s, a]:
                continue

            counts = np.zeros(mdp.cfg.n_states)
            rewards = np.empty(n)
            for i in range(n):
                s_next, r = mdp.step(s, a, rng)
                counts[s_next] += 1
                rewards[i] = r

            p_hat = counts / n
            p_true = np.asarray(P[s * N_ACTIONS + a].todense()).ravel()
            # Total-variation distance. With at most 14 successors, the expected
            # MC error is ~sqrt(k/2*pi*n) ~ 0.017, so 0.05 is a comfortable bound.
            tv = 0.5 * np.abs(p_hat - p_true).sum()
            assert tv < 0.05, (
                f"transition mismatch at {(level, hour, grid, fuel)}/{a}: TV={tv:.4f}")

            # The reward tolerance must come from the SAMPLING ERROR, not from a
            # magic constant. An empty tank at the morning peak has a reward std
            # near 30 (a shortage costs 20 per unit and demand runs to 6), so a
            # fixed tolerance would either fail there or be useless everywhere
            # else. Four standard errors, with a small floor for the degenerate
            # zero-variance states (a full tank at 3am never goes short).
            r_hat = float(rewards.mean())
            tol = 4.0 * float(rewards.std()) / np.sqrt(n) + 0.02
            assert abs(r_hat - R[s, a]) < tol, (
                f"reward mismatch at {(level, hour, grid, fuel)}/{a}: "
                f"simulator {r_hat:.3f} vs model {R[s, a]:.3f} (tol {tol:.3f})")


# ---------------------------------------------------------------- value iteration
def test_vi_residuals_decay_at_gamma(mdp, solved):
    """The Bellman operator is a gamma-contraction, so the residual has to fall
    by a factor of gamma per sweep once the transient has washed out."""
    _, _, res = solved
    mid = len(res) // 2
    ratios = res[mid + 1: mid + 20] / res[mid: mid + 19]
    assert np.allclose(ratios, mdp.cfg.gamma, atol=0.01)


def test_vi_agrees_with_exact_policy_evaluation(mdp, model, solved):
    """V* from iterating the Bellman optimality operator must equal V^{pi*}
    from solving the linear system directly. Two independent routes, one
    answer - if they disagree, one of them is wrong."""
    P, R = model
    V, pi, _ = solved
    V_exact = policy_evaluation(P, R, pi, mdp.cfg.gamma)
    assert np.abs(V - V_exact).max() < 1e-6


def test_policies_only_choose_legal_actions(mdp, model, solved):
    P, R = model
    _, pi, _ = solved
    checked = [pi, policy_myopic(mdp, R), policy_random(mdp, 0),
               policy_always_idle(mdp), policy_caretaker(mdp, 8, 3)]
    for pol in checked:
        assert mdp.available[np.arange(mdp.cfg.n_states), pol].all()


# ---------------------------------------------------------------- the science
def test_optimal_policy_beats_every_baseline(mdp, model, solved):
    P, R = model
    _, pi, _ = solved
    gamma = mdp.cfg.gamma
    opt = score_policy(P, R, pi, gamma)
    for pol in [policy_myopic(mdp, R), policy_random(mdp, 0),
                policy_always_idle(mdp), policy_caretaker(mdp, 8, 3)]:
        assert score_policy(P, R, pol, gamma) < opt


def test_lookahead_is_actually_necessary(mdp, model, solved):
    """If the myopic (gamma=0) policy were near-optimal, this MDP would have no
    long-horizon structure and the whole assignment would be vacuous. It is not:
    greedy loses badly. This test is what entitles us to say so."""
    P, R = model
    _, pi, _ = solved
    gamma = mdp.cfg.gamma
    opt = score_policy(P, R, pi, gamma)
    myopic = score_policy(P, R, policy_myopic(mdp, R), gamma)
    assert regret_percent(myopic, opt) > 50.0


def test_optimal_policy_sometimes_pumps_at_a_short_term_loss(mdp, model, solved):
    """Anticipation, made concrete: there must exist a state where pumping is
    strictly worse THIS hour and the optimal policy pumps anyway."""
    P, R = model
    _, pi, _ = solved
    found = any(
        pi[s] == PUMP_GRID and R[s, PUMP_GRID] < R[s, IDLE] - 1e-9
        for s in range(mdp.cfg.n_states)
    )
    assert found, "no state where pi* accepts a worse immediate reward"


def test_q_learning_learns_something_useful(mdp, model, solved):
    P, R = model
    _, pi, _ = solved
    gamma = mdp.cfg.gamma
    opt = score_policy(P, R, pi, gamma)

    ql = q_learning(mdp, QLearningConfig(n_episodes=4000), gamma, seed=1)
    ql_score = score_policy(P, R, ql["policy"], gamma)
    rand_score = score_policy(P, R, policy_random(mdp, 0), gamma)

    assert ql_score > rand_score            # learned better than nothing
    assert ql_score < opt                   # but did not beat the exact optimum
    assert ql["coverage"] > 0.95            # exploring starts really do cover A(s)


def test_certainty_equivalence_beats_q_learning_on_the_same_samples(mdp, model, solved):
    """The report's central claim, as a test. Given exactly the same transitions,
    estimating a model and planning on it extracts far more than tabular
    Q-learning does. If this ever flips, the report's conclusion is wrong and we
    want to know immediately."""
    P, R = model
    _, pi, _ = solved
    gamma = mdp.cfg.gamma
    opt = score_policy(P, R, pi, gamma)

    ql = q_learning(mdp, QLearningConfig(n_episodes=4000, record_counts=True),
                    gamma, seed=1)
    ce = certainty_equivalence(mdp, ql["counts"], ql["reward_sums"],
                               ql["visits"], gamma)

    ql_regret = regret_percent(score_policy(P, R, ql["policy"], gamma), opt)
    ce_regret = regret_percent(score_policy(P, R, ce, gamma), opt)
    assert ce_regret < ql_regret


def test_exploring_starts_matter(mdp, model, solved):
    """Turning them off should measurably hurt: without random resets the agent
    never sees large parts of the state space, so its argmax there is garbage."""
    P, R = model
    _, pi, _ = solved
    gamma = mdp.cfg.gamma

    on = q_learning(mdp, QLearningConfig(n_episodes=3000,
                                         exploring_starts=True), gamma, seed=2)
    off = q_learning(mdp, QLearningConfig(n_episodes=3000,
                                          exploring_starts=False), gamma, seed=2)
    assert on["coverage"] > off["coverage"]
    assert score_policy(P, R, on["policy"], gamma) > \
           score_policy(P, R, off["policy"], gamma)


def test_q_learning_is_reproducible(mdp):
    gamma = mdp.cfg.gamma
    a = q_learning(mdp, QLearningConfig(n_episodes=500), gamma, seed=42)
    b = q_learning(mdp, QLearningConfig(n_episodes=500), gamma, seed=42)
    assert np.array_equal(a["policy"], b["policy"])


def test_tuned_caretaker_is_the_best_version_of_itself(mdp, model):
    """We beat the TUNED threshold rule, not a hand-crippled one. Confirm the
    tuner really did find the best thresholds."""
    P, R = model
    gamma = mdp.cfg.gamma
    (gt, ft), best = tune_caretaker(mdp, P, R, gamma)
    for probe in [(3, 1), (5, 2), (10, 0)]:
        assert score_policy(P, R, policy_caretaker(mdp, *probe), gamma) <= best + 1e-9
