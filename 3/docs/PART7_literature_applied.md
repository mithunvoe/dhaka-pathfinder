# Part 7 — What we took from the literature, and where it shows up

> The honest question is not "did you cite papers", it is "would this project be
> different if you had not read them". This file answers that, one paper at a
> time, with the line of code or the experiment that changed. Where a paper made
> a claim we could test, we tested it, and we say what happened — including the
> two places where we got it wrong first.

---

## 7.1 The short version

| Paper | What we took from it | Where it lives | Did we test its claim? |
|---|---|---|---|
| Kennedy & Eberhart 1995 | the velocity update itself | `pso_wifi_placement.py` §3 | — (it is the algorithm) |
| Shi & Eberhart 1998 | inertia weight `w`: 0.9 → 0.4 | `optimize()` | yes, via the diversity curve |
| **Kennedy & Mendes 2002** | **swarm topology is a variable, not a given** | `topology` / `_social_attractor()` | **yes — and it corrected us** |
| Wolpert & Macready 1997 (NFL) | you must *demonstrate* fit to the landscape, not assert it | the entire ablation | it is why the ablation exists |
| Rappaport 2002 | log-distance path loss + wall attenuation | `WifiFloorProblem.fitness()` | — (it is the objective) |
| Bellman 1957 / Howard 1960 | value iteration; the contraction bound | `value_iteration()` | yes — measured rate = 0.9900 = γ |
| Watkins & Dayan 1992 | the TD update; the `max` that makes it off-policy | `q_learning()` | — (it is the algorithm) |
| Jaakkola et al. 1994 / Tsitsiklis 1994 | "every (s,a) infinitely often" | exploring starts | yes — off = 76.7% vs on = 25.1% |
| **Even-Dar & Mansour 2003** | α = 1/(1+n)^0.7, and *why constant α cannot work* | `alpha_omega` | **yes — constant α stalls at 50.4%** |
| **Agarwal et al. 2020 + Li et al. 2024** | **the sample-complexity gap that explains our headline result** | `certainty_equivalence()` | yes — predicted gap, observed gap |
| Sutton 1990 (Dyna) | the model-free/model-based boundary is a spectrum | framing of Part B | noted as future work |

Two of these did not merely inform the project. They **changed a result we had
already written down.** Those two are worth the rest of this document.

---

## 7.2 Kennedy & Mendes (2002) — the paper that caught us over-claiming

**What we had.** Our first ablation was binary: communication on (every particle
pulled towards the single global best) or off (`c2 = 0`). On, 89.91. Off, 87.30 —
indistinguishable from random search. Communication matters. Done.

**What the paper says.** Kennedy & Mendes systematically compare 70 different
swarm communication topologies, and their finding is narrower and more useful than
the folklore:

> *"greater connectivity speeds up convergence, though it does not tend to improve
> the population's ability to discover global optima"*

Concretely, they measure the fully-connected `gbest` swarm hitting their criterion
in a median of **404.8 iterations but only 85% of the time**, against a ring
`lbest` swarm needing **755.5 iterations but succeeding 94% of the time**.

They also warn about the other end, which is exactly our `c2 = 0` row:

> *"inhibiting communication too much results in inefficient allocation of trials,
> as individual particles wander cluelessly through the search space"*

**What we did with it.** We implemented a ring topology (`ring_k` neighbours each
side, information walks around the circle one hop per iteration instead of
teleporting) and re-ran the ablation at the same 3,030-evaluation budget.

**What we got, and how we got it wrong first.** Our first pass reported that the
ring was simply *better* — mean 89.99 against gbest's 89.91. A unit test asserting
that failed on a different set of seeds, which was the correct outcome. We then
tested it properly, on 30 paired seeds:

| | mean | sd | stagnates at |
|---|---|---|---|
| fully connected (gbest) | 89.831 | 0.317 | iteration **83** / 100 |
| ring, k = 1 | 89.961 | **0.129** | iteration **100** / 100 |

- **Wilcoxon on the mean: p = 0.92.** Not significant. The ring does **not** find a
  better optimum.
- **F-test on the variance: 6× smaller, p = 6 × 10⁻⁶.** Highly significant.
- The fully-connected swarm **gives up at iteration 83**. The ring is still
  improving when the budget runs out.

That is the paper's claim, reproduced: more connectivity converges *faster*, not
*better*. The gain from the ring is **reliability**, not peak performance. Our first
write-up said "better" and the statistics did not support it.

**The bit we would have got wrong if we had only skimmed it.** The paper does *not*
recommend the ring. It ranks ring-with-self **64th of the 70** topologies tested
("slow and inaccurate") and recommends the **von Neumann** lattice instead. So we
must not say "the literature says use a ring". We say the narrower true thing: the
literature predicts a speed-versus-reliability trade between dense and sparse
topologies, and on our landscape that trade appears exactly as described. Trying
von Neumann is the obvious next experiment and we have not run it.

---

## 7.3 Even-Dar & Mansour (2003) — a prediction we could falsify

The Robbins–Monro conditions require Σα = ∞ and Σα² < ∞. A **constant** α satisfies
the first and fails the second, so it provably converges only to a *neighbourhood*
of Q\*, never to Q\* itself. A polynomial rate α_n = 1/(1+n)^ω with ω ∈ (0.5, 1]
satisfies both.

This is a falsifiable prediction about *our* problem, so we put both in the sweep
rather than just citing the theorem:

| α schedule | regret |
|---|---|
| polynomial, ω = 0.7 | **25.1%** |
| polynomial, ω = 0.5 | 28.2% |
| constant α = 0.10 | 50.4% |
| constant α = 0.50 | 57.9% |

The constant-α runs stall exactly as the theory says they must. We did not have to
take the theorem on faith, and neither does the reader.

---

## 7.4 Agarwal et al. (2020) and Li et al. (2024) — why our headline result happens

This is the one that turned an empirical curiosity into an explanation.

**Our result.** Take the 720,000 transitions Q-learning consumed, estimate the
maximum-likelihood model P̂ from them, and plan on it. That *certainty-equivalence*
planner reaches **1.43%** regret against the Q-learner's **15.13%** — a factor of ten,
on identical data. We could describe this but not explain it.

**The near-miss.** Our first instinct was to cite Azar, Munos & Kappen (2013), which
proves a minimax bound on model-based value iteration. When we checked what it
actually proves, it does **not** support our claim: its lower bound is
information-theoretic and binds *every* algorithm, model-free included. Citing it
for a model-based-beats-model-free claim would have been wrong, and an examiner who
knows the paper would have said so.

**The papers that do support it.** The separation is real, and it is sharper than we
expected:

- **Agarwal, Kakade & Yang (2020)** prove the *plug-in* (certainty-equivalence)
  estimator — build P̂, plan optimally in the empirical MDP — is **minimax optimal**,
  achieving Θ̃(|S||A| / ((1−γ)³ε²)).
- **Li, Cai, Chen, Wei & Chi (2024)** settle vanilla synchronous Q-learning at
  Θ̃(|S||A| / ((1−γ)⁴ε²)) and state plainly that this "unveils the strict
  sub-optimality of Q-learning when |A| ≥ 2".

So certainty-equivalence sits at 1/(1−γ)³ and vanilla tabular Q-learning is stuck at
1/(1−γ)⁴ — **a full factor of 1/(1−γ) worse in the effective horizon.**

**Now put our numbers in.** We chose γ = 0.99, so 1/(1−γ) = **100**. The theory
predicts our Q-learner should be roughly two orders of magnitude behind in sample
efficiency, purely because of the horizon. That is not a curiosity about our hall.
It is a known, proved separation, and our 1.43%-versus-15.13% is what it looks like
from the inside.

**The caveat we must not drop.** The separation is against *vanilla* Q-learning, not
model-free RL in general. Variance-reduced Q-learning variants do reach the optimal
rate. So the correct sentence is "tabular Q-learning is provably sample-inefficient
here", **not** "model-free learning is worse". We say the first.

**The mechanism, in one line.** A Q-learning update spends a transition on one cell
of the table and discards it. A learned model *stores* it, and every subsequent
Bellman backup re-reads it — so one sample propagates across the whole state space
instead of into the single cell where it landed.

---

## 7.5 Wolpert & Macready (1997) — the paper that shaped the method, not the code

No Free Lunch says that averaged over all objective functions, every optimiser is
identical. The practical consequence is that "PSO is good" is not a claim you can
make; only "PSO is good *on this landscape*, and here is the evidence" is.

That single idea is why Part A is built the way it is. It is why we do not report
"PSO found a good layout" and stop; it is why there is a matched-budget random
search, a matched-budget grid search, a swarm-size sweep, a communication ablation
and a topology comparison. Every one of those exists to answer "compared to what?".

It is also why the Part B result survives. We went looking for evidence that
model-free learning wins, ran the experiment that could embarrass us, and it did.
Reporting that is the same discipline.

---

## 7.6 What we read and did not use

Being straight about this is part of the point.

- **The twelve applications in [PART2](PART2_applications.md)** are a genuine
  literature survey with verified citations, but not one of them changed a design
  decision here. They are context, not input. If we were doing it again we would
  mine them for *encoding* tricks (several of those papers are interesting precisely
  because of how they encode a constraint into the representation rather than
  penalising it) and see whether our AP-position encoding could absorb the
  separation constraint structurally instead of as a penalty term.
- **The pump-scheduling RL papers** (Hajgató et al. 2020; Hu et al. 2023; Pei et al.
  2025) all use deep RL on large water-distribution networks. We deliberately went
  the other way — small enough that value iteration gives exact ground truth, so
  every other method can be scored against the true optimum rather than against
  another approximation. That is a design decision *informed* by them, but it is a
  decision to do the opposite.
- **Clerc & Kennedy (2002)** — the constriction factor χ = 0.7298 is an alternative
  to the inertia weight, and Eberhart & Shi (2000) found it performs better on their
  benchmarks (while still keeping velocity clamping, contrary to a common
  misreading). We use the inertia weight because it is what the course teaches and
  what the slide shows. Comparing the two on our landscape is a clean experiment we
  did not run.
- **Sutton (1990), Dyna** — Dyna interleaves real experience with simulated experience
  from a learned model, and sits exactly on the boundary our Part B is about. Our
  certainty-equivalence arm is the batch, one-shot version of the same idea. Dyna-Q
  is the incremental version and would be the natural third arm.

---

## 7.7 If he asks "which paper actually changed your mind?"

Two, and say both.

**Kennedy & Mendes (2002)** turned our communication ablation from a yes/no question
into a "how much, and in what structure" question — and then corrected us when we
over-read our own numbers. The mean difference between the ring and the fully
connected swarm is not significant. The variance difference is, by six times. We had
written "better" and had to change it to "steadier".

**Li et al. (2024)** with **Agarwal et al. (2020)** gave us the reason our headline
result happens at all. We had the measurement — a learned model beats Q-learning by
ten times on identical samples — and no explanation. The explanation is a proved
sample-complexity separation of 1/(1−γ), which at our γ = 0.99 is a factor of a
hundred. We also learned, checking it, that the paper we *nearly* cited for this
does not support the claim, which is its own lesson about reading past the abstract.
