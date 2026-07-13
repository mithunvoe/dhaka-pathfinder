# Part 1 — Deep Foundation & Core Swarm Pillars

> Self-contained theory of population-based metaheuristics, written to
> be defended line-by-line in a viva and to seed the arXiv "Learning Journey"
> report. Notation is fixed once in §1.1 and reused throughout. Every update
> equation is attributed to its seminal source.

---

## 1.1 What a population-based metaheuristic *is* (formal template)

### 1.1.1 The optimization problem

We are given a search space $\mathcal{S}$ and an objective (fitness) function

$$
f : \mathcal{S} \rightarrow \mathbb{R}, \qquad
\text{find } x^\star \in \arg\min_{x \in \mathcal{S}} f(x)
$$

(maximization is the sign-flip $f \mapsto -f$; our Lab-3 code maximizes coverage, so we negate mentally where needed). $\mathcal{S}$ may be continuous ($\mathcal{S}\subseteq\mathbb{R}^n$, bounded by a box $[\ell,u]$), combinatorial (permutations, binary strings $\{0,1\}^n$), or mixed. We assume only that $f$ can be **evaluated pointwise** — an *oracle* or *black box*. We do **not** assume $f$ is convex, differentiable, continuous, or even given in closed form.

A **metaheuristic** is a problem-independent, stochastic, iterative strategy that samples $\mathcal{S}$, guided only by the fitness values it has observed, to drive the best-so-far sample toward $x^\star$. A **population-based** metaheuristic maintains not one incumbent but a *set* (population) of $\mu$ candidate solutions simultaneously.

### 1.1.2 State, operators, and the generic template

Let the **population** at iteration $t$ be

$$
P_t = \big(x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(\mu)}\big) \in \mathcal{S}^{\mu}.
$$

Every algorithm in this dossier (GA, PSO, ACO, ABC, FA, GWO, CS, WOA) is an instance of the same three-operator loop:

1. **Evaluate** — compute $f(x_t^{(i)})$ for all $i$ (the only place $f$ is touched; the *budget* is counted here).
2. **Vary** — produce candidate offspring/moves by a stochastic operator $\mathcal{V}$ that reads the current population (and possibly external memory $M_t$ such as pheromone or personal bests):
   $$
   \tilde P_t \sim \mathcal{V}(\,\cdot \mid P_t, M_t).
   $$
   $\mathcal{V}$ is where the algorithms differ: recombination + mutation (GA), velocity update (PSO/FA), pheromone-biased construction (ACO), neighbour perturbation (ABC), leader-guided steps (GWO), Lévy flights (CS), spiral/encircling (WOA).
3. **Select** — choose the survivors $P_{t+1}$ from $P_t \cup \tilde P_t$ by a rule $\mathcal{Q}$ biased toward lower $f$ (fitter):
   $$
   P_{t+1} = \mathcal{Q}(P_t, \tilde P_t).
   $$

```
initialise P_0 ~ Uniform(S),  M_0 = empty
evaluate f(P_0)
for t = 0, 1, 2, ... until stop:
    P~ ← Vary(P_t, M_t)          # stochastic reproduction / movement
    evaluate f(P~)               # <-- one "generation" of budget
    P_{t+1} ← Select(P_t, P~)    # fitness-biased survival
    M_{t+1} ← UpdateMemory(...)  # pheromone / pbest / gbest / leaders
return best-so-far x*
```

The engine is **variation ⊕ selection**: variation *injects entropy* (proposes unseen points), selection *removes entropy* (concentrates probability mass on good regions). Optimization is the controlled tension between the two — this is the exploration/exploitation axis formalized in §1.4.

### 1.1.3 The Markov-chain view (why this is analyzable)

Because $P_{t+1}$ depends only on $(P_t, M_t)$ and fresh randomness, the sequence of populations is a **Markov chain** on the (finite or measurable) state space $\Omega = \mathcal{S}^\mu \times \mathcal{M}$:

$$
\Pr\big(P_{t+1} = q \mid P_t = p, P_{t-1}, \dots\big) = \Pr\big(P_{t+1} = q \mid P_t = p\big) =: K(p, q),
$$

where $K$ is the transition kernel induced by $\mathcal{V}$ and $\mathcal{Q}$. For a finite $\mathcal{S}$ (e.g. bitstrings), $K$ is a stochastic matrix and the whole apparatus of Markov-chain theory (irreducibility, ergodicity, absorbing states) applies — this is exactly how convergence is *proved* (§1.6). Two structural facts follow immediately:

- If mutation assigns **strictly positive probability** to every point of $\mathcal{S}$ from every state (global reachability), the chain can never be permanently trapped: $\Pr(\exists\, t: x_t^{(i)} = x^\star) \to 1$. This is the *global-search property*.
- If selection is **elitist** (the best-so-far is never discarded), the best-so-far fitness is a monotone, bounded process and therefore converges (Rudolph 1994; §1.6.2).

### 1.1.4 Generational vs. steady-state

- **Generational**: the entire population is replaced each iteration ($P_{t+1}$ built from $\tilde P_t$). Canonical GA, PSO, WOA, GWO, FA are generational.
- **Steady-state**: only one/few individuals are replaced per step (e.g. replace-worst). ABC is effectively steady-state per food source; ACO updates pheromone incrementally.
- **Elitism** is an orthogonal switch: retain the top-$k$ verbatim. It converts a merely *exploratory* chain into a *convergent* one (§1.6.2) at the cost of selection pressure (§1.5). Our PSO keeps `gbest`, an implicit elitism of size 1.

---

## 1.2 Why and when they beat gradient-based and exact solvers

### 1.2.1 What gradient methods assume — and when it breaks

Gradient descent, Newton, quasi-Newton (L-BFGS), and conjugate-gradient all update

$$
x_{k+1} = x_k - \eta_k\, H_k^{-1}\, \nabla f(x_k)
$$

and therefore **require** (i) $f$ differentiable with a computable (or finite-difference-approximable) $\nabla f$, and (ii) local gradient information to be *informative* about the global optimum. They are *local* methods with (super)linear convergence **on convex or locally convex, smooth** landscapes. They fail — provably or practically — when:

- **No gradient exists**: $f$ is discontinuous, piecewise-constant, or defined by a simulation/`max`/threshold. *Our Lab-3 objective* uses $\max_j \mathrm{RSSI}_{ij}$ (a kink), a logistic soft-threshold, and integer **wall-crossing counts** — the last makes $f$ literally piecewise-constant in AP position, so $\nabla f = 0$ almost everywhere and undefined on the kinks. A gradient step has *nothing to descend*.
- **Non-convex / multimodal**: many local minima; gradient descent converges to whichever basin it starts in. Restarts help only $\propto$ (number of basins)$^{-1}$.
- **Ill-conditioned or deceptive**: the gradient points *away* from the global optimum (deceptive functions, §1.7.4).
- **Black-box, expensive, noisy**: finite-difference gradients cost $n{+}1$ evaluations per step and amplify noise.

### 1.2.2 What exact/deterministic solvers assume — and when it breaks

Exact methods (branch-and-bound, MILP, dynamic programming, exhaustive enumeration, A\*/backtracking as in Labs 1–2) return a *provably* optimal solution but with worst-case cost that **explodes**:

- Combinatorial spaces grow factorially/exponentially: $k$-of-$G$ placement has $\binom{G}{k}$ options; a permutation of $n$ has $n!$. Our Grid-Search baseline is exactly this — a $\binom{25}{3}=2300$-way enumeration of a *coarse* grid; a grid fine enough to rival a continuous optimizer needs $\binom{G}{k}\to\infty$. This is the **curse of dimensionality** made concrete.
- Continuous global optimization is NP-hard in general; exact solvers need special structure (linearity, convexity, submodularity) that black-box $f$ does not provide.

### 1.2.3 The regime where metaheuristics win — and where they lose (honest)

Population metaheuristics are the rational choice when **all** of these hold: no reliable gradient; non-convex/multimodal or combinatorial; black-box or simulation-based $f$; a "good enough, fast enough" near-optimum is acceptable in place of a certificate of optimality; and the per-evaluation cost permits thousands of samples.

They **lose** — and one must say so in a viva — when:

- $f$ is **convex and smooth**: gradient/Newton converge (super)linearly to the *global* optimum; a swarm is slower and gives no certificate.
- $f$ is **low-dimensional and cheap**: dense grid or multi-start local search dominates.
- The problem has **exploitable structure** (LP/MILP/convex/DP/submodular): an exact solver returns a *provably* optimal answer with a bound; a metaheuristic cannot.
- Evaluations are **extremely expensive** ($\sim$minutes each): Bayesian optimization / surrogate models use the budget better than a 30-particle swarm.

Metaheuristics buy **robustness to landscape pathology** at the price of **optimality guarantees**. That trade is the thesis of this lab.

---

## 1.3 The No Free Lunch (NFL) theorem

**Source.** D. H. Wolpert & W. G. Macready, "No Free Lunch Theorems for Optimization," *IEEE Transactions on Evolutionary Computation* 1(1):67–82, 1997, DOI [10.1109/4235.585893](https://doi.org/10.1109/4235.585893).

**Setup.** Fix finite $\mathcal{S}$ and finite value set $\mathcal{Y}$. An algorithm $a$ generates a time-ordered sample (trace) $d_m = \big((x_1,y_1),\dots,(x_m,y_m)\big)$ of $m$ distinct visited points; let $d_m^y$ be the sequence of *fitness values* seen. Performance is any functional $\Phi(d_m^y)$ (e.g. best value found).

**Statement (NFL-1).** For any two algorithms $a_1, a_2$, averaged **uniformly over all possible objective functions** $f:\mathcal{S}\to\mathcal{Y}$,

$$
\sum_{f} \Pr\big(d_m^y \mid f, m, a_1\big) \;=\; \sum_{f} \Pr\big(d_m^y \mid f, m, a_2\big).
$$

Consequently $\sum_f \Phi(d_m^y \mid f,m,a_1) = \sum_f \Phi(d_m^y \mid f,m,a_2)$ for **every** performance measure $\Phi$: *summed over all functions, all non-revisiting algorithms are identical.* Any above-average performance on one class of $f$ is paid for by exactly equal below-average performance on the complementary class. (An analogous NFL holds for supervised learning.)

**Why it is true, in one line.** The uniform average over all $f$ makes the histogram of unseen fitness values independent of *how* points were chosen; an algorithm's cleverness is about *which point to sample next*, but under the uniform prior the value at any unseen point is equally likely to be anything, so ordering gains nothing.

**Implications for algorithm selection.**

1. **There is no universally best optimizer.** Claims that "algorithm X dominates" are meaningful only *relative to a function class*. Benchmarking on a biased suite (e.g. only separable, smooth functions) and generalizing is an NFL fallacy.
2. **Performance is bought with matched priors.** An algorithm beats another *iff* its inductive bias matches the structure of the target functions. GWO's leader-hierarchy exploitation wins on smooth low-D redundant landscapes (robot IK, Part 2); PSO's continuous drift wins on our coverage surface. *Match the operator to the landscape.*
3. **Real problems are not uniformly random.** NFL's uniform prior includes mostly incompressible, structureless functions. Real objectives have exploitable regularity (locality, gradient-of-fitness correlation, decomposability). Metaheuristics work *in practice* precisely because they encode weak, broadly-correct priors (nearby solutions have correlated fitness — §1.7.2).
4. **The engineering job is choosing a bias that fits the problem; no algorithm wins everywhere.** This is what justifies the "choose the algorithm to fit the problem" methodology of Part 4.

**Caveat (sharpened NFL).** Later work (e.g. Igel & Toussaint 2003; Auger & Teytaud 2010) shows NFL holds exactly only for sets of functions **closed under permutation** of the domain; over structured subsets some algorithms *do* dominate. This does not rescue a "best algorithm," but it explains why structure-exploiting heuristics beat blind search on real data.

---

## 1.4 Exploration vs. exploitation (diversification vs. intensification)

### 1.4.1 Definitions

- **Exploration / diversification**: sampling far from current incumbents to discover new basins of attraction; increases population **diversity** and the chance of escaping local optima; slows refinement.
- **Exploitation / intensification**: concentrating samples around the best-known solutions to refine them; accelerates convergence; risks **premature convergence** to a local optimum if diversity dies first.

Formally, let $D(P_t)$ be a diversity measure, e.g. mean distance to the population centroid $\bar x_t$:

$$
D(P_t) = \frac{1}{\mu}\sum_{i=1}^{\mu} \big\lVert x_t^{(i)} - \bar x_t \big\rVert_2 .
$$

Healthy search shows $D(P_t)$ **high early** (exploration) decaying **monotonically** to $\approx 0$ (exploitation/convergence). *Our Lab-3 `diversity.png` plots exactly this collapse* — it is empirical evidence the balance was tuned correctly, not left at library defaults.

### 1.4.2 The balance is a schedule, and every algorithm has a knob

The state of the art controls the balance by **annealing** a single parameter from "explorative" to "exploitative" over the run:

| Algorithm | Balance mechanism | Explorative $\to$ exploitative |
|---|---|---|
| PSO | inertia weight $w$ | $w$ linearly $0.9 \to 0.4$ (Shi & Eberhart 1998); large $w$ = momentum/roaming, small $w$ = local damping |
| GA | mutation rate $p_m$, selection pressure | high $p_m$/low pressure early; anneal $p_m$ down |
| GWO | coefficient $a$ | $a$ linearly $2 \to 0$; $\lvert A\rvert{>}1$ diverge (explore), $\lvert A\rvert{<}1$ converge (exploit) |
| WOA | $a$ (same) + spiral/encircle switch $p$ | $a:2\to0$; probabilistic switch keeps both modes |
| ACO | evaporation $\rho$ | high $\rho$ forgets, keeps exploring; low $\rho$ locks in trails |
| FA | randomization $\alpha$, absorption $\gamma$ | anneal $\alpha$; $\gamma$ sets attraction range ($\gamma\to0$ global, $\gamma\to\infty$ local) |
| CS | discovery prob. $p_a$, Lévy scale $\alpha$ | Lévy heavy tails give rare long explorative jumps throughout |
| ABC | `limit` (abandonment) | scouts re-inject exploration when a source stagnates |

Our PSO uses the inertia schedule $w_t = w_{\max} - (w_{\max}-w_{\min})\,t/(T-1)$ — the direct answer to "how did you avoid premature convergence?"

---

## 1.5 Diversity maintenance & premature convergence

**Premature convergence** = the population collapses onto a single (sub-optimal) basin while budget remains; $D(P_t)\to 0$ *before* $x^\star$ is found, after which variation can no longer escape (crossover of near-identical parents is a no-op; PSO velocities $\to 0$). Signs (from the course slides): population becomes uniform; no best-fitness improvement over many generations. Mechanisms that counter it:

- **Mutation / immigration.** Non-zero mutation guarantees global reachability (§1.1.3). Random *immigration* injects fresh individuals when stagnation is detected — a hard reset of diversity.
- **Controlled elitism.** Keep only a *small* elite fraction; retaining too many clones the population.
- **Selection-pressure control.** Tournament size, rank-based instead of fitness-proportional selection (bounds the takeover rate so one super-individual cannot dominate in a few generations).
- **Niching / fitness sharing** (Goldberg & Richardson 1987). Penalize crowding so multiple optima coexist. Shared fitness:
  $$
  f'_i = \frac{f_i}{\sum_{j=1}^{\mu} \mathrm{sh}(d_{ij})}, \qquad
  \mathrm{sh}(d) = \begin{cases} 1 - (d/\sigma_{\text{share}})^{\alpha}, & d < \sigma_{\text{share}} \\ 0, & d \ge \sigma_{\text{share}} \end{cases}
  $$
  where $d_{ij}=\lVert x_i - x_j\rVert$ and $\sigma_{\text{share}}$ is the niche radius. Individuals in a crowd share (dilute) their fitness, so selection spreads them across niches.
- **Crowding / restricted replacement** (De Jong; Deb & Goldberg 1989): offspring replace the *most similar* parent, preserving spatial spread.
- **Velocity clamping / re-init** (PSO): clamp $\lvert v\rvert \le v_{\max}$ to stop explosion (over-exploration) *and* GCPSO-style re-randomization of `gbest` to stop stagnation (over-exploitation).

Our optimizer relies on: velocity clamping ($v_{\max}=0.2\times$range), the inertia schedule, and the swarm's own stochastic $r_1,r_2$; the `diversity.png` curve is the evidence that these keep $D(P_t)>0$ through the exploration phase.

---

## 1.6 Mathematical definitions of convergence

### 1.6.1 What "convergence" means

Let $x_t^{\text{best}} = \arg\min_{i,\,s\le t} f(x_s^{(i)})$ be the best-so-far and $f^\star = f(x^\star)$.

- **Convergence in value (in probability):** $\forall \varepsilon>0,\ \Pr\big(f(x_t^{\text{best}}) - f^\star > \varepsilon\big) \xrightarrow{t\to\infty} 0.$
- **Almost-sure convergence:** $\Pr\big(\lim_{t\to\infty} f(x_t^{\text{best}}) = f^\star\big) = 1.$
- **Global-search guarantee:** the (weaker, prerequisite) property that every region of positive measure is sampled infinitely often, so $x^\star$ is *reached* with probability 1. Global reachability + elitism $\Rightarrow$ a.s. convergence.

Note the distinction between converging to a **fixed point** (the swarm agrees — *stagnation*, which may be sub-optimal) and converging to the **global optimum** (the object we actually want). Conflating them is a classic viva trap (§Part 3).

### 1.6.2 Elitist GA converges; canonical GA does not

**Source.** G. Rudolph, "Convergence Analysis of Canonical Genetic Algorithms," *IEEE Transactions on Neural Networks* 5(1):96–101, 1994, DOI [10.1109/72.265964](https://doi.org/10.1109/72.265964).

Modeling the GA as a homogeneous Markov chain on $\{0,1\}^{\ell\mu}$, Rudolph proves: the **canonical GA** (fitness-proportional selection + crossover + bit mutation, *no elitism*) does **not** converge to the global optimum — its best-so-far can oscillate because selection can lose the optimum once found. Adding **elitism** (maintain the best individual across generations) yields a chain whose best-so-far is monotone and **converges to the global optimum with probability 1**. Takeaway: *the convergence guarantee comes from elitism + positive mutation, not from crossover.*

### 1.6.3 PSO can stagnate at a non-optimum

**Source.** F. van den Bergh, "An Analysis of Particle Swarm Optimizers," PhD thesis, Univ. of Pretoria, 2001; and F. van den Bergh & A. P. Engelbrecht, "A study of particle swarm optimization particle trajectories," *Information Sciences* 176(8):937–971, 2006, DOI [10.1016/j.ins.2005.02.003](https://doi.org/10.1016/j.ins.2005.02.003).

Standard `gbest` PSO is **not** a global (nor even local) optimizer: once all particles coincide with `gbest`, the cognitive and social terms vanish and velocity decays as $v\!\to\!0$, so the swarm halts at `gbest` **regardless of whether it is a local minimum** — pure *stagnation*. The **Guaranteed Convergence PSO (GCPSO)** fix forces the global-best particle to sample a shrinking random neighbourhood, restoring convergence to a local minimizer. Clerc & Kennedy (2002, DOI [10.1109/4235.985692](https://doi.org/10.1109/4235.985692)) give the *constriction* analysis that yields stable trajectories via $\chi\approx0.7298,\ c_1{=}c_2{=}2.05$; the inertia form $w{\in}[0.4,0.9],\ c_1{=}c_2{\approx}1.49$ we use is its practical equivalent.

The honest position for the viva: PSO offers **no** guarantee of global optimality; we mitigate stagnation empirically (diversity monitoring, multiple seeds, matched-budget baselines, Wilcoxon test) rather than claim a proof we do not have.

---

## 1.7 Topology of the fitness landscape

A landscape is the triple $(\mathcal{S}, \mathcal{N}, f)$ where $\mathcal{N}$ is a neighbourhood/move structure. Its geometry dictates which operator wins (NFL, §1.3).

### 1.7.1 Modality
The number of local optima. **Unimodal**: one optimum (gradient/hill-climbing suffices). **Multimodal**: many (population search + niching needed). Formally $x$ is a local minimum if $f(x)\le f(y)\ \forall y\in\mathcal{N}(x)$. Our coverage surface is multimodal: distinct AP configurations (which AP covers which wing) are separate basins of comparable quality.

### 1.7.2 Ruggedness (fitness correlation)
How fast fitness de-correlates along a walk. Take a random walk $x_0,x_1,\dots$ over $\mathcal{N}$ and the series $f_t=f(x_t)$; its **autocorrelation** at lag $s$ is

$$
\rho(s) = \frac{\mathbb{E}\big[(f_t-\bar f)(f_{t+s}-\bar f)\big]}{\operatorname{Var}(f)},
\qquad
\ell = -\frac{1}{\ln \lvert \rho(1)\rvert}
$$

(Weinberger 1990, *Biological Cybernetics* 63:325–336). Large **correlation length** $\ell$ $\Rightarrow$ smooth, "nearby solutions have similar fitness" $\Rightarrow$ local search is informative. Small $\ell$ $\Rightarrow$ rugged, needle-like $\Rightarrow$ near-random, population diversity is essential. Ruggedness is *why* metaheuristics encode the locality prior NFL says you need.

### 1.7.3 Neutrality
Extended regions of **equal** fitness — *neutral networks* — over which selection is blind and only drift moves the population (Kimura's neutral theory, ported to EC by Barnett, Reidys & Stadler). A landscape with plateaus starves gradient and fitness-proportional selection of signal; the search must *drift* (mutation) across the plateau to its edge. Our integer wall-crossing term creates exactly such plateaus (moving an AP within a shadow-free region does not change the crossing count), so $f$ is locally flat in patches — another reason gradient methods stall and population drift helps.

### 1.7.4 Deceptiveness
A landscape is **deceptive** (Goldberg 1989, *Genetic Algorithms in Search, Optimization, and Machine Learning*) when low-order statistics point *away* from the global optimum: the average fitness of the schemata (building blocks) containing $x^\star$ is *lower* than that of competing schemata, so selection actively drives the population away from $x^\star$. Deceptive problems break gradient methods *and* naive GAs; they motivate linkage learning and diversity preservation. Most real landscapes are partially deceptive in places.

### 1.7.5 Why population metaheuristics fit non-convex / non-diff / multimodal / high-D / black-box
- **Non-differentiable**: they use only $f$ values, never $\nabla f$ — immune to kinks, `max`, thresholds, integer counts (our objective has all four).
- **Non-convex / multimodal**: a *population* samples many basins in parallel; selection + variation perform implicit basin-hopping; niching keeps optima co-resident.
- **High-dimensional**: no $\binom{G}{k}$ enumeration; the swarm follows fitness-correlation structure (§1.7.2) rather than gridding, so cost scales with iterations, not with $\dim(\mathcal{S})^{}$ exponentially. (This is precisely why our PSO beats Grid Search on the 6-D placement problem under a matched budget.)
- **Black-box**: they need only the oracle; no model, gradient, or convexity.
- **Robust to noise/neutrality**: population averaging and drift tolerate flat/noisy regions that stall single-point deterministic methods.

---

## 1.8 The eight algorithms — mechanics, one by one

Common notation: $x$ = a solution/position; $f$ = objective; $t$ = iteration; $r,r_1,r_2\sim U(0,1)$ fresh randoms; boldface = vectors.

### 1.8.1 Genetic Algorithm (GA)
- **Analogy.** Darwinian evolution: a population of chromosomes; fitter individuals reproduce more; crossover recombines parental "genes," mutation introduces novelty; over generations, good building blocks (schemata) proliferate.
- **Core equations.** Fitness-proportional (roulette) selection probability
  $$ p_i = \frac{f_i}{\sum_{j=1}^{\mu} f_j}. $$
  Single-point crossover exchanges the tail substrings of two parents past a random cut; bit-flip mutation flips each gene independently with probability $p_m$. The **Schema Theorem** bounds building-block growth:
  $$
  \mathbb{E}[m(H,t{+}1)] \ \ge\ m(H,t)\,\frac{f(H)}{\bar f}\Big[1 - p_c\,\frac{\delta(H)}{\ell-1} - o(H)\,p_m\Big],
  $$
  where $m(H,t)$ = instances of schema $H$ at gen $t$, $f(H)$ = mean fitness of $H$, $\bar f$ = population mean fitness, $\delta(H)$ = defining length, $o(H)$ = order, $\ell$ = string length, $p_c$ = crossover rate. Short, low-order, above-average schemata grow exponentially.
- **Parameters & sensitivity.** Population $\mu\!\in\![30,100]$; crossover $p_c\!\in\![0.6,0.95]$; mutation $p_m\!\in\![0.001,0.01]$ (values $>0.05$ usually harmful — devolves to random walk). Selection scheme (tournament/rank) controls takeover speed; encoding choice dominates everything.
- **Best-fit landscape.** Combinatorial / discrete / mixed spaces with meaningful recombination structure (feature selection, scheduling, permutations).
- **Signature strength.** Extreme representational flexibility — encode almost anything (binary, permutation, tree, real vector) and design matched operators.
- **Main limitation.** Premature convergence via diversity loss; performance highly sensitive to encoding and operator design.
- **Seminal cite.** J. H. Holland, *Adaptation in Natural and Artificial Systems*, Univ. of Michigan Press, 1975 (2nd ed. MIT Press 1992, DOI [10.7551/mitpress/1090.001.0001](https://doi.org/10.7551/mitpress/1090.001.0001)).

### 1.8.2 Particle Swarm Optimization (PSO)
- **Analogy.** A flock/school foraging: each bird (particle) balances its own memory of the best food it found (`pbest`, nostalgia) against the flock's best (`gbest`, social influence), with momentum (inertia) carrying it forward.
- **Core equations** (inertia form; the course slide's rule):
  $$
  \mathbf{v}_i^{t+1} = w\,\mathbf{v}_i^{t} + c_1 r_1(\mathbf{p}_i - \mathbf{x}_i^{t}) + c_2 r_2(\mathbf{g} - \mathbf{x}_i^{t}), \qquad
  \mathbf{x}_i^{t+1} = \mathbf{x}_i^{t} + \mathbf{v}_i^{t+1}.
  $$
  $\mathbf{v}_i$ = velocity, $\mathbf{x}_i$ = position, $\mathbf{p}_i$ = personal best, $\mathbf{g}$ = global best, $w$ = inertia, $c_1$ = cognitive (personal) coefficient, $c_2$ = social coefficient, $r_1,r_2\sim U(0,1)$. Inertia $w$ was added by Shi & Eberhart 1998 (DOI [10.1109/ICEC.1998.699146](https://doi.org/10.1109/ICEC.1998.699146)); it is absent from the 1995 original.
- **Parameters & sensitivity.** $w\!\in\![0.4,0.9]$ (often linearly decreasing); $c_1{,}c_2\!\approx\!2.0$ (range 0.5–2.5); swarm 20–50; velocity clamp $v_{\max}$. Balance is very sensitive to $w$ and to the $c_1{:}c_2$ ratio; $w$ too large $\Rightarrow$ divergence, too small $\Rightarrow$ premature convergence. Clerc–Kennedy constriction ($\chi{=}0.7298$) is the stability-guaranteed alternative.
- **Best-fit landscape.** Continuous, real-valued, moderately multimodal optimization — *our Wi-Fi placement is a textbook fit*.
- **Signature strength.** Few operators, few parameters, fast empirical convergence on continuous problems; trivially parallel.
- **Main limitation.** Stagnation/velocity collapse at a non-guaranteed point (van den Bergh 2001); weaker on combinatorial spaces without re-encoding.
- **Seminal cite.** J. Kennedy & R. C. Eberhart, "Particle Swarm Optimization," *Proc. IEEE ICNN'95*, vol. 4, pp. 1942–1948, DOI [10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).

### 1.8.3 Ant Colony Optimization (ACO)
- **Analogy.** Ants deposit pheromone on trails; shorter routes are traversed faster and reinforced more; evaporation erases stale trails; the colony collectively converges on short paths — *stigmergy* (indirect coordination via the environment).
- **Core equations.** Probabilistic edge choice by ant $k$ at node $i$:
  $$
  p_{ij}^{k} = \frac{\tau_{ij}^{\alpha}\,\eta_{ij}^{\beta}}{\sum_{l\in \mathcal{N}_i^{k}} \tau_{il}^{\alpha}\,\eta_{il}^{\beta}},
  \qquad
  \tau_{ij} \leftarrow (1-\rho)\,\tau_{ij} + \sum_{k}\Delta\tau_{ij}^{k},\quad \Delta\tau_{ij}^{k}=\tfrac{Q}{L_k}.
  $$
  $\tau_{ij}$ = pheromone on edge $(i,j)$, $\eta_{ij}$ = heuristic desirability ($\approx 1/d_{ij}$), $\alpha,\beta$ = weight exponents, $\mathcal{N}_i^k$ = feasible next nodes, $\rho$ = evaporation rate, $Q$ = constant, $L_k$ = tour length of ant $k$.
- **Parameters & sensitivity.** $\alpha\!\approx\!1$, $\beta\!\in\![2,5]$, $\rho\!\approx\!0.5$, ants $m\!\approx\!n$, $Q$. Over-emphasizing $\tau$ (large $\alpha$, small $\rho$) causes early lock-in; MAX–MIN Ant System (Stützle & Hoos 2000) bounds $\tau\in[\tau_{\min},\tau_{\max}]$ to prevent it.
- **Best-fit landscape.** Discrete graph/combinatorial construction problems: routing, TSP, scheduling, sequencing.
- **Signature strength.** Constructs solutions incrementally with an explicit *learned* memory (pheromone) — excellent on shortest-path/graph problems and dynamic re-routing.
- **Main limitation.** Pheromone stagnation / premature convergence; sensitive to $\rho,\alpha,\beta$; mostly discrete.
- **Seminal cite.** M. Dorigo, V. Maniezzo, A. Colorni, "Ant System: Optimization by a Colony of Cooperating Agents," *IEEE Trans. SMC-B* 26(1):29–41, 1996, DOI [10.1109/3477.484436](https://doi.org/10.1109/3477.484436) (concept: Dorigo PhD thesis, Politecnico di Milano, 1992).

### 1.8.4 Artificial Bee Colony (ABC)
- **Analogy.** Honeybee foraging with a division of labour: *employed* bees exploit known food sources, *onlookers* probabilistically pick richer sources to exploit further, *scouts* abandon exhausted sources and explore randomly.
- **Core equations.** Employed/onlooker neighbour step and onlooker selection:
  $$
  v_{ij} = x_{ij} + \phi_{ij}\,(x_{ij}-x_{kj}), \quad \phi_{ij}\in[-1,1], \qquad
  p_i = \frac{\mathrm{fit}_i}{\sum_{n=1}^{SN}\mathrm{fit}_n}.
  $$
  $x_{ij}$ = dimension $j$ of source $i$; $x_{kj}$ = a random other source $k\neq i$; $\phi_{ij}$ = uniform random; $SN$ = number of food sources. A source not improved within `limit` trials is abandoned and re-initialized (scout): $x_{ij}=x^{\min}_j + r\,(x^{\max}_j-x^{\min}_j)$.
- **Parameters & sensitivity.** Colony/source count $SN$; abandonment `limit` (governs explore/exploit — small `limit` over-explores); employed=onlookers=$SN/2$.
- **Best-fit landscape.** Continuous, multimodal numerical optimization.
- **Signature strength.** Strong global exploration via scouts; only one real control parameter (`limit`) beyond size.
- **Main limitation.** Good exploration but *poor exploitation* — slow final convergence near optima (motivating GABC and hybrids).
- **Seminal cite.** D. Karaboga, "An Idea Based on Honey Bee Swarm for Numerical Optimization," Tech. Rep. TR06, Erciyes Univ., 2005; Karaboga & Basturk, *J. Global Optimization* 39(3):459–471, 2007, DOI [10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x).

### 1.8.5 Firefly Algorithm (FA)
- **Analogy.** Fireflies attract mates by bioluminescence; a dimmer firefly moves toward a brighter one; perceived brightness falls with distance (light absorption), so attraction is local — producing automatic subswarming around multiple optima.
- **Core equations.** Attractiveness and movement:
  $$
  \beta(r) = \beta_0\,e^{-\gamma r^2}, \qquad
  \mathbf{x}_i \leftarrow \mathbf{x}_i + \beta_0\,e^{-\gamma r_{ij}^2}(\mathbf{x}_j-\mathbf{x}_i) + \alpha\,\boldsymbol{\varepsilon}_i.
  $$
  $\beta_0$ = attractiveness at $r{=}0$; $\gamma$ = light-absorption coefficient; $r_{ij}=\lVert \mathbf{x}_i-\mathbf{x}_j\rVert$; $\alpha$ = randomization scale; $\boldsymbol{\varepsilon}_i$ = random vector.
- **Parameters & sensitivity.** $\alpha\!\approx\!0.2$ (often annealed); $\beta_0\!=\!1$; $\gamma\!\in\![0.1,10]$: $\gamma\!\to\!0$ makes it PSO-like (global), $\gamma\!\to\!\infty$ makes it a random walk. $\gamma$ is the critical knob.
- **Best-fit landscape.** Highly **multimodal** continuous problems where finding *many* optima matters (the local-attraction property partitions the swarm into niches).
- **Signature strength.** Automatic multimodal subdivision — locates multiple optima simultaneously without explicit niching.
- **Main limitation.** Premature convergence/stagnation on high-D problems; $O(\mu^2)$ pairwise comparisons per iteration.
- **Seminal cite.** X.-S. Yang, "Firefly Algorithms for Multimodal Optimization," *SAGA 2009*, LNCS 5792:169–178, DOI [10.1007/978-3-642-04944-6_14](https://doi.org/10.1007/978-3-642-04944-6_14) (origin: *Nature-Inspired Metaheuristic Algorithms*, Luniver Press, 2008).

### 1.8.6 Grey Wolf Optimizer (GWO)
- **Analogy.** Grey-wolf pack hierarchy and hunting: the three best solutions are the $\alpha,\beta,\delta$ wolves; the rest ($\omega$) update their positions by encircling the estimated prey (optimum) as directed by the three leaders.
- **Core equations.**
  $$
  \mathbf{D} = \lvert \mathbf{C}\cdot\mathbf{X}_p - \mathbf{X}\rvert, \qquad \mathbf{X}(t{+}1) = \mathbf{X}_p - \mathbf{A}\cdot\mathbf{D},
  $$
  with leader aggregation $\mathbf{X}_1=\mathbf{X}_\alpha-\mathbf{A}_1\mathbf{D}_\alpha$, $\mathbf{X}_2=\mathbf{X}_\beta-\mathbf{A}_2\mathbf{D}_\beta$, $\mathbf{X}_3=\mathbf{X}_\delta-\mathbf{A}_3\mathbf{D}_\delta$, and $\mathbf{X}(t{+}1)=\tfrac13(\mathbf{X}_1+\mathbf{X}_2+\mathbf{X}_3)$. Coefficients $\mathbf{A}=2\mathbf{a}\cdot\mathbf{r}_1-\mathbf{a}$, $\mathbf{C}=2\mathbf{r}_2$, where $\mathbf{a}$ decreases **linearly from 2 to 0** across iterations; $\lvert\mathbf{A}\rvert{>}1$ = explore, $\lvert\mathbf{A}\rvert{<}1$ = exploit.
- **Parameters & sensitivity.** Population ($\approx$30), max-iterations, and the $\mathbf{a}$ schedule (the sole explore/exploit control). Few parameters $\Rightarrow$ easy to tune, but the linear $\mathbf{a}$ schedule is a rigid prior.
- **Best-fit landscape.** Smooth-ish, low-to-moderate-dimensional, *redundant* continuous problems (e.g. robot inverse kinematics — Part 2) where strong late-stage exploitation pays.
- **Signature strength.** Excellent exploitation with almost no parameters; the three-leader consensus damps premature lock to a single incumbent.
- **Main limitation.** Premature convergence on high-D multimodal problems (no mutation/diversity operator; driven by the leaders).
- **Seminal cite.** S. Mirjalili, S. M. Mirjalili, A. Lewis, "Grey Wolf Optimizer," *Advances in Engineering Software* 69:46–61, 2014, DOI [10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007).

### 1.8.7 Cuckoo Search (CS)
- **Analogy.** Brood parasitism: cuckoos lay eggs in host nests; the best eggs (solutions) survive; hosts discover and discard a fraction $p_a$ of alien eggs (abandonment); new nests are sought by **Lévy flights** — heavy-tailed random walks that mix frequent short hops with rare long jumps.
- **Core equations.**
  $$
  \mathbf{x}_i^{t+1} = \mathbf{x}_i^{t} + \alpha \oplus \mathrm{L\acute{e}vy}(\lambda), \qquad \mathrm{L\acute{e}vy}(\lambda)\sim u = s^{-\lambda},\ 1<\lambda\le 3,
  $$
  where $\alpha$ = step-size scaling, $\oplus$ = entrywise product, and a fraction $p_a$ of worst nests is replaced by a biased random walk $\mathbf{x}_i^{t+1}=\mathbf{x}_i^{t}+r(\mathbf{x}_j^{t}-\mathbf{x}_k^{t})$.
- **Parameters & sensitivity.** Discovery probability $p_a\!\approx\!0.25$ (authors report low sensitivity), Lévy exponent $\lambda\!\approx\!1.5$, nests $n\!\in\![15,40]$. Notably *few* parameters and robust defaults.
- **Best-fit landscape.** Multimodal continuous global optimization; the Lévy heavy tail gives persistent long-range exploration that resists premature convergence.
- **Signature strength.** Lévy-flight global exploration escapes local optima that Gaussian-step methods miss; very few parameters.
- **Main limitation.** Fixed Lévy step size $\Rightarrow$ slow late convergence on complex/high-D problems.
- **Seminal cite.** X.-S. Yang & S. Deb, "Cuckoo Search via Lévy Flights," *World Congress on Nature & Biologically Inspired Computing (NaBIC 2009)*, IEEE, pp. 210–214, DOI [10.1109/NABIC.2009.5393690](https://doi.org/10.1109/NABIC.2009.5393690).

### 1.8.8 Whale Optimization Algorithm (WOA)
- **Analogy.** Humpback-whale *bubble-net* feeding: whales encircle prey and swim along a shrinking spiral while blowing bubbles; the algorithm alternates between shrinking-encircle and logarithmic-spiral moves toward the best solution.
- **Core equations.** Encircle vs. spiral, chosen by a coin flip $p$:
  $$
  \mathbf{X}(t{+}1) = \begin{cases}
  \mathbf{X}^{*} - \mathbf{A}\cdot\lvert \mathbf{C}\,\mathbf{X}^{*} - \mathbf{X}\rvert, & p<0.5 \\[4pt]
  \mathbf{D}'\,e^{bl}\cos(2\pi l) + \mathbf{X}^{*}, & p\ge 0.5
  \end{cases}
  $$
  where $\mathbf{X}^{*}$ = best solution so far, $\mathbf{D}'=\lvert\mathbf{X}^{*}-\mathbf{X}\rvert$, $b$ = spiral shape constant, $l\in[-1,1]$, $\mathbf{A}=2\mathbf{a}\mathbf{r}-\mathbf{a}$, $\mathbf{C}=2\mathbf{r}$, and $\mathbf{a}$ decreases linearly $2\to0$. When $\lvert\mathbf{A}\rvert\ge 1$ the encircle term targets a *random* whale instead of $\mathbf{X}^*$ (exploration).
- **Parameters & sensitivity.** Population, max-iterations, spiral constant $b$, and the $\mathbf{a}$ schedule; like GWO, very few knobs.
- **Best-fit landscape.** Continuous global optimization; the spiral gives a distinctive exploitation trajectory useful on engineering design problems.
- **Signature strength.** The dual encircle/spiral operator balances explore/exploit with minimal parameters; strong on many engineering benchmarks.
- **Main limitation.** Slow convergence and local-optima trapping on complex high-D problems; diversity loss in late iterations.
- **Seminal cite.** S. Mirjalili & A. Lewis, "The Whale Optimization Algorithm," *Advances in Engineering Software* 95:51–67, 2016, DOI [10.1016/j.advengsoft.2016.01.008](https://doi.org/10.1016/j.advengsoft.2016.01.008).

---

## 1.9 Side-by-side summary table

| Algorithm | Inspiration | Core update (essence) | Key parameters | Best-fit landscape | Signature strength | Main limitation | Seminal cite |
|---|---|---|---|---|---|---|---|
| **GA** | Darwinian evolution | roulette $p_i{=}f_i/\sum f$; crossover; bit-flip mutation $p_m$ | $\mu$ 30–100, $p_c$ 0.6–0.95, $p_m$ 0.001–0.01 | discrete/combinatorial/mixed w/ recombination structure | representational flexibility (encode anything) | premature convergence; encoding-sensitive | Holland 1975 |
| **PSO** | bird flock / fish school | $\mathbf v{=}w\mathbf v{+}c_1r_1(\mathbf p{-}\mathbf x){+}c_2r_2(\mathbf g{-}\mathbf x)$ | $w$ 0.4–0.9, $c_1{,}c_2{\approx}2$, swarm 20–50 | continuous, moderately multimodal | few params, fast on continuous, parallel | stagnation, no global guarantee | Kennedy & Eberhart 1995 |
| **ACO** | ant stigmergy (pheromone) | $p{\propto}\tau^\alpha\eta^\beta$; $\tau{\leftarrow}(1{-}\rho)\tau{+}\sum Q/L_k$ | $\alpha{\approx}1$, $\beta$ 2–5, $\rho{\approx}0.5$ | discrete graph/routing/sequencing | incremental construction w/ learned memory | pheromone stagnation; mostly discrete | Dorigo et al. 1996 |
| **ABC** | honeybee foraging roles | $v{=}x{+}\phi(x{-}x_k)$; scouts after `limit` | $SN$, `limit` | continuous multimodal | strong exploration; one main knob | weak exploitation (slow finish) | Karaboga 2005 |
| **FA** | firefly bioluminescence | $\mathbf x{+}{=}\beta_0e^{-\gamma r^2}(\mathbf x_j{-}\mathbf x_i){+}\alpha\varepsilon$ | $\alpha{\approx}0.2$, $\beta_0{=}1$, $\gamma$ 0.1–10 | highly multimodal (many optima) | automatic multimodal niching | premature convergence; $O(\mu^2)$/iter | Yang 2008/2009 |
| **GWO** | grey-wolf pack hierarchy | 3 leaders $\alpha\beta\delta$; $\mathbf X{=}\mathbf X_p{-}\mathbf A\mathbf D$; $a{:}2{\to}0$ | pop, iters, $a$ schedule | smooth/redundant low–mid-D continuous | strong exploitation, few params | high-D multimodal premature convergence | Mirjalili et al. 2014 |
| **CS** | cuckoo brood parasitism | $\mathbf x{+}{=}\alpha\oplus\mathrm{L\acute evy}(\lambda)$; abandon $p_a$ | $p_a{\approx}0.25$, $\lambda{\approx}1.5$, $n$ 15–40 | multimodal continuous global | Lévy long-jumps escape local optima | fixed step $\Rightarrow$ slow finish | Yang & Deb 2009 |
| **WOA** | humpback bubble-net | encircle/spiral switch; $a{:}2{\to}0$ | pop, iters, $b$, $a$ schedule | continuous engineering-design global | dual operator, minimal params | slow, local-optima trap on high-D | Mirjalili & Lewis 2016 |

**Cross-cutting reading.** All eight are the §1.1.2 template with different $(\mathcal{V},\mathcal{Q},M)$: GA/ABC use explicit selection + recombination/neighbour variation; PSO/FA/GWO/WOA use *attraction toward memory* (pbest/gbest/brighter/leaders/best) as implicit selection; ACO uses environmental memory (pheromone). All expose exactly one dominant explore/exploit knob (§1.4.2). None escapes NFL (§1.3): each is a *bet* that the target landscape matches its operator's locality/attraction prior — the bet we place, and justify, for our Wi-Fi problem in Part 4.
