# Part 2 — Exceptional Real-World Applications (Literature Review)

Twelve applications of population/swarm metaheuristics
from peer-reviewed venues, spanning seven domains. Each entry gives (a) the
problem, (b) the algorithm and why it fit, (c) the solution encoding, (d) the
fitness/objective, (e) constraint handling, (f) reported results, and (g) the
full citation.

**Sourcing honesty.** Every citation and headline number below was verified
against the primary source (or, where noted, the publisher abstract/record).
Figures the source did not let us confirm to the digit are tagged
`[unverified]`; they are reported as the authors state them, not as our claims.
This transparency is deliberate: an arXiv-bound report must not launder
unverifiable numbers as fact.

## Index

| # | Domain | Problem | Algorithm | Year | Venue |
|---|---|---|---|---|---|
| 1 | Medicine / Bioinformatics | Cancer gene selection | Master–Slave Binary GWO | 2021 | BioMed Research International |
| 2 | Medicine | 4D lung-SBRT radiotherapy planning | PSO (virtual-search) | 2016 | IEEE Trans. Biomed. Eng. |
| 3 | Aerospace / Structural | Spacecraft multi-gravity-assist trajectory | Self-adaptive DE | 2022 | Acta Astronautica |
| 4 | Aerospace / Structural | Truss topology + sizing | Novelty-driven Binary PSO | 2021 | arXiv (Assimi et al.) |
| 5 | Energy | Distribution-net reconfiguration + DG siting | PSO | 2025 | PLOS ONE |
| 6 | Energy | Wind-farm micro-siting (position + hub height) | PSO | 2021 | Applied Sciences |
| 7 | Robotics | 6-DOF welding-arm inverse kinematics | Grey Wolf Optimizer | 2022 | IJORAS |
| 8 | NAS | ImageNet classifier architecture search | Regularized (aging) Evolution | 2019 | AAAI |
| 9 | 5G Wireless | Small-cell base-station siting (real field data) | Multi-objective Cuckoo Search + K-means | 2023 | PeerJ Comp. Sci. |
| 10 | Edge Computing | Edge-server placement | Hybrid GA + PSO | 2024 | PeerJ Comp. Sci. |
| 11 | Logistics | Container-terminal berth allocation | Genetic Algorithm | 2018 | Pesquisa Operacional |
| 12 | Logistics | Cold-chain refrigerated vehicle routing | Hybrid Tabu–GWO | 2024 | PLOS ONE |

---

## 1 · Gene selection for cancer classification — Master–Slave Binary GWO
- **(a) Problem.** Select the few informative genes from high-dimensional
  microarray/biomedical datasets (thousands of features, tens of samples) so a
  downstream classifier diagnoses cancer accurately without noise from redundant
  genes.
- **(b) Algorithm + fit.** A **Master–Slave Binary Grey Wolf Optimizer
  (MSBGWO)**. Feature selection is a binary, high-dimensional, deceptive search
  (the "small-n, large-p" regime); GWO's α/β/δ leadership gives strong
  exploitation, and the master–slave twist (each weak "slave" wolf learns from an
  assigned "master") injects the diversity plain GWO lacks, escaping the
  local optima that trap BPSO/BGA here.
- **(c) Encoding.** Binary feature mask of length $D$ = number of genes; bit
  $=1$ keeps the gene. Continuous wolf positions are binarised through a sigmoid
  transfer function.
- **(d) Fitness.** $\text{fit}=1-\alpha\frac{S}{D}-\alpha\,\overline{\text{Acc}}$
  with $\alpha=0.8$, $S$ = number of selected genes, $D$ = total genes,
  $\overline{\text{Acc}}$ = mean KNN ($k{=}5$) accuracy — jointly rewards accuracy
  and sparsity.
- **(e) Constraints.** Unconstrained wrapper; feasibility is inherent to the
  binary mask (any subset is valid).
- **(f) Results.** Accuracy **1.000** on Leukemia and DLBCL, **0.996** on
  Ovarian, **0.957** on Colon, **0.850** on CNS, while selecting the fewest
  features; beats BGWO2, BGA, BPSO, DE, and SCA on accuracy, precision, recall,
  F-measure, and feature count `[verified]`.
- **(g) Citation.** Momanyi, E.; Segera, D. "A Master-Slave Binary Grey Wolf
  Optimizer for Optimal Feature Selection in Biomedical Data Classification."
  *BioMed Research International*, 2021:5556941. DOI 10.1155/2021/5556941.

## 2 · 4D lung-SBRT radiotherapy planning — PSO with a "virtual search" strategy
- **(a) Problem.** In 4D conformal radiotherapy for lung SBRT, choose the beam
  aperture monitor-unit (MU) weights across all respiratory phases to deliver a
  lethal tumour dose while sparing healthy tissue.
- **(b) Algorithm + fit.** **PSO** with a novel constraint-handling ("virtual
  search") strategy. The dose objective is a high-dimensional, constrained,
  non-linear function of continuous weights with no clean gradient — PSO searches
  it directly; the virtual-search normalisation keeps the swarm in the clinically
  feasible region.
- **(c) Encoding.** A particle is the weight vector
  $\vartheta=[\vartheta_1,\dots,\vartheta_{N_{\text{ap}}}]$, $0\le\vartheta_k\le3.3$;
  dimension **90–110** apertures for the tested cases.
- **(d) Fitness.** Dose-based MSE
  $F=\sum_{i}^{N_{\text{struct}}}\frac1{N_{\text{vox},i}}\big(\sum F^{\text{up}}_{i,f}+\sum F^{\text{low}}_{i,f}+F^{\max}_i\big)$
  penalising over-, under-, and max-dose voxel violations; dose
  $\mathbf D=\sum_k \vartheta_k\mathbf d_k$.
- **(e) Constraints.** Three schemes compared; the proposed one iteratively
  rescales dose by $NF^t=D_{95}^{\text{presc}}/D_{95}^t$ (feasibility-restoring
  normalisation), vs a hard penalty of $10^{26}$ for violating 95 % tumour
  coverage at 54 Gy.
- **(f) Results.** Across 5 lung-SBRT patients the virtual-search PSO reached the
  smallest objective values and tightest convergence ranges; a dose-projection
  baseline found the global optimum in only 2/5 cases and needed ~200 vs 30
  iterations `[verified]`.
- **(g) Citation.** Modiri, A.; Gu, X.; Hagan, A.M.; Sawant, A. "Radiotherapy
  Planning Using an Improved Search Strategy in Particle Swarm Optimization."
  *IEEE Trans. Biomedical Engineering*, 64(5):980–989, 2016 (landmark clinical
  reference; ~1 yr outside the 8-yr window). DOI 10.1109/TBME.2016.2585114.

## 3 · Deep-space multi-gravity-assist trajectory — Self-adaptive DE
- **(a) Problem.** Design propellant-efficient interplanetary trajectories that
  chain multiple planetary gravity assists plus deep-space manoeuvres — ESA's
  GTOP benchmark suite, built precisely because these are severely multimodal.
- **(b) Algorithm + fit.** **Self-adaptive ("self-learning") Differential
  Evolution** — mutation strategy and $(F,CR)$ switch mid-run based on recent
  success, plus re-initialisation of stagnated sub-populations. DE's
  vector-difference mutation suits the continuous decision variables; the
  adaptive+restart machinery targets DE's premature-convergence weakness that
  makes GTOP hard.
- **(c) Encoding.** Real-valued MGA-1DSM vector: launch epoch $t_0$; hyperbolic
  excess velocity $[V_\infty,\text{RA},\text{dec}]$; per-leg times-of-flight;
  per-flyby swing-by parameters; dimension $\approx 4+3(n-1)$ for $n$ planets
  `[unverified — standard GTOP chromosome, not itemised in the paper text obtained]`.
- **(d) Fitness.** Minimise total mission $\Delta v$ (equiv. maximise delivered
  mass), a highly multimodal scalar of the trajectory vector `[unverified exact form]`.
- **(e) Constraints.** Box bounds (launch window, TOF ranges, min flyby
  altitude); continuity satisfied by the encoding; explicit re-initialisation to
  escape local optima.
- **(f) Results.** Solved 6 well-known ESA GTOP problems; matched the known
  best on **5 of 6** and set a **new best-known on 4 of 6** `[verified headline;
  per-problem delta-v paywalled, unverified]`.
- **(g) Citation.** Choi, J.H.; Lee, J.; Park, C. "Deep-space trajectory
  optimizations using differential evolution with self-learning." *Acta
  Astronautica*, 191:258–269, 2022. DOI 10.1016/j.actaastro.2021.11.014.

## 4 · Truss topology + sizing — Novelty-driven Binary PSO
- **(a) Problem.** Minimum-weight design of trusses, choosing both topology
  (which members exist) and sizing (discrete cross-sections) — a combinatorial,
  multimodal structural-optimisation problem.
- **(b) Algorithm + fit.** **Novelty-driven Binary PSO** in a bilevel scheme:
  the upper level uses binary PSO to *discover diverse topologies by maximising
  novelty* (not just fitness), the lower level does discrete member sizing. The
  novelty drive combats the premature convergence that plagues weight-only search
  on this deceptive landscape.
- **(c) Encoding.** Binary particle at the upper level (member presence);
  discrete-catalogue section indices at the lower level.
- **(d) Fitness.** Minimise structural weight subject to stress/displacement
  limits; the upper level additionally maximises a novelty metric over the
  archive of found designs.
- **(e) Constraints.** Stress and nodal-displacement limits (standard
  penalty/repair in truss BPSO).
- **(f) Results.** Authors report the method "outperforms the current
  state-of-the-art" and returns *multiple* high-quality designs `[verified
  qualitative claim; the abstract obtained lists no per-truss weights]`.
- **(g) Citation.** Assimi, H.; Neumann, F.; Wagner, M.; Li, X.
  "Novelty-Driven Binary Particle Swarm Optimisation for Truss Optimisation
  Problems." arXiv:2112.07875, 2021.

## 5 · Distribution-network reconfiguration + DG siting — PSO (real Ethiopian feeder)
- **(a) Problem.** Jointly choose tie-switch states (reconfiguration) and the
  location/size of distributed generation (DG) on the **real Wolaita Sodo MV
  feeder, Ethiopia** (114 transformers, ≈27 MVA) to cut losses and fix severe
  under-voltage — not a toy IEEE 33/69-bus feeder.
- **(b) Algorithm + fit.** **PSO**. The search mixes a combinatorial part
  (switch states that must keep the feeder radial — a discontinuous graph
  constraint) with a continuous part (DG MW), evaluated through a
  backward/forward-sweep load flow that jumps discontinuously on topology change.
  No usable gradient exists; PSO searches switch+DG combinations directly.
- **(c) Encoding.** Each particle encodes 4 tie-switch open/closed assignments
  (restricted to radial/fundamental-loop combinations) plus DG bus index + MW
  rating; swarm $n=20$, $I_{\max}=20$.
- **(d) Fitness.** Weighted sum
  $F=w_1\Delta P_{\text{loss}}+w_2\Delta Q_{\text{loss}}+w_3\,\text{VD}$ (real
  loss $\sum I^2R$, reactive loss $\sum I^2X$, voltage-deviation index).
- **(e) Constraints.** Radiality enforced by *restricting the search space* to
  spanning-tree combinations (encoding-level repair, not a penalty); voltage/
  current/DG-size limits checked as hard feasibility in the load flow.
- **(f) Results.** Combined reconfig+DG cut active loss **1631.10 → 455.66 kW
  (−72.06 %)**, raised min voltage **0.7537 → 0.9550 p.u.**, cut max deviation
  **24.63 % → 4.5 %**, annual energy loss **16.669 → 4.164 GWh**, ≈16.40 M ETB/yr
  savings on a 22.07 M ETB investment (~6-yr payback) `[verified]`.
- **(g) Citation.** Alemayehu, B.; Mishra, S.; Tejani, G.G.; Tripathi, S.
  "Network reconfiguration and DG based compensation of Wolaita Sodo distribution
  system by using particle swarm optimisation." *PLOS ONE*, 20(10):e0335512,
  2025. DOI 10.1371/journal.pone.0335512.

## 6 · Wind-farm micro-siting with variable hub height — PSO (real Manjil site)
- **(a) Problem.** Re-optimise the layout of Iran's **Manjil/Rudbar** wind
  complex by jointly choosing each turbine's $(x,y)$ position **and** hub height,
  using the site's real wind regime, to cut wake losses and cost of energy.
- **(b) Algorithm + fit.** **PSO**. Jensen wake interactions make even the 2D
  layout multimodal; adding hub height as a third variable means a taller turbine
  can clear a shorter one's wake, so the optimum changes *discontinuously* as
  heights cross wake cones — a non-smooth landscape ideal for derivative-free
  swarm search.
- **(c) Encoding.** Per turbine a triple $(x_i,y_i,h_i)$ — planar coordinates +
  hub height from a discrete menu $\{40,50,60\}$ m; $\approx 3\times21=63$-D for
  the 21-turbine case `[dimension is our derivation, unverified]`.
- **(d) Fitness.** Minimise cost of energy $CoE=(C_{\text{cap}}(h)+C_{\text{O\&M}})/AEP(x,y,h)$,
  with AEP from the wind-rose-weighted power curve minus pairwise Jensen wake
  deficits.
- **(e) Constraints.** Min inter-turbine spacing and site-boundary limits;
  $h_i$ restricted to the discrete menu; infeasible layouts penalised
  `[penalty coefficients unverified]`.
- **(f) Results.** Optimised layout **+10.75 %** power and **−9.42 %** levelised
  cost vs baseline; the mixed-height 21-turbine config (2×60 m, 4×50 m, rest 40 m)
  produced 7.75 MW `[verified headline; granular AEP/$MWh omitted as unverifiable]`.
- **(g) Citation.** Yeghikian, M.; Ahmadi, A.; Dashti, R.; Esmaeilion, F.;
  Mahmoudan, A.; Hoseinzadeh, S.; Garcia, D.A. "Wind Farm Layout Optimization with
  Different Hub Heights in Manjil Wind Farm Using PSO." *Applied Sciences*,
  11(20):9746, 2021. DOI 10.3390/app11209746.

## 7 · 6-DOF welding-arm inverse kinematics — Grey Wolf Optimizer
- **(a) Problem.** Solve the inverse kinematics of a new 6-DOF revolute arm for
  automated oil-and-gas pipeline welding: find the six joint angles driving the
  end-effector along a rectangular weld path — analytically messy because many
  joint combinations reach the same pose.
- **(b) Algorithm + fit.** **GWO** (vs PSO/WOA/JFO and an improved I-GWO). On the
  smooth-but-redundant, low-dimensional, box-constrained IK landscape, GWO's
  α/β/δ exploitation plus its $a:2\to0$ decay converge far more reliably than the
  competitors.
- **(c) Encoding.** Continuous 6-D angle vector $[\theta_1,\dots,\theta_6]$,
  box-bounded per Denavit–Hartenberg limits; population 50, 10 iterations, solved
  per waypoint over 6 waypoints.
- **(d) Fitness.** Minimise position MSE
  $=\frac1N\sum_{i\in\{x,y,z\}}(P_i^{\text{ref}}-P_i)^2$ (forward-kinematics vs
  reference).
- **(e) Constraints.** Hard box-clipping of each joint angle to its DH range
  during updates.
- **(f) Results.** GWO per-waypoint MSE **8.4×10⁻⁷–6.5×10⁻⁵**; I-GWO best
  (**1.6×10⁻⁷–9.8×10⁻⁷**); PSO 2–4 orders worse (**1.7×10⁻³–3.4×10⁻³**); only
  GWO/I-GWO "effectively solved" the task `[verified]`.
- **(g) Citation.** Nyong-Bassey, B.E.; Epemu, A.M. "Inverse Kinematics Analysis
  of Novel 6-DOF Robotic Arm Manipulator for Oil and Gas Welding Using
  Meta-Heuristic Algorithms." *IJORAS*, 4:13–22, 2022. DOI 10.33093/ijoras.2022.4.3.

## 8 · ImageNet architecture search — Regularized (aging) Evolution → AmoebaNet
- **(a) Problem.** Automatically discover a convolutional image classifier that
  beats the best human- and reinforcement-learning-designed architectures on
  ImageNet under a fixed search budget.
- **(b) Algorithm + fit.** **Regularized ("aging") Evolution** — a
  tournament-selection GA that each cycle discards the *oldest* member (not the
  worst), biasing survival toward genotypes with sustained recent success. Suits
  the huge, discrete, non-differentiable architecture space and, at equal
  compute, beat RL controllers — a genuinely non-obvious result.
- **(c) Encoding.** Two cell types (normal, reduction), each a sequence of 5
  blocks; each block picks 2 hidden states + one op each from a discrete op set;
  network = stacked cells. Genome = discrete (state, state, op, op) tuples.
- **(d) Fitness.** Validation top-1 accuracy after a truncated training proxy;
  population $P=100$, tournament sample $S=25$, aging removal.
- **(e) Constraints.** No penalty — structural validity guaranteed by the
  mutation operator (one pointer/op changed at a time); age-based culling is the
  regulariser.
- **(f) Results.** AmoebaNet-A **82.8 %** top-1 / **96.1 %** top-5 at 86.7 M
  params (≈ NASNet-A RL 82.7/96.2 at 88.9 M); scaled to 469 M → **83.9 % / 96.6 %**
  ImageNet SOTA `[verified]`; the "evolution beats RL faster early" claim
  `[unverified — no exact multiplier]`.
- **(g) Citation.** Real, E.; Aggarwal, A.; Huang, Y.; Le, Q.V. "Regularized
  Evolution for Image Classifier Architecture Search." *AAAI-19*, 2019.
  arXiv:1802.01548; DOI 10.1609/aaai.v33i01.33014780.

## 9 · 5G small-cell base-station siting — Multi-objective Cuckoo Search + K-means
- **(a) Problem.** Site 5G small cells across real neighbourhoods of **Belém,
  Brazil** to maximise coverage while minimising transmit power, driven by real
  OpenCellID field data rather than a synthetic grid.
- **(b) Algorithm + fit.** **MOCS-KM** (multi-objective Cuckoo Search + K-means).
  Cuckoo Search's heavy-tailed Lévy flights escape the many local optima of a
  noisy real-field coverage surface; K-means turns the metaheuristic's choices
  into concrete cell clusters/powers. (Closest published cousin of this lab's own
  Wi-Fi placement project.)
- **(c) Encoding.** Low-dimensional mixed vector $[k,P_t]$: $k$ = number of
  cells (integer 1–100) seeding K-means over real coordinates; $P_t$ = transmit
  power (continuous, 30–40 dBm).
- **(d) Fitness.** $f=0.7\,N_{\text{outage}}+0.3\,P_{\text{diff}}$ (percent users
  in outage vs power above the 30 dBm floor).
- **(e) Constraints.** $P_r\ge-90$ dBm defines connected-vs-outage, enforced in
  fitness; $k,P_t$ box-bounded.
- **(f) Results.** **0 %** outage @700 MHz (12 cells, 30 dBm), **2.43 %** @2.3
  GHz, **2.29 %** @3.5 GHz; max coverage **97.4 %**; runtime 552–993 s `[verified]`.
- **(g) Citation.** Ferreira, F.H.; Barros, F.J.B.; de Alcântara Neto, M.C.;
  Cardoso, E.; Francês, C.R.L.; Araújo, J. "Hybrid computational and real
  data-based positioning of small cells in 5G networks." *PeerJ Computer
  Science*, 2023. DOI 10.7717/peerj-cs.1412.

## 10 · Edge-server placement — Hybrid GA + PSO (GP4ESP)
- **(a) Problem.** Decide which base/edge stations to equip with limited edge
  servers so that overall request **response time** is minimised in a
  heterogeneous edge-computing network — crucially accounting for server
  *computing delay*, which prior work ignored (over-estimating service quality).
- **(b) Algorithm + fit.** A **hybrid GA + PSO** that fuses PSO's swarm cognition
  into GA's evolutionary operators. ESP is a combinatorial assignment problem
  with a rugged objective; the hybrid balances GA's global recombination with
  PSO's directed convergence.
- **(c) Encoding.** Selection/assignment representation over candidate edge sites
  (which stations host servers) `[standard ESP encoding; the paper's exact vector
  layout not quoted here]`.
- **(d) Fitness.** Minimise overall response time = network transmission delay +
  edge-server computing/queueing delay (the paper's key modelling addition).
- **(e) Constraints.** Limited edge-resource budget (number/capacity of servers).
- **(f) Results.** **18.2 %–20.7 %** shorter overall response time vs **eleven**
  up-to-date ESP algorithms, stable as problem scale varies `[verified via
  publisher record/PMC]`.
- **(g) Citation.** Han, et al. "GP4ESP: a hybrid genetic algorithm and particle
  swarm optimization algorithm for edge server placement." *PeerJ Computer
  Science*, 2024. DOI 10.7717/peerj-cs.2439 (PMC11623000).

## 11 · Container-terminal berth allocation — Genetic Algorithm
- **(a) Problem.** Discrete berth allocation at the real **TECON Rio Grande**
  container terminal (Brazil): assign arriving vessels to berths and fix
  berthing order/time to minimise service delay and time-window penalties.
- **(b) Algorithm + fit.** **GA** (heuristic crossover), benchmarked head-to-head
  vs Simulated Annealing. GA is a population-based global search over the
  combinatorial, multimodal berthing-sequence space; SA is the single-point foil.
- **(c) Encoding.** Permutation chromosome — the ordered vessel sequence per
  berth; primary case 62 vessels over 3 berths (also 35/55/62 vessels × 5/10
  berths).
- **(d) Fitness.** Minimise
  $Z^{*}=w_0\!\sum v_i(\cdot)+w_1\!\sum[\max(0,a_i{-}T){+}\max(0,T{+}t{-}b_i)]+w_2\sum[\cdot]$
  (service cost + vessel and berth time-window penalties), weights $[1,10,10]$.
- **(e) Constraints.** Penalty relaxation — hard time-window constraints become
  soft penalty terms, allowing temporarily infeasible intermediate solutions.
- **(f) Results.** Primary case (30 runs): GA avg **185,951** vs SA **194,074**
  (**~4.2 %** lower), GA **20′50″** vs SA **29′04″** (**~28 %** faster), GA std
  **2,202** vs SA **3,267**; GA ≥ SA across all six scenarios `[verified]`.
- **(g) Citation.** Pereira, E.D.; Coelho, A.S.; Longaray, A.A.; Machado, C.M.S.;
  Munhoz, P.R. "Metaheuristic Analysis Applied to the Berth Allocation Problem:
  Case Study in a Port Container Terminal." *Pesquisa Operacional*, 38(2), 2018.
  DOI 10.1590/0101-7438.2018.038.02.0247.

## 12 · Cold-chain refrigerated vehicle routing — Hybrid Tabu–GWO
- **(a) Problem.** Route refrigerated vehicles for a real fresh-food e-commerce
  operator in China under vehicle-capacity, driving-range, dual-time-window, and
  perishable-freshness constraints.
- **(b) Algorithm + fit.** **Hybrid Tabu Search – Grey Wolf Optimizer (TGWO)**.
  GWO converges fast but has weak local search; grafting Tabu neighbourhood moves
  (insert/swap/2-opt) sharpens exploitation and escapes local optima on the
  multi-cost cold-chain landscape.
- **(c) Encoding.** Each wolf is a TSP-style permutation from the depot, decoded
  into multi-vehicle routes by sequential assignment; case dimension $D=19$
  (1 depot + 18 customers).
- **(d) Fitness.** Minimise $Z=C_1+C_2+C_3+C_4$ (fixed vehicle + transport incl.
  refrigeration energy + cargo spoilage + time-window penalty).
- **(e) Constraints.** Large-coefficient penalty ($M{=}10^8$) for time-window
  breach; a repair-like decomposition opens a new route when capacity/range/
  freshness would be violated.
- **(f) Results.** Total cost TS **17,314.8**, GWO **7,754.9**, TGWO **7,112.5**;
  TGWO saves **50.34 %** / **30.66 %** in travel distance vs TS / GWO `[verified;
  the authors' "143.44 %" cost-saving figure uses a non-standard denominator and
  is flagged, not endorsed]`.
- **(g) Citation.** Zhang, H.; Yan, J.; Wang, L. "Hybrid Tabu-Grey wolf optimizer
  algorithm for enhancing fresh cold-chain logistics distribution." *PLOS ONE*,
  19(8):e0306166, 2024. DOI 10.1371/journal.pone.0306166.

---

## Ranking — the top 3 most exceptional

**#1 — AmoebaNet / Regularized Evolution (App 8).** The clever trick is
*age-based regularization*: culling the oldest genotype instead of the worst
turns a plain GA into a search that resists overfitting to lucky early winners.
It was the first evolved architecture to *beat*
both human- and RL-designed networks at ImageNet scale (83.9 % top-1), overturning
the assumption that reinforcement learning was the right NAS paradigm. The
mechanism is one line of code, and it changed how AutoML search is done.

**#2 — Wolaita Sodo reconfiguration + DG (App 5).** The clever move is
*constraint-as-encoding*: rather than penalising non-radial grids, the search
space itself is restricted to spanning-tree/fundamental-loop combinations, so
every particle is feasible by construction — eliminating an entire class of
wasted evaluations. Coupled with a real deployed 114-transformer Ethiopian
feeder and audited economics (72 % loss cut, ~6-year payback in real ETB), it is
the entry with the hardest real-world grounding and the largest tangible impact.

**#3 — Manjil variable-hub-height wind siting (App 6).** The clever trick is the
*encoding*: promoting hub height to a free decision variable per turbine lets a
tall turbine physically climb out of an upstream turbine's wake — turning a
2-D layout problem into a 3-D one where the extra dimension buys energy that no
same-height layout can. It is a rare case where enlarging the search space
*reduces* effective difficulty, and it is validated on a real, historic wind
farm (+10.75 % power, −9.42 % cost).
