# AI Lab

Course assignments for the CSEDU Artificial Intelligence Lab. Three assignments, all set in
Dhaka.

| Folder | Assignment | Method |
|--------|------------|--------|
| [`1/`](1/) | Dhaka Pathfinder | Uninformed and informed search (BFS, DFS, UCS, Greedy, A\*, Weighted A\*) over the real OpenStreetMap road graph, under a twelve-factor cost model |
| [`2/`](2/) | Urban fuel-crisis allocator | Constraint satisfaction / optimisation (backtracking, MRV, LCV, forward checking, AC-3, min-conflicts) |
| [`3/`](3/) | Swarm + decision making | **Part A:** Particle Swarm Optimization for Wi-Fi placement. **Part B:** Value Iteration and Q-Learning for pump control under load-shedding |

**The written submission covering all three is [`3/report/main.pdf`](3/report/main.pdf)**, built
from the instructor's template.

---

## Where to start

- **New to the project, or revising for the viva?** Read
  [`3/docs/START_HERE.md`](3/docs/START_HERE.md). It assumes you remember nothing.
- **Showing this to the instructor?** Read [`DEMO_DAY.md`](DEMO_DAY.md).

## Running each one

```bash
cd 1 && ./run.sh ui          # Streamlit map UI
cd 2 && ./run.sh ui          # Streamlit UI
cd 3 && ./run.sh swarm       # Part A: PSO              (~30 s)
cd 3 && ./run.sh rl          # Part B: VI vs Q-learning (~4 min)
cd 3 && ./run.sh report      # build report/main.pdf
```

Every lab has `./run.sh test`. Labs 1 and 2 use `uv`; lab 3 uses plain `pip`
(`numpy` / `scipy` / `matplotlib` / `pandas` only — every algorithm written from scratch).

All data is cached locally, so nothing needs the network at demo time.
