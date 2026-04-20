"""Search algorithm registry."""

from dhaka_pathfinder.algorithms.base import SearchResult, SearchStats  # noqa: F401
from dhaka_pathfinder.algorithms.informed import (  # noqa: F401
    astar_search,
    greedy_best_first,
    weighted_astar_search,
)
from dhaka_pathfinder.algorithms.uninformed import (  # noqa: F401
    bfs_search,
    dfs_search,
    ucs_search,
)

ALGORITHMS = {
    "bfs": bfs_search,
    "dfs": dfs_search,
    "ucs": ucs_search,
    "greedy": greedy_best_first,
    "astar": astar_search,
    "weighted_astar": weighted_astar_search,
}

INFORMED = {"greedy", "astar", "weighted_astar"}
UNINFORMED = {"bfs", "dfs", "ucs"}
