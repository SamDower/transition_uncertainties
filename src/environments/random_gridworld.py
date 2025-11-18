import numpy as np
from typing import Any, Tuple, Dict, List
from abc import ABC
from .mdp import MDP

class RandomGridworldMDP(MDP):
    """
    Random stochastic gridworld MDP with traps, walls, teleporters,
    and region-dependent stochasticity.
    """

    ACTIONS = {
        0: (-1, 0),   # UP
        1: (1, 0),    # DOWN
        2: (0, -1),   # LEFT
        3: (0, 1),    # RIGHT
    }

    def __init__(
        self,
        N: int,
        p_slip: float = 0.1,
        trap_rate: float = 0.05,
        wall_rate: float = 0.1,
        risky_region_bias: float = 2.0,
        reward_params: dict = None,
        teleporter_rate: float = 0.02,
        small_goal_rate: float = 0.0,
        discount_factor: float = 0.99,
        seed: int = None,
    ):
        super().__init__(discount_factor)
        self.N = N
        self.grid_size = N  # For compatibility with other code
        self.rng = np.random.default_rng(seed)

        if reward_params is None:
            reward_params = {
                "trap_range": (-5.0, -1.0),
                "normal_range": (-0.1, 0.1),
                "goal": 1.0,
                "small_goal": 0.3
            }
        self.reward_params = reward_params

        # Internal maps
        self.cell_type = {}     # (i,j) -> "WALL" | "TRAP" | "TELEPORT" | "NORMAL" | "SMALL_GOAL"
        self.teleport_target = {}  # (i,j) -> (ti,tj) if teleporter

        # Transition + rewards:
        # T[s][a] = list of (next_state, prob)
        self.T: Dict[Tuple[int,int], Dict[int, List[Tuple[Tuple[int,int], float]]]] = {}
        # R[s][a][s'] = reward
        self.R: Dict[Tuple[int,int], Dict[int, Dict[Tuple[int,int], float]]] = {}

        # Start and goal are fixed:
        self.start_state = (1, 1)
        self.goal_state = (N - 2, N - 2)
        self.small_goal_states = set()  # Set of small goal states

        self._generate_cell_types(trap_rate, wall_rate, teleporter_rate, small_goal_rate)
        self._generate_transitions(p_slip, risky_region_bias)
        self._generate_rewards()

        self.state = self.start_state

    # ------------------------------------------------------------------
    # GENERATION PROCEDURES
    # ------------------------------------------------------------------

    def _generate_cell_types(self, trap_rate, wall_rate, teleporter_rate, small_goal_rate):
        N = self.N
        # First pass: assign wall/trap/normal types
        for i in range(N):
            for j in range(N):
                s = (i, j)
                r = self.rng.random()

                if r < wall_rate:
                    self.cell_type[s] = "WALL"
                elif r < wall_rate + trap_rate:
                    self.cell_type[s] = "TRAP"
                else:
                    self.cell_type[s] = "NORMAL"

        # Ensure start & goal aren't walls
        self.cell_type[self.start_state] = "NORMAL"
        self.cell_type[self.goal_state] = "NORMAL"

        # Second pass: assign small goals in regions with x > N/2 or y > N/2 (away from start)
        if small_goal_rate > 0:
            for i in range(N):
                for j in range(N):
                    s = (i, j)
                    # Only place small goals in the regions x > N/2 or y > N/2
                    if (i > N // 2 or j > N // 2) and s != self.goal_state and s != self.start_state:
                        if self.cell_type[s] != "WALL" and self.rng.random() < small_goal_rate:
                            self.cell_type[s] = "SMALL_GOAL"
                            self.small_goal_states.add(s)

        # Third pass: assign teleporters (after all cells have been typed)
        for i in range(N):
            for j in range(N):
                s = (i, j)
                if (
                    self.cell_type[s] != "WALL" and self.cell_type[s] != "SMALL_GOAL"
                    and self.rng.random() < teleporter_rate
                ):
                    self.cell_type[s] = "TELEPORT"
                    # Must teleport to a non-wall state
                    possible = [(x, y) for x in range(N) for y in range(N)
                                if (x, y) != s and self.cell_type[(x, y)] != "WALL"]
                    self.teleport_target[s] = possible[self.rng.integers(len(possible))]

    def _move(self, s, a):
        """Deterministic movement ignoring slip."""
        di, dj = RandomGridworldMDP.ACTIONS[a]
        i, j = s
        ni, nj = i + di, j + dj

        if not (0 <= ni < self.N and 0 <= nj < self.N):
            return s
        if self.cell_type[(ni, nj)] == "WALL":
            return s
        return (ni, nj)

    def _generate_transitions(self, p_slip, risky_region_bias):
        N = self.N
        # Initialize T for all non-wall, non-small-goal states
        for i in range(N):
            for j in range(N):
                s = (i, j)
                if self.cell_type[s] != "WALL" and self.cell_type[s] != "SMALL_GOAL":
                    self.T[s] = {}

        for i in range(N):
            for j in range(N):
                s = (i, j)

                if self.cell_type[s] == "WALL" or self.cell_type[s] == "SMALL_GOAL":
                    continue

                # T[s] already initialized above

                for a in RandomGridworldMDP.ACTIONS.keys():
                    # Teleporter cells override everything
                    if self.cell_type[s] == "TELEPORT":
                        t = self.teleport_target[s]
                        self.T[s][a] = [(t, 1.0)]
                        continue

                    # Compute intended / slip moves
                    intended = self._move(s, a)
                    slip_moves = []
                    for a2 in RandomGridworldMDP.ACTIONS.keys():
                        if a2 == a:
                            continue
                        slip_moves.append(self._move(s, a2))

                    # Apply region-dependent stochasticity
                    local_p_slip = p_slip
                    if (i >= N // 2 and j >= N // 2):  # bottom-right quadrant
                        local_p_slip = np.clip(p_slip * risky_region_bias, 0, 1)

                    p_intended = 1 - local_p_slip
                    p_slip_each = local_p_slip / len(slip_moves)

                    # Build prob distribution
                    dist = {}
                    dist[intended] = dist.get(intended, 0) + p_intended
                    for nxt in slip_moves:
                        dist[nxt] = dist.get(nxt, 0) + p_slip_each

                    # Convert dict → list
                    self.T[s][a] = [(ns, float(p)) for ns, p in dist.items()]

    def _generate_rewards(self):
        N = self.N
        rp = self.reward_params

        # Initialize R for all non-wall, non-small-goal states
        for i in range(N):
            for j in range(N):
                s = (i, j)
                if self.cell_type[s] != "WALL" and self.cell_type[s] != "SMALL_GOAL":
                    self.R[s] = {}

        for i in range(N):
            for j in range(N):
                s = (i, j)
                if self.cell_type[s] == "WALL":
                    continue

                # R[s] already initialized above

                for a in RandomGridworldMDP.ACTIONS.keys():
                    if s not in self.T or a not in self.T[s]:
                        continue
                    self.R[s][a] = {}

                    for (s_next, _) in self.T[s][a]:
                        # Reward logic:
                        if s_next == self.goal_state:
                            r = rp["goal"]
                        elif s_next in self.small_goal_states:
                            r = rp["small_goal"]
                        elif self.cell_type[s] == "TRAP":
                            lo, hi = rp["trap_range"]
                            r = self.rng.uniform(lo, hi)
                        else:
                            lo, hi = rp["normal_range"]
                            r = self.rng.uniform(lo, hi)

                        self.R[s][a][s_next] = float(r)

    # ------------------------------------------------------------------
    # MDP API IMPLEMENTATION
    # ------------------------------------------------------------------

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        """Sample from transition distribution and return next state, reward, done."""
        # Skip if current state is a small goal or main goal (should not happen in normal use)
        if state in self.small_goal_states or state == self.goal_state:
            # Return a dummy transition (stay in place with zero reward)
            return state, 0.0, True

        # Check if state has transitions defined
        if state not in self.T or action not in self.T[state]:
            raise KeyError(f"No transitions defined for state {state}, action {action}")

        transitions = self.T[state][action]
        next_states, probs = zip(*transitions)
        idx = self.rng.choice(len(next_states), p=probs)
        next_state = next_states[idx]
        reward = self.R[state][action][next_state]
        # Done if we reach the main goal or any small goal
        done = (next_state == self.goal_state or next_state in self.small_goal_states)
        return next_state, reward, done

    def get_num_states(self) -> int:
        return self.N * self.N

    def get_num_actions(self) -> int:
        return len(RandomGridworldMDP.ACTIONS)

    def get_all_transitions(self) -> List[Tuple[Tuple[int, int], int, Tuple[int, int], float]]:
        """
        Get all possible transitions (s, a, s', prob) in the MDP.

        Returns:
            List of (state, action, next_state, probability) tuples.
            Only includes transitions with non-zero probability.
        """
        transitions = []
        for state in self.T:
            for action in self.T[state]:
                for next_state, prob in self.T[state][action]:
                    transitions.append((state, action, next_state, prob))
        return transitions

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """
        Convert a linear state index to a (row, col) state tuple.

        Args:
            index: Linear index from 0 to N*N-1.

        Returns:
            Tuple of (row, col).
        """
        row = index // self.N
        col = index % self.N
        return (row, col)

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """
        Convert a (row, col) state tuple to a linear state index.

        Args:
            state: Tuple of (row, col).

        Returns:
            Linear index from 0 to N*N-1.
        """
        row, col = state
        return row * self.N + col

    def visualize_grid(self):
        """
        Print an ASCII visualization of the gridworld.

        Legend:
            S = Start
            G = Goal
            g = Small goal
            # = Wall
            T = Trap
            P = Teleporter
            . = Normal cell
        """
        N = self.N
        print("\n=== GRIDWORLD MAP ===")
        print("Legend: S=Start  G=Goal  g=SmallGoal  #=Wall  T=Trap  P=Teleporter  .=Normal\n")

        for i in range(N):
            row_str = ""
            for j in range(N):
                s = (i, j)

                if s == self.start_state:
                    row_str += " S "
                elif s == self.goal_state:
                    row_str += " G "
                elif s in self.small_goal_states:
                    row_str += " g "
                else:
                    ctype = self.cell_type.get(s, "NORMAL")
                    if ctype == "WALL":
                        row_str += " # "
                    elif ctype == "TRAP":
                        row_str += " T "
                    elif ctype == "TELEPORT":
                        row_str += " P "
                    else:
                        row_str += " . "
            print(row_str)
        print()