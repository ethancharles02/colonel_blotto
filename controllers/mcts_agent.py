from enum import Enum

import numpy as np

from simulation.environment import ColonelBlottoEnv


class SimulationPlayerType(Enum):
    DEFENDER = 1
    ATTACKER = 2


def get_random_move(env: ColonelBlottoEnv, player_type: SimulationPlayerType) -> int:
    return int(np.random.randint(0, get_num_troops(env, player_type) + 1))


def get_num_troops(env: ColonelBlottoEnv, player_type: SimulationPlayerType) -> int:
    return env.n_att if player_type is SimulationPlayerType.ATTACKER else env.n_def


class MCTSNode:
    def __init__(self, env: ColonelBlottoEnv, player_type: SimulationPlayerType, parent, c=np.sqrt(2), num_simulations=100):
        self.num_simulations = num_simulations
        self.env: ColonelBlottoEnv = env
        self.player_type: SimulationPlayerType = player_type
        self.other_player_type: SimulationPlayerType = SimulationPlayerType.ATTACKER if player_type is SimulationPlayerType.DEFENDER else SimulationPlayerType.DEFENDER
        self.parent: MCTSNode = parent
        self.num_troops = get_num_troops(env, self.player_type)
        self.terminal = self.num_troops == 0
        self.children = {}
        for m in range(self.num_troops + 1):
            self.children[m] = None

        self.num_visits = 0
        self.total_utility = 0
        self.c = c

    def max_child(self) -> int:
        max_n = 0
        max_m = None

        for m in range(self.num_troops + 1):
            if self.children[m].num_visits > max_n:
                max_n = self.children[m].num_visits
                max_m = m
        return max_m

    def upper_bound(self, num_visits: int):
        bound = (self.total_utility / self.num_visits) + (self.c * np.sqrt(np.log(num_visits) / self.num_visits))
        return bound

    def make_move(self, env: ColonelBlottoEnv, cur_move: int, parent_move: int) -> int:
        if self.player_type is SimulationPlayerType.ATTACKER:
            a_move, d_move = cur_move, parent_move
        else:
            a_move, d_move = parent_move, cur_move
        _, reward_a, reward_d, _ = env.step(a_move, d_move)
        return reward_a if self.player_type is SimulationPlayerType.ATTACKER else reward_d

    def select(self):
        if self.terminal:
            return self

        max_ub = -np.inf
        max_child = None

        for move in range(self.num_troops + 1):
            if self.children[move] is None:
                new_env = self.env.copy()
                utility = self.make_move(new_env, move, get_random_move(new_env, self.other_player_type))

                self.children[move] = MCTSNode(new_env, self.other_player_type, self, self.c, self.num_simulations)
                score = utility * self.num_simulations
                self.children[move].total_utility += score
                return self.children[move]

            current_ub = self.children[move].upper_bound(self.num_visits)

            if current_ub > max_ub:
                max_ub = current_ub
                max_child = move

        return self.children[max_child].select()

    def simulate(self):
        new_env = self.env.copy()
        utility = 0
        orig_num_troops = get_num_troops(new_env, self.player_type)
        orig_other_num_troops = get_num_troops(new_env, self.other_player_type)
        n_att = new_env.n_att
        n_def = new_env.n_def

        for _ in range(self.num_simulations):
            num_troops = orig_num_troops
            other_num_troops = orig_other_num_troops
            new_env.n_att = n_att
            new_env.n_def = n_def
            i = 0
            while (num_troops != 0 and other_num_troops != 0) or i == 0:
                move = np.random.randint(0, num_troops + 1)
                other_move = np.random.randint(0, other_num_troops + 1)
                utility -= self.make_move(new_env, move, other_move)
                num_troops = get_num_troops(new_env, self.player_type)
                other_num_troops = get_num_troops(new_env, self.other_player_type)
                i += 1

        self.total_utility += utility
        self.num_visits += 1
        self.parent.back(-utility)

    def back(self, score):
        self.num_visits += 1
        self.total_utility += score
        if self.parent is not None:
            self.parent.back(-score)


class MCTSAgent:
    def __init__(self, side: str, depth: int = 100, num_simulations_per_depth: int = 5):
        if side not in {"attacker", "defender"}:
            raise ValueError(f"Unsupported side: {side}")
        self.mixture = 1
        self.depth = depth
        self.num_simulations_per_depth = num_simulations_per_depth
        self.player_type = SimulationPlayerType.ATTACKER if side == "attacker" else SimulationPlayerType.DEFENDER

        self.simulation_env = ColonelBlottoEnv(0, 0, 0, 0, 0, 0, True)

    def get_action(self, state: dict[str, int | float]) -> int:
        self.simulation_env.n_def = state["n_def"]
        self.simulation_env.n_att = state["n_att"]
        self.simulation_env.m = state["m"]
        self.simulation_env.p = state["p"]
        self.simulation_env.alpha = 0.5
        self.simulation_env.c_0 = 0.1

        if get_num_troops(self.simulation_env, self.player_type) == 0:
            return 0
        root = MCTSNode(self.simulation_env, self.player_type, None, np.sqrt(2), self.num_simulations_per_depth)
        for _ in range(self.depth):
            cur_node = root.select()
            cur_node.simulate()

        return int(root.max_child())
