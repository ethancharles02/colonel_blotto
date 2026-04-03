from simulation.environment import ColonelBlottoEnv
import numpy as np

# Not set up to handle retaining simulations
class MCAgent:
    def __init__(self, side: str, alpha: float = 0.5, is_static: bool=True):
        if side not in {"attacker", "defender"}:
            raise ValueError(f"Unsupported side: {side}")
        self.is_attacker = side == "attacker"
        self.alpha = alpha
        self.is_static = is_static
        self.best_amount = None
        self.mixture = 1
        self.simulation_env = ColonelBlottoEnv(0, 0, 0, 0, 0, 0, False)

    def get_action(self, state: dict[str, int|float]) -> int:
        """Gets the highest one-step utility decision for a non-retaining simulation

        Arguments:
            state {dict[str, int | float]} -- State

        Returns:
            int -- Move to make
        """
        if self.is_static and self.best_amount is not None:
            return self.best_amount

        self.simulation_env.n_def = state["n_def"]
        self.simulation_env.n_att = state["n_att"]
        self.simulation_env.m = state["m"]
        self.simulation_env.p = state["p"]
        self.simulation_env.alpha = self.alpha
        self.simulation_env.c_0 = state["c_t"]

        simulation_data: dict[int, list[int]] = {}
        # Loop through all possible attacker amounts and then check that amount against all the
        # possible defender amounts. Picks the one that gets the highest utility for all defender
        # strategies. It is worth noting that this really isn't a monte carlo simulation since there
        # isn't any randomization implemented here. It is just checking all possibilities since
        # there is a limited number of them
        player_troops = state["n_att"] if self.is_attacker else state["n_def"]
        other_player_troops = state["n_def"] if self.is_attacker else state["n_att"]
        for player_amount in range(player_troops):
            if player_amount not in simulation_data:
                simulation_data[player_amount] = [0, 0]
            simulation_data[player_amount][0] += 1
            for other_player_amount in range(other_player_troops):
                attacker_amount = player_amount if self.is_attacker else other_player_amount
                defender_amount = other_player_amount if self.is_attacker else player_amount
                _, reward_a, reward_d, _ = self.simulation_env.step(attacker_amount, defender_amount)
                reward = reward_a if self.is_attacker else reward_d
                simulation_data[player_amount][1] += reward

        averaged_scores = [[amount, data[1] / data[0] if data[0] > 0 else -np.inf] for amount, data in simulation_data.items()]
        sorted_amounts = sorted(averaged_scores, key=lambda x: x[1], reverse=True)

        best_amount = sorted_amounts[0][0]

        if self.is_static:
            self.best_amount = best_amount

        return best_amount