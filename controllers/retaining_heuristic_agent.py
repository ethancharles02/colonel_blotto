import numpy as np

class RetainingHeuristicAgent:
    def __init__(self, side: str):
        if side not in {"attacker", "defender"}:
            raise ValueError(f"Unsupported side: {side}")
        self.is_attacker = side == "attacker"

    def get_action(self, state: dict[str, int|float]) -> int:
        """Gets an action for the attacker when it has retaining troops. It will prioritize winning
        troops and then start distributing troops when it is a guaranteed win

        Arguments:
            state {dict[str, int | float]} -- State dictionary

        Returns:
            int -- Action to make
        """
        player_troops = state["n_att"] if self.is_attacker else state["n_def"]
        other_player_troops = state["n_def"] if self.is_attacker else state["n_att"]
        return np.random.choice([0, player_troops]) if player_troops <= other_player_troops * 2 else player_troops // 2