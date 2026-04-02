class EvenAgent:
    def __init__(self, side: str):
        if side not in {"attacker", "defender"}:
            raise ValueError(f"Unsupported side: {side}")
        self.side = side

    def get_action(self, state: dict[str, int | float]) -> int:
        troop_key = "n_att" if self.side == "attacker" else "n_def"
        return int(state[troop_key] // 2)
