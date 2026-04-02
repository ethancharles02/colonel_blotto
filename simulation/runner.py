from controllers.even_agent import EvenAgent
from controllers.mcts_agent import MCTSAgent
from controllers.random_agent import RandomAgent
from simulation.environment import ColonelBlottoEnv


CONTROLLER_REGISTRY = {
    "random": RandomAgent,
    "even": EvenAgent,
    "mcts": MCTSAgent,
    "dp": None,
}


def create_environment(n_att: int, n_def: int, m: float, p: float, alpha: float, c0: float, retain: bool) -> ColonelBlottoEnv:
    return ColonelBlottoEnv(n_def=n_def, n_att=n_att, m=m, p=p, alpha=alpha, c_0=c0, retain=retain)


def create_controller(
    name: str,
    side: str,
    n_att: int,
    n_def: int,
    m: float,
    p: float,
    alpha: float,
    num_steps: int,
    retain: bool,
):
    if name == "dp":
        from controllers.dp_agent import DPAgent

        return DPAgent(
            side=side,
            n_a=n_att,
            n_d=n_def,
            theta=m / p,
            memory=alpha,
            n_stages=num_steps,
            retain_troops=retain,
        )

    return CONTROLLER_REGISTRY[name](side)


def run_batch_simulation(
    attacker_name: str,
    defender_name: str,
    sim_iters: int,
    num_steps: int,
    n_att: int,
    n_def: int,
    m: float,
    p: float,
    alpha: float,
    c0: float,
    retain: bool,
):
    all_records = []
    attacker = create_controller(attacker_name, "attacker", n_att, n_def, m, p, alpha, num_steps, retain)
    defender = create_controller(defender_name, "defender", n_att, n_def, m, p, alpha, num_steps, retain)

    for sim_id in range(sim_iters):
        env = create_environment(n_att, n_def, m, p, alpha, c0, retain)
        state = env.reset()

        for step in range(num_steps):
            controller_state = dict(state)
            controller_state["stage"] = step + 1

            act_a = attacker.get_action(controller_state)
            act_d = defender.get_action(controller_state)

            next_state, reward_a, reward_d, info = env.step(act_a, act_d)

            all_records.append(
                {
                    "sim_id": sim_id,
                    "step": step,
                    "act_a": act_a,
                    "act_d": act_d,
                    "reward_a": reward_a,
                    "reward_d": reward_d,
                    "capture_rate": info["capture_rate"],
                    "captured_troops": info["capture_rate"] * state["n_att"],
                    "n_att": state["n_att"],
                    "n_def": state["n_def"],
                }
            )

            state = next_state

    metadata = {
        "attacker": attacker_name,
        "defender": defender_name,
        "sim_iters": sim_iters,
        "num_steps": num_steps,
        "n_att": n_att,
        "n_def": n_def,
        "m": m,
        "p": p,
        "alpha": alpha,
        "c0": c0,
        "retain": retain,
    }

    return all_records, metadata
