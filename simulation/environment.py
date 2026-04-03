import copy


class Controller:
    def get_action(self, state):
        raise NotImplementedError("Not yet implemented")


class ColonelBlottoEnv:
    def __init__(self, n_def=2, n_att=2, m=1.0, p=2.0, alpha=0.5, c_0=0.1, retain=False):
        self.n_def = n_def
        self.n_att = n_att
        self.m = m
        self.p = p
        self.alpha = alpha
        self.c_0 = c_0
        self.retain = retain

        self.c_t = self.c_0
        self.history = []

    def reset(self):
        self.c_t = self.c_0
        self.history = []
        return self._get_state()

    def copy(self):
        new_env = ColonelBlottoEnv(self.n_def, self.n_att, self.m, self.p, self.alpha, self.c_0, self.retain)
        new_env.history = copy.deepcopy(self.history)
        new_env.c_t = self.c_t
        return new_env

    def _get_state(self):
        return {
            "n_def": self.n_def,
            "n_att": self.n_att,
            "c_t": self.c_t,
            "m": self.m,
            "p": self.p,
        }

    def step(self, action_att, action_def, m_override=None):
        current_m = m_override if m_override is not None else self.m
        theta = current_m / self.p
        defect = self.c_t > theta
        attacker_captured = 0
        defender_captured = 0

        if defect:
            def_utility = 1
            att_utility = -1
            capture_rate = 0.0

        else:
            att_1 = action_att
            att_2 = self.n_att - action_att

            def_1 = action_def
            def_2 = self.n_def - action_def

            def_wins_1 = def_1 >= att_1
            attacker_captured_1 = att_1 if def_wins_1 else 0
            defender_captured_1 = 0 if def_wins_1 else def_1
            num_captures_1 = attacker_captured_1 - defender_captured_1
            def_wins_2 = def_2 >= att_2
            attacker_captured_2 = att_2 if def_wins_2 else 0
            defender_captured_2 = 0 if def_wins_2 else def_2
            num_captures_2 = attacker_captured_2 - defender_captured_2

            total_captures = num_captures_1 + num_captures_2
            attacker_captured = attacker_captured_1 + attacker_captured_2
            defender_captured = defender_captured_1 + defender_captured_2

            if def_wins_1 and def_wins_2:
                def_utility = 1
            elif def_wins_1 or def_wins_2:
                def_utility = 0
            else:
                def_utility = -1

            att_utility = -def_utility

            capture_rate = attacker_captured / self.n_att if self.n_att > 0 else 0

            if self.retain:
                self.n_def += total_captures
                self.n_att -= total_captures
                self.n_att += 0

        self.c_t = self.alpha * self.c_t + (1 - self.alpha) * capture_rate

        step_info = {
            "defect": defect,
            "def_utility": def_utility,
            "att_utility": att_utility,
            "capture_rate": capture_rate,
            "attacker_captured": attacker_captured,
            "defender_captured": defender_captured,
            "new_c_t": self.c_t,
        }
        self.history.append(step_info)

        return self._get_state(), att_utility, def_utility, step_info
