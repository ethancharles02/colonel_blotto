# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linprog

# Dynamic Programming Solution

class DPAgent:
  # Initializes the agent with the parameters of the game
  def __init__(self, side, n_a, n_d, theta, memory, n_stages, retain_troops=False, capture_resolution=100,
               exploit=False):
    # Make sure variables are valid
    assert side in ["attacker", "defender"]
    assert n_a > 0 and n_d > 0
    assert theta >= 0 and theta <= 1
    assert memory >= 0 and memory <= 1
    assert n_stages > 0
    assert capture_resolution > 5
    # Store variables
    self.side = side
    self.n_a = n_a
    self.n_d = n_d
    self.total_troops = n_a + n_d
    self.theta = theta
    self.memory = memory
    self.n_stages = n_stages
    self.retain_troops = retain_troops
    self.capture_resolution = capture_resolution
    self.exploit = exploit
    self.valid_capture_probs = np.linspace(0, 1, capture_resolution + 1)[:-1] + 0.5 / capture_resolution
    # Solve the game
    if not retain_troops:
      print(f"Solving each stage with N_a={n_a}, N_d={n_d}, theta={theta}, memory={memory}, exploit={exploit}...")
      self.solve_baseline()
    else:
      print(f"Solving each stage with total troops={n_a + n_d}, theta={theta}, memory={memory}, exploit={exploit}, and troop retention...")
      self.solve_with_retention()

  # Returns a sample for the configured player
  def get_action(self, state):
    stage = state["stage"]
    perceived_capture = state["c_t"]
    attacking_troops = state["n_att"]
    # Make sure stage and perceived capture are valid
    assert stage >= 1 and stage <= self.n_stages
    assert perceived_capture >= 0 and perceived_capture <= 1
    # Make sure attacking troops are set if retain is true
    if self.retain_troops:
      assert attacking_troops >= 0 and attacking_troops <= self.total_troops
      defending_troops = self.total_troops - attacking_troops
    # Get the perceived capture index
    capture_index = np.argmin(np.abs(self.valid_capture_probs - perceived_capture))
    # Check if baseline
    if not self.retain_troops:
      if self.side == "attacker":
        mixture = self.attack_policies[stage - 1, capture_index, :]
        strategy = np.random.choice(np.arange(self.n_a + 1), p=mixture)
      else:
        mixture = self.defense_policies[stage - 1, capture_index, :]
        strategy = np.random.choice(np.arange(self.n_d + 1), p=mixture)
      return int(strategy)
    else:
      if self.side == "attacker":
        mixture = self.attack_policies[stage - 1, attacking_troops, capture_index, :attacking_troops + 1]
        strategy = np.random.choice(np.arange(attacking_troops + 1), p=mixture)
      else:
        mixture = self.defense_policies[stage - 1, attacking_troops, capture_index, :defending_troops + 1]
        strategy = np.random.choice(np.arange(defending_troops + 1), p=mixture)
      return int(strategy)
  
  # Solves the mixed strategies of each side
  def solve_mixtures(self, payoff_matrix):
    # Get the choices
    attack_choices, defend_choices = payoff_matrix.shape
    # Define the primal LP
    c = np.r_[np.zeros(attack_choices), 1]
    A_ub = np.hstack([payoff_matrix.T, -np.ones((defend_choices, 1))])
    b_ub = np.zeros(defend_choices)
    A_eq = np.hstack([np.ones((1, attack_choices)), [[0]]])
    b_eq = [1]
    bounds = [(0, None)]*attack_choices + [(None, None)]
    # Solve minimization
    res_row = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method="highs")
    # Check for error
    if not res_row.success:
      raise RuntimeError(res_row.message)
    # Store the attack strategy and payoff
    attack_strategy, attack_payoff = res_row.x[:attack_choices], res_row.x[-1]
    # Define the dual LP
    c = np.r_[np.zeros(defend_choices), -1]
    A_ub = np.hstack([-payoff_matrix,  np.ones((attack_choices, 1))])
    b_ub = np.zeros(attack_choices)
    A_eq = np.hstack([np.ones((1, defend_choices)), [[0]]])
    b_eq = [1]
    bounds = [(0, None)]*defend_choices + [(None, None)]
    # Solve maximization
    res_col = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method="highs")
    # Check for error
    if not res_col.success:
      raise RuntimeError(res_col.message)
    # Store the defense strategy and payoff
    defend_strategy, defend_payoff = res_col.x[:defend_choices], res_col.x[-1]
    # Average the payoffs (in case they are slightly different)
    value = float(0.5*(attack_payoff + defend_payoff))
    attack_strategy = (attack_strategy + attack_strategy[::-1]) / 2
    defend_strategy = (defend_strategy + defend_strategy[::-1]) / 2
    # Clip strategies for safety
    attack_strategy = np.clip(attack_strategy, 0, 1)
    attack_strategy /= np.sum(attack_strategy)
    defend_strategy = np.clip(defend_strategy, 0, 1)
    defend_strategy /= np.sum(defend_strategy)
    # Return payoff matrix, both strategies, and the expected payoff
    return attack_strategy, defend_strategy, value
  
  # Solves equilibrium without retention
  def solve_baseline(self):
    # Initialize equilibrium policies and utilities
    self.attack_policies = np.zeros((self.n_stages, self.capture_resolution, self.n_a + 1))
    self.defense_policies = np.zeros((self.n_stages, self.capture_resolution, self.n_d + 1))
    self.expected_utility = np.zeros((self.n_stages, self.capture_resolution))
    # Iterate through each stage
    for stage in tqdm(range(self.n_stages, 0, -1)):
      # Iterate through each perceived capture probability
      for capture_index in range(self.capture_resolution):
        perceived_capture = self.valid_capture_probs[capture_index]
        # Set the default policies
        if self.n_a % 2 == 0:
          self.attack_policies[stage - 1, capture_index, self.n_a // 2] = 1.0
        else:
          self.attack_policies[stage - 1, capture_index, self.n_a // 2:self.n_a // 2 + 2] = 0.5
        if self.n_d % 2 == 0:
          self.defense_policies[stage - 1, capture_index, self.n_d // 2] = 1.0
        else:
          self.defense_policies[stage - 1, capture_index, self.n_d // 2:self.n_d // 2 + 2] = 0.5
        # Check if deterrence is currently active
        if perceived_capture > self.theta:
          # Compute the expected utility of the next round
          if stage == self.n_stages:
            next_expected_utility = 0.0
          else:
            next_perceived_capture = self.memory * perceived_capture
            next_capture_index = np.argmin(np.abs(self.valid_capture_probs - next_perceived_capture))
            next_expected_utility = self.expected_utility[stage, next_capture_index]
          # Store the current expected utility
          self.expected_utility[stage - 1, capture_index] = next_expected_utility + 1.0
        else:
          # Contruct the payoff matrix
          payoff_matrix = np.zeros((self.n_a + 1, self.n_d + 1))
          for i in range(self.n_a + 1):
            for j in range(self.n_d + 1):
              # Compute the utility and captures for this round
              current_utility = 1.0
              current_capture = 0
              if i > j:
                current_utility -= 1.0
              else:
                current_capture += i
              if self.n_a - i > self.n_d - j:
                current_utility -= 1.0
              else:
                current_capture += self.n_a - i
              current_capture /= self.n_a
              # Compute the expected utility of the next round
              if stage == self.n_stages:
                next_expected_utility = 0.0
              else:
                next_perceived_capture = self.memory * perceived_capture + (1 - self.memory) * current_capture
                next_capture_index = np.argmin(np.abs(self.valid_capture_probs - next_perceived_capture))
                next_expected_utility = self.expected_utility[stage, next_capture_index]
              # Update the payoff matrix
              payoff_matrix[i, j] = current_utility + next_expected_utility
          # Compute the equilibrium mixed strategies
          if not self.exploit:
            attack_strategy, defend_strategy, value = self.solve_mixtures(payoff_matrix)
          elif self.side == "attacker":
            defend_strategy = np.ones(self.n_d + 1) / (self.n_d + 1)
            pure_values = payoff_matrix @ defend_strategy
            value = np.min(pure_values)
            attack_strategy = np.zeros(self.n_a + 1)
            attack_strategy[np.abs(pure_values - value) < 1e-3] = 1.0
            attack_strategy /= np.sum(attack_strategy)
          elif self.side == "defender":
            attack_strategy = np.ones(self.n_a + 1) / (self.n_a + 1)
            pure_values = attack_strategy @ payoff_matrix
            value = np.max(pure_values)
            defend_strategy = np.zeros(self.n_d + 1)
            defend_strategy[np.abs(pure_values - value) < 1e-3] = 1.0
            defend_strategy /= np.sum(defend_strategy)
          # Store the equilibrium policies
          self.attack_policies[stage - 1, capture_index, :] = attack_strategy
          self.defense_policies[stage - 1, capture_index, :] = defend_strategy
          self.expected_utility[stage - 1, capture_index] = value

  # Solves retention strategy
  def solve_with_retention(self):
    # Initialize equilibrium policies and utilities
    self.attack_policies = np.zeros((self.n_stages, self.total_troops + 1, self.capture_resolution, self.total_troops + 1))
    self.defense_policies = np.zeros((self.n_stages, self.total_troops + 1, self.capture_resolution, self.total_troops + 1))
    self.expected_utility = np.zeros((self.n_stages, self.total_troops + 1, self.capture_resolution))
    # Iterate through each stage
    for stage in tqdm(range(self.n_stages, 0, -1)):
      # Iterate through each number of attacking troops
      for attacking_troops in range(self.total_troops + 1):
        defending_troops = self.total_troops - attacking_troops
        # Iterate through each perceived capture probability
        for capture_index in range(self.capture_resolution):
          perceived_capture = self.valid_capture_probs[capture_index]
          # Set the default policies
          if attacking_troops % 2 == 0:
            self.attack_policies[stage - 1, attacking_troops, capture_index, attacking_troops // 2] = 1.0
          else:
            self.attack_policies[stage - 1, attacking_troops, capture_index, attacking_troops // 2:attacking_troops // 2 + 2] = 0.5
          if defending_troops % 2 == 0:
            self.defense_policies[stage - 1, attacking_troops, capture_index, defending_troops // 2] = 1.0
          else:
            self.defense_policies[stage - 1, attacking_troops, capture_index, defending_troops // 2:defending_troops // 2 + 2] = 0.5
          # Check if deterrence is currently active
          if perceived_capture > self.theta:
            # Compute the expected utility of the next round
            if stage == self.n_stages:
              next_expected_utility = 0.0
            else:
              next_perceived_capture = self.memory * perceived_capture
              next_capture_index = np.argmin(np.abs(self.valid_capture_probs - next_perceived_capture))
              next_expected_utility = self.expected_utility[stage, attacking_troops, next_capture_index]
            # Store the current expected utility
            self.expected_utility[stage - 1, attacking_troops, capture_index] = next_expected_utility + 1.0
          # Check if the attacker has all of the troops
          elif attacking_troops == self.total_troops:
            # Compute the expected utility of the next round
            if stage == self.n_stages:
              next_expected_utility = 0.0
            else:
              next_perceived_capture = self.memory * perceived_capture
              next_capture_index = np.argmin(np.abs(self.valid_capture_probs - next_perceived_capture))
              next_expected_utility = self.expected_utility[stage, attacking_troops, next_capture_index]
            # Store the current expected utility
            self.expected_utility[stage - 1, attacking_troops, capture_index] = next_expected_utility - (1.0 if attacking_troops > 1 else 0)
          # Check if the defender has all of the troops
          elif attacking_troops == 0:
            # Compute the expected utility of the next round
            if stage == self.n_stages:
              next_expected_utility = 0.0
            else:
              next_perceived_capture = self.memory * perceived_capture
              next_capture_index = np.argmin(np.abs(self.valid_capture_probs - next_perceived_capture))
              next_expected_utility = self.expected_utility[stage, attacking_troops, next_capture_index]
            # Store the current expected utility
            self.expected_utility[stage - 1, attacking_troops, capture_index] = next_expected_utility + 1.0
          # Otherwise, compute equilibrium strategies
          else:
            # Contruct the payoff matrix
            payoff_matrix = np.zeros((attacking_troops + 1, defending_troops + 1))
            for i in range(attacking_troops + 1):
              for j in range(defending_troops + 1):
                # Compute the utility and captures for this round
                current_utility = 1.0
                attackers_captured = 0
                defenders_captured = 0
                if i > j:
                  current_utility -= 1.0
                  defenders_captured += j
                else:
                  attackers_captured += i
                if attacking_troops - i > defending_troops - j:
                  current_utility -= 1.0
                  defenders_captured += defending_troops - j
                else:
                  attackers_captured += attacking_troops - i
                current_capture = attackers_captured / attacking_troops
                # Compute the expected utility of the next round
                if stage == self.n_stages:
                  next_expected_utility = 0.0
                else:
                  next_perceived_capture = self.memory * perceived_capture + (1 - self.memory) * current_capture
                  next_capture_index = np.argmin(np.abs(self.valid_capture_probs - next_perceived_capture))
                  next_attacking_troops = attacking_troops + defenders_captured - attackers_captured
                  next_expected_utility = self.expected_utility[stage, next_attacking_troops, next_capture_index]
                # Update the payoff matrix
                payoff_matrix[i, j] = current_utility + next_expected_utility
            # Compute the equilibrium mixed strategies
            if not self.exploit:
              attack_strategy, defend_strategy, value = self.solve_mixtures(payoff_matrix)
            elif self.side == "attacker":
              defend_strategy = np.ones(defending_troops + 1) / (defending_troops + 1)
              pure_values = payoff_matrix @ defend_strategy
              value = np.min(pure_values)
              attack_strategy = np.zeros(attacking_troops + 1)
              attack_strategy[np.abs(pure_values - value) < 1e-3] = 1.0
              attack_strategy /= np.sum(attack_strategy)
            elif self.side == "defender":
              attack_strategy = np.ones(attacking_troops + 1) / (attacking_troops + 1)
              pure_values = attack_strategy @ payoff_matrix
              value = np.max(pure_values)
              defend_strategy = np.zeros(defending_troops + 1)
              defend_strategy[np.abs(pure_values - value) < 1e-3] = 1.0
              defend_strategy /= np.sum(defend_strategy)
            # Store the equilibrium policies
            self.attack_policies[stage - 1, attacking_troops, capture_index, :attacking_troops + 1] = attack_strategy
            self.defense_policies[stage - 1, attacking_troops, capture_index, :defending_troops + 1] = defend_strategy
            self.expected_utility[stage - 1, attacking_troops, capture_index] = value
