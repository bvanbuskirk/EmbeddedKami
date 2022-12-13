#!/usr/bin/python3

import numpy as np
from utils import dimensions


class MPCPolicy:
    def __init__(self, action_range=None, simulate_fn=None, cost_fn=None, params=None, cost_weights_dict=None):
        self.action_dim = dimensions["action_dim"]
        self.simulate = simulate_fn
        self.compute_costs = cost_fn
        self.params = params
        self.cost_weights_dict = cost_weights_dict

        if action_range is None:
            self.action_range = np.array([[-1, -1], [1, 1]]) * 0.999
        else:
            self.action_range = action_range

    def update_params_and_weights(self, params, cost_weights_dict):
        self.params = params
        self.cost_weights_dict = cost_weights_dict

    def compute_total_costs(self, predicted_state_sequence, sampled_actions, goals, robot_goals):
        cost_dict = self.compute_costs(predicted_state_sequence, sampled_actions, goals, robot_goals=robot_goals)
        ensemble_size, n_samples, horizon, _ = predicted_state_sequence.shape

        ensemble_costs = np.zeros((ensemble_size, n_samples, horizon))
        for cost_type in cost_dict:
            ensemble_costs += cost_dict[cost_type] * self.cost_weights_dict[cost_type]

        # discount costs through time
        # discount = (1 - 1 / (4 * horizon)) ** np.arange(horizon)

        discount = 0.75 ** np.arange(horizon)
        ensemble_costs *= discount[None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        total_costs = ensemble_costs.mean(axis=(0, 2))
        total_costs -= total_costs.min()
        total_costs /= total_costs.max()

        return total_costs

    def get_action(self):
        return None, None


class RandomShootingPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, initial_state, goals):
        horizon = self.params["horizon"]
        n_samples = self.params["sample_trajectories"]
        robot_goals = self.params["robot_goals"]

        sampled_actions = np.random.uniform(*self.action_range, size=(n_samples, horizon, self.action_dim))
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_total_costs(predicted_state_sequence, sampled_actions, goals, robot_goals)

        best_idx = total_costs.argmin()
        best_action = sampled_actions[best_idx, 0]
        predicted_next_state = predicted_state_sequence[:, best_idx, 0].squeeze()

        return best_action, predicted_next_state


class CEMPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def get_action(self, initial_state, goals):
        horizon = self.params["horizon"]
        n_samples = self.params["sample_trajectories"]
        refine_iters = self.params["refine_iters"]
        alpha = self.params["alpha"]
        n_best = self.params["n_best"]
        robot_goals = self.params["robot_goals"]
        action_trajectory_dim = self.action_dim * horizon

        trajectory_mean = np.zeros(action_trajectory_dim)
        trajectory_std = np.zeros(action_trajectory_dim)
        sampled_actions = np.random.uniform(*self.action_range, size=(n_samples, horizon, self.action_dim))

        for i in range(refine_iters):
            if i > 0:
                sampled_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(n_samples, action_trajectory_dim))
                sampled_actions = sampled_actions.reshape(n_samples, horizon, self.action_dim)

            predicted_state_sequence = self.simulate(initial_state, sampled_actions)
            total_costs = self.compute_total_costs(predicted_state_sequence, sampled_actions, goals, robot_goals)

            action_trajectories = sampled_actions.reshape((n_samples, action_trajectory_dim))
            best_costs_idx = np.argsort(-total_costs)[-n_best:]
            best_trajectories = action_trajectories[best_costs_idx]
            best_trajectories_mean = best_trajectories.mean(axis=0)
            best_trajectories_std = best_trajectories.std(axis=0)

            trajectory_mean = alpha * best_trajectories_mean + (1 - alpha) * trajectory_mean
            trajectory_std = alpha * best_trajectories_std + (1 - alpha) * trajectory_std

            if trajectory_std.max() < 0.02:
                break

        best_action = trajectory_mean[:self.action_dim]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).squeeze()

        return best_action, predicted_next_state


class MPPIPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        self.trajectory_mean = None
        return super().__init__(**kwargs)

    def get_action(self, initial_state, goals):
        horizon = self.params["horizon"]
        n_samples = self.params["sample_trajectories"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        noise_std = self.params["noise_std"]
        robot_goals = self.params["robot_goals"]

        if self.trajectory_mean is None:
            self.trajectory_mean = np.zeros((horizon, self.action_dim))

        just_executed_action = self.trajectory_mean[0].copy()
        self.trajectory_mean[:-1] = self.trajectory_mean[1:]

        sampled_actions = np.empty((n_samples, horizon, self.action_dim))
        noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples, horizon, self.action_dim))

        for t in range(horizon):
            if t == 0:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * just_executed_action
            else:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * sampled_actions[:, t-1]

        sampled_actions = np.clip(sampled_actions, *self.action_range)
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_total_costs(predicted_state_sequence, sampled_actions, goals, robot_goals)

        action_trajectories = sampled_actions.reshape((n_samples, -1))

        weights = np.exp(gamma * -total_costs)
        weighted_trajectories = (weights[:, None] * action_trajectories).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / weights.sum()

        best_action = self.trajectory_mean[:self.action_dim]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).mean(axis=0).squeeze()

        # if np.linalg.norm(best_action) < np.sqrt(2) * 0.5:
        #     import pdb;pdb.set_trace()

        return best_action, predicted_next_state
