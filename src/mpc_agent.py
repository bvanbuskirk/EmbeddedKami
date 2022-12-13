#!/usr/bin/python3

import numpy as np
import torch

from dynamics_network import DynamicsNetwork
from mpc_policies import MPPIPolicy, CEMPolicy, RandomShootingPolicy
from utils import DataUtils, signed_angle_difference, dimensions


class MPCAgent:
    def __init__(self, seed=1, mpc_method='mppi', sensory_model=False, hidden_dim=200, hidden_depth=2, lr=0.001,
                 std=0.01, dist=True, scale=True, ensemble=1, use_object=False,
                 action_range=None, mpc_params=None, cost_weights_dict=None, device=torch.device("cpu")):
        assert ensemble > 0

        input_dim = dimensions["action_dim"]
        output_dim = dimensions["robot_output_dim"]
        self.state_dim = dimensions["state_dim"]

        if use_object:
            input_dim += dimensions["object_input_dim"]
            output_dim += dimensions["object_output_dim"]
            self.state_dim += dimensions["state_dim"]

        self.dtu = DataUtils(use_object=use_object)
        self.models = [DynamicsNetwork(input_dim, output_dim, self.dtu, hidden_dim=hidden_dim, hidden_depth=hidden_depth,
                                       lr=lr, std=std, dist=dist, use_object=use_object, scale=scale)
                        for _ in range(ensemble)]
        for model in self.models:
            model.to(device)

        if mpc_method == 'mppi':
            policy = MPPIPolicy
        elif mpc_method == 'cem':
            policy = CEMPolicy
        elif mpc_method == 'shooting':
            policy = RandomShootingPolicy
        else:
            raise NotImplementedError

        if sensory_model:
            sensory_in = dimensions["sensory_input_dim"]
            sensory_out = output_dim
            self.state_estimator = DynamicsNetwork(sensory_in, sensory_out, self.dtu, hidden_dim=hidden_dim, hidden_depth=hidden_depth,
                                       lr=lr, std=std, dist=dist, use_object=use_object, scale=scale)

        self.policy = policy(action_range=action_range, simulate_fn=self.simulate, cost_fn=self.compute_costs,
                             params=mpc_params, cost_weights_dict=cost_weights_dict)

        self.seed = seed
        self.scale = scale
        self.ensemble = ensemble
        self.use_object = use_object

    @property
    def model(self):
        return self.models[0]

    def get_action(self, state, goals):
        return self.policy.get_action(state, goals)

    def simulate(self, initial_state, action_sequence):
        n_samples, horizon, _ = action_sequence.shape
        initial_state = np.tile(initial_state, (n_samples, 1))
        state_sequence = np.empty((len(self.models), n_samples, horizon, self.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            for t in range(horizon):
                action = action_sequence[:, t]
                with torch.no_grad():
                    if t == 0:
                        state_sequence[i, :, t] = model(initial_state, action, sample=False, delta=False)
                    else:
                        state_sequence[i, :, t] = model(state_sequence[i, :, t-1], action, sample=False, delta=False)

        # if n_samples > 1:
        #     if np.linalg.norm(state_sequence[0, 0] - state_sequence[0, 1]) == 0:
        #         import pdb;pdb.set_trace()

        return state_sequence

    def compute_costs(self, state, action, goals, robot_goals=False, signed=False):
        state_dim = dimensions["state_dim"]
        if self.use_object:
            robot_state = state[:, :, :, :state_dim]
            object_state = state[:, :, :, state_dim:2*state_dim]

            effective_state = robot_state if robot_goals else object_state
        else:
            effective_state = state[:, :, :, :state_dim]

        # distance to goal position
        state_to_goal_xy = (goals - effective_state)[:, :, :, :-1]
        dist_cost = np.linalg.norm(state_to_goal_xy, axis=-1)
        if signed:
            dist_cost *= forward

        # difference between current and goal heading
        current_angle = effective_state[:, :, :, 2]
        target_angle = np.arctan2(state_to_goal_xy[:, :, :, 1], state_to_goal_xy[:, :, :, 0])
        heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (heading_cost > 0) * 2 - 1
        forward = (np.abs(heading_cost) < np.pi / 2) * 2 - 1

        heading_cost[forward == -1] = (heading_cost[forward == -1] + np.pi) % (2 * np.pi)
        heading_cost = np.stack((heading_cost, 2 * np.pi - heading_cost)).min(axis=0)

        if signed:
            heading_cost *= left * forward
        else:
            heading_cost = np.abs(heading_cost)

        # object-robot separation
        if self.use_object:
            object_to_robot_xy = (robot_state - object_state)[:, :, :, :-1]
            sep_cost = np.linalg.norm(object_to_robot_xy, axis=-1)
        else:
            sep_cost = np.array([0.])

        # object-robot heading difference
        if self.use_object:
            robot_theta, object_theta = robot_state[:, :, :, -1], object_state[:, :, :, -1]
            heading_diff = (robot_theta - object_theta) % (2 * np.pi)
            heading_diff_cost = np.stack((heading_diff, 2 * np.pi - heading_diff), axis=1).min(axis=1)
        else:
            heading_diff_cost = np.array([0.])

        # action magnitude
        norm_cost = -np.linalg.norm(action, axis=-1)

        cost_dict = {
            "distance": dist_cost,
            "heading": heading_cost,
            "action_norm": norm_cost,
            "separation": sep_cost,
            "heading_difference": heading_diff_cost,
        }

        return cost_dict
