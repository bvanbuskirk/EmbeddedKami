#!/usr/bin/python3

import os
import argparse
import pickle as pkl
import numpy as np
import rospy
from tqdm import trange
import wandb
from pdb import set_trace

from mpc_agent import MPCAgent
from replay_buffer import ReplayBuffer
from logger import Logger
from train_utils import AGENT_PATH, train_from_buffer
from utils import build_action_msg, signed_angle_difference, dimensions

from ros_stuff.msg import ProcessedStates, RobotCmd
from std_msgs.msg import Time
import tf2_ros

# seed for reproducibility
# SEED = 0
# import torch; torch.manual_seed(SEED)
# np.random.seed(SEED)
SEED = np.random.randint(0, 1e9)


class Experiment():
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp,
                 robot_id, object_id, mpc_horizon, mpc_samples, n_rollouts, tolerance, steps_per_lap,
                 calibrate, plot, new_buffer, pretrain, robot_goals, scale, mpc_method, save_freq,
                 online, mpc_refine_iters, pretrain_samples, consecutive, random_steps, use_all_data, debug,
                 save_agent, load_agent, train_epochs, mpc_gamma, ensemble, batch_size, rand_ac_mean,
                 meta, warmup_test, trajectory, **kwargs):
        # flags for different stages of eval
        self.started = False
        self.done = False

        # counters
        self.total_steps = 0
        self.lap_step = 0
        self.laps = 0

        # AR tag ids for state lookup
        self.robot_id = robot_id
        self.object_id = object_id

        # states
        self.robot_vel = robot_vel
        self.object_vel = object_vel
        self.state_dict = {
            "robot": robot_pos,
            "object": object_pos,
            "corner": corner_pos,
        }
        self.action_timestamp = action_timestamp
        self.last_action_timestamp = self.action_timestamp.copy()

        # online data collection/learning params
        self.random_steps = 0 if pretrain or load_agent else random_steps
        self.gradient_steps = 2
        self.online = online
        self.save_freq = save_freq
        self.pretrain_samples = pretrain_samples

        # system params
        self.debug = debug
        self.use_object = (self.object_id >= 0)
        self.duration = 0.4 if self.use_object else 0.2
        # self.duration = 0.2
        self.action_range = np.array([[-1, -1], [1, 1]]) * 0.999
        self.rand_ac_mean = rand_ac_mean
        # self.post_action_sleep_time = 0.7
        self.post_action_sleep_time = 0.5
        self.max_vel_magnitude = 0.1

        # train params
        self.pretrain = pretrain
        self.consecutive = consecutive
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.use_all_data = use_all_data
        self.save_agent = save_agent
        self.load_agent = load_agent

        # misc
        self.trajectory = trajectory     #'square' or 'fig8'
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.steps_per_lap = steps_per_lap
        self.robot_goals = robot_goals
        self.scale = scale
        self.debug = debug
        self.meta = meta
        self.do_warmup_test = warmup_test
        self.start_lap_step = np.floor(np.random.rand() * steps_per_lap)
        self.reverse_lap = False
        self.out_of_bounds = False
        self.predicted_next_state = np.zeros(6) if self.use_object else np.zeros(3)

        if not robot_goals:
            assert self.use_object

        self.all_actions = []
        self.n_costs = 3
        self.costs = np.empty((0, self.n_costs))      # dist, heading, total

        self.logger = Logger(self, plot, corner_pos, **kwargs)
        self.replay_buffer, self.validation_buffer = self.logger.load_buffer(robot_id)

        if new_buffer or self.replay_buffer is None:
            state_dim = 2 * dimensions["state_dim"] if self.use_object else dimensions["state_dim"]
            self.replay_buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=dimensions["action_dim"])

        print("setting up tf buffer/listener")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        print("finished setting up tf buffer/listener")

        if not self.debug:
            self.action_publisher = rospy.Publisher("/action_topic", RobotCmd, queue_size=1)

        # self.yaw_offset_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/yaw_offsets.npy"
        # if not os.path.exists(self.yaw_offset_path) or calibrate:
        #     self.yaw_offsets = np.zeros(10)
        #     self.calibrate()

        # self.yaw_offsets = np.load(self.yaw_offset_path)
        self.yaw_offsets = np.zeros(10)

        self.define_goal_trajectory()

        # weights for MPC cost terms
        cost_weights_dict = {
            "distance": 1.,
            "heading": 0.,
            "action_norm": 0.,
            "distance_bonus": 0.,
            "separation": 0.,
            "heading_difference": 0.,
        }

        # parameters for MPC methods
        mpc_params = {
            "horizon": mpc_horizon,
            "sample_trajectories": mpc_samples,
            "robot_goals": robot_goals,
        }
        if mpc_method == 'mppi':
            mpc_params.update({
                "beta": 0.8,
                "gamma": mpc_gamma,
                "noise_std": 2.,
            })
        elif mpc_method == 'cem':
            mpc_params.update({
                "alpha": 0.8,
                "n_best": 30,
                "refine_iters": mpc_refine_iters,
            })
        elif mpc_method == 'shooting':
            mpc_params.update({})
        else:
            raise NotImplementedError

        # load or initialize and potentially pretrain new agent
        if load_agent:
            with open(AGENT_PATH, "rb") as f:
                self.agent = pkl.load(f)

            self.agent.policy.update_params_and_weights(mpc_params, cost_weights_dict)
        else:
            self.agent = MPCAgent(seed=SEED, mpc_method=mpc_method, dist=True, scale=self.scale,
                                  hidden_dim=200, hidden_depth=1, lr=0.001, std=0.1,
                                  ensemble=ensemble, use_object=self.use_object,
                                  action_range=self.action_range, mpc_params=mpc_params,
                                  cost_weights_dict=cost_weights_dict)

            if pretrain:
                train_from_buffer(
                    self.agent, self.replay_buffer, validation_buffer=self.validation_buffer,
                    pretrain=pretrain, pretrain_samples=pretrain_samples, consecutive=consecutive, save_agent=save_agent,
                    train_epochs=train_epochs, use_all_data=use_all_data, batch_size=batch_size,
                    meta=meta,
                )

        np.set_printoptions(suppress=True)

    def run(self):
        if self.do_warmup_test:
            self.warmup_test()

        self.take_warmup_steps()

        while not rospy.is_shutdown():
            t = rospy.get_time()

            if not self.started and self.dist_to_start() < self.tolerance \
                        and (self.pretrain or self.replay_buffer.size >= self.random_steps):
                self.started = True

            state, action = self.step()
            self.logger.log_states(self.state_dict, self.get_goal(), self.started)

            if self.done:
                rospy.signal_shutdown("Finished! All robots reached goal.")
                return

            print("TIME:", rospy.get_time() - t)

            if not self.debug:
                if self.out_of_bounds:
                    self.out_of_bounds = False
                else:
                    next_state = self.get_state()
                    self.collect_training_data(state, action, next_state)

                if self.started:
                    self.print_prediction_errors(state, next_state)

            if self.started:
                self.all_actions.append(action.tolist())

            if self.online:
                self.update_model_online()

    def step(self):
        if not self.pretrain and not self.load_agent and self.replay_buffer.size == self.random_steps:
            train_from_buffer(
                self.agent, self.replay_buffer, validation_buffer=self.validation_buffer,
                save_agent=self.save_agent, train_epochs=self.train_epochs,
                use_all_data=self.use_all_data, batch_size=self.batch_size,
                meta=self.meta,
            )

        state = self.get_state()
        action = self.get_take_action(state)

        self.check_rollout_finished()

        self.total_steps += 1
        return state, action

    def print_prediction_errors(self, state, next_state):
        distance_travelled = np.linalg.norm((next_state - state)[:2])
        heading_travelled = signed_angle_difference(next_state[2], state[2])
        distance_error = np.linalg.norm((next_state - self.predicted_next_state)[:2])
        heading_error = signed_angle_difference(next_state[2], self.predicted_next_state[2])
        robot_prediction_error = np.array([
            distance_error,
            distance_error / (distance_travelled + 1e-8),
            heading_error,
            np.abs(heading_error / (heading_travelled + 1e-8)),
        ])
        print("\nROBOT PREDICTION ERROR (distance, distance normalized, heading, heading_normalized):", robot_prediction_error)
        self.logger.log_model_errors(robot_prediction_error, object=False)
        error_dict = {"robot_dist_error" : distance_error,
                      "robot_heading_error" : heading_error}

        if self.use_object:
            distance_travelled = np.linalg.norm((next_state - state)[3:5])
            heading_travelled = signed_angle_difference(next_state[5], state[5])
            distance_error = np.linalg.norm((next_state - self.predicted_next_state)[3:5])
            heading_error = signed_angle_difference(next_state[5], self.predicted_next_state[5])
            object_prediction_error = np.array([
                distance_error,
                distance_error / distance_travelled,
                heading_error,
                np.abs(heading_error / heading_travelled),
            ])
            print("OBJECT PREDICTION ERROR (distance, distance normalized, heading):", object_prediction_error, "\n")
            self.logger.log_model_errors(object_prediction_error, object=True)
            error_dict["object_dist_error"] = distance_error
            error_dict["object_heading_error"] = heading_error
        wandb.log(error_dict, step=self.total_steps)

    def take_warmup_steps(self):
        if self.debug:
            return

        for _ in trange(5, desc="Warmup Steps"):
            idx = np.random.randint(0, 2)
            negate = np.random.randint(0, 2)
            actions = np.array([[1, 1], [1, -1]]) * 0.999
            action = actions[idx] * (-1 if negate else 1)
            action_msg = build_action_msg(action, self.duration, -1)
            self.action_publisher.publish(action_msg)
            rospy.sleep(0.001)

            while self.last_action_timestamp == self.action_timestamp:
                rospy.wait_for_message("/action_timestamps", Time, timeout=10)
            self.last_action_timestamp = self.action_timestamp.copy()
            rospy.sleep(self.duration + self.post_action_sleep_time)

    def get_state(self):
        robot_pos = self.state_dict["robot"].copy()
        object_pos = self.state_dict["object"].copy()
        corner_pos = self.state_dict["corner"].copy()

        if np.any(robot_pos[:2] > corner_pos[:2]) or np.any(robot_pos[:2] < 0):
            print("\nOUT OF BOUNDS\n")
            self.out_of_bounds = True
            import pdb;pdb.set_trace()

        # apply yaw offset
        robot_pos[2] = (robot_pos[2] + self.yaw_offsets[self.robot_id]) % (2 * np.pi)

        if self.use_object:
            object_pos[2] = (object_pos[2] + self.yaw_offsets[self.object_id]) % (2 * np.pi)

            return np.concatenate((robot_pos, object_pos), axis=0)
        else:
            return robot_pos

    def get_take_action(self, state):
        goals = self.get_next_n_goals(self.agent.policy.params["horizon"])
        curr_goal = goals[0]

        if self.replay_buffer.size >= self.random_steps:
            action, self.predicted_next_state = self.agent.get_action(state, goals)

            self.lap_step += (-1 if self.reverse_lap else 1) if self.started else 0
        else:
            print("TAKING RANDOM ACTION")
            # idx = self.replay_buffer.size % 2
            # locs = np.array([[1, 1], [1, -1]]) * self.rand_ac_mean
            # if self.replay_buffer.size < self.random_steps / 2:
            #     locs *= -1
            # scale = 0.7 if self.rand_ac_mean == 0 else 0.3
            # # action = np.random.normal(loc=locs[idx], scale=scale, size=dimensions["action_dim"])

            # action = np.random.uniform(-1.1, 1.1, size=dimensions["action_dim"]).squeeze()

            # BETA DIST
            # action = np.random.beta(0.6, 0.6, size=2)
            # if np.random.uniform(0.0, 1.0) > 0.5:
            #     action[0] *= -1
            # if np.random.uniform(0.0, 1.0) > 0.5:
            #     action[1] *= -1

            # # MESHGRID
            rng = np.linspace(-1, 1, np.floor(np.sqrt(self.random_steps)))
            left, right = np.meshgrid(rng, rng)
            actions = np.stack((left, right)).transpose(2, 1, 0).reshape(-1, 2)
            action = actions[self.replay_buffer.size]
        if np.any(np.isnan(action)):
            import pdb;pdb.set_trace()

        action = np.clip(action, *self.action_range)
        if not self.debug:
            print("SENDING ACTION")

            # if self.online:
            #     self.scaler = 1.0
            # elif self.total_steps % 50 == 0:
            #     # scaler = np.random.rand() * 5
            #     lst_left = [0.7, 0.9, 2.4, 4.1]
            #     lst_right = [1.4, 2.8, 1.8, 0.8]
            #     self.scaler = np.array([lst_left[self.total_steps // 50], lst_right[self.total_steps // 50]])

            # ac = np.clip(action * self.scaler, *self.action_range)
            # print("SCALER:", self.scaler)

            ac = action

            # if self.online:
            #     self.cent = 0.5
            #     self.shift = 0.7
            # elif self.total_steps % 20 == 0:
            #     self.cent = np.random.uniform()
            #     self.shift = np.random.uniform()

            # print(f"CENT: {self.cent}, SHIFT: {self.shift}")

            # if self.shift < 0.5:
            #     ac = np.clip(np.power((action + self.cent), 3), *self.action_range)
            # else:
            #     ac = np.clip(np.power((action - self.cent), 3), *self.action_range)

            action_msg = build_action_msg(ac, self.duration, self.total_steps)
            self.action_publisher.publish(action_msg)
            rospy.sleep(0.001)

            while self.last_action_timestamp == self.action_timestamp:
                rospy.wait_for_message("/action_timestamps", Time, timeout=1)
            self.last_action_timestamp = self.action_timestamp.copy()
            rospy.sleep(self.duration + self.post_action_sleep_time)

        print(f"\n\n\n\nNO. {self.total_steps}")
        print("/////////////////////////////////////////////////")
        print("=================================================")
        print("GOAL:", curr_goal)
        print("STATE:", state)
        print("ACTION:", action)
        print("ACTION NORM:", np.linalg.norm(action) / np.sqrt(2), "\n")

        cost_dict, total_cost = self.record_costs(curr_goal)

        for cost_type, cost in cost_dict.items():
            print(f"{cost_type}: {cost}")
        # print("TOTAL:", total_cost)

        print("\nREPLAY BUFFER SIZE:", self.replay_buffer.size)
        print("=================================================")
        print("/////////////////////////////////////////////////")

        return action

    def calibrate(self):
        yaw_offsets = np.zeros(10)

        input(f"Place robot/object on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.get_state()
        input(f"Place robot/object on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.get_state()

        robot_left_state, robot_right_state = left_state[:3], right_state[:3]
        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])
        measured_robot_angle = robot_left_state[2]
        yaw_offsets[self.robot_id] = true_robot_angle - measured_robot_angle

        if self.use_object:
            object_left_state, object_right_state = left_state[3:6], right_state[3:6]
            true_object_vector = (object_left_state - object_right_state)[:2]
            true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])
            measured_object_angle = object_left_state[2]
            yaw_offsets[self.object_id] = true_object_angle - measured_object_angle

        np.save(self.yaw_offset_path, yaw_offsets)

    def define_goal_trajectory(self):
        rospy.sleep(0.2)        # wait for states to be published and set
        front_corner_pos = self.state_dict["corner"].copy()
        if self.trajectory == "square":
            if self.robot_goals:
                front_corner_rel = np.array([0.8, 0.8])
                back_corner_rel = np.array([0.2, 0.25])
            else:
                front_corner_rel = np.array([0.7, 0.7])
                back_corner_rel = np.array([0.3, 0.3])
            square_front_left = front_corner_rel * front_corner_pos[:2]
            square_back_right = back_corner_rel * front_corner_pos[:2]
            square_front_right = np.array([square_back_right[0], square_front_left[1]])
            square_back_left = np.array([square_front_left[0], square_back_right[1]])
            corners = [square_front_left, square_front_right, square_back_right, square_back_left]

            side_lengths = [np.linalg.norm(corners[0]-corners[1]),
                            np.linalg.norm(corners[1]-corners[2]),
                            np.linalg.norm(corners[2]-corners[3]),
                            np.linalg.norm(corners[3]-corners[0])]

            unit_vectors = [(corners[1]-corners[0])/side_lengths[0],
                            (corners[2]-corners[1])/side_lengths[1],
                            (corners[3]-corners[2])/side_lengths[2],
                            (corners[0]-corners[3])/side_lengths[3]]

            step_increment = np.sum(side_lengths) / self.steps_per_lap
            left_over_len = 0
            self.goals = [square_front_left]
            for i in range(4):
                if left_over_len > 0:
                    self.goals.append(corners[i] + unit_vectors[i] * left_over_len)
                steps = ((side_lengths[i] - left_over_len)  // step_increment).astype(int)
                for _ in range(steps):
                    self.goals.append(self.goals[-1] + unit_vectors[i]*step_increment)
                left_over_len = (side_lengths[i] - left_over_len)  % step_increment
            self.goals = np.array(self.goals)

        elif self.trajectory == "fig8":
            if self.robot_goals:
                back_circle_center_rel = np.array([0.7, 0.5])
                front_circle_center_rel = np.array([0.4, 0.5])
            else:
                back_circle_center_rel = np.array([0.65, 0.5])
                front_circle_center_rel = np.array([0.4, 0.5])

            self.back_circle_center = back_circle_center_rel * front_corner_pos[:2]
            self.front_circle_center = front_circle_center_rel * front_corner_pos[:2]
            self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def record_costs(self, goal):
        cost_dict = self.agent.compute_costs(
                self.get_state()[None, None, None, :], np.array([[[[0., 0.]]]]), goal[None, :], robot_goals=self.robot_goals
                )

        total_cost = 0
        for cost_type, cost in cost_dict.items():
            cost_dict[cost_type] = cost.squeeze()
            total_cost += cost.squeeze() * self.agent.policy.cost_weights_dict[cost_type]

        if self.started:
            costs_to_record = np.array([[cost_dict["distance"], cost_dict["heading"], total_cost]])
            self.costs = np.append(self.costs, costs_to_record, axis=0)
            wandb.log(cost_dict, step=self.total_steps)

        return cost_dict, total_cost

    def dist_to_start(self):
        state = self.get_state().squeeze()
        state = state[:3] if self.robot_goals else state[3:]
        return np.linalg.norm((state - self.get_goal(step_override=0))[:2])

    def get_goal(self, step_override=None):
        if step_override is not None:
            lap_step = step_override
        else:
            lap_step = self.lap_step

        t_rel = ((lap_step + self.start_lap_step) % self.steps_per_lap) / self.steps_per_lap

        if self.trajectory == 'square':
            # square trajectory
            lap_step = lap_step % self.steps_per_lap
            goal = self.goals[lap_step]

        elif self.trajectory == 'fig8':
            # figure-eight trajectory
            if t_rel < 0.5:
                theta = t_rel * 2 * 2 * np.pi
                center = self.front_circle_center
            else:
                theta = np.pi - ((t_rel - 0.5) * 2 * 2 * np.pi)
                center = self.back_circle_center

            goal = center + np.array([np.cos(theta), np.sin(theta)]) * self.radius
        return np.block([goal, 0.])

    def get_next_n_goals(self, n):
        goals = np.empty((n, 3))
        for i in range(n):
            step = self.lap_step + i * (-1 if self.reverse_lap else 1)
            goals[i] = self.get_goal(step_override=step)

        return goals

    def check_rollout_finished(self):
        if np.abs(self.lap_step) == self.steps_per_lap:
            self.laps += 1
            # Print current cumulative loss per lap completed
            dist_costs, heading_costs, total_costs = self.costs.T
            data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])
            print("lap:", self.laps)
            print("rows: (dist, heading, total)")
            print("cols: (mean, std, min, max)")
            print("DATA:", data, "\n")
            self.lap_step = 0
            self.started = False

            start_state = self.get_goal(step_override=0)
            self.logger.plot_states(save=True, laps=self.laps, replay_buffer=self.replay_buffer,
                                    start_state=start_state, reverse_lap=self.reverse_lap)
            self.logger.reset_plot_states()

            # self.logger.plot_model_errors()
            # self.logger.reset_model_errors()

            self.logger.log_performance_metrics(self.costs, self.all_actions)

            if self.online:
                self.logger.dump_agent(self.agent, self.laps, self.replay_buffer)
                self.costs = np.empty((0, self.n_costs))
            elif(not self.online and self.laps == self.n_rollouts):
                self.logger.dump_agent(self.agent, self.laps, self.replay_buffer)

            self.start_lap_step = np.floor(np.random.rand() * self.steps_per_lap)
            self.reverse_lap = not self.reverse_lap

        if self.laps == self.n_rollouts:
            self.done = True

    def collect_training_data(self, state, action, next_state):
        self.replay_buffer.add(state, action, next_state)

        if self.replay_buffer.idx % self.save_freq == 0:
            print(f"\nSAVING REPLAY BUFFER\n")
            self.logger.dump_buffer(self.replay_buffer)

    def update_model_online(self):
        if self.replay_buffer.size > self.random_steps:
            for model in self.agent.models:
                for _ in range(self.gradient_steps):
                    # if self.meta:
                    #     states, actions, next_states = self.replay_buffer.sample_recent(30)
                    # else:

                    states, actions, next_states = self.replay_buffer.sample(self.batch_size)

                    # sample_size = min(self.total_steps, 10)
                    # states, actions, next_states = self.replay_buffer.sample_recent(sample_size)
                    # print(len(states), len(actions), len(next_states))

                    model.update(states, actions, next_states)

    def warmup_test(self):
        norms = []
        heading_diffs = []
        n_steps = 2000

        for i in trange(n_steps, desc="Warmup Test"):
            state = self.get_state()

            # alternate between forward and backward steps
            action = np.array([1, 1]) * 0.7 * (-1 if i % 2 == 0 else 1)
            action_msg = build_action_msg(action, self.duration, -1)

            self.action_publisher.publish(action_msg)
            rospy.sleep(0.001)

            while self.last_action_timestamp == self.action_timestamp:
                rospy.wait_for_message("/action_timestamps", Time, timeout=1)
            self.last_action_timestamp = self.action_timestamp.copy()
            rospy.sleep(self.duration + self.post_action_sleep_time)

            next_state = self.get_state()
            norms.append(np.linalg.norm((next_state - state)[:2]))
            heading_diffs.append(signed_angle_difference(next_state[2], state[2]))

        import matplotlib.pyplot as plt
        steps = np.arange(n_steps // 2)
        norms_cm = np.array(norms) * 100
        heading_diffs_deg = np.array(heading_diffs) * 180 / np.pi

        plt.figure()
        plt.plot(steps, norms_cm[::2], color="blue", label="Forward Step Distance")
        plt.plot(steps, norms_cm[1::2], color="green", label="Backward Step Distance")
        plt.xlabel("Step")
        plt.ylabel("Distance (cm)")
        plt.title("Warmup Test: Forward and Backward Step Distances Over Time")
        plt.legend()

        plt.figure()
        plt.plot(steps, heading_diffs_deg[::2], color="blue", label="Forward Step Heading Change")
        plt.plot(steps, heading_diffs_deg[1::2], color="green", label="Backward Step Heading Change")
        plt.xlabel("Step")
        plt.ylabel("Angle (deg)")
        plt.title("Warmup Test: Forward and Backward Step Heading Changes Over Time")
        plt.legend()

        plt.show()

        import pdb;pdb.set_trace()

def main(args):
    rospy.init_node("laptop_client_mpc")

    """
    get states from state publisher
    """
    pos_dim = 3
    vel_dim = 3
    robot_pos = np.empty(pos_dim)
    object_pos = np.empty(pos_dim)
    corner_pos = np.empty(pos_dim)
    robot_vel = np.empty(vel_dim)
    object_vel = np.empty(vel_dim)

    def update_state(msg):
        rs, os, cs = msg.robot_state, msg.object_state, msg.corner_state

        robot_pos[:] = np.array([rs.x, rs.y, rs.yaw])
        object_pos[:] = np.array([os.x, os.y, os.yaw])
        corner_pos[:] = np.array([cs.x, cs.y, cs.yaw])

        robot_vel[:] = np.array([rs.x_vel, rs.y_vel, rs.yaw_vel])
        object_vel[:] = np.array([os.x_vel, os.y_vel, os.yaw_vel])

    print("waiting for /processed_state topic from state publisher")
    rospy.Subscriber("/processed_state", ProcessedStates, update_state, queue_size=1)
    print("subscribed to /processed_state")


    """
    get action timestamps from kamigami
    """
    action_timestamp = np.zeros(1)

    def update_timestamp(msg):
        action_timestamp[:] = msg.data.to_sec()

    print("waiting for /action_timestamps topic from kamigami")
    rospy.Subscriber("/action_timestamps", Time, update_timestamp, queue_size=1)
    print("subscribed to /action_timestamps")


    """
    run experiment
    """
    experiment = Experiment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **vars(args))
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-object_id', type=int, default=-1, help='object id for rollout')
    parser.add_argument('-mpc_method', default='mppi')
    parser.add_argument('-mpc_horizon', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-mpc_refine_iters', type=int, default=1)
    parser.add_argument('-mpc_gamma', type=float, default=50000)
    parser.add_argument('-n_rollouts', type=int)
    parser.add_argument('-tolerance', type=float, default=0.05)
    parser.add_argument('-steps_per_lap', type=int)
    parser.add_argument('-calibrate', action='store_true')
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-new_buffer', action='store_true')
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-consecutive', action='store_true')
    parser.add_argument('-robot_goals', action='store_true')
    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-online', action='store_true')
    parser.add_argument('-save_freq', type=int, default=50)
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-random_steps', type=int, default=500)
    parser.add_argument('-use_all_data', action='store_true')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')
    parser.add_argument('-train_epochs', type=int, default=200)
    parser.add_argument('-ensemble', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)
    parser.add_argument('-rand_ac_mean', type=float, default=0.)
    parser.add_argument('-rand_rot', action='store_true')
    parser.add_argument('-exp_name')
    parser.add_argument('-meta', action='store_true')
    parser.add_argument('-warmup_test', action='store_true')
    parser.add_argument('-trajectory', default='fig8')



    args = parser.parse_args()
    main(args)
