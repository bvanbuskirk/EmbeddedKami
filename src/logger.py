#!/usr/bin/python3

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import csv


class Logger:
    def __init__(self, experiment, plot, corner, exp_name=None, **kwargs):
        if exp_name is None:
            exp_name = f"{'robot' if experiment.robot_goals else 'object'}goals"
            if experiment.pretrain or experiment.load_agent:
                exp_name += f"_pretrain{experiment.pretrain_samples}"
            if experiment.online:
                exp_name += "_online"

        self.exp_path = f"/home/bvanbuskirk/Desktop/experiments/{'object' if experiment.use_object else 'robot'}/{exp_name}/"
        self.buffer_path = "/home/bvanbuskirk/Desktop/experiments/buffers/"
        self.plot_path = self.exp_path + "plots/"
        self.state_path = self.exp_path + "states/"
        self.agent_path = self.exp_path + "agents/"

        Path(self.buffer_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        Path(self.state_path).mkdir(parents=True, exist_ok=True)
        Path(self.agent_path).mkdir(parents=True, exist_ok=True)

        self.robot_states = []
        self.object_states = []
        self.goal_states = []

        self.use_object = experiment.use_object
        self.plot = plot
        self.corner = corner
        self.logged_costs = self.logged_actions = False
        self.object_or_robot = 'object' if self.use_object else 'robot'

    def log_performance_metrics(self, costs, actions):
        dist_costs, heading_costs, total_costs = costs.T
        data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])

        print("rows: (dist, heading, total)")
        print("cols: (mean, std, min, max)")
        print("DATA:", data, "\n")

        self.log_costs(dist_costs, heading_costs, total_costs, costs)
        self.log_actions(actions)

    def log_costs(self, dist_costs, heading_costs, total_costs, costs_np):
        if not self.logged_costs:
            write_option = "w"
            self.logged_costs = True
        else:
            write_option = "a"

        with open(self.exp_path + "costs.csv", write_option, newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for total_loss, dist_loss, heading_loss in zip(total_costs, dist_costs, heading_costs):
                fwriter.writerow([total_loss, dist_loss, heading_loss])
            fwriter.writerow([])

        with open(self.exp_path + "costs.npy", "wb") as f:
            np.save(f, costs_np)

    def log_actions(self, actions):
        if not self.logged_actions:
            write_option = "w"
            self.logged_actions = True
        else:
            write_option = "a"

        with open(self.exp_path + "actions.csv", write_option, newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for action in actions:
                fwriter.writerow(action)
            fwriter.writerow([])

    def reset_plot_states(self):
        self.robot_states = []
        self.object_states = []
        self.goal_states = []

    def log_states(self, state_dict, goal_pos, started):
        robot_pos, object_pos = state_dict["robot"], state_dict["object"]
        if started or self.plot:
            self.robot_states.append(robot_pos.copy())
            self.object_states.append(object_pos.copy())
            self.goal_states.append(goal_pos.copy())

        if self.plot:
            self.plot_states(save=False)

    def dump_agent(self, agent, laps, replay_buffer):
        with open(self.agent_path + f"lap{laps}_rb{replay_buffer.size}.npy", "wb") as f:
            pkl.dump(agent, f)

    def dump_buffer(self, replay_buffer):
        with open(self.buffer_path + f"{self.object_or_robot}_buffer.pkl", "wb") as f:
            pkl.dump(replay_buffer, f)

    def load_buffer(self, robot_id):
        pkl_path = self.buffer_path + f"{self.object_or_robot}_buffer.pkl"
        validation_path = self.buffer_path + f"{self.object_or_robot}{robot_id}_validation_buffer.pkl"

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                replay_buffer = pkl.load(f)
        else:
            replay_buffer = None

        if os.path.exists(validation_path):
            with open(validation_path, 'rb') as f:
                validation_buffer = pkl.load(f)
        else:
            validation_buffer = None

        return replay_buffer, validation_buffer

    def plot_states(self, save=False, laps=None, replay_buffer=None, start_state=None, reverse_lap=False):
        plot_goals = np.array(self.goal_states)
        plot_robot_states = np.array(self.robot_states)
        plot_object_states = np.array(self.object_states)

        if self.plot and len(self.goal_states) != 1:
            plot_goals = plot_goals[-2:]
            plot_robot_states = plot_robot_states[-2:]
            plot_object_states = plot_object_states[-2:]

        plt.plot(plot_goals[:, 0], plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
        plt.plot(plot_robot_states[:, 0], plot_robot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")

        if self.use_object:
            plt.plot(plot_object_states[:, 0], plot_object_states[:, 1], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")

        if len(self.goal_states) == 1 or not self.plot:
            ax = plt.gca()
            ax.axis('equal')
            plt.xlim((self.corner[0], 0))
            plt.ylim((self.corner[1], 0))
            plt.legend()
            plt.ion()
            plt.show()

        if save:
            plt.plot(start_state[0], start_state[1], color="orange", marker="D", label="Starting Point", markersize=6)
            plt.title(f"{'Reversed' if reverse_lap else 'Standard'} Trajectory")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.draw()
            plt.pause(1.)

            plt.savefig(self.plot_path + f"lap{laps}_rb{replay_buffer.size}.png")
            plt.close()

            state_dict = {"robot": plot_robot_states, "object": plot_object_states, "goal": plot_goals}
            for name in ["robot", "object", "goal"]:
                with open(self.state_path + f"/{name}_lap{laps}.npy", "wb") as f:
                    np.save(f, state_dict[name])

        else:
            plt.draw()
            plt.pause(0.001)