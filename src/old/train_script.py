import argparse
import pickle as pkl
from pdb import set_trace

import numpy as np
import torch
from matplotlib import pyplot as plt

from mpc_agent import MPCAgent
from utils import dcn

parser = argparse.ArgumentParser(description='Train/load agent and do MPC.')
parser.add_argument('-load_agent_path', type=str,
            help='path/file to load old agent from')
parser.add_argument('-save_agent_path', type=str,
            help='path/file to save newly-trained agent to')
parser.add_argument('-new_agent', '-n', action='store_true',
            help='flag to train new agent')
parser.add_argument('-hidden_dim', type=int, default=512,
            help='dimension of hidden layers for dynamics network')
parser.add_argument('-hidden_depth', type=int, default=2,
            help='number of hidden layers for dynamics network')
parser.add_argument('-epochs', type=int, default=10,
            help='number of training epochs for new agent')
parser.add_argument('-batch_size', type=int, default=128,
            help='batch size for training new agent')
parser.add_argument('-learning_rate', type=float, default=7e-4,
            help='batch size for training new agent')
parser.add_argument('-seed', type=int, default=1,
            help='random seed for numpy and pytorch')
parser.add_argument('-dist', action='store_true',
            help='flag to have the model output a distribution')
parser.add_argument('-dropout', type=float, default=0.5,
            help='dropout probability')
parser.add_argument('-scale', action='store_true',
            help='flag to preprocess data with standard scaler')
parser.add_argument('-save', action='store_true',
            help='flag to save model after training')
parser.add_argument('-retrain', action='store_true',
            help='flag to load existing model and continue training')
parser.add_argument('-std', type=float, default=0.02,
            help='standard deviation for model distribution')
parser.add_argument('-ensemble', type=int, default=1,
            help='how many networks to use for an ensemble')
parser.add_argument('-use_all_data', action='store_true')
parser.add_argument('-use_object', action='store_true')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cpu")

if args.hidden_dim * args.hidden_depth >= 4000:
    if torch.backends.mps.is_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

if args.use_object:
    with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
        buffer = pkl.load(f)

    states = buffer.states[:buffer.idx-1]
    action = buffer.action[:buffer.idx-1]
    next_states = buffer.states[1:buffer.idx]
else:
    data = np.load("/Users/Obsidian/Desktop/eecs106b/projects/MPCDynamicsKamigami/sim/data/real_data.npz")
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]

    states = states[:, 0, :-1]
    actions = actions[:, 0, :-1]
    next_states = next_states[:, 0, :-1]

if args.retrain:
    online_data = np.load("../../sim/data/real_data_online400.npz")

    online_states = online_data['states']
    online_action = online_data['actions']
    online_next_states = online_data['next_states']

    n_repeat = int(len(states) / len(online_states))
    n_repeat = 1 if n_repeat == 0 else n_repeat
    online_states = np.tile(online_states, (n_repeat, 1))
    online_actions = np.tile(online_action, (n_repeat, 1))
    online_next_states = np.tile(online_next_states, (n_repeat, 1))

    # states = np.append(states, online_states, axis=0)
    # actions = np.append(actions, online_actions, axis=0)
    # next_states = np.append(next_states, online_next_states, axis=0)

    states = online_states
    actions = online_actions
    next_states = online_next_states

plot_data = False
if plot_data:
    plotstart = 10
    plotend = plotstart + 20

    actions_plot = actions[plotstart:plotend]

    states_x = states[plotstart:plotend, 0]
    states_y = states[plotstart:plotend, 1]
    next_states_x = next_states[plotstart:plotend, 0]
    next_states_y = next_states[plotstart:plotend, 1]

    states_theta = states[plotstart:plotend, -1]
    states_theta += np.pi
    states_sin = np.sin(states_theta)
    states_cos = np.cos(states_theta)

    next_states_theta = next_states[plotstart:plotend, -1]
    next_states_theta += np.pi
    next_states_sin = np.sin(next_states_theta)
    next_states_cos = np.cos(next_states_theta)

    plt.quiver(states_x[1:], states_y[1:], -states_cos[1:], -states_sin[1:], color="green")
    plt.quiver(next_states_x[:-1], next_states_y[:-1], -next_states_cos[:-1], -next_states_sin[:-1], color="purple")
    plt.plot(states_x[1:], states_y[1:], color="green", linewidth=1.0)
    plt.plot(next_states_x[:-1], next_states_y[:-1], color="purple", linewidth=1.0)
    for i, (x, y) in enumerate(zip(states_x, states_y)):
        if i == 0:
            continue
        plt.annotate(f"{i-1}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(str(actions_plot[i]), textcoords="offset points", xytext=(-10, -10), ha='center')

    for i, (x, y) in enumerate(zip(next_states_x, next_states_y)):
        if i == len(next_states_x) - 1:
            continue
        label = f"{i}"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

if args.new_agent:
    agent = MPCAgent(seed=args.seed, dist=args.dist,
                        scale=args.scale, hidden_dim=args.hidden_dim,
                        hidden_depth=args.hidden_depth, lr=args.learning_rate,
                        dropout=args.dropout, std=args.std, ensemble=args.ensemble,
                        use_object=args.use_object)

    # batch_sizes = [args.batch_size, args.batch_size * 10, args.batch_size * 100, args.batch_size * 1000]
    batch_sizes = [args.batch_size]
    for batch_size in batch_sizes:
        training_losses, test_losses, test_idx = agent.train(
                    states, actions, next_states, set_scalers=True, epochs=args.epochs,
                    batch_size=batch_size, use_all_data=args.use_all_data
                    )

        if args.ensemble == 1:
            training_losses = np.array(training_losses).squeeze()
            test_losses = np.array(test_losses).squeeze()

            print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
            print("MIN TEST LOSS:", test_losses.min())

            fig, axes = plt.subplots(1, 2)
            axes[0].plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
            axes[1].plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")

            axes[0].set_yscale('log')
            axes[1].set_yscale('log')

            axes[0].set_title('Training Loss')
            axes[1].set_title('Test Loss')

            for ax in axes:
                ax.grid()

            axes[0].set_ylabel('Loss')
            axes[1].set_xlabel('Epoch')
            fig.set_size_inches(15, 5)

            plt.show()

    if args.save:
        agent_path = args.save_agent_path
        print(f"\nSAVING MPC AGENT: {agent_path}\n")
        with open(agent_path, "wb") as f:
            pkl.dump(agent, f)
else:
    agent_path = args.load_agent_path
    with open(agent_path, "rb") as f:
        agent = pkl.load(f)

    # agent.model.eval()
    # diffs = []
    # pred_next_states = agent.get_prediction(test_states, test_actions)

    # error = abs(pred_next_states - test_state_delta)
    # print("\nERROR MEAN:", error.mean(axis=0))
    # print("ERROR STD:", error.std(axis=0))
    # print("ERROR MAX:", error.max(axis=0))
    # print("ERROR MIN:", error.min(axis=0))

    # diffs = abs(test_states - test_state_delta)
    # print("\nACTUAL MEAN:", diffs.mean(axis=0))
    # print("ACTUAL STD:", diffs.std(axis=0))
    # set_trace()

    if args.retrain:
        training_losses, test_losses, test_idx = agent.train(states, actions, next_states,
                                                epochs=args.epochs, batch_size=args.batch_size)

        training_losses = np.array(training_losses).squeeze()
        test_losses = np.array(test_losses).squeeze()
        ignore = 20
        print("\nMIN TEST LOSS EPOCH:", test_losses[ignore:].argmin() + ignore)
        print("MIN TEST LOSS:", test_losses[ignore:].min())
        plt.plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
        plt.plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Dynamics Model Loss')
        plt.legend()
        plt.grid()
        plt.show()

        if args.save:
            print("\nSAVING MPC AGENT\n")
            agent_path = args.save_agent_path
            with open(agent_path, "wb") as f:
                pkl.dump(agent, f)

for model in agent.models:
    model.eval()

state_delta = agent.dtu.compute_relative_delta_xysc(states, next_states)

test_state, test_action = states[test_idx], actions[test_idx]
test_state_delta = dcn(state_delta[test_idx])

pred_state_delta = dcn(model(test_state, test_action, sample=False))
# pred_state_delta = agent.get_prediction(test_states, test_actions, sample=False, scale=args.scale, delta=True, use_ensemble=False)

error = abs(pred_state_delta - test_state_delta)
print("\nERROR MEAN:", error.mean(axis=0))
print("ERROR STD:", error.std(axis=0))
print("ERROR MAX:", error.max(axis=0))
print("ERROR MIN:", error.min(axis=0))

diffs = abs(test_state_delta)
print("\nACTUAL MEAN:", diffs.mean(axis=0))
print("ACTUAL STD:", diffs.std(axis=0))

set_trace()

for k in range(20):
    slist = []
    alist = []
    start, end = 10 * k, 10 * k + 10
    state = states[start]
    for i in range(start, end):
        action = action[i]
        slist.append(state.squeeze())
        alist.append(action.squeeze())
        state = agent.get_prediction(state, action)
        state[2:] = np.clip(state[2:], -1., 1.)

    slist = np.array(slist)
    alist = np.array(alist)

    plt.quiver(slist[:, 0], slist[:, 1], -slist[:, 2], -slist[:, 3], color="green")
    plt.quiver(states[start:end, 0], states[start:end, 1], -states[start:end, 2], -states[start:end, 3], color="purple")
    plt.plot(slist[:, 0], slist[:, 1], color="green", linewidth=1.0, label="Predicted Trajectory")
    plt.plot(states[start:end, 0], states[start:end, 1], color="purple", linewidth=1.0, label="Actual Trajectory")

    for i, (x, y) in enumerate(zip(slist[:, 0], slist[:, 1])):
        plt.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    for i, (x, y) in enumerate(zip(states[start:end, 0], states[start:end, 1])):
        label = f"{i}"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.legend()
    plt.show()
    set_trace()
