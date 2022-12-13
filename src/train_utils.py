#!/usr/bin/python3

import numpy as np
import torch
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pickle as pkl
from tqdm import tqdm

from utils import dcn, as_tensor

AGENT_PATH = "/home/bvanbuskirk/Desktop/experiments/agent.pkl"


def train_from_buffer(agent, replay_buffer, validation_buffer=None, pretrain=False, consecutive=False, pretrain_samples=500,
                     save_agent=False, train_epochs=100, use_all_data=False, batch_size=500, meta=False):
    if pretrain or consecutive:
        n_samples = min(pretrain_samples, replay_buffer.size)

        if consecutive:
            print("sampling consecutive actions from buffer\n")
            states = replay_buffer.states[:n_samples]
            actions = replay_buffer.actions[:n_samples]
            next_states = replay_buffer.next_states[:n_samples]
        else:
            states, actions, next_states = replay_buffer.sample(n_samples)
    else:
        states, actions, next_states = replay_buffer.sample(replay_buffer.size)

    training_losses, test_losses, test_idx = train(
            agent, states, actions, next_states, validation_buffer=validation_buffer,
            set_scalers=True, epochs=train_epochs, batch_size=batch_size, use_all_data=use_all_data,
            meta=meta,
    )

    if save_agent:
        with open(AGENT_PATH, "wb") as f:
            pkl.dump(agent, f)

    training_losses = np.array(training_losses).squeeze()
    test_losses = np.array(test_losses).squeeze()

    print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
    print("MIN TEST LOSS:", test_losses.min())

    state_delta = agent.dtu.compute_relative_delta_xysc(states, next_states)

    # test_state, test_action = states[test_idx], actions[test_idx]
    # test_state_delta = dcn(state_delta[test_idx])

    if validation_buffer is not None:
        val_state, val_action, val_next_state = validation_buffer.sample(validation_buffer.capacity)
    else:
        val_state, val_action, val_next_state = states, actions, next_states

    val_state_delta = agent.dtu.compute_relative_delta_xysc(val_state, val_next_state)
    val_state, val_action, val_next_state = as_tensor(val_state, val_action, val_next_state)

    with torch.no_grad():
        agent.models[-1].eval()
        pred_state_delta = agent.models[-1](val_state, val_action, sample=False, delta=True)

    error = abs(pred_state_delta - val_state_delta)
    print("\nERROR MEAN:", error.mean(axis=0))
    print("ERROR STD:", error.std(axis=0))
    print("ERROR MAX:", error.max(axis=0)[0])
    print("ERROR MIN:", error.min(axis=0)[0])

    diffs = abs(val_state_delta)
    print("\nACTUAL MEAN:", diffs.mean(axis=0))
    print("ACTUAL STD:", diffs.std(axis=0))

    fig, axes = plt.subplots(1, 4)
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

def train(agent, state, action, next_state, validation_buffer=None, epochs=5, batch_size=256, set_scalers=False,
          use_all_data=False, meta=False):
    state, action, next_state = as_tensor(state, action, next_state)
    if validation_buffer is not None:
        print("\nUSING VALIDATION BUFFER\n")
        val_state, val_action, val_next_state = validation_buffer.sample(validation_buffer.size)
        val_state_delta = agent.dtu.compute_relative_delta_xysc(val_state, val_next_state)
        val_state, val_action, val_next_state = as_tensor(val_state, val_action, val_next_state)

    n_test = int(len(state) * 0.1) if not use_all_data else 2
    all_idx = torch.arange(len(state))

    for k, model in enumerate(agent.models):
        if meta:
            train_idx, test_idx = all_idx, all_idx
        else:
            train_idx, test_idx = train_test_split(all_idx, test_size=n_test, random_state=agent.seed + k)

        test_state, test_action, test_next_state = state[test_idx], action[test_idx], next_state[test_idx]
        test_state_delta = agent.dtu.compute_relative_delta_xysc(test_state, test_next_state)

        if use_all_data:
            train_idx = all_idx

        train_state = state[train_idx]
        train_action = action[train_idx]
        train_next_state = next_state[train_idx]

        if set_scalers:
            model.set_scalers(train_state, train_action, train_next_state)

        train_losses, test_losses = [], []
        n_batches = int(np.ceil(len(train_state) / batch_size))

        # evaluate
        with torch.no_grad():
            model.eval()
            if validation_buffer is not None:
                pred_state_delta = model(val_state, val_action, sample=False, delta=True)
                test_loss_mean = F.mse_loss(pred_state_delta, val_state_delta, reduction='mean')
            else:
                pred_state_delta = model(test_state, test_action, sample=False, delta=True)
                test_loss_mean = F.mse_loss(pred_state_delta, test_state_delta, reduction='mean')

        test_losses.append(dcn(test_loss_mean))
        tqdm.write(f"Pre-Train: mean test loss: {test_loss_mean}")

        print("\n\nTRAINING MODEL\n")
        for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
            # train
            if meta:
                train_loss_mean = model.update_meta(train_state, train_action, train_next_state)
            else:
                shuffle_idx = np.random.permutation(len(train_state))
                train_state, train_action, train_next_state = train_state[shuffle_idx], train_action[shuffle_idx], train_next_state[shuffle_idx]

                for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                    start, end = j * batch_size, (j + 1) * batch_size
                    batch_state, batch_action, batch_next_state = train_state[start:end], train_action[start:end], train_next_state[start:end]
                    train_loss_mean = model.update(batch_state, batch_action, batch_next_state)

            # evaluate
            with torch.no_grad():
                model.eval()
                if validation_buffer is not None:
                    pred_state_delta = model(val_state, val_action, sample=False, delta=True)
                    test_loss_mean = F.mse_loss(pred_state_delta, val_state_delta, reduction='mean')
                else:
                    pred_state_delta = model(test_state, test_action, sample=False, delta=True)
                    test_loss_mean = F.mse_loss(pred_state_delta, test_state_delta, reduction='mean')

            train_losses.append(train_loss_mean)
            test_losses.append(dcn(test_loss_mean))
            tqdm.write(f"{i+1}: train loss: {train_loss_mean:.5f} | test loss: {test_loss_mean:.5f}")

    if meta:
        for g in model.optimizer.param_groups:
            # g['lr'] = model.update_lr
            g['lr'] = torch.clamp(model.update_lr, 1e-4, 1)
            print("\nUPDATE LEARNING RATE:", model.update_lr)

        # losses = []
        # for i in range(100):
        #     model.update(train_state[:20], train_action[:20], train_next_state[:20])
        #     pred_state_delta = model(val_state, val_action, sample=False, delta=True)
        #     losses.append(dcn(F.mse_loss(pred_state_delta, val_state_delta, reduction='mean')))

        # print("\nMIN META TEST ITER:", np.argmin(losses))
        # print("MIN META TEST LOSS:", np.min(losses), "\n")

    model.eval()
    return train_losses, test_losses, test_idx
