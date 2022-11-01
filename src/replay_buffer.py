#!/usr/bin/python3

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=10000, state_dim=3, action_dim=2):
        self.states = np.empty((capacity, state_dim))
        self.actions = np.empty((capacity, action_dim))
        self.next_states = np.empty((capacity, state_dim))

        self.capacity = capacity
        self.full = False
        self.idx = 0

    @property
    def size(self):
        return self.capacity if self.full else self.idx

    def add(self, state, action, next_state):
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.next_states[self.idx], next_state)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, sample_size):
        sample_size = min(self.size, sample_size)
        stored_idxs = np.arange(self.size)
        sample_idx = np.random.choice(stored_idxs, sample_size)

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        next_states = self.next_states[sample_idx]

        return states, actions, next_states

    # assumes buffer never fills up/wraps
    def sample_recent(self, sample_size):
        idxs = np.arange(self.size)
        sample_idx = idxs[-sample_size:]

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        next_states = self.next_states[sample_idx]

        return states, actions, next_states
