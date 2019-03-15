import numpy as np
import random

class randomReplays(object):
    def __init__(self, sizes):
        super(randomReplays, self).__init__()
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        
    def add(self, obs_t, action, pre_la, reward, la, obs_tp1, done):
        data = (obs_t, action, pre_la, reward, la, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, pre_la, reward, la, obs_tp1, done):
        data = (obs_t, action, pre_la, reward, la, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, pre_leagalActions, rewards, leagalActions, obses_tp1, dones = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, pre_la, reward, la, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            pre_leagalActions.append(pre_la)
            rewards.append(reward)
            leagalActions.append(la)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (np.array(obses_t),
                np.array(actions),
                pre_leagalActions,
                np.array(rewards),
                leagalActions,
                np.array(obses_tp1),
                np.array(dones))

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
