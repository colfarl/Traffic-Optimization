# Author: Colin Farley, 2025‑04‑20
# Description: Double‑DQN with soft target updates and gradient clipping.

from __future__ import annotations
import random
from collections import deque
from typing import Tuple, Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DQN(nn.Module):
    """Two layer MLP Q-network."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # type: ignore[override]
        return self.net(x)


class DQNAgent:
    """Double DQN agent with soft update and optional PER."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 50_000,
        tau: float = 0.01,
        grad_clip: float = 1.0,
        device: str = "cpu",
        seed: int = 42
    ) -> None:

        set_global_seeds(seed)
        self.state_size, self.action_size = state_size, action_size
        self.gamma, self.tau = gamma, tau
        self.epsilon, self.epsilon_min, self.epsilon_decay = (
            epsilon,
            epsilon_min,
            epsilon_decay,
        )
        self.batch_size, self.grad_clip = batch_size, grad_clip
        self.device = torch.device(device)

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=memory_size
        )

    def remember(self, s, a, r, s2, done) -> None:
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state_t)).item()

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        with torch.no_grad():
            a_prime = self.policy_net(s2).argmax(dim=1, keepdim=True)
            q_next = self.target_net(s2).gather(1, a_prime).squeeze()
            target_q = r + (1 - d) * self.gamma * q_next

        q_vals = self.policy_net(s).gather(1, a).squeeze()
        loss = nn.SmoothL1Loss()(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft‑update target net
        self._soft_update()

        # ε‑greedy decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_p, policy_p in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_p.data.mul_(1 - self.tau).add_(self.tau * policy_p.data)

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())