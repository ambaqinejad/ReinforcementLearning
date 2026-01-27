# env_attention_quad.py

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from pygame.examples.go_over_there import target_position

# Roles
ROLE_SCOUT = 0
ROLE_SURROUND = 1
ROLE_ATTACKER = 2

# --------------------------------------------------
# Obstacle Manager
# --------------------------------------------------
class Obstacle:
    def __init__(self, pos, radius=0.3, is_dynamic=False, uid=None):
        self.x, self.y, self.z = pos
        self.radius = radius
        self.is_dynamic = is_dynamic
        self.uid = uid


class ObstacleManager:
    def __init__(self, arena_bounds):
        self.arena_x, self.arena_y, self.arena_z = arena_bounds
        self.obstacles = []

    def create_buildings(self, n_buildings=5):
        self.buildings = []
        for _ in range(n_buildings):
            pos = [
                np.random.uniform(*self.arena_x),
                np.random.uniform(*self.arena_y),
                np.random.uniform(*self.arena_z)
            ]
            radius = np.random.uniform(0.3, 0.7)
            uid = p.loadURDF("cube_small.urdf", pos, globalScaling=radius)
            p.changeVisualShape(uid, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
            self.obstacles.append(Obstacle(pos, radius, False, uid))

    def create_trees(self, n_trees=8):
        for _ in range(n_trees):
            pos = [
                np.random.uniform(*self.arena_x),
                np.random.uniform(*self.arena_y),
                np.random.uniform(*self.arena_z)
            ]
            radius = np.random.uniform(0.1, 0.2)
            uid = p.loadURDF("cube_small.urdf", pos, globalScaling=radius)
            p.changeVisualShape(uid, -1, rgbaColor=[0, 1, 0, 1])
            self.obstacles.append(Obstacle(pos, radius, False, uid))

    def create_dynamic_obstacles(self, n_dyn=3):
        for _ in range(n_dyn):
            pos = [
                np.random.uniform(*self.arena_x),
                np.random.uniform(*self.arena_y),
                np.random.uniform(*self.arena_z)
            ]
            radius = 0.2
            uid = p.loadURDF("cube_small.urdf", pos, globalScaling=radius)
            p.changeVisualShape(uid, -1, rgbaColor=[1, 0, 0, 1])
            self.obstacles.append(Obstacle(pos, radius, True, uid))

    def update(self, step_count):
        for obs in self.obstacles:
            if obs.is_dynamic:
                dx = 0.1 * np.sin(0.1 * step_count)
                dy = 0.1 * np.cos(0.1 * step_count)
                dz = 0.0
                obs.x += dx
                obs.y += dy
                p.resetBasePositionAndOrientation(obs.uid, [obs.x, obs.y, obs.z], [0, 0, 0, 1])

    def get_nearby_obstacles(self, position, max_count=5):
        pos = np.array(position)
        dists = []
        for obs in self.obstacles:
            o_pos = np.array([obs.x, obs.y, obs.z])
            dists.append((np.linalg.norm(o_pos - pos), obs))
        dists.sort(key=lambda x: x[0])
        return [obs for _, obs in dists[:max_count]]


# --------------------------------------------------
# Agent Data Structure
# --------------------------------------------------
class QuadAgent:
    def __init__(self, uid, role=ROLE_SCOUT):
        self.uid = uid
        self.role = role
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0


# --------------------------------------------------
# Main Environment
# --------------------------------------------------
class MultiAgentQuadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render=True,
        roles=[ROLE_SCOUT, ROLE_SCOUT, ROLE_SCOUT, ROLE_SCOUT, ROLE_SURROUND, ROLE_SURROUND, ROLE_SURROUND, ROLE_SURROUND, ROLE_ATTACKER],
        max_steps=500,
    ):
        super().__init__()

        self.render_mode = render
        self.roles = roles
        self.n_agents = len(roles)
        self.max_steps = max_steps
        self.step_count = 0

        self.dt = 0.1
        self.action_scale = 0.1

        self.arena_x = [-5.0, 5.0]
        self.arena_y = [-5.0, 5.0]
        self.arena_z = [0.5, 5.0]

        # PyBullet setup
        if self.render_mode:
            try:
                p.connect(p.GUI)
            except:
                p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.loadURDF("plane.urdf")

        # Obstacle Manager
        self.obstacle_manager = ObstacleManager(
            arena_bounds=(self.arena_x, self.arena_y, self.arena_z)
        )
        self.obstacle_manager.create_buildings(n_buildings=6)
        self.obstacle_manager.create_trees(n_trees=10)
        self.obstacle_manager.create_dynamic_obstacles(n_dyn=5)

        # Agents
        self.agents = []
        urdf_path = os.path.join(os.path.dirname(__file__), "../quadrotor.urdf")
        for i, role in enumerate(self.roles):
            pos = [0.0, i * 0.5, 1.0]
            uid = p.loadURDF(urdf_path, pos)
            if role == ROLE_SCOUT:
                p.changeVisualShape(uid, -1, rgbaColor=[1, 0, 0, 1])
            elif role == ROLE_SURROUND:
                p.changeVisualShape(uid, -1, rgbaColor=[0, 1, 0, 1])
            elif role == ROLE_ATTACKER:
                p.changeVisualShape(uid, -1, rgbaColor=[0, 0, 1, 1])
            agent = QuadAgent(uid, role)
            agent.x, agent.y, agent.z = pos
            self.agents.append(agent)

        # Target (evader)
        self.target_uid = p.loadURDF("cube_small.urdf", [3.0, 0.0, 1.5])
        p.changeVisualShape(self.target_uid, -1, rgbaColor=[0, 1, 0, 1])

        # Spaces
        self.max_neighbors = 4
        self.max_obstacles = 6
        self.obs_dim = 6 + self.max_neighbors * (6 + 3) + self.max_obstacles * 5

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_agents, 3), dtype=np.float32
        )

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Reset agents
        for i, agent in enumerate(self.agents):
            x = np.random.uniform(*self.arena_x)
            y = np.random.uniform(*self.arena_y)
            z = np.random.uniform(1.0, 2.0)
            p.resetBasePositionAndOrientation(agent.uid, [x, y, z], [0, 0, 0, 1])
            agent.x, agent.y, agent.z = x, y, z
            agent.vx, agent.vy, agent.vz = 0.0, 0.0, 0.0

        # Reset target
        x = np.random.uniform(*self.arena_x)
        y = np.random.uniform(*self.arena_y)
        z = np.random.uniform(1.0, 2.0)
        p.resetBasePositionAndOrientation(self.target_uid, [x, y, z], [0, 0, 0, 1])

        return self.get_obs(), {}

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.n_agents)
        terminated = False
        truncated = False

        # Move agents
        for i, agent in enumerate(self.agents):
            act = np.clip(actions[i], -1, 1) * self.action_scale
            agent.x += act[0]
            agent.y += act[1]
            agent.z += act[2]
            p.resetBasePositionAndOrientation(agent.uid, [agent.x, agent.y, agent.z], [0, 0, 0, 1])

        # Move target randomly
        t_pos, _ = p.getBasePositionAndOrientation(self.target_uid)
        t_pos = np.array(t_pos)
        t_pos += np.random.uniform(-0.05, 0.05, size=3)
        t_pos[2] = np.clip(t_pos[2], 1.0, 2.0)
        p.resetBasePositionAndOrientation(self.target_uid, t_pos, [0, 0, 0, 1])

        # Update dynamic obstacles
        self.obstacle_manager.update(self.step_count)

        # Rewards: negative distance to target
        for i, agent in enumerate(self.agents):
            rewards[i] = -np.linalg.norm([agent.x - t_pos[0], agent.y - t_pos[1], agent.z - t_pos[2]])

        # Termination
        dists = [np.linalg.norm([agent.x - t_pos[0], agent.y - t_pos[1], agent.z - t_pos[2]]) for agent in self.agents]
        if np.min(dists) < 0.3:
            terminated = True
            rewards += 50.0  # bonus for capture

        if self.step_count >= self.max_steps:
            truncated = True

        return self.get_obs(), rewards, terminated, truncated, {}

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def get_obs(self):
        obs = []
        for i, agent in enumerate(self.agents):
            obs.append(self._get_agent_obs(i))
        return np.array(obs, dtype=np.float32)

    def _get_agent_obs(self, agent_idx):
        agent = self.agents[agent_idx]
        obs = []

        # Self state
        obs.extend([0.0, 0.0, 0.0, agent.vx, agent.vy, agent.vz])

        # Neighbors
        neighbors = []
        for j, other in enumerate(self.agents):
            if j == agent_idx:
                continue
            dx, dy, dz = other.x - agent.x, other.y - agent.y, other.z - agent.z
            dist = np.linalg.norm([dx, dy, dz])
            neighbors.append((dist, other, dx, dy, dz))
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:self.max_neighbors]

        for _, other, dx, dy, dz in neighbors:
            obs.extend([dx, dy, dz, other.vx - agent.vx, other.vy - agent.vy, other.vz - agent.vz])
            role_onehot = [0.0, 0.0, 0.0]
            role_onehot[other.role] = 1.0
            obs.extend(role_onehot)
        while len(neighbors) < self.max_neighbors:
            obs.extend([0.0] * 9)

        # Obstacles
        obstacles = self.obstacle_manager.get_nearby_obstacles([agent.x, agent.y, agent.z], self.max_obstacles)
        for obs_item in obstacles:
            dx, dy, dz = obs_item.x - agent.x, obs_item.y - agent.y, obs_item.z - agent.z
            obs.extend([dx, dy, dz, obs_item.radius, float(obs_item.is_dynamic)])
        while len(obstacles) < self.max_obstacles:
            obs.extend([0.0]*5)

        return np.array(obs, dtype=np.float32)

    # --------------------------------------------------
    # Render
    # --------------------------------------------------
    def render(self):
        if not self.render_mode:
            return
        # Simple FPV camera for agent 0
        agent = self.agents[0]
        cam_pos = [agent.x, agent.y, agent.z + 0.2]
        cam_target = [agent.x + 1.0, agent.y, agent.z]
        view = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 20)
        w, h, rgb, _, _ = p.getCameraImage(96, 96, view, proj)
        img = np.reshape(rgb, (h, w, 4))[:, :, :3]
        return img

    def get_state(self):
        """
        Global state for centralized critic (CTDE)
        """

        state_parts = []

        # -----------------------------
        # Agents
        # -----------------------------
        for agent in self.agents:
            pos, orn = p.getBasePositionAndOrientation(agent.uid)
            vel, ang_vel = p.getBaseVelocity(agent.uid)

            state_parts.append(np.array(pos, dtype=np.float32))  # (3,)
            state_parts.append(np.array(vel, dtype=np.float32))  # (3,)

        for obs in self.obstacle_manager.obstacles:
            state_parts.append([obs.x, obs.y, obs.z, obs.radius, obs.is_dynamic])
        # -----------------------------
        # Goal
        # -----------------------------
        target_position, target_orn = p.getBasePositionAndOrientation(self.target_uid)
        state_parts.append(target_position)

        # -----------------------------
        # Time
        # -----------------------------
        state_parts.append(
            np.array([self.step_count / self.max_steps], dtype=np.float32)
        )

        return np.concatenate(state_parts, axis=0)

