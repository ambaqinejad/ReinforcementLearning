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
ROLE_GOAL = 3


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
            radius = np.random.uniform(1, 5)
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
            radius = np.random.uniform(1, 2)
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


class AgentManager:
    def __init__(self):
        self.agents = []

    def create_agents(self, roles=None):
        # Agents
        self.agents = []
        urdf_path = os.path.join(os.path.dirname(__file__), "../quadrotor.urdf")
        for i, role in enumerate(roles):
            pos = [0.0, i * 0.5, 1.0]
            uid = p.loadURDF(urdf_path, pos)
            if role == ROLE_SCOUT:
                p.changeVisualShape(uid, -1, rgbaColor=[1, 0, 0, 1])
            elif role == ROLE_SURROUND:
                p.changeVisualShape(uid, -1, rgbaColor=[0, 1, 0, 1])
            elif role == ROLE_ATTACKER:
                p.changeVisualShape(uid, -1, rgbaColor=[0, 0, 1, 1])
            agent = QuadAgent(uid, role, len(roles))
            agent.x, agent.y, agent.z = pos
            agent.battery = 100
            agent.battery_alarm = False
            agent.see_goal = False
            self.agents.append(agent)


# --------------------------------------------------
# Agent Data Structure
# --------------------------------------------------
class QuadAgent:
    def __init__(self, uid, role=ROLE_SCOUT, n_quad_agents=9):
        self.uid = uid
        self.role = role
        self.n_quad_agents = n_quad_agents
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.battery = 0.0
        self.battery_alarm = False
        self.see_goal = False
        self.messages = [0.0] * n_quad_agents * 6

    # def send_message(self, sender_agent, receiver_agents, message):
    #     for receiver_agent in receiver_agents:
    #         receiver_agent.receive_message(message)
    #
    # def receive_message(self, message):
    #     self.messages.append(message)


# --------------------------------------------------
# Main Environment
# --------------------------------------------------
class MultiAgentQuadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            render=True,
            roles=None,
            max_steps=500,
            n_scout=4,
            n_surround=4,
            n_attacker=1,
            n_goal=1,
    ):
        super().__init__()

        self.render_mode = render
        self.roles = roles
        self.not_goal_roles = [r for r in roles if r != ROLE_GOAL]
        self.goal_roles = [r for r in roles if r == ROLE_GOAL]
        self.max_steps = max_steps
        self.goal_role = roles[-1]

        self.n_agents = len(self.roles)
        self.n_quad_agents = self.n_agents - n_goal
        self.n_scout = n_scout
        self.n_surround = n_surround
        self.n_attacker = n_attacker
        self.n_goal = n_goal
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

        # ---------------------------------------------

        # Obstacle Manager
        self.obstacle_manager = ObstacleManager(
            arena_bounds=(self.arena_x, self.arena_y, self.arena_z)
        )
        self.obstacle_manager.create_buildings(n_buildings=6)
        self.obstacle_manager.create_trees(n_trees=10)
        self.obstacle_manager.create_dynamic_obstacles(n_dyn=5)

        # Agent Manager
        self.agent_manager = AgentManager()
        self.agent_manager.create_agents(roles=self.not_goal_roles)

        # Target (evader)
        self.goal_uid = p.loadURDF("cube_small.urdf", [3.0, 0.0, 1.5])
        p.changeVisualShape(self.goal_uid, -1, rgbaColor=[0, 1, 0, 1])

        # Spaces
        self.max_neighbors = 4
        self.max_obstacles = 6
        self.each_agent_message = 6
        self.obs_dim = 6 + self.max_neighbors * 6 + self.max_obstacles * 3 + self.n_quad_agents * 6
        # 6 -> self x, y, z, vx, vy, vz
        # 6 -> for each neighbor -> dx, dy, dz, dvx, dvy, dvz -> neighbor x - self x
        # 3 -> for each obstacle -> dx, dy, dz -> obstacle x - self x
        # 6 -> for each agent    -> see goal 1, alarm 1, battery 1, goal x, y, z
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
        for i, agent in enumerate(self.agent_manager.agents):
            x = np.random.uniform(*self.arena_x)
            y = np.random.uniform(*self.arena_y)
            z = np.random.uniform(1.0, 2.0)
            p.resetBasePositionAndOrientation(agent.uid, [x, y, z], [0, 0, 0, 1])
            agent.x, agent.y, agent.z = x, y, z
            agent.vx, agent.vy, agent.vz = 0.0, 0.0, 0.0
            agent.battery = 100
            agent.battery_alarm = False
            agent.see_goal = False
            agent.n_quad_agents = self.n_quad_agents
            agent.messages = [0.0] * agent.n_quad_agents * 6

        # Reset target
        x = np.random.uniform(*self.arena_x)
        y = np.random.uniform(*self.arena_y)
        z = np.random.uniform(1.0, 2.0)
        p.resetBasePositionAndOrientation(self.goal_uid, [x, y, z], [0, 0, 0, 1])

        return self.get_obs(), {}

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def scout_handler(self, actions, scout_agents):
        scout_rewards = np.zeros(len(scout_agents))
        terminated = False
        for i, agent in enumerate(scout_agents):
            act = np.clip(actions[i], -1, 1) * self.action_scale
            pos, _ = p.getBasePositionAndOrientation(agent.uid)
            new_pos = np.clip(pos + act,
                              [self.arena_x[0], self.arena_y[0], self.arena_z[0]],
                              [self.arena_x[1], self.arena_y[1], self.arena_z[1]])
            p.resetBasePositionAndOrientation(agent.uid, new_pos, [0, 0, 0, 1])

        # check scout near to goal
        is_near = [False] * len(scout_agents)
        dists_to_goal = [0.0] * len(scout_agents)
        # dists_to_other_scout_agents = np.array([
        #     np.linalg.norm(self._get_pos(pid) - ev_new)
        #     for pid in self.pursuers
        # ])
        for i, agent in enumerate(scout_agents):
            agent_pos, _ = p.getBasePositionAndOrientation(agent.uid)
            goal_pos, _ = p.getBasePositionAndOrientation(self.goal_uid)
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            dists_to_goal[i] = dist_to_goal
            scout_rewards[i] -= dist_to_goal
            if np.abs(dist_to_goal) < 5:
                if np.abs(dist_to_goal) <= 3:
                    scout_rewards[i] -= dist_to_goal * 1.7
                else:
                    is_near[i] = True
                    for _agent in scout_agents:
                        _agent.messages = [float(True), float(False), 0.0]
                        _agent.messages.extend(goal_pos)

        if all(is_near):
            scout_rewards[:] += 100
            terminated = True
        return scout_rewards, terminated


    def step(self, actions):
        rewards = np.zeros(self.n_agents)
        terminated = False
        truncated = False

        not_goal_agents = self.agent_manager.agents[0:-1]
        goal_agents = self.agent_manager.agents[-1]

        scout_agents = self.agent_manager.agents[0:self.n_scout]
        surround_agents = self.agent_manager.agents[0 + self.n_scout:0 + self.n_scout + self.n_surround]
        attacker_agents = self.agent_manager.agents[
            0 + self.n_scout + self.n_surround:0 + self.n_scout + self.n_surround + self.n_attacker]
        goal_agents = self.agent_manager.agents[
            0 + self.n_scout + self.n_surround + self.n_attacker:0 + self.n_scout + self.n_surround + self.n_attacker + self.n_goal]

        # ---- Move evader ----
        ev_act = np.clip(actions[-1], -1, 1) * self.action_scale * 1.2
        ev_pos, _ = p.getBasePositionAndOrientation(self.goal_uid)
        ev_new = np.clip(ev_pos + ev_act,
                         [self.arena_x[0], self.arena_y[0], self.arena_z[0]],
                         [self.arena_x[1], self.arena_y[1], self.arena_z[1]])
        p.resetBasePositionAndOrientation(self.goal_uid, ev_new, [0, 0, 0, 1])

        # ---- Move pursuers ----
        rewards, terminated = self.scout_handler(actions, scout_agents)
        rewards.extend()


        #
        # # ---- Rewards ----
        # dists = np.array([
        #     np.linalg.norm(self._get_pos(pid) - ev_new)
        #     for pid in self.pursuers
        # ])
        #
        # # Individual shaping for each pursuer
        # rewards[:self.n_pursuers] = -dists
        #
        # # Evader wants to maximize average distance
        # rewards[-1] = np.mean(dists)
        #
        # min_dist = np.min(dists)
        #
        # if np.min(dists) < self.capture_radius:
        #     rewards[:self.n_pursuers] += 50.0
        #     rewards[-1] -= 50.0
        #     terminated = True
        # rewards[:self.n_pursuers] -= 0.01
        #
        # self._move_obstacles()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        p.stepSimulation()
        time.sleep(0.01 if self.render_mode else 0.0)
        rewards /= 10.0

        return self._get_obs(), rewards, terminated, truncated, {}

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def get_obs(self):
        # 6 -> self x, y, z, vx, vy, vz
        # 6 -> for each neighbor -> dx, dy, dz, dvx, dvy, dvz -> neighbor x - self x
        # 3 -> for each obstacle -> dx, dy, dz -> obstacle x - self x
        # 6 -> for each agent    -> see goal 1, alarm 1, battery 1, goal x, y, z
        obs = []
        # دید بقیه به هدف
        for i, agent in enumerate(self.agent_manager.agents):
            obs.append(self._get_agent_obs(i))

        # دید هدف به بقیه
        goal_pos, _ = p.getBasePositionAndOrientation(self.goal_uid)
        goal_velocity, _ = p.getBaseVelocity(self.goal_uid)
        goal_obs = []
        goal_obs.extend(goal_pos)
        goal_obs.extend(goal_velocity)
        neighbors = []
        for i, agent in enumerate(self.agent_manager.agents):
            dx, dy, dz = agent.x - goal_pos[0], agent.y - goal_pos[1], agent.z - goal_pos[2]
            dist = np.linalg.norm([dx, dy, dz])
            neighbors.append((dist, agent, dx, dy, dz))
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:self.max_neighbors]

        for _, agent, dx, dy, dz in neighbors:
            goal_obs.extend(
                [dx, dy, dz, agent.vx - goal_velocity[0], agent.vy - goal_velocity[1], agent.vz - goal_velocity[2]])
        while len(neighbors) < self.max_neighbors:
            goal_obs.extend([0.0] * 6)

        # Obstacles
        obstacles = self.obstacle_manager.get_nearby_obstacles(goal_pos, self.max_obstacles)
        for obs_item in obstacles:
            dx, dy, dz = obs_item.x - goal_pos[0], obs_item.y - goal_pos[1], obs_item.z - goal_pos[2]
            goal_obs.extend([dx, dy, dz])
        while len(obstacles) < self.max_obstacles:
            goal_obs.extend([0.0] * 3)

        goal_obs.extend([0.0] * self.n_quad_agents * 6)
        obs.append(goal_obs)
        return np.array(obs, dtype=np.float32)

    def _get_agent_obs(self, agent_idx):
        # 6 -> self x, y, z, vx, vy, vz
        # 6 -> for each neighbor -> dx, dy, dz, dvx, dvy, dvz -> neighbor x - self x
        # 3 -> for each obstacle -> dx, dy, dz -> obstacle x - self x
        # 6 -> for each agent    -> see goal 1, alarm 1, battery 1, goal x, y, z
        agent = self.agent_manager.agents[agent_idx]
        obs = []

        # Self state
        obs.extend([0.0, 0.0, 0.0, agent.vx, agent.vy, agent.vz])

        # Neighbors
        neighbors = []
        for j, other in enumerate(self.agent_manager.agents):
            if j == agent_idx:
                continue
            dx, dy, dz = other.x - agent.x, other.y - agent.y, other.z - agent.z
            dist = np.linalg.norm([dx, dy, dz])
            neighbors.append((dist, other, dx, dy, dz))
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:self.max_neighbors]

        for _, other, dx, dy, dz in neighbors:
            obs.extend([dx, dy, dz, other.vx - agent.vx, other.vy - agent.vy, other.vz - agent.vz])
        while len(neighbors) < self.max_neighbors:
            obs.extend([0.0] * 6)

        # Obstacles
        obstacles = self.obstacle_manager.get_nearby_obstacles([agent.x, agent.y, agent.z], self.max_obstacles)
        for obs_item in obstacles:
            dx, dy, dz = obs_item.x - agent.x, obs_item.y - agent.y, obs_item.z - agent.z
            obs.extend([dx, dy, dz])
        while len(obstacles) < self.max_obstacles:
            obs.extend([0.0] * 3)

        obs.extend(agent.messages)
        return np.array(obs, dtype=np.float32)

    # --------------------------------------------------
    # Render
    # --------------------------------------------------
    def render(self):
        if not self.render_mode:
            return
        # Simple FPV camera for agent 0
        agent = self.agent_manager.agents[0]
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
        for agent in self.agent_manager.agents:
            pos, _ = p.getBasePositionAndOrientation(agent.uid)
            vel, _ = p.getBaseVelocity(agent.uid)

            state_parts.append(np.array(pos, dtype=np.float32))  # (3,)
            state_parts.append(np.array(vel, dtype=np.float32))  # (3,)

        for obs in self.obstacle_manager.obstacles:
            state_parts.append([obs.x, obs.y, obs.z])
        # -----------------------------
        # Goal
        # -----------------------------
        target_position, target_orn = p.getBasePositionAndOrientation(self.goal_uid)
        state_parts.append(target_position)

        return np.concatenate(state_parts, axis=0)
