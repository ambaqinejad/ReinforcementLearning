import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import cv2


class MultiAgentQuadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            render=True,
            n_pursuers=3,
            n_evaders=1,
            num_obs=6,
            max_steps=3000
    ):
        super().__init__()
        self.render_mode = render
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_total_agents = n_pursuers + n_evaders
        self.num_obs = num_obs
        self.max_steps = max_steps
        self.step_count = 0

        self.dt = 0.1
        self.action_scale = 0.08
        self.capture_radius = 0.35

        self.arena_x = [-1.0, 5.0]
        self.arena_y = [-2.5, 2.5]
        self.arena_z = [0.8, 5.5]

        # ======================================================
        # PyBullet setup
        # ======================================================
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

        # ======================================================
        # Load pursuer quadrotors
        # ======================================================
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "../quadrotor.urdf"
        )

        self.pursuers = []
        for i in range(self.n_pursuers):
            start_pos = [0.0, i * 0.6, 1.0]
            quad_id = p.loadURDF(urdf_path, start_pos, globalScaling=1.2)
            color = [0.2, 0.2 + 0.25 * i, 1.0 - 0.3 * i, 1]
            p.changeVisualShape(quad_id, -1, rgbaColor=color)
            self.pursuers.append(quad_id)

        # ======================================================
        # Evader (former goal)
        # ======================================================
        self.evader_id = p.loadURDF(
            "cube_small.urdf",
            [3.0, 0.0, 2.0],
            globalScaling=1.4
        )
        p.changeVisualShape(self.evader_id, -1, rgbaColor=[0, 1, 0, 1])

        # ======================================================
        # Obstacles (unchanged)
        # ======================================================
        self.obs_ids = []
        self.obs_positions = []
        self.obs_speeds = []
        self._generate_obstacles()

        # ======================================================
        # Spaces
        # ======================================================
        self.obs_dim = 18

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.n_total_agents, self.obs_dim),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_total_agents, 3),
            dtype=np.float32
        )

    # ======================================================
    # Utility
    # ======================================================
    def _get_pos(self, body_id):
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return np.array(pos, dtype=np.float32)

    def _random_point(self):
        return np.array([
            np.random.uniform(*self.arena_x),
            np.random.uniform(*self.arena_y),
            np.random.uniform(*self.arena_z)
        ])

    # ======================================================
    # Obstacles
    # ======================================================
    def _generate_obstacles(self):
        for oid in self.obs_ids:
            p.removeBody(oid)

        self.obs_ids, self.obs_positions, self.obs_speeds = [], [], []

        for _ in range(self.num_obs):
            pos = self._random_point()
            oid = p.loadURDF("cube_small.urdf", pos, globalScaling=1.0)
            p.changeVisualShape(oid, -1, rgbaColor=[1, 0, 0, 1])
            speed = np.random.uniform(0.1, 0.4)
            phase = np.random.uniform(0, 2 * np.pi)
            self.obs_ids.append(oid)
            self.obs_positions.append(pos)
            self.obs_speeds.append((speed, phase))

    def _move_obstacles(self):
        for i, oid in enumerate(self.obs_ids):
            base = self.obs_positions[i]
            speed, phase = self.obs_speeds[i]

            dx = np.sin(self.step_count * speed * 0.1 + phase) * 0.4
            dy = np.cos(self.step_count * speed * 0.1 + phase) * 0.4
            dz = np.sin(self.step_count * speed * 0.07 + phase) * 0.2

            new_pos = base + np.array([dx, dy, dz])
            p.resetBasePositionAndOrientation(oid, new_pos, [0, 0, 0, 1])

    # ======================================================
    # Observation
    # ======================================================
    def _get_obs(self):
        obs_all = []

        ev_pos = self._get_pos(self.evader_id)

        # ---- Pursuers ----
        for i in range(self.n_pursuers):
            pos = self._get_pos(self.pursuers[i])
            obs = []

            obs.extend((ev_pos - pos).tolist())      # evader relative
            obs.extend(pos.tolist())                 # self position

            obs_all.append(obs + [0.0] * (self.obs_dim - len(obs)))

        # ---- Evader ----
        obs_e = []
        dists = [(np.linalg.norm(ev_pos - self._get_pos(p)), self._get_pos(p))
                 for p in self.pursuers]
        dists.sort(key=lambda x: x[0])

        for _, ppos in dists[:2]:
            obs_e.extend((ppos - ev_pos).tolist())

        obs_e.extend(ev_pos.tolist())
        obs_all.append(obs_e + [0.0] * (self.obs_dim - len(obs_e)))

        return np.array(obs_all, dtype=np.float32)

    # ======================================================
    # Centralized state (MADDPG)
    # ======================================================
    def get_state(self):
        state = []
        for p_id in self.pursuers:
            state.extend(self._get_pos(p_id).tolist())
        state.extend(self._get_pos(self.evader_id).tolist())
        for pos in self.obs_positions:
            state.extend(pos.tolist())
        return np.array(state, dtype=np.float32)

    # ======================================================
    # Gym API
    # ======================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        for i, pid in enumerate(self.pursuers):
            p.resetBasePositionAndOrientation(pid, [0.0, i * 0.6, 1.0], [0, 0, 0, 1])

        p.resetBasePositionAndOrientation(
            self.evader_id, self._random_point(), [0, 0, 0, 1]
        )

        self._generate_obstacles()

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.n_total_agents)
        terminated = False
        truncated = False

        # ---- Move pursuers ----
        for i, pid in enumerate(self.pursuers):
            act = np.clip(actions[i], -1, 1) * self.action_scale
            pos = self._get_pos(pid)
            new_pos = np.clip(pos + act,
                              [self.arena_x[0], self.arena_y[0], self.arena_z[0]],
                              [self.arena_x[1], self.arena_y[1], self.arena_z[1]])
            p.resetBasePositionAndOrientation(pid, new_pos, [0, 0, 0, 1])

        # ---- Move evader ----
        ev_act = np.clip(actions[-1], -1, 1) * self.action_scale * 1.2
        ev_pos = self._get_pos(self.evader_id)
        ev_new = np.clip(ev_pos + ev_act,
                         [self.arena_x[0], self.arena_y[0], self.arena_z[0]],
                         [self.arena_x[1], self.arena_y[1], self.arena_z[1]])
        p.resetBasePositionAndOrientation(self.evader_id, ev_new, [0, 0, 0, 1])

        # ---- Rewards ----
        dists = np.array([
            np.linalg.norm(self._get_pos(pid) - ev_new)
            for pid in self.pursuers
        ])

        # Individual shaping for each pursuer
        rewards[:self.n_pursuers] = -dists

        # Evader wants to maximize average distance
        rewards[-1] = np.mean(dists)

        min_dist = np.min(dists)

        if np.min(dists) < self.capture_radius:
            rewards[:self.n_pursuers] += 50.0
            rewards[-1] -= 50.0
            terminated = True
        rewards[:self.n_pursuers] -= 0.01

        self._move_obstacles()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        p.stepSimulation()
        time.sleep(0.01 if self.render_mode else 0.0)
        rewards /= 10.0

        return self._get_obs(), rewards, terminated, truncated, {}

    # ======================================================
    # FPV render (unchanged)
    # ======================================================
    def render(self):
        if not self.render_mode:
            return
        pos = self._get_pos(self.pursuers[0])
        cam_pos = pos + np.array([0, 0, 0.3])
        cam_target = cam_pos + np.array([1, 0, 0])
        view = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 20)
        w, h, rgb, _, _ = p.getCameraImage(96, 96, view, proj)
        img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
        cv2.imshow("Agent-0 FPV", img)
        cv2.waitKey(1)
