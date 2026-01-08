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
        n_agents=3,
        num_obs=6,
        max_steps=3000
    ):
        super().__init__()

        # ----------------------------
        # General settings
        # ----------------------------
        self.render_mode = render
        self.n_agents = n_agents
        self.num_obs = num_obs
        self.max_steps = max_steps
        self.step_count = 0

        self.dt = 0.1
        self.action_scale = 0.08

        self.arena_x = [-1.0, 5.0]
        self.arena_y = [-2.5, 2.5]
        self.arena_z = [0.8, 5.5]

        # ----------------------------
        # PyBullet setup
        # ----------------------------
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

        # ----------------------------
        # Load quadrotors
        # ----------------------------
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "quadrotor.urdf"
        )

        self.quads = []
        for i in range(self.n_agents):
            start_pos = [0.0, i * 0.6, 1.0]
            quad_id = p.loadURDF(
                urdf_path,
                start_pos,
                globalScaling=1.2
            )
            color = [0.2, 0.2 + 0.25 * i, 1.0 - 0.3 * i, 1]
            p.changeVisualShape(quad_id, -1, rgbaColor=color)
            self.quads.append(quad_id)

        # ----------------------------
        # Goal
        # ----------------------------
        self.goal_speed = 0.002
        self.goal_target = None
        self.goal_id = p.loadURDF(
            "cube_small.urdf",
            [3.0, 0.0, 2.0],
            globalScaling=1.5
        )
        p.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 1, 0, 1])
        self.goal_pos = np.zeros(3)

        # ----------------------------
        # Obstacles
        # ----------------------------
        self.obs_ids = []
        self.obs_positions = []
        self.obs_speeds = []

        self._generate_obstacles()

        # ----------------------------
        # Spaces
        # ----------------------------
        self.obs_dim = 18  # per agent

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, 3),
            dtype=np.float32
        )

    # ======================================================
    # Utility functions
    # ======================================================
    def _get_agent_pos(self, agent_id):
        pos, _ = p.getBasePositionAndOrientation(self.quads[agent_id])
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

        self.obs_ids = []
        self.obs_positions = []
        self.obs_speeds = []

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
    # Goal
    # ======================================================
    def _move_goal(self):
        if self.goal_target is None or np.linalg.norm(self.goal_pos - self.goal_target) < 0.2:
            self.goal_target = self._random_point()

        direction = self.goal_target - self.goal_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.goal_pos += (direction / dist) * self.goal_speed

        p.resetBasePositionAndOrientation(self.goal_id, self.goal_pos, [0, 0, 0, 1])

    # ======================================================
    # Observation
    # ======================================================
    def _get_nearest_obstacles(self, agent_pos, k=2):
        dists = [(np.linalg.norm(agent_pos - p), p) for p in self.obs_positions]
        dists.sort(key=lambda x: x[0])

        vec = []
        for _, pos in dists[:k]:
            vec.extend((agent_pos - pos).tolist())
        return vec

    def _get_nearest_agents(self, agent_id, agent_pos, k=1):
        dists = []
        for j in range(self.n_agents):
            if j != agent_id:
                pos = self._get_agent_pos(j)
                dists.append((np.linalg.norm(agent_pos - pos), pos))

        dists.sort(key=lambda x: x[0])
        vec = []
        for _, pos in dists[:k]:
            vec.extend((agent_pos - pos).tolist())
        return vec

    def _get_obs(self):
        obs_all = []

        for i in range(self.n_agents):
            pos = self._get_agent_pos(i)
            obs = []

            obs.extend((self.goal_pos - pos).tolist())
            obs.extend(pos.tolist())
            obs.extend(self._get_nearest_obstacles(pos, k=2))
            obs.extend(self._get_nearest_agents(i, pos, k=1))

            obs_all.append(obs)

        return np.array(obs_all, dtype=np.float32)

    # ======================================================
    # State (for centralized critic)
    # ======================================================
    def get_state(self):
        state = []
        for i in range(self.n_agents):
            state.extend(self._get_agent_pos(i).tolist())
        state.extend(self.goal_pos.tolist())
        for pos in self.obs_positions:
            state.extend(pos.tolist())
        return np.array(state, dtype=np.float32)

    # ======================================================
    # Gym API
    # ======================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        for i in range(self.n_agents):
            start_pos = [0.0, i * 0.6, 1.0]
            p.resetBasePositionAndOrientation(self.quads[i], start_pos, [0, 0, 0, 1])

        self.goal_pos = self._random_point()
        p.resetBasePositionAndOrientation(self.goal_id, self.goal_pos, [0, 0, 0, 1])

        self._generate_obstacles()

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.n_agents)
        terminated = False
        truncated = False

        for i in range(self.n_agents):
            action = np.clip(actions[i], -1, 1) * self.action_scale
            pos = self._get_agent_pos(i)
            new_pos = pos + action

            new_pos[0] = np.clip(new_pos[0], *self.arena_x)
            new_pos[1] = np.clip(new_pos[1], *self.arena_y)
            new_pos[2] = np.clip(new_pos[2], *self.arena_z)

            p.resetBasePositionAndOrientation(self.quads[i], new_pos, [0, 0, 0, 1])

            dist = np.linalg.norm(new_pos - self.goal_pos)
            rewards[i] -= dist

        # Team reward
        min_dist = min(
            np.linalg.norm(self._get_agent_pos(i) - self.goal_pos)
            for i in range(self.n_agents)
        )

        if min_dist < 0.3:
            rewards += 20.0
            terminated = True

        self._move_obstacles()
        self._move_goal()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        p.stepSimulation()
        time.sleep(0.01 if self.render_mode else 0.0)

        return self._get_obs(), rewards, terminated, truncated, {}

    # ======================================================
    # Render (FPV style)
    # ======================================================
    def render(self):
        pos = self._get_agent_pos(0)
        cam_pos = pos + np.array([0, 0, 0.3])
        cam_target = cam_pos + np.array([1, 0, 0])
        view = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 20)
        w, h, rgb, _, _ = p.getCameraImage(96, 96, view, proj)
        img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
        cv2.imshow("Agent-0 FPV", img)
        cv2.waitKey(1)




# =========================================================
# TRAINING (slow moving goal)
# =========================================================
env = DynamicQuadFPVEnv(render=True, num_obs=5)
TRAIN = True
if TRAIN:
    env.goal_speed_scale = 0.0
    env.goal_radius = 0.3

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=150000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",  # 🔑 controlled entropy
        verbose=1
    )

    model.learn(total_timesteps=40000)
    model.save("sac_quad_static_goal")

    # ===== Phase 2: Slow Moving Goal =====
    env.goal_speed_scale = 0.002

    model = SAC.load("sac_quad_static_goal", env)
    model.learn(total_timesteps=30_000)
    model.save("sac_quad_slow_goal")

    # ===== Phase 3: Fast Moving Goal =====
    env.goal_speed_scale = 0.02

    model = SAC.load("sac_quad_slow_goal", env)
    model.learn(total_timesteps=40_000)
    model.save("sac_quad_fpvsim")

    print("-------------------------------------------------")
    model.save("sac_quad_static_goal")
else:
    env.goal_speed_scale = 0.02
    env.goal_radius = 0.5
    model = SAC.load("sac_quad_fpvsim", env)
# =========================================================
# TESTING (faster moving goal)
# =========================================================


env.goal_speed_scale = 0.02
model = SAC.load("sac_quad_fpvsim", env)

obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        break

cv2.destroyAllWindows()
# =========================================================
# PLOTS (Path & Reward)
# =========================================================
trail = np.array(env.trail)

plt.figure(figsize=(10, 4))

# ---- Top-down trajectory ----
plt.subplot(1, 2, 1)
if len(trail) > 0:
    plt.plot(trail[:, 0], trail[:, 1], "-y", label="Drone Path")

plt.scatter(env.goal_pos[0], env.goal_pos[1],
            c="g", s=60, marker="*", label="Final Goal Position")

plt.scatter([p[0] for p in env.obs_positions],
            [p[1] for p in env.obs_positions],
            c="r", s=40, label="Obstacles")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Top-Down View (Trajectory)")
plt.legend()
plt.grid()

# ---- Reward curve ----
plt.subplot(1, 2, 2)
plt.plot(env.reward_history, "-b")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.grid()

plt.tight_layout()
plt.show()
