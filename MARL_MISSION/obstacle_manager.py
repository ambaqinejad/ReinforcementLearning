import pybullet as p
import numpy as np
import pybullet_data


class ObstacleManager:
    def __init__(self, arena_bounds):
        self.arena_x, self.arena_y, self.arena_z = arena_bounds
        self.static_ids = []
        self.dynamic_ids = []
        self.dynamic_base = []

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _rand_pos(self, z_low=1.0, z_high=2.5):
        return [
            np.random.uniform(*self.arena_x),
            np.random.uniform(*self.arena_y),
            np.random.uniform(z_low, z_high)
        ]

    # ----------------------------
    # Buildings
    # ----------------------------
    def create_buildings(self, n_buildings=5):
        for _ in range(n_buildings):
            pos = self._rand_pos(1.5, 2.5)
            bid = p.loadURDF("cube.urdf", pos, globalScaling=3.0)
            p.changeVisualShape(bid, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
            self.static_ids.append(bid)

    # ----------------------------
    # Trees
    # ----------------------------
    def create_trees(self, n_trees=8):
        for _ in range(n_trees):
            pos = self._rand_pos(1.0, 2.0)
            tid = p.loadURDF("cylinder.urdf", pos, globalScaling=1.5)
            p.changeVisualShape(tid, -1, rgbaColor=[0.1, 0.6, 0.1, 1])
            self.static_ids.append(tid)

    # ----------------------------
    # Dynamic obstacles
    # ----------------------------
    def create_dynamic_obstacles(self, n_dyn=4):
        for _ in range(n_dyn):
            pos = self._rand_pos(1.2, 2.2)
            oid = p.loadURDF("sphere_small.urdf", pos, globalScaling=1.0)
            self.dynamic_ids.append(oid)
            self.dynamic_base.append(np.array(pos))

    def update(self, step_count):
        for i, oid in enumerate(self.dynamic_ids):
            base = self.dynamic_base[i]
            dx = np.sin(step_count * 0.05 + i) * 0.6
            dy = np.cos(step_count * 0.05 + i) * 0.6
            dz = np.sin(step_count * 0.03) * 0.3
            new_pos = base + np.array([dx, dy, dz])
            p.resetBasePositionAndOrientation(oid, new_pos, [0, 0, 0, 1])

    def get_all_obstacles(self):
        return self.static_ids + self.dynamic_ids
