import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        if target_pos is None :
            print("Setting default init pose")
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        return np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()

    def step(self, rotor_speeds):
        recompensa = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) 
            recompensa += self.get_reward()
            pose_all.append(self.sim.pose)
            if done: recompensa += 10
        return np.concatenate(pose_all), recompensa, done

    def reset(self):
        self.sim.reset()
        return np.concatenate([self.sim.pose] * self.action_repeat) 