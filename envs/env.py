from metadrive import MultiAgentRoundaboutEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np

class RLLibMetaDriveRoundabout(MultiAgentEnv):
    def __init__(self, config: dict):
        self.env = MultiAgentRoundaboutEnv(config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_distances = {}  # Track previous distances for progress reward

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action_dict):
        obs, rew, term, trunc, info = self.env.step(action_dict)

        # Custom reward computation
        custom_rew = {}
        for agent_id in obs.keys():
            agent_info = info.get(agent_id, {})
            reward = 0.0

            # Success: Reaching destination
            if agent_info.get('arrive_dest', False):
                reward += 100.0

            # Collision penalty
            if agent_info.get('crash', False):
                reward -= 50.0

            # Off-roading penalty
            if agent_info.get('out_of_road', False):
                reward -= 20.0

            # Forward progress: Reward reduction in distance to destination
            curr_dist = agent_info.get('distance_to_destination', 0.0)
            if np.isnan(curr_dist):
                curr_dist = 0.0
            prev_dist = self.prev_distances.get(agent_id, float('inf'))
            if curr_dist < prev_dist:
                reward += (prev_dist - curr_dist) * 0.1  # Scale progress reward
            self.prev_distances[agent_id] = curr_dist

            # Speed sign compliance: Assume speed limit of 10 m/s
            velocity = agent_info.get('velocity', 0.0)
            if isinstance(velocity, (list, tuple)):
                velocity = velocity[0]  # Use x-component if vector
            if np.isnan(velocity):
                velocity = 0.0
            speed_limit = 10.0
            if velocity > speed_limit:
                reward -= (velocity - speed_limit) * 0.5  # Penalty for speeding
            else:
                reward += 0.1  # Small bonus for compliance

            # Lane keeping: Reward for staying in lane (not off-road and moving)
            if not agent_info.get('out_of_road', False) and velocity > 0.1:
                reward += 0.05  # Small bonus for lane keeping

            custom_rew[agent_id] = reward

            # Reset distance tracking if episode ends for this agent
            if term.get(agent_id, False) or trunc.get(agent_id, False):
                self.prev_distances[agent_id] = float('inf')

        # Ensure __all__ exists (RLlib compat)
        if "__all__" not in term:
            term["__all__"] = all(term.get(a, False) for a in obs.keys())
        if "__all__" not in trunc:
            trunc["__all__"] = all(trunc.get(a, False) for a in obs.keys())

        return obs, custom_rew, term, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def build_env_config(num_agents: int, render: bool) -> dict:
    return {
        "use_render": render,
        "manual_control": False,
        "log_level": 50,
        "num_agents": num_agents,
    }


def make_env(config: dict):
    return RLLibMetaDriveRoundabout(config)
