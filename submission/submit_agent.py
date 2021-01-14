import os

import numpy as np
import torch
import torch.nn.functional as F
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from .model import ActorCritic

hidden_size = 256

class a3cAgent_submit(AgentWithConverter):
    def __init__(self, action_space, env, path):
        super().__init__(action_space, action_space_converter=IdToAct)
        self.action_space.filter_action(self._filter_action)
        self.rhoth = 0.8
        self.model = ActorCritic(env.observation_space.size(), self.action_space, hidden_size)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

    def convert_obs(self, observation):
        if np.all(observation.rho<self.rhoth):
            return None
        obs_vec = observation.to_vect()
        return obs_vec

    def my_act(self, transformed_observation, reward, done=False):
        if transformed_observation is None:
            return 0
        cstate = torch.from_numpy(transformed_observation)
        with torch.no_grad():
            _, logit = self.model((cstate.unsqueeze(0)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        return action[0,0]

def make_agent(env, submission_dir):

    res = a3cAgent_submit(env.action_space, env, os.path.join(submission_dir, "model.pth"))
    return res
