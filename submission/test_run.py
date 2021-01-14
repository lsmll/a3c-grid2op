import os
import grid2op
from submit_agent import make_agent
env = grid2op.make('l2rpn_neurips_2020_track1_small')
agent = make_agent(env,os.getcwd())
NB_EPISODE = 10  # assess the performance for 10 episodes, for example
for i in range(NB_EPISODE):
    reward = env.reward_range[0]
    done = False
    obs = env.reset()
    while not done:
        act = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)
        print(i,reward,done)
