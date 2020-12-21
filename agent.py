import time
#from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import grid2op
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Reward import GameplayReward, L2RPNReward

from model import ActorCritic
import my_optim

class a3cAgent(AgentWithConverter):
    def __init__(self, action_space, args):
        super().__init__(action_space, action_space_converter=IdToAct)
        #print('Filtering actions..')
        self.action_space.filter_action(self._filter_action)
        #print('Done')
        self.args=args


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
        obs_vec = observation.to_vect()
        return obs_vec

    def my_act(self, transformed_obs, reward=None, done=False, model=None):
        action = self.select_action(transformed_obs, model)

        return action

    def select_action(self, obs, model):
        _, logit = model((obs.unsqueeze(0)))
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1).detach()
        return action

    def ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                    shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad


    def do_train(self, rank, args, shared_model, counter, lock, optimizer=None):
        torch.manual_seed(args.seed + rank)

        env = grid2op.make(args.env_name, test=args.for_test, reward_class=L2RPNReward)
        env.seed(args.seed + rank)

        model = ActorCritic(env.observation_space.size(), self.action_space, args.hidden_size)

        if optimizer is None:
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

        model.train()

        state = self.convert_obs(env.reset())
        state = torch.from_numpy(state)
        done = True

        episode_length = 0
        while True:
            # Sync with the shared model
            model.load_state_dict(shared_model.state_dict())

            values = []
            log_probs = []
            rewards = []
            entropies = []

            for step in range(args.num_steps):
                episode_length += 1
                value, logit = model((state.unsqueeze(0)))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                state, reward, done, _ = env.step(self.convert_act(action[0,0]))
                state = self.convert_obs(state)
                done = done or episode_length >= args.max_episode_length
                reward = max(min(reward, 1), -1)

                with lock:
                    counter.value += 1

                if done:
                    episode_length = 0
                    state = self.convert_obs(env.reset())

                state = torch.from_numpy(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _ = model((state.unsqueeze(0)))
                R = value.detach()

            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = rewards[i] + args.gamma * \
                    values[i + 1] - values[i]
                gae = gae * args.gamma * args.gae_lambda + delta_t

                policy_loss = policy_loss - \
                    log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

            optimizer.zero_grad()

            (policy_loss + args.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            self.ensure_shared_grads(model, shared_model)
            optimizer.step()

    def do_test(self, rank, args, shared_model, counter):
        torch.manual_seed(args.seed + rank)
        if args.run_name is None:
            rn=None
        else:
            rn='runs/'+args.run_name
        writer = SummaryWriter(log_dir=rn, flush_secs=60)

        cnt = 0
        env = grid2op.make(args.env_name, test=args.for_test, reward_class=L2RPNReward)
        env.seed(args.seed + rank)

        model = ActorCritic(env.observation_space.size(), self.action_space, args.hidden_size)

        model.eval()

        state = self.convert_obs(env.reset())
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        start_time = time.time()

        # a quick hack to prevent the agent from stucking
        #actions = deque(maxlen=100)
        episode_length = 0
        while True:
            episode_length += 1
            # Sync with the shared model
            if done:
                model.load_state_dict(shared_model.state_dict())

            with torch.no_grad():
                _, logit = model((state.unsqueeze(0)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].numpy()

            state, reward, done, _ = env.step(self.convert_act(action[0,0]))
            state = self.convert_obs(state)
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            #actions.append(action[0, 0])
            #if actions.count(actions[0]) == actions.maxlen:
            #    done = True

            if done:
                print("Time {}, num steps {}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    counter.value,
                    reward_sum, episode_length), flush=True)

                writer.add_scalar('Main/Reward', reward_sum, cnt)
                writer.add_scalar('Main/Episode Length', episode_length, cnt)
                writer.add_scalar('Stats/Global steps', counter.value, cnt)
                cnt+=1

                reward_sum = 0
                episode_length = 0
                #actions.clear()
                state = self.convert_obs(env.reset())
                time.sleep(args.test_interval)

            state = torch.from_numpy(state)

    def train(self):
        args=self.args
        torch.manual_seed(args.seed)
        env = env = grid2op.make(args.env_name, test=args.for_test, reward_class=L2RPNReward)
        shared_model = ActorCritic(env.observation_space.size(), self.action_space, args.hidden_size)
        shared_model.share_memory()

        if args.no_shared:
            optimizer = None
        else:
            optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
            optimizer.share_memory()

        processes = []

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        p = mp.Process(target=self.do_test, args=(args.num_processes, args, shared_model, counter))
        p.start()
        processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(target=self.do_train, args=(rank, args, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            