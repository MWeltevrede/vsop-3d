# MIT License

# Copyright (c) 2019 CleanRL developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import random
import time
from collections import deque
from distutils.util import strtobool
from pathlib import Path
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import cProfile, pstats

from expgen.model import Policy, ImpalaModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="unique experiment id to be shared over different seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        help="directory to write results",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="starpilot",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(25e6),
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=256,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="the number of frames to stack",
    )
    parser.add_argument(
        "--num-weight-decay",
        type=float,
        default=64 * 256,
        help="Number of examples used to calculate weight decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.881,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=32,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=1,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=1e-5,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--thompson",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="do thompson sampling",
    )
    parser.add_argument(
        "--num-advantage-samples",
        type=int,
        default=1,
        help="number of samples from value posterior to calculate advantages",
    )

    # Agent specific arguments
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.075,
        help="dropout rate",
    )
    parser.add_argument(
        "--spectral-norm",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="apply spectral normalization",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2,
        help="width multiplier",
    )
    parser.add_argument(
        "--max_pure_expl_steps",
        type=int,
        default=0,
        help="maximum number of pure exploration steps to take each episode",
    )
    parser.add_argument(
        "--num_starting_states",
        type=int,
        default=1,
        help="number of starting states to pre-sample with pure-exploration"
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=20,
        help="total number of checkpoints to save"
    )
    parser.add_argument(
        "--reload_checkpoint",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to start from checkpoint",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="name of the checkpoint file to load from",
    )
    parser.add_argument(
        "--num_training_levels",
        type=int,
        default=200,
        help="number of levels to use for training"
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ConsistentDropout(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ConsistentDropout, self).__init__()
        self.q = 1 - p
        self.inplace = inplace

    def forward(self, x, seed=None):
        if self.q == 1.0:
            return x
        if self.training:
            mask = torch.distributions.Bernoulli(probs=self.q).sample(
                torch.Size([1]) + x.shape[1:]
            ).to(x.device) / (self.q)
            return x * mask
        return x


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=[3, 3, 3],
            padding=[1, 1, 1],
        )
        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=[3, 3, 3],
            padding=[1, 1, 1],
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv3d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=[3, 3, 3],
            padding=[1, 1, 1],
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool3d(
            x,
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2],
            padding=[1, 1, 1],
        )
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, d, h, w = self._input_shape
        return self._out_channels, (d + 1) // 2, (h + 1) // 2, (w + 1) // 2


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        dropout_rate=0.0,
        spectral_norm=False,
        width=1,
    ):
        super().__init__()
        s, h, w, c = envs.observation_space.shape
        shape = (c, s, h, w)
        conv_seqs = []
        dim_model = width * 256
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels * width)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.cnn = nn.Sequential(*conv_seqs)
        ll = nn.Linear(
            in_features=shape[0] * shape[2] * shape[3], out_features=dim_model
        )
        if spectral_norm:
            ll = nn.utils.spectral_norm(ll)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            ConsistentDropout(p=dropout_rate),
            ll,
            nn.ReLU(),
            ConsistentDropout(p=dropout_rate),
        )
        self.actor = layer_init(
            nn.Linear(dim_model, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(dim_model, 1), std=1)

    def embed_image(self, x):
        x = x.permute(1, 4, 0, 2, 3)  # sehwc -> ecshw
        x = self.cnn(x / 255.0)
        return x.sum(2)

    def get_value(self, x):
        x = self.embed_image(x)
        x = self.network(x)
        return self.critic(x)

    def get_action(self, x, action=None):
        x = self.embed_image(x)
        x = self.network(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
        )

    def get_action_and_value(self, x, action=None):
        x = self.embed_image(x)
        x = self.network(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class VecSettableProcgen(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """
        Environment wrapper that defines a function that allows for getting and setting states in a Procgen VecEnv
        """
        super(VecSettableProcgen, self).__init__(env)

    def get_states(self):
        unwrapped = self.env
        while isinstance(unwrapped, gym.Wrapper):
            unwrapped = unwrapped.env
        return unwrapped.env.get_state()
        
    def set_states(self, states, env_indices):
        new_states = self.get_states()
        for c, v in enumerate(env_indices):
            new_states[v] = states[c]
        
        unwrapped = self.env
        while isinstance(unwrapped, gym.Wrapper):
            unwrapped = unwrapped.env
        unwrapped.env.set_state(new_states)

        if isinstance(self.env, gym.wrappers.FrameStack):
            # reset framestack for env_indices whose states got set
            obs = unwrapped.env.observe()[1]['rgb'][env_indices]
            for _ in range(self.env.num_stack):
                current_obs = self.env.frames[0]
                current_obs[env_indices] = obs
                self.env.frames.append(current_obs)

        return self.env.observation()
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
def make_envs(num_envs, env_id, num_levels, gamma, distribution_mode="easy", frames=4, start_level=0):
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.observation_space.low = envs.single_observation_space.low
    envs.observation_space.high = envs.single_observation_space.high
    envs.observation_space.dtype = envs.single_observation_space.dtype
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    envs = gym.wrappers.FrameStack(envs, frames)
    envs = VecSettableProcgen(envs)
    return envs

def main():
    args = parse_args()
    exp_name = f"vsop-3d-{args.experiment_id}"
    run_experiment(exp_name=exp_name, args=args)

def run_experiment(exp_name, args):
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(job_dir / "checkpoint" / f"{args.env_id}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.env_id}__{exp_name}__{args.seed}__{args.max_pure_expl_steps}__{int(time.time())}"
    config = vars(args)
    config["exp_name"] = exp_name
    if args.track:
        import wandb

        wandb.init(
            dir=job_dir,
            project="phasic-exploration--vsop-3d",
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            tags=[args.env_id],
        )
    writer = SummaryWriter(job_dir / "runs" / f"{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_envs(
        num_envs=args.num_envs,
        env_id=args.env_id,
        num_levels=args.num_training_levels,
        gamma=args.gamma,
        frames=args.num_frames,
    )
    envs_test = make_envs(
        num_envs=args.num_envs,
        env_id=args.env_id,
        num_levels=0,
        gamma=args.gamma,
        frames=args.num_frames,
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(
        envs,
        dropout_rate=args.dropout_rate,
        spectral_norm=args.spectral_norm,
        width=args.width,
    ).to(device)
    agent = torch.compile(agent)
    if args.num_weight_decay < np.inf:
        optimizer = optim.AdamW(
            agent.parameters(),
            lr=args.learning_rate,
            weight_decay=(1 - args.dropout_rate) / (2 * args.num_weight_decay),
        )
    else:
        optimizer = optim.Adam(
            agent.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )

    # create pure exploration actor-critic
    recurrent_hidden_size = int(256)
    gray_scale = False
    epsilon_RPO = 0

    pure_actor_critic = Policy(
        tuple(np.array(envs.observation_space.shape[1:])[[2,0,1]]),
        envs.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': True,
                        'hidden_size': recurrent_hidden_size, 'gray_scale': gray_scale},
        epsilon_RPO=epsilon_RPO
    )
    pure_actor_critic.to(device)

    pure_actor_critic_weights = torch.load(f"/expgen/{args.env_id}-expgen.pt")
    # pure_actor_critic_weights = torch.load(f"{args.env_id}-expgen.pt")
    pure_actor_critic.load_state_dict(pure_actor_critic_weights['state_dict'])

    # Pre-sample reachable starting states
    starting_states = [list() for _ in range(args.num_training_levels)] 
    sequence_lengths = np.linspace(0, args.max_pure_expl_steps, args.num_starting_states).astype(int)
    for i in range(args.num_training_levels):
        for j in range(args.num_starting_states-1):
            temp_envs = make_envs(
                num_envs=1,
                env_id=args.env_id,
                num_levels=1,
                start_level=i,
                gamma=0.999,
                frames=16,
            )
            next_obs = torch.Tensor(np.array(temp_envs.reset())).to(device)
            pure_masks = torch.ones(1, 1, device=device)
            pure_recurrent_hidden_states = torch.zeros(1, pure_actor_critic.recurrent_hidden_state_size, device=device)

            # pure exploration
            if len(starting_states[i]) == 0:
                starting_states[i].append((temp_envs.get_states(), 0))

            step = 0
            while step < sequence_lengths[j+1]:
                with torch.no_grad():
                    _, pure_action, _, _, pure_recurrent_hidden_states = pure_actor_critic.act(
                        next_obs[-1].permute(0,3,1,2) / 255.,
                        pure_recurrent_hidden_states,
                        pure_masks
                        )

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, _, done, _ = temp_envs.step(pure_action.squeeze(-1).cpu().numpy())
                step += 1

                next_obs, next_done = (
                    torch.Tensor(np.array(next_obs)).to(device),
                    torch.Tensor(done).to(device),
                )
                pure_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

                if done[0]:
                    # print("Done when exploring")
                    step = 0
            
            starting_states[i].append((temp_envs.get_states(), sequence_lengths[j+1]))

    del pure_actor_critic
    del pure_actor_critic_weights
    del pure_masks
    del pure_recurrent_hidden_states
    torch.cuda.empty_cache()

    # ALGO Logic: Storage setup
    obs = (
        torch.zeros(
            (args.num_steps, args.num_frames, args.num_envs)
            + envs.single_observation_space.shape
        )
        .byte()
        .to("cpu")
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    non_teleport_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(np.array(envs.reset())).to(device)
    next_obs_test = torch.Tensor(np.array(envs_test.reset())).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    def sample_starting_states(starting_states, num_to_sample):
        states = []
        total_steps = 0
        for _ in range(num_to_sample):
            s, l = starting_states[np.random.randint(args.num_training_levels)][np.random.randint(args.num_starting_states)]
            states.append(s)
            total_steps += l
        return states, total_steps
    
    new_states, total_steps = sample_starting_states(starting_states, args.num_envs)
    non_teleport_step += total_steps
    next_obs = torch.Tensor(np.array(envs.set_states(new_states, list(range(args.num_envs))))).to(device)

    test_returns = []

    # reload from checkpoint
    if args.reload_checkpoint:
        checkpoint = torch.load(checkpoint_dir / f"{args.checkpoint_name}.cp")
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        non_teleport_step = checkpoint['non_teleport_step']
        num_updates = (args.total_timesteps - global_step) // args.batch_size

    for update in range(1, num_updates + 1):
        if not args.thompson:
            agent.eval()
        episode_returns = []
        episode_len = []
        test_episode_returns = []
        test_episode_len = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            non_teleport_step += 1 * args.num_envs
            obs[step] = next_obs.byte().to("cpu")
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                agent.train()
                action, logprob, _ = agent.get_action(next_obs)
                agent.eval()
                action_test, _, _ = agent.get_action(next_obs_test)
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs_test, _, _, info_test = envs_test.step(action_test.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done, next_obs_test = (
                torch.Tensor(np.array(next_obs)).to(device),
                torch.Tensor(done).to(device),
                torch.Tensor(np.array(next_obs_test)).to(device),
            )

            for item in info:
                if "episode" in item.keys():
                    episode_returns.append(item["episode"]["r"])
                    episode_len.append(item["episode"]["l"])
                    break

            for item in info_test:
                if "episode" in item.keys():
                    test_returns.append(item["episode"]["r"])
                    test_episode_returns.append(item["episode"]["r"])
                    test_episode_len.append(item["episode"]["l"])
                    break
                    
            if np.any(done):
                new_states, total_steps = sample_starting_states(starting_states, done.sum())
                non_teleport_step += total_steps
                next_obs = torch.Tensor(np.array(envs.set_states(new_states, np.where(done)[0]))).to(device)

        # bootstrap value if not done
        b_obs = obs.permute(1, 0, 2, 3, 4, 5).reshape(
            (
                args.num_frames,
                -1,
            )
            + envs.single_observation_space.shape
        )
        b_advantages, b_returns = [], []
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.cat([b_obs, next_obs.byte().to("cpu")], dim=1).permute(
                    1, 0, 2, 3, 4
                )
            ),
            batch_size=256,
            shuffle=False,
        )
        state = torch.get_rng_state()
        for sample_idx in range(args.num_advantage_samples):
            with torch.no_grad():
                values = []
                for batch in dl:
                    torch.manual_seed(sample_idx)
                    values.append(
                        agent.get_value(
                            batch[0].permute(1, 0, 2, 3, 4).float().to(device)
                        )
                    )
                values = torch.cat(values, dim=0).reshape(args.num_steps + 1, -1)
                next_value = values[-1:]
                values = values[:-1]
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
                b_advantages.append(advantages.unsqueeze(0))
                b_returns.append(returns.unsqueeze(0))
        torch.set_rng_state(state)

        # flatten the batch
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = torch.cat(b_advantages, dim=0).mean(0).reshape(-1)
        b_returns = torch.cat(b_returns, dim=0).mean(0).reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        agent.train()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if len(mb_inds) != args.minibatch_size:
                    break

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[:, mb_inds].float().to(device), b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss = -(torch.nn.functional.relu(mb_advantages) * newlogprob).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm < torch.inf:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # save checkpoint
        subtract_checkpoint_array = global_step - (np.arange(0, args.total_timesteps+1, (args.total_timesteps // args.num_checkpoints)) - (args.num_envs*args.num_steps/2.))
        checkpoint_id = np.where(subtract_checkpoint_array > 0, subtract_checkpoint_array, np.inf).argmin()
        checkpoint_path = checkpoint_dir / f"{run_name}_{checkpoint_id}.cp"
        if not os.path.isfile(checkpoint_path):
            torch.save({
                'global_step': global_step,
                'non_teleport_step': non_teleport_step,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("charts/num_non_teleport_steps", non_teleport_step, global_step)
        writer.add_scalar("train/episodic_return", np.array(episode_returns).mean(), global_step)
        writer.add_scalar("train/episodic_length", np.array(episode_len).mean(), global_step)
        writer.add_scalar("test/episodic_return", np.array(test_episode_returns).mean(), global_step)
        writer.add_scalar("test/episodic_length", np.array(test_episode_len).mean(), global_step)
        writer.add_histogram(
            "histograms/values",
            b_values.flatten(),
            global_step,
            bins=128,
        )
        writer.add_histogram(
            "histograms/returns",
            b_returns.flatten(),
            global_step,
            bins=128,
        )
        writer.add_histogram(
            "histograms/rewards",
            rewards.flatten(),
            global_step,
            bins=128,
        )

    envs.close()
    writer.close()

    return np.mean(test_returns[-100:])


if __name__ == "__main__":
    main()
