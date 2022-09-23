#!/usr/bin/env python3
"""This is an example to add a simple VPG algorithm."""
import numpy as np
import torch

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import PointEnv
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np import discount_cumsum
from garage.sampler import LocalSampler
# from garage.torch.policies import GaussianMLPPolicy
# from Policy import GaussianMLPPolicy
from policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.trainer import Trainer


# pylint: disable=too-few-public-methods
class VPG:
    """Simple Vanilla Policy Gradient.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        sampler (garage.sampler.Sampler): Sampler.

    """

    def __init__(self, env_spec, policy, sampler, alpha=0.1):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        # self.max_episode_length = env_spec.max_episode_length
        self.max_episode_length = 200

        self._discount = 0.99
        # self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.alpha = alpha
        self._n_samples = 1

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        """
        for epoch in trainer.step_epochs():
            samples = trainer.obtain_samples(epoch)
            log_performance(epoch,
                            EpisodeBatch.from_list(self.env_spec, samples),
                            self._discount)
            self._train_once(samples)

    def _train_once(self, samples):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            samples (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        losses = []
        # avg_grad = torch.zeros_like(self.policy.get_param_values())
        avg_grad = 0
        # self._policy_opt.zero_grad()
        for path in samples:
            returns_numpy = discount_cumsum(path['rewards'], self._discount)
            returns = torch.Tensor(returns_numpy.copy())
            obs = torch.Tensor(path['observations'])
            actions = torch.Tensor(path['actions'])
            dist = self.policy(obs)[0]
            log_likelihoods = dist.log_prob(actions)
            loss = (-log_likelihoods * returns).mean()
            loss.backward()
            losses.append(loss.item())
            grad = self.policy.get_grads()
            avg_grad += grad
        # self._policy_opt.step()
        policy_param = self.policy.get_parameter_values()
        avg_grad /= len(samples)
        policy_param -=self.alpha * avg_grad
        # momentem=self.alpha * avg_grad
        self.policy.set_parameter_values(policy_param)
        return np.mean(losses)


@wrap_experiment
def run_vpg(ctxt=None, alpha = 0.1):
    """Train VPG with PointEnv environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.

    """
    set_seed(100)
    trainer = Trainer(ctxt)
    # env = PointEnv()
    env = GymEnv('InvertedDoublePendulum-v4')
    # policy = GaussianMLPPolicy(env.spec)
    policy = GaussianMLPPolicy(env.spec,
                                hidden_sizes=[32, 32],
                                hidden_nonlinearity=torch.tanh,
                                output_nonlinearity=None)
    policy.reset()

    sampler = LocalSampler(agents=policy,
                           envs=env,
                        #    max_episode_length=env.spec.max_episode_length)
                           max_episode_length=200)

    algo = VPG(env.spec, policy, sampler, alpha=alpha)
    trainer.setup(algo, env)
    trainer.train(n_epochs=200, batch_size=4000)


# for a in [0.01, 0.1, 1]:
run_vpg(alpha = 0.01)