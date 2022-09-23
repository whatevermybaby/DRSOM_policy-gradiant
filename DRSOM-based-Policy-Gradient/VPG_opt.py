
"""This is an example to add a simple VPG algorithm."""
import numpy as np
import torch
from garage.envs import GymEnv
import gym

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import PointEnv
from garage.experiment.deterministic import set_seed
from garage.np import discount_cumsum
from garage.sampler import LocalSampler
# from garage.torch.policies import GaussianMLPPolicy
from policies.gaussian_mlp_policy import GaussianMLPPolicy

from garage.trainer import Trainer
from drsom3d import DRSOMF


# pylint: disable=too-few-public-methods
class VPG_opt:
    """Simple Vanilla Policy Gradient.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        sampler (garage.sampler.Sampler): Sampler.

    """

    def __init__(self, env_spec, policy, sampler, opt = 'drsom'):
        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        # self.max_episode_length = env_spec.max_episode_length
        self.max_episode_length = 200

        self._discount = 0.99
        self._n_samples = 1

        if opt == 'adam':
            self._policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay = 5e-4)

        elif opt == 'drsom':
            self._policy_opt = DRSOMF(  self.policy.parameters(), 
                                        option_tr='p', 
                                        gamma=1e-3,
                                        beta1=4,
                                        beta2=2,
                                        hessian_window=1,
                                        thetas=(0.9, 0.999),)
        elif opt == 'sgd':
            self._policy_opt = torch.optim.SGD(policy.parameters(), lr=1e-2, momentum=0.9, weight_decay = 5e-4)
        elif opt == 'adagrad':
            self._policy_opt = torch.optim.Adagrad(policy.parameters(), lr=1e-2, weight_decay = 5e-4)

        


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
        # print(self.policy.get_parameter_values())

        def closure(backward=True):
            self._policy_opt.zero_grad()

            total_loss = torch.Tensor([0])
            # total_loss = []

            for path in samples:
                returns_numpy = discount_cumsum(path['rewards'], self._discount)
                returns = torch.Tensor(returns_numpy.copy())
                obs = torch.Tensor(path['observations'])
                actions = torch.Tensor(path['actions'])
                # print('原始：',actions)
                # actions =np.array(actions)
                # print('np.array后：',actions)
                # actions=list(map(int,actions))
                # print('np.array最后：',actions)
                dist = self.policy(obs)[0]
                log_likelihoods = dist.log_prob(actions)
                loss = (-log_likelihoods * returns).mean()
                total_loss = total_loss.add(loss)
                # total_loss.append(loss)

            avg_loss = total_loss.div(len(samples))
            if not backward:
                return avg_loss
            avg_loss.backward(create_graph=True)

            return avg_loss 


        self._policy_opt.step(closure=closure)

        return closure(backward = False)


@wrap_experiment
def run_VPG_opt(ctxt=None, opt ='drsom'):
    """Train VPG with PointEnv environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.

    """
    set_seed(100)
    trainer = Trainer(ctxt)
    env = GymEnv('InvertedDoublePendulum-v4')
    # env = GymEnv('CartPole-v1')
    # env=GymEnv('Taxi-v3')
    # policy = GaussianMLPPolicy(env.spec)
    # env=GymEnv('BipedalWalker-v3')
    policy = GaussianMLPPolicy(env.spec,
                            hidden_sizes=[32, 32],
                            hidden_nonlinearity=torch.tanh,
                            output_nonlinearity=None)
    policy.reset()

    sampler = LocalSampler(agents=policy,
                           envs=env,
                        #    max_episode_length=env.spec.max_episode_length)
                           max_episode_length=200)

    algo = VPG_opt(env.spec, policy, sampler, opt = opt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=200, batch_size=4000)


# for solver in ['adam', 'sgd', 'adagrad', 'drsom']:
#     run_VPG_opt(opt = solver)

run_VPG_opt(opt = 'drsom')
