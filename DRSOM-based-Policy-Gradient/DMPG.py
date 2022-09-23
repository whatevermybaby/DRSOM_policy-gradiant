import numpy as np
import torch
from garage.envs import GymEnv
import gym
import time
import os.path

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import PointEnv
from garage.experiment.deterministic import set_seed
from garage.np import discount_cumsum
from garage.sampler import LocalSampler
# from garage.torch.policies import GaussianMLPPolicy
# from policies.gaussian_mlp_policy import GaussianMLPPolicy, CategoricalMLPPolicy
from policies.gaussian_mlp_policy import GaussianMLPPolicy
# from policies.CategoricalMLPPolicy import CategoricalMLPPolicy
from garage.envs import normalize

from garage.trainer import Trainer
from drsom3d import DRSOMF
# from rsomfa import RSOMF

# from drsom_norm import DRSOMF



# pylint: disable=too-few-public-methods
class VPG_opt:
    """Simple Vanilla Policy Gradient.
    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        sampler (garage.sampler.Sampler): Sampler.
    """

    def __init__(self, 
                env_spec, 
                policy, 
                sampler, 
                opt = 'drsom', 
                option_tr='p', 
                gamma=1e-12,
                beta1=5e1,
                beta2=3e1,
                hessian_window=1,
                thetas=(0.9, 0.999),
                env = 'unknown',
                batch_size = 5e4,
                n_timestep = 1e7,
                log_dir = 'log'):

        self.env_spec = env_spec
        self.policy = policy
        self._sampler = sampler
        # self.max_episode_length = env_spec.max_episode_length
        self._discount = 0.99

        self.batch_size = batch_size
        self.n_timestep = n_timestep
        self.log_name = log_dir + '/%s_%s_bs_%d_nstep_%d_0' % (opt, env, batch_size, n_timestep)

        num = 1
        while os.path.isfile(self.log_name + '.txt'):
            self.log_name = self.log_name[:-1] + str(num)
            num += 1



        if opt == 'adam':
            self._policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay = 5e-4)

        elif opt == 'drsom':
            self._policy_opt = DRSOMF(  self.policy.parameters(), 
                                        option_tr=option_tr, 
                                        gamma=gamma,
                                        beta1=beta1,
                                        beta2=beta2,
                                        hessian_window=hessian_window,
                                        thetas=thetas)
        # elif opt == 'drsom_fd':
        #     self._policy_opt = RSOMF(  self.policy.parameters(), 
        #                                 option_tr=option_tr, 
        #                                 # gamma=gamma,
        #                                 theta1=beta1,
        #                                 theta2=beta2,
        #                                 hessian_window=hessian_window,
        #                                 betas=thetas)
        elif opt == 'sgd':
            self._policy_opt = torch.optim.SGD(policy.parameters(), lr=1e-3, momentum=0.9, weight_decay = 5e-4)
        elif opt == 'adagrad':
            self._policy_opt = torch.optim.Adagrad(policy.parameters(), lr=1e-3, weight_decay = 5e-4)

        
    def get_log_name(self):
        return self.log_name

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.
        Args:
            trainer (Trainer): Experiment trainer.
        """
        j = 0
        start = time.time()
        with open(self.log_name + 'bigzata_2d.txt', 'a') as file:

            while (self.batch_size < self.n_timestep - j):
                # print("我在跑呢")
                samples = trainer.obtain_samples(j)
                j += sum([len(path["rewards"]) for path in samples])

                for path in samples:
                    path['returns'] = discount_cumsum(path['rewards'], self._discount)

                def closure(backward=True):
                    self._policy_opt.zero_grad()

                    total_loss = torch.Tensor([0])
                    for path in samples:
                        # returns_numpy = discount_cumsum(path['rewards'], self._discount)
                        returns_numpy = path['returns']
                        returns = torch.Tensor(returns_numpy.copy())
                        obs = torch.Tensor(path['observations'])
                        actions = torch.Tensor(path['actions'])
                        dist = self.policy(obs)[0]
                        log_likelihoods = dist.log_prob(actions)
                        loss = (-log_likelihoods * returns).mean()
                        total_loss = total_loss.add(loss)

                    avg_loss = total_loss.div(len(samples))
                    if not backward:
                        return avg_loss
                    avg_loss.backward(create_graph=True)

                    return avg_loss 

                avg_returns = np.mean([p['returns'][0] for p in samples])
                end = time.time()
                file.write("%d %.4f %.4f\n" % (j, avg_returns, end-start))
                print("timesteps: " + str(j) + " average return: ", avg_returns, "time", end-start)
                
                self._policy_opt.step(closure=closure)

        file.close()





@wrap_experiment
def run_VPG_opt(ctxt=None, 
                opt ='drsom',
                option_tr='p', 
                gamma=1e-8,
                beta1=5,
                beta2=3,
                hessian_window=1,
                thetas=(0.9, 0.999),
                environment = 'InvertedDoublePendulum-v4',
                batch_size = 5e4,
                n_timestep = 1e7,
                max_path_length = 500,
                log_dir = 'log'):
    """Train VPG with PointEnv environment.
    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
    """
    set_seed(100)
    trainer = Trainer(ctxt)
    # env = GymEnv(environment)
    env = normalize(GymEnv(environment))

    policy = GaussianMLPPolicy(env.spec,
                            hidden_sizes=[32, 32],
                            hidden_nonlinearity=torch.tanh,
                            output_nonlinearity=None)


    # if 'CartPole' in environment:
    #     policy = CategoricalMLPPolicy(env.spec,
    #                             hidden_sizes=[8, 8],
    #                             hidden_nonlinearity=torch.tanh,
    #                             output_nonlinearity=None)

    policy.reset()

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=max_path_length)

    algo = VPG_opt( env.spec, 
                    policy, 
                    sampler, 
                    opt = opt,
                    option_tr=option_tr, 
                    gamma=gamma,
                    beta1=beta1,
                    beta2=beta2,
                    hessian_window=hessian_window,
                    thetas=thetas,
                    env=environment,
                    batch_size=batch_size,
                    n_timestep=n_timestep,
                    log_dir=log_dir
                   )

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=batch_size)

    # return log_dir + '/%s_%s_bs_%d_nstep_%d.txt' % (opt, environment, batch_size, n_timestep)
    return algo.get_log_name()
run_VPG_opt(opt='drsom')