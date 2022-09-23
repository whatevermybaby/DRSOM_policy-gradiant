import torch
import garage
import os
from garage.envs import GymEnv
# from garage.experiment import run_experiment
from garage.experiment import Snapshotter
from garage.np.baselines import LinearFeatureBaseline
# from garage.tf.envs import TfEnv
# from Policy import GaussianMLPPolicy, CategoricalMLPPolicy
from Algorithms.MBPG import MBPG_IM
from gym.envs.classic_control import CartPoleEnv

from garage.envs import normalize

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import PointEnv
from garage.experiment.deterministic import set_seed
from garage.np import discount_cumsum
from garage.sampler import LocalSampler
# from garage.torch.policies import GaussianMLPPolicy
from Policy.GaussianMLPPolicy import GaussianMLPPolicy

from garage.trainer import Trainer
# from drsom import DRSOMF

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
# from policies.gaussian_mlp_policy import GaussianMLPPolicy

from garage.trainer import Trainer
# # from drsom import DRSOMF
import argparse
parser = argparse.ArgumentParser(description='IS-MBPG')
parser.add_argument('--env', default='InvertedDoublePendulum', type=str, help='choose environment from [CartPole, Walker, Hopper, HalfCheetah]')
parser.add_argument('--IS_MBPG_star', default=False, type=bool, help='whether to use IS-MBPG*')

args = parser.parse_args()

@wrap_experiment
def run_task(ctxt=None):
    """Set up environment and algorithm and run the task.
    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters
    """

    #count = 1
    th = 1.8
    g_max = 0.05
    star_version = args.IS_MBPG_star
    if args.env == 'InvertedDoublePendulum':
    #CartPole
    
        set_seed(100)
        
        trainer = Trainer(ctxt)
        env = GymEnv('Hopper-v2')
        # runner = Snapshotter()
        batch_size = 5000
        max_length = 100
        n_timestep = 5e4
        n_counts = 1
        name = 'InvertedDoublePendulum'
        #grad_factor = 5
        grad_factor = 100
        th = 1.2
        # # batchsize:1
        # lr = 0.1
        # w = 1.5
        # c = 15

        #batchsize:50
        lr = 0.75
        c = 1
        w = 1

        # for MBPG+:
        # lr = 1.2

        #g_max = 0.03
        discount = 0.995
        path = './init/InvertedDoublePendulum_policy.pth'

    # if args.env == 'Walker':
    #     #Walker_2d
    #     env = TfEnv(normalize(Walker2dEnv()))
    #     # runner = LocalRunner(snapshot_config)
    #     runner = Snapshotter()
    #     batch_size = 50000
    #     max_length = 500

    #     th = 1.2

    #     n_timestep = 1e7
    #     n_counts = 5
    #     lr = 0.75
    #     w = 2
    #     c = 5
    #     grad_factor = 10

    #     # for MBPG+:
    #     #lr = 0.9

    #     discount = 0.999

    #     name = 'Walk'
    #     path = './init/Walk_policy.pth'

    # if args.env == 'Hopper':
    #     #Hopper
    #     env = TfEnv(normalize(HopperEnv()))
    #     # runner = LocalRunner(snapshot_config)
    #     runner = Snapshotter()

    #     batch_size = 50000

    #     max_length = 1000
    #     th = 1.5
    #     n_timestep = 1e7
    #     n_counts = 5
    #     lr = 0.75
    #     w = 1
    #     c = 3
    #     grad_factor = 10
    #     g_max = 0.15
    #     discount = 0.999

    #     name = 'Hopper'
    #     path = './init/Hopper_policy.pth'

    # if args.env == 'HalfCheetah':
    #     env = TfEnv(normalize(HalfCheetahEnv()))
    #     runner = Snapshotter()
    #     batch_size = 10000
    #     #batch_size = 50000
    #     max_length = 500

    #     n_timestep = 1e7
    #     n_counts = 5
    #     lr = 0.6
    #     w = 3
    #     c =7
    #     grad_factor = 10
    #     th = 1.2
    #     g_max = 0.06

    #     discount = 0.999

    #     name = 'HalfCheetah'
    #     path = './init/HalfCheetah_policy.pth'
    for i in range(n_counts):
        print(env.spec)



        if args.env == 'CartPole':
            policy = CategoricalMLPPolicy(env.spec,
                                       hidden_sizes=[8, 8],
                                       hidden_nonlinearity=torch.tanh,
                                       output_nonlinearity=None)
        elif args.env=="InvertedDoublePendulum":
            policy= GaussianMLPPolicy(env.spec,
                            hidden_sizes=[32, 32],
                            hidden_nonlinearity=torch.tanh,
                            output_nonlinearity=None)
            policy.reset()
        # # # else:
        #     policy = GaussianMLPPolicy(env.spec,
        #                                hidden_sizes=[64, 64],
        #                                hidden_nonlinearity=torch.tanh,
        #                                output_nonlinearity=None)
        # # policy.load_state_dict(torch.load(path))
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        sampler = LocalSampler(agents=policy,
                           envs=env,
                        #    max_episode_length=env.spec.max_episode_length)
                           max_episode_length=200)
        algo = MBPG_IM(
                   sampler=sampler,
                   env_spec=env.spec,
                   env = env,
                   env_name= name,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=max_length,
                   discount=discount,
                   grad_factor=grad_factor,
                   policy_lr= lr,
                   c = c,
                   w = w,
                   n_timestep=n_timestep,
                   #count=count,
                   th = th,
                   batch_size=batch_size,
                   center_adv=True,
                   g_max = g_max,
                   #decay_learning_rate=d_lr,
                   star_version=star_version
                   )

        trainer.setup(algo, env)
        trainer.train(n_epochs=20, batch_size=batch_size)
        # runner.setup(algo, env)
        # runner.train(n_epochs=100, batch_size=batch_size)


run_task()


