import pandas as pd
import matplotlib.pyplot as plt
# from DBPG import run_VPG_opt
import numpy as np
# from VPG_new import run_vpg
import argparse
plt.figure(dpi=300,figsize=(20,10))
plt.title('Results on Invertedpendulum')

path = 'drsom_InvertedDoublePendulum-v4_bs_50000_nstep_10000000_0samllzata_2d_true.txt'
data = pd.read_csv(path,sep=' ',header=None)
plt.plot(data.iloc[:,0], data.iloc[:,1], label='small_zeta1')

path = 'drsom_InvertedDoublePendulum-v4_bs_50000_nstep_10000000_0bigzata_2d.txt'
data = pd.read_csv(path,sep=' ',header=None)
plt.plot(data.iloc[:,0], data.iloc[:,1], label='big_zeta1')


path = 'drsom_InvertedDoublePendulum-v4_bs_50000_nstep_10000000_0samllzata_2d_nesterov.txt'
data = pd.read_csv(path,sep=' ',header=None)
plt.plot(data.iloc[:,0], data.iloc[:,1], label='small_zeta1_nesterov')

path = 'adam_InvertedDoublePendulum-v4_bs_50000_nstep_10000000_0smallzata.txt'
data = pd.read_csv(path,sep=' ',header=None)
plt.plot(data.iloc[:,0], data.iloc[:,1], label='adam')


plt.legend(loc=2,fontsize=15,ncol=5,frameon=False)

plt.ylabel('AVGReturn')
plt.xlabel('timestep #')
plt.savefig('fig/results_on_Inverteddoublependulum_2nd.jpg')