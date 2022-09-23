import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

epsoids = 200
col_name = 'Evaluation/AverageReturn'
# path1 = 'experiment/run_vpg/progress.csv'       #vpg alpha = 1e-2
# path2 = 'experiment/run_vpg_1/progress.csv'     #vpg alpha = 1e-1
# path3 = 'experiment/run_VPG_opt/progress.csv'     #vpg adam
# path4 = 'experiment/run_VPG_opt_1/progress.csv'     #vpg sgd
# path5 = 'experiment/run_VPG_opt_2/progress.csv'     #vpg adagrad
# path6 = 'experiment/run_VPG_opt_3/progress.csv'     #vpg drsom without hessian
# path7 = 'experiment/run_VPG_opt_4/progress.csv'     #vpg drsom
path1='experiment/run_VPG_opt/progress.csv'
path2='experiment/run_VPG_opt_1/progress.csv'
path3='experiment/run_VPG_opt_2/progress.csv'
path4='experiment/run_VPG_opt_5/progress.csv'
path5='experiment/run_VPG_opt_4/progress.csv'
path6='experiment/run_vpg/progress.csv'
path7='experiment/experiment/run_task/progress.csv'
path8='experiment/run_VPG_opt_47/progress.csv'
path9='experiment/run_VPG_opt_48/progress.csv'



drsom3d = pd.read_csv(path1)
drsom_Nesterov = pd.read_csv(path2)
drsom_heavyball=pd.read_csv(path3)
drsom_justg=pd.read_csv(path4)
Adam=pd.read_csv(path5)
vanilla_g=pd.read_csv(path6)
MBPG1=pd.read_csv(path7)
drsom_unconstrianed=pd.read_csv(path8)
drsom_ucbeter=pd.read_csv(path9)
# adam_lgbatch=pd.read_csv(path8)
# vpg_adam = pd.read_csv(path3)
# vpg_sgd = pd.read_csv(path4)
# vpg_adagrad = pd.read_csv(path5)
# vpg_drsom_no_hessian = pd.read_csv(path6)
# vpg_drsom = pd.read_csv(path7)

plt.figure(dpi=300,figsize=(8,4))
plt.title('Result on Inverted DoublePendulum')
plt.plot(np.arange(1, epsoids + 1), drsom3d[col_name], label='drsom3d')
plt.plot(np.arange(1, epsoids + 1), drsom_Nesterov[col_name], label='drsom_Nesterov')
plt.plot(np.arange(1, epsoids + 1), drsom_heavyball[col_name], label='drsom_heavyball')
plt.plot(np.arange(1, epsoids + 1), drsom_justg[col_name], label='drsom_justg')
plt.plot(np.arange(1, epsoids + 1), Adam[col_name], label='Adam')
plt.plot(np.arange(1, epsoids + 1), vanilla_g[col_name], label='vanilla_g')
plt.plot(np.arange(1, epsoids + 1), MBPG1[col_name], label='MBPG')
plt.plot(np.arange(1, epsoids + 1), drsom_unconstrianed[col_name], label='drsom_unconstrianed')
plt.plot(np.arange(1, epsoids + 1), drsom_ucbeter[col_name], label='drsom_ucbeter')
# plt.plot(np.arange(1, epsoids + 1), vpg_adagrad[col_name], label='adagrad')
# plt.plot(np.arange(1, epsoids + 1), vpg_drsom_no_hessian[col_name], label='drsom without hessian')
# plt.plot(np.arange(1, epsoids + 1), vpg_drsom[col_name], label='drsom')


plt.legend()
plt.ylabel('AVGReturn')
plt.xlabel('Episode #')
plt.savefig('InvertedDoublePendulum_momentummethod_revised4.jpg')
plt.show()