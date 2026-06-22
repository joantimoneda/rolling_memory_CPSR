

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import trange

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, "test_fs"))

# RUN 1, GPT4o

files = os.listdir()
files = [f for f in files if "xlsx" in f]
files
len(files)

f1_1 = []

for i in trange(0, len(files)):
    
    file = pd.read_excel(files[i], index_col=0)

    f1_split_1 = []

    for j in range(0, (len(file)-200)): 
        split = file.iloc[j:j+200,:]
        f1_split_1.append([f1_score(split.nostalgic, split.prediction, labels=[1], average="weighted")]) 
           
    f1_1.append(f1_split_1)   


f1_1 = pd.concat([pd.DataFrame(x).T for x in f1_1], ignore_index=True)
f1_1
means_r1_1 = f1_1.mean()
means_r1_1

#MEMORY:

files = os.listdir("../test_basic_memory")
files = [f for f in files if "xlsx" in f]
files
len(files)

f1_1_mem = []

for i in trange(0, len(files)):
    
    file = pd.read_excel("../test_basic_memory/" + files[i], index_col=0)

    f1_split_1 = []

    for j in range(0, (len(file)-200)): 
        split = file.iloc[j:j+200,:]
        f1_split_1.append([f1_score(split.nostalgic, split.prediction, labels=[1], average="weighted")]) 
           
    f1_1_mem.append(f1_split_1)   


f1_1_mem = pd.concat([pd.DataFrame(x).T for x in f1_1_mem], ignore_index=True)
f1_1_mem
means_r1_1_mem = f1_1_mem.mean()
means_r1_1_mem


### 1:
plt.figure(figsize=(8,5),facecolor = 'white').patch.set_facecolor('white')
plt.plot(f1_1_mem.iloc[0,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[1,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[2,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[3,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[4,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[5,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[6,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[7,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[8,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[9,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[10,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[11,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[12,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[13,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[14,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_mem.iloc[15,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(means_r1_1_mem, color="darkred", alpha=0.75, linewidth=2, label = "Memory (F1 {})".format(round(np.mean(means_r1_1_mem), 3)))

# OVERLAY 

#plt.figure(figsize=(8,5),facecolor = 'white').patch.set_facecolor('white')
plt.plot(f1_1.iloc[0,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[1,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[2,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[3,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[4,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[5,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[6,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[7,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[8,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(f1_1.iloc[9,:], color="darkblue", alpha=0.25, linewidth=1)
plt.plot(means_r1_1, color="darkblue", alpha=0.75, linewidth=2, label = "No Memory (FS) (F1 {})".format(round(np.mean(means_r1_1), 3)))
plt.ylabel("Nostalgia Running F1-score", fontsize=11)
plt.xlabel("Cumulative Observations", fontsize=11)
plt.rc('font',family='Arial')
plt.grid(False)
plt.xticks([0, 100, 200, 300, 400], labels=["0 - 200","100 - 300", "200 - 400", "300 - 500", "400 - 600"], fontsize=10)
plt.yticks(fontsize=10)
plt.legend(frameon=False, loc=(0.55, 0.05), fontsize=11)
#plt.savefig('../fig_fs_memory_reshuffle_overlay_1.pdf', format="pdf", bbox_inches='tight', transparent=True)
plt.savefig('../../../figures/fig2.pdf', format="pdf", bbox_inches='tight', transparent=True)
plt.show()
# NOTE: exploratory per-class plots from the original script were removed;
# this script reproduces only the paper figure saved above.
