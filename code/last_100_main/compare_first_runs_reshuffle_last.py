

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import trange

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, "last_100_reshuffle"))


# RUN 1, GPT4o

files = os.listdir()
files = [f for f in files if "run1" in f]
files
files_peak = os.listdir("../../peak_100/peak_reshuffle")
files_peak = [f for f in files_peak if "run1" in f]
files_peak = ["../../peak_100/peak_reshuffle/" + x for x in files_peak]
files_peak
len(files_peak)
files = sorted(files + files_peak)
len(files)

f1_all = []
f1_0 = []
f1_1 = []

for i in trange(0, len(files)):
    
    file = pd.read_excel(files[i], index_col=0)

    f1_split_0 = []
    f1_split_1 = []
    f1_split_ov = []

    for j in range(0, (len(file)-200)): 
        split = file.iloc[j:j+200,:]
        f1_split_ov.append([f1_score(split.nostalgic, split.prediction, average="weighted")]) 
        f1_split_0.append([f1_score(split.nostalgic, split.prediction, labels=[0], average="weighted")]) 
        f1_split_1.append([f1_score(split.nostalgic, split.prediction, labels=[1], average="weighted")]) 
           
    f1_all.append(f1_split_ov)   
    f1_0.append(f1_split_0)   
    f1_1.append(f1_split_1)   


f1_r1 = pd.concat([pd.DataFrame(x).T for x in f1_all], ignore_index=True)
f1_r1
f1_0_r1 = pd.concat([pd.DataFrame(x).T for x in f1_0], ignore_index=True)
f1_0_r1
f1_1_r1 = pd.concat([pd.DataFrame(x).T for x in f1_1], ignore_index=True)
f1_1_r1

means_r1 = f1_r1.mean()
means_r1
means_r1_0 = f1_0_r1.mean()
means_r1_0
means_r1_1 = f1_1_r1.mean()
means_r1_1

plt.plot(means_r1, color="darkred", alpha=0.75, linewidth=1)
plt.plot(means_r1_0, color="darkred", alpha=0.5, linewidth=1)
plt.plot(means_r1_1, color="darkred", alpha=0.5, linewidth=1)





# RUN 2

files = os.listdir()
files = sorted([f for f in files if "run2" in f])
files 
len(files)

f1_all = []
f1_0 = []
f1_1 = []

for i in trange(0, len(files)):
    
    file = pd.read_excel(files[i], index_col=0)

    f1_split_0 = []
    f1_split_1 = []
    f1_split_ov = []

    for j in range(0, (len(file)-200)): 
        split = file.iloc[j:j+200,:]
        f1_split_ov.append([f1_score(split.nostalgic, split.prediction, average="weighted")]) 
        f1_split_0.append([f1_score(split.nostalgic, split.prediction, labels=[0], average="weighted")]) 
        f1_split_1.append([f1_score(split.nostalgic, split.prediction, labels=[1], average="weighted")]) 
           
    f1_all.append(f1_split_ov)   
    f1_0.append(f1_split_0)   
    f1_1.append(f1_split_1)   


f1 = pd.concat([pd.DataFrame(x).T for x in f1_all], ignore_index=True)
f1
f1_0 = pd.concat([pd.DataFrame(x).T for x in f1_0], ignore_index=True)
f1_0
f1_1 = pd.concat([pd.DataFrame(x).T for x in f1_1], ignore_index=True)
f1_1.iloc[:,180:185]

means_r2 = f1.mean()
means_r2
means_r2_0 = f1_0.mean()
means_r2_0
means_r2_1 = f1_1.mean()
means_r2_1

plt.plot(means_r2, color="darkred", alpha=0.75, linewidth=1)
plt.plot(means_r2_0, color="darkred", alpha=0.5, linewidth=1)
plt.plot(means_r2_1, color="darkred", alpha=0.5, linewidth=1)


plt.plot(means_r1_1, color="darkred", alpha=0.5, linewidth=1)
plt.plot(means_r2_1, color="darkblue", alpha=0.5, linewidth=1)

means_r1_1.mean()
means_r2_1.mean()


### MAIN FIGURE COMPARING RUNS:

plt.figure(figsize=(8,5),facecolor = 'white').patch.set_facecolor('white')
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
plt.plot(means_r2_1, color="darkblue", alpha=0.75, linewidth=2, label = "Run 2 (F1 {})".format(round(np.mean(means_r2_1), 3)))

plt.plot(f1_1_r1.iloc[0,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[1,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[2,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[3,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[4,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[5,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[6,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[7,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[8,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[9,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[10,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[11,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[12,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[13,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[14,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[15,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[16,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[17,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[18,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(f1_1_r1.iloc[19,:], color="darkred", alpha=0.25, linewidth=1)
plt.plot(means_r1_1, color="darkred", alpha=0.75, linewidth=2, label = "Run 1 (F1 {})".format(round(np.mean(means_r1_1), 3)))

plt.ylabel("Nostalgia Running F1-score", fontsize=11)
plt.xlabel("Cumulative Observations", fontsize=11)
plt.rc('font',family='Arial')
plt.grid(False)
plt.xticks([0, 100, 200, 300, 400], labels=["0 - 200","100 - 300", "200 - 400", "300 - 500", "400 - 600"], fontsize=10)
plt.yticks(fontsize=10)
plt.legend(frameon=False, loc=(0.7, 0.1), fontsize=11)
#plt.savefig('fig_reshuffle_last100_overlay_all_r1.pdf', format="pdf", bbox_inches='tight', transparent=True)
plt.savefig('../../../figures/fig3a.pdf', format="pdf", bbox_inches='tight', transparent=True)
plt.show()

# NOTE: exploratory per-class plots from the original script were removed;
# this script reproduces only the paper figure saved above.
