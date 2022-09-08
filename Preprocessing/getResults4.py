import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
with open("s0nytstar0.1.txt","r",encoding="utf-8") as f:
    i=0
    acc=[]
    for line in f:
        aaaa=re.findall(r"t_ent_sample_acc: ([\d.]+), t_head_rel_sample_acc: ([\d.]+), t_tail_rel_sample_acc: ([\d.]+)",line)
        if len(aaaa):
            acc.append(aaaa)
            
with open("s1nytstar0.1.txt","r",encoding="utf-8") as f1:
    i=0
    acc1=[]
    for line in f1:
        aaaa=re.findall(r"t_ent_sample_acc: ([\d.]+), t_head_rel_sample_acc: ([\d.]+), t_tail_rel_sample_acc: ([\d.]+)",line)
        if len(aaaa):
            acc1.append(aaaa)

ar1=np.array(acc1)
ar1=ar1.astype(np.float)
draw1=np.squeeze(ar1,axis=1)
draw1=np.array(draw1)

ar=np.array(acc)
ar=ar.astype(np.float)
draw=np.squeeze(ar,axis=1)
draw=np.array(draw)


df=pd.DataFrame(draw,columns=["No strategy entity","No strategy head relation","No strategy tail relation"])

df1=pd.DataFrame(draw1,columns=["Strategy entity","Strategy head relation","Strategy tail relation"])
#df.index=range(2,len(df) + 2)
#plt.figure(figsize=(16,6))

#sns.lineplot(data=df[["Overall","EPO","Normal","SEO"]],markers=True)
#sns.lineplot(data=df[["N=1","N=2","N=3","N=4","N=5"]],markers=True)
sns.set_palette("flare")
sns.lineplot(data=df)
sns.set_palette('cool')
#sns.set_palette(sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True))
sns.lineplot(data=df1.iloc[:df.shape[0]+1,:])
plt.xlabel("Iteration number")
plt.ylabel("Accuracy")
#plt.xlim((0, 30))
sns.despine()
       
        
        
        
    
#    print(" ".join(results[-3:]+results[1:-3]))
        
