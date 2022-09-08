import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
with open("resultsaaa.txt","r",encoding="utf-8") as f:
    i=0
    results=[]
    result=None
    for line in f:
        aaaa=re.findall(r"3YF2qsDm",line)
        if aaaa != []:
            if result is not None:
                results.append(result)
            result=[]
        aaa=re.findall(r"test_triples",line)
        b=r"(0\.\d+)"
        num=re.findall(b,line)
        if aaa != []:
            flag=True
            i=0
        if flag:
            i+=1
            if i==3:
                result.append(num[0])
                
ar=np.array(results)
ar=ar.astype(np.float)
need=ar.T

draw=[]
#
for i in range(need.shape[0]//2):
    draw.append(np.max(need[i:i+2,:],axis=0))

draw=np.array(draw).T

df=pd.DataFrame(draw[1:,:],columns=["Overall","N=1","N=2","N=3","N=4","N=5","EPO","Normal","SEO"])
df.index=range(2,len(df) + 2)
plt.figure(figsize=(16,6))

sns.lineplot(data=df[["Overall","EPO","Normal","SEO"]],markers=True)
#sns.lineplot(data=df[["N=1","N=2","N=3","N=4","N=5"]],markers=True)
#sns.lineplot(data=df,markers=True)
plt.xlabel("Epoch")
plt.ylabel("F1 score")
sns.despine()
#plt.xlim((0, 30))
            
       
        
        
        
    
#    print(" ".join(results[-3:]+results[1:-3]))
        
