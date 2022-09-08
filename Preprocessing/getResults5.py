import re
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
with open("train_log1-strategy.txt","r",encoding="utf-8") as f:
    i=0
    results={"te":[],"stu":[]}
    flag_stu=False
    flag_te=False
    tmp_list=[]
    for line in f:
#        b=r"test_triples_(\d).*?\(([\d\.]+)\,"
        ent=re.findall("'student val_ent_seq_acc': ([\d.]+), 'student val_head_rel_acc': ([\d.]+), 'student val_tail_rel_acc': ([\d.]+),",line)
#        head=re.findall("'student val_head_rel_acc': ([\d.]+),",line)
#        tail=re.findall("'student val_tail_rel_acc': ([\d.]+),",line)
        if ent != []:
            tmp_list.append(ent[0])
            flag_stu=True
        else:
            ent=re.findall("'val_ent_seq_acc': ([\d.]+), 'val_head_rel_acc': ([\d.]+), 'val_tail_rel_acc': ([\d.]+)",line)
            if ent != []:
                tmp_list.append(ent[0])
                flag_te=True
        if flag_stu:
            result=re.findall("generate pseudo label,Z_RATIO: ([\d.]+), NUMBER: ([\d.]+) ",line)
            if len(result):
                results["stu"].append(tmp_list+[float(result[0][0])])
                flag_stu=False
                tmp_list=[]
        elif flag_te:
            result=re.findall("generate pseudo label,Z_RATIO: ([\d.]+), NUMBER: ([\d.]+) ",line)
            if len(result):
                results["te"].append(tmp_list+[float(result[0][0])])
                flag_te=False
                tmp_list=[]
draw=[]
for (ent,head,tail),score in results["te"]:
    draw.append([float(ent)+float(head)+float(tail),float(ent),float(head),float(tail),score])

draw1=[]
for (ent,head,tail),score in results["stu"][1:]:
    draw1.append([float(ent)+float(head)+float(tail),float(ent),float(head),float(tail),score])

df=pd.DataFrame(draw,columns=["all","ent","head","tail","score"])
df1=pd.DataFrame(draw1,columns=["all","ent","head","tail","score"])

sns.lineplot(data=df)
sns.set_palette('cool')

plt.xlabel("Iteration number")
plt.ylabel("Accuracy")
#plt.xlim((0, 30))
sns.despine()
plt.figure()
sns.lineplot(data=df1)
sns.set_palette('cool')

#    print(" ".join(results[2:]))
with open("resultsaaa.txt","w",encoding="utf-8") as f:
    f.write("".join(results))

        