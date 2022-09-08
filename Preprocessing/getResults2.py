import re
with open("train_log.txt","r",encoding="utf-8") as f:
    i=0
    results=[]
    flag=True
    
    for line in f:
#        b=r"test_triples_(\d).*?\(([\d\.]+)\,"
        
        if flag:
            start=re.findall("---------------- Results -----------------------",line)
            if start != []:
                flag=False
        else:
            a=re.findall("Total epoch(\d*):",line)
            if a != []:
                flag=True
                results.append(str(int(a[0])-1))
            else:
                results.append(line)
        

        

#    print(" ".join(results[2:]))
with open("resultsaaa.txt","w",encoding="utf-8") as f:
    f.write("".join(results))

        