import re
with open("test.txt","r",encoding="utf-8") as f:
    i=0
    results=[]
    for line in f:
#        b=r"test_triples_(\d).*?\(([\d\.]+)\,"
        
        if not i%6:
            name=re.findall("\'test_triples(.*?)\'",line)
        
        
        b=r".*?([\d\.]+)[\,\)]"
        a=re.findall(b,line)
        if not (i+1)%6:
            results.append(a[0])
        i+=1
    results[-1],results[-2]=results[-2],results[-1]
    results[-1],results[-3]=results[-3],results[-1]
    
    print(" ".join(results[-3:]+results[1:-3]))
        
