import re
with open("test1.txt","r") as f:
    i=0
    results=[]
    for line in f:
#        b=r"test_triples_(\d).*?\(([\d\.]+)\,"
        
        if not i%3:
            name=re.findall("\'test_triples(.*?)\'",line)
        
        
        b=r"(\d+\.\d+)"
        a=re.findall(b,line)
#        results.append(a[0])
#        results.append(a[1])
        results.append(a[2])
        

    print(" ".join(results[2:]))
#    print(" ".join(results))
    