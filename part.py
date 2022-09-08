import json
import random
def read_json(filename):
        data = []
        with open('' + filename, 'r') as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data
train_data_path="/home/hcw/RSAN-master/data/multiNYT/origin/train.json"
train_data =read_json(train_data_path)
def sampling_data(data_list,ratio=0.3):
    import random
    random.shuffle(data_list)
    return data_list[0:int(len(data_list)*ratio)]

with open("train_sample3.json","w",encoding="utf-8") as f:
    print(f"initial length:{len(train_data)}")
    sample_data=sampling_data(train_data,ratio=0.3)
    print(f"sample length:{len(sample_data)}")
    f.write(json.dumps(sample_data))

with open("train_sample2.json","w",encoding="utf-8") as f:
    print(f"initial length:{len(train_data)}")
    sample_data=sampling_data(train_data,ratio=0.2)
    print(f"sample length:{len(sample_data)}")
    f.write(json.dumps(sample_data))

with open("train_sample1.json","w",encoding="utf-8") as f:
    print(f"initial length:{len(train_data)}")
    sample_data=sampling_data(train_data,ratio=0.1)
    print(f"sample length:{len(sample_data)}")
    f.write(json.dumps(sample_data))

