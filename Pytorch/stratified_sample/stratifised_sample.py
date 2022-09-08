def split_sample(dataset,n_part):
    """
    Given pytorch dataset
    return index of n_part
    """
    factor=len(dataset)//n_part
    res=len(dataset)%n_part
    return [factor + int((factor * (i + 1) + res) // len(dataset)) * res for i in range(n_part)]


def stratified_sample(dataset, ratio, display=False):
    """
    根据样本中数据的比例来采样
    """
    import collections

    data_dict = collections.defaultdict(list)
    for i in range(len(dataset)):
        j=dataset[i][-1][1]#每个例子中每个类加入一个#every sentence multi relations
        if len(j) != 0:
            rel_record=[]
            for every_rel_ins in j:
                rel_id=every_rel_ins[0]
                if rel_id not in rel_record:
                    data_dict[rel_id].append(i)# data_dict[label]+=train_index
                    rel_record.append(rel_id)
        else:
            data_dict[-1].append(i)# 对应无tuple那种关系

    sampled_indices = []
    rest_indices = []


    if display:
        plot_samples=[]
        plot_rest=[]
        for rel_id,indices in data_dict.items():
            random.shuffle(indices)
            index = int(len(indices) * ratio + 1)
            sampled_indices += indices[:index]
            plot_samples.append(len(indices[:index]))
            rest_indices += indices[index:]
            plot_rest.append(len(indices[index:]))
        import matplotlib.pyplot as plt
        plt.subplot(212)
        plt.bar(data_dict.keys(), [len(x) for x in data_dict.values()])
        plt.title("origin training data")
        plt.subplot(221)
        plt.bar(data_dict.keys(), plot_samples)
        plt.title("labeled data")
        plt.subplot(222)
        plt.bar(data_dict.keys(), plot_rest)
        plt.title("unlabeled data")
        plt.show()
    else:
        for indices in data_dict.values():
            random.shuffle(indices)
            index = int(len(indices) * ratio + 1)
            sampled_indices += indices[:index]
            rest_indices += indices[index:]
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices)]