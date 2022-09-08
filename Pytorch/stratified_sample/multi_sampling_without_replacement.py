def c(self,list_collection, n,shuffle=True):
    if shuffle:
        list_collection=random.shuffle(list_collection)
    return list(self.split_list_by_n(list_collection,n,last_to_n=True))

def split_list_by_n(self,list_collection, n,last_to_n=False):
    """
    将list均分，每份n个元素
    :return:返回的结果为评分后的每份可迭代对象
    """
    for i in range(0, len(list_collection), n):
        if last_to_n:
            if (i+n)>len(list_collection):
                yield list_collection[i:]+random.choice(list_collection,i+n-len(list_collection)-1)
            else:
                yield list_collection[i: i + n]

        else:
            yield list_collection[i: i + n]