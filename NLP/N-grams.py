
def gen_ngrams(words, num = 5):
    split_words = { }
    lens = len(words)
    for i in range(0, lens):
        for j in range(1, num + 1):
            if i + j < lens - num - 2:
                if words[i:i + j] in split_words:
                    split_words[words[i:i + j]][0] += 1
                    split_words[words[i:i + j]][1] = float(split_words[words[i:i + j]][0]) / float(lens)
                    split_words[words[i:i + j]][6].append(words[i - 1])
                    split_words[words[i:i + j]][7].append(words[i + j])
                else:
                    split_words[words[i:i + j]] = [1,  
                                                   1 / float(lens)
                                                   words[i:i + j], 
                                                   1, 
                                                   1, 
                                                   0,
                                                   [words[i - 1]],
                                                   [words[i + j]]
                                                   ]  
        if (i % 10000 == 0):
            print("完成 :" + str(float(i) / float(len(words)) * 100) + " %")

    return split_words