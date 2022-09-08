filename1 = 'no_strategy_single_model.txt'
filename2 = 'no_strategy_two_model.txt'

def merge(filename1, filename2,filename3=None):
    """
    if filename3 ==None:
        Append filename2 to filename1.
    else:
        Append filename2 to filename1 ouput filename3
    """
    if filename3:
        
        with open(filename1, 'r', encoding='utf-8') as f1, open(filename2, 'r', encoding='utf-8') as f2, open(filename3, 'w', encoding='utf-8') as f3:
            for i in f1:
                f3.write(i)            
            for i in f2:
                f3.write(i)
    else:
        
        with open(filename2, 'r', encoding='utf-8') as f2, open(filename1, 'a+', encoding='utf-8') as f1:
            for i in f2:
                f1.write(i)


merge(filename1, filename2,"nostrategy_all.txt")
