def intellgent_value_control(lst_eva_val,cur_eva_val,lst_value,change_value,is_random,direction=1,epsilon=1e-2):
    """
    direction =1 plus else add
    lst_value is value need to be controled.
    """
    if cur_val-lst_val<epsilon:# find opt
        pass
    else:# generate random
        return random.gauss(0, 0.3)