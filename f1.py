# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:54:03 2021

@author: 11457
"""

def get_prf_scores(precision,recall):
    minimini = 1e-10
    minimini = 0
    precision = 0.774345395704607
    recall = 0.668275862068957
    f1 = 2 * precision * recall / (precision + recall + minimini)
    return precision, recall, f1

def get_metric(p_num: int, total_num: int, total_predicted_num: int) -> Tuple[float, float, float]:
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, fscore