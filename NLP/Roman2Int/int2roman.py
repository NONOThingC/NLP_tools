# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:46:22 2021

@author: 11457
"""

def intToRoman(num):
    d = {1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I'} 
    
    res = ""
    
    for i,value in d.items():
        res += (num//i) * value
        num %= i
    
    return res