# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:37:00 2021

@author: 11457
"""

def romanToInt(s):
    roman = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
    z = 0
    for i in range(0, len(s) - 1):
        if roman[s[i]] < roman[s[i+1]]:
            z -= roman[s[i]]
        else:
            z += roman[s[i]]
    return z + roman[s[-1]]

def romanToInt( s):
    n=len(s)
    i=0
    ans=0
    roman_left={"I":{"V":4,"X":9},"X":{"L":40,"C":90},"C":{"D":400,"M":900}}
    roman2int={
        "I":1,
        "V":5,
        "X":10,
        "L":50,
        "C":100,
        "D":500,
        "M":1000
    }
    while i<n:
        if s[i] in roman_left and i+1<n and s[i+1] in roman_left[s[i]]:
            ans+=roman_left[s[i]][s[i+1]]
            i+=1
        else:
            ans+=roman2int[s[i]]
        i+=1
    return ans