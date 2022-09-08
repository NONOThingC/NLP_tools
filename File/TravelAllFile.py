# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:21:26 2021

@author: 11457
"""
import os

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.xls'):
                fullname = os.path.join(root, f)
                yield fullname

def main():
    base = './base/'
    for i in findAllFile(base):
        print(i)

if __name__ == '__main__':
    main()


# def findAllFile(base):
#     for root, ds, fs in os.walk(base):
#         for f in fs:
#             fullname = os.path.join(root, f)
#             yield fullname