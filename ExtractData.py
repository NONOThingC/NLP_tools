# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:02:04 2020

@author: 11457
"""

import json
import sys,os
import collections
import re
import pathlib
from argparse import ArgumentParser



if __name__ == '__main__':
    parser = ArgumentParser(
        description="Extract filename")
    parser.add_argument('--filename', default='nohup.txt')
    args = parser.parse_args()
    filename=args.filename
    Pattern="\sf1\s=([\s\d\.]+)\n"
#    filename=pathlib.Path.cwd() / "nohup5.txt"
    save_file=[]
    save_filename="Ex_"+filename
    with open(filename,'r',encoding='utf-8') as f:
        a=f.read()
        all_result=re.findall(Pattern,a)
        with open(save_filename,'w',encoding='utf-8') as wf:
        
            for i,result in enumerate(all_result):
                save_file.append((i,result))
                wf.write(str(i)+'\t'+result+'\n')
