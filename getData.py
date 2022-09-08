# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:02:04 2020

@author: 11457
"""

import json
import sys,os
import collections
class GetFile():
    
    def __init__(self,entity_filename='entities.json',attrs_filename='attrs.json',relationships_filename='relationships.json',schema_filename='schema.json'):
        BASE_DIR=os.path.dirname(os.path.realpath(__file__))
        self._entities=BASE_DIR+os.path.sep+entity_filename
        self._attrs=BASE_DIR+os.path.sep+attrs_filename
        self._relations=BASE_DIR+os.path.sep+relationships_filename
        self._schema=BASE_DIR+os.path.sep+schema_filename
        self._func_code=1
        

    def get_data(self,fun_code)->tuple:
        filename=''
        if fun_code==1:# 实体
            filename=self._entities
        elif fun_code==2:# 关系
            filename=self._relations
        elif fun_code==3:# 属性
            filename=self._attrs
        else:# schema
            filename=self._schema
        with open(filename,'r',encoding='utf-8') as f:
            a=json.load(f)
            for line in a.items():
                yield line
    
    def get_document(self,start,end,way='line'):#
        # 注意，文档名字是[start,end]的闭区间
        prefix="./yanbao_txt/yanbao"
        for i in range(start,end+1):
            num=str(i).rjust(3,'0')
            filename=prefix+str(num)+'.txt'
            try:
                with open(filename,'r',encoding='utf-8') as f:
                    if(way=='line'):#一次得到一个文档的每行，每行都是一个str
                        for line in f:
                            yield line
                    elif(way=='document_list'):#一次得到一个文档，每篇文档都是一个list
                        yield f.readlines()
                    elif(way=='document_str'):#一次得到一个文档，每篇文档都是一个str
                        yield f.read()
                    else:
                        yield f.read()#一次得到一个文档，一整篇是一个str
            except FileNotFoundError:
                print("file {} not exists".format(filename))
                break

    def get_entity_dict(self,reverse=False):
        entity_dict=collections.defaultdict(list)
        if reverse:
            for i,list_j in self.GetFile().get_data(1):
                for j in list_j:
                    entity_dict[j].append(i)#ins->type
        else:
            for i,list_j in self.GetFile().get_data(1):
                for j in list_j:
                    entity_dict[i].append(j)#type->ins
        return entity_dict

def chunked_file_reader(fp, block_size=1024 * 8):
    """生成器函数：分块读取文件内容
    """
    while True:
        chunk = fp.read(block_size)
        # 当文件没有更多内容时，read 调用将会返回空字符串 ''
        if not chunk:
            break
        yield chunk


def return_count_v3(fname):
    count = 0
    with open(fname) as fp:
        for chunk in chunked_file_reader(fp):
            count += 1
    return count

def retrun_count(fname):
    """计算文件有多少行
    """
    count = 0
    with open(fname) as file:
        for line in file:
            count += 1
    return count



if __name__ == '__main__':
    
    # 用法
    ## 读取实体，关系，属性，模式
    entities=GetFile().get_data(1)
    for i in entities:
        print(i)
    print("-"*30)
    relations=GetFile().get_data(2)
    for i in relations:
        print(i)
    print("-"*30)
    
    ## 读取文档
    document=GetFile().get_document(0,5)#按照行来读取
    for i in document:
        print(i)
    document=GetFile().get_document(0,5,way='document_list')#按照文档来读取，一次得到一个文档，每篇文档都是一个list
    for i in document:
        print(i)
    document=GetFile().get_document(0,5,way='document_str')#按照文档来读取，一次得到一个文档，一整篇是一个str
    for i in document:
        print(i)

