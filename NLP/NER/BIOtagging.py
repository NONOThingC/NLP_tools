#!/usr/bin/python
# -*- coding: UTF-8 -*-
import collections
import pandas as pd
import re
import os
from pip import main
from tqdm import tqdm
import json
from typing import List
import logging
# import process.tag_scheme_converter import BIO2BIOES,BIOES2BIO
import nltk
import ast
import sklearn
import pickle
import random

class QCClean(object):
     
    def __init__(self,qc_data_path,handle_attr=None,convert_to_BIO=True,split_way=["new-value","random"]) -> None:
        #读取和清理，生成(indexed_text, attribute_type, start_idx, end_idx)集合
        df=pd.read_csv(qc_data_path)
        df=df[['Category ID', 'item_id', 'shop_id', 'model_id','title',  'description','extraction_right_value_list_all']]
        
        if handle_attr is None:
            handle_attr=self._get_attr_type(df)
        # handle_attr2idx={t:i for i,t in enumerate(handle_attr)}
        self.handle_attr2spe_attr_type={t:self._replace_space_to_spe_token(t) for t in handle_attr}
        self.type2value_counter=collections.defaultdict(dict)
        self.handle_attr=handle_attr
        # clean and get attr_group
        df["title_tagged"]=df.apply(lambda x:self._clean_text(description_text=x['description'],title_text=x['title'],taggings=x['extraction_right_value_list_all'],line_break="\x01",handle_attr=handle_attr),axis=1)
        self.df=df #  function below would change df
        self._drop_duplication()
        self.type2value_counter=self.statistic_attr_type2value2cnt(df)#update self.type2value_counter
        self.path = os.getcwd()
        self.is_convert_to_BIO=convert_to_BIO
        if self.is_convert_to_BIO:
            self.convert_to_BIO()
        
        if "random" in split_way:
            self.save_random_dataset(test_ratio=0.2)
        if "new-value" in split_way:
            self.save_gen_new_words_dataset(test_ratio=0.2,new_word_threshold=40)
            

    def _drop_duplication(self):
        self.df=self.df.loc[self.df.apply(lambda x:len(x['title_tagged'])!=0,axis=1)].reset_index(drop=True)#去除空值
        
        column_name="title_tagged"
        self.df[column_name+"_str"]=self.df[column_name].map(str)
        self.df=self.df.loc[self.df[[column_name+"_str",'title']].assign(C=self.df[[column_name+"_str",'title']].apply(lambda x: tuple(sorted(x)), axis=1)).drop_duplicates('C').index]
        self.df.drop(labels=column_name+"_str",axis=1)
        # self.df[["title","title_tagged"]].duplicated()
        # self.df[["title","title_tagged"]].assign(C=self.df.apply(lambda x: tuple(sorted(x)), axis=1))

    def save_gen_new_words_dataset(self,test_ratio=0.2,new_word_threshold=40,dataset_name="new-value"):
        output_path=os.path.join(self.path,"output/"+str(dataset_name))
        os.makedirs(output_path, exist_ok=True)
        train_dataset_df,test_dataset_df=self._split_by_new_words(test_ratio=test_ratio,select_threshold=new_word_threshold)
        print(f"training data value:{self.statistic_attr_type2value2cnt(train_dataset_df)}")
        print(f"test data value:{self.statistic_attr_type2value2cnt(test_dataset_df)}")
        self._save_information(train_dataset_df,test_dataset_df,output_path)
        
    def save_random_dataset(self,test_ratio=0.2,dataset_name="normal-value"):
        output_path=os.path.join(self.path,"output/"+str(dataset_name))
        os.makedirs(output_path, exist_ok=True)
        train_dataset_df,test_dataset_df=self._split_by_instances(test_ratio)
        self._save_information(train_dataset_df,test_dataset_df,output_path)
        
    def _save_information(self,train_dataset_df,test_dataset_df,output_path):
        if self.is_convert_to_BIO:
            self._save_BIO_txt(train_dataset_df,test_dataset_df,output_path)
        print(f"train amount:{train_dataset_df.shape[0]},test amount:{test_dataset_df.shape[0]}")
        
        train_dataset_df=train_dataset_df.reset_index(drop=True)
        test_dataset_df=test_dataset_df.reset_index(drop=True)
        train_dataset_df.to_csv(os.path.join(output_path,"train.csv"))
        test_dataset_df.to_csv(os.path.join(output_path,"test.csv"))
        train_type2cnt=self.get_attr_type2cnt(train_dataset_df)
        train_type2cnt["train_amount"]=train_dataset_df.shape[0]
        test_type2cnt=self.get_attr_type2cnt(test_dataset_df)
        test_type2cnt["test_amount"]=test_dataset_df.shape[0]
        
        with open(os.path.join(output_path,'train_attr_count.json'), 'w') as fp:
            json.dump(train_type2cnt, fp)
        with open(os.path.join(output_path,'test_attr_count.json'), 'w') as fp:
            json.dump(test_type2cnt, fp)
        
        with open(os.path.join(output_path,"handle_attr.pkl"),"wb") as f:
            pickle.dump(self.handle_attr,f)
        with open(os.path.join(output_path,'handle_attr2spe_attr_type.json'), 'w') as fp:
            json.dump(self.handle_attr2spe_attr_type, fp)
            
        # with open(os.path.join(output_path,"handle_attr2spe_attr_type.pkl"),"wb") as f:
        #     pickle.dump(self.handle_attr2spe_attr_type,f)
        
    def _split_by_new_words(self,test_ratio=0.2,select_threshold=40):
        # To dataframe
        dict2dataframe=collections.defaultdict(list)
        for attr_type in self.type2value_counter.keys():
            for value_content,cnt in self.type2value_counter[attr_type].items():
                dict2dataframe["attr_type"].append(attr_type)
                dict2dataframe["value_content"].append(value_content)
                dict2dataframe["count"].append(cnt)
        statistic_df=pd.DataFrame.from_dict(dict2dataframe)#"attr_type","value_content","count"
        type2test_value,no_new_words_attrtype,(all_value_occur_count,test_value_occur_count)=self._select_attr_until_threshold(statistic_df,self.df,test_ratio=0.2,select_threshold=40,column_name="title_tagged")
        # split dataset 
        self.df["is_test"]=self.df.apply(lambda x:self._filter_new_value(x["title_tagged"],type2test_value,no_new_words_attrtype),axis=1)
        train_dataset_df,test_dataset_df=self.df.loc[self.df["is_test"]==False ],self.df.loc[self.df["is_test"]==True]
        return train_dataset_df,test_dataset_df
    
    def _split_by_instances(self,test_ratio_of_all_data=0.2):
        train_dataset_df,test_dataset_df=sklearn.model_selection.train_test_split(self.df,train_size=1-test_ratio_of_all_data, test_size=test_ratio_of_all_data)
        return train_dataset_df,test_dataset_df
    
    def convert_to_BIO(self):
        self.df["title_tagged_str"],self.df["num_overlap"]=zip(*self.df.apply(lambda x:self._tagging_for_multirows(x["title_tagged"],x['title'],attr_type2idx=self.handle_attr2spe_attr_type,label_format="BIO"),axis=1))

    def _save_BIO_txt(self,train_dataset_df,test_dataset_df,output_path):
        train_data=train_dataset_df["title_tagged_str"].loc[pd.notnull(train_dataset_df["title_tagged_str"])].tolist()
        test_data=test_dataset_df["title_tagged_str"].loc[pd.notnull(test_dataset_df["title_tagged_str"])].tolist()
        print(f"train amount:{len(train_data)},test amount:{len(test_data)}")
        train_data="\n\n".join(train_data)
        test_data="\n\n".join(test_data)
        with open(os.path.join(output_path,"train.txt"),"w",encoding="utf-8") as f:
            f.write(train_data)
        with open(os.path.join(output_path,"test.txt"),"w",encoding="utf-8") as f:
            f.write(test_data)
        with open(os.path.join(output_path,"num_overlap.txt"),"w",encoding="utf-8") as f:
            f.write(str(train_dataset_df["num_overlap"].sum())+"#"+str(test_dataset_df["num_overlap"].sum()))

    def _filter_new_value(self,filter_data,type2test_value,no_new_words_attrtype):
        for indexed_text, attribute_type, start_idx, end_idx in filter_data:
            if attribute_type in no_new_words_attrtype:
                if random.random()<0.05:
                    return True
            elif indexed_text in type2test_value[attribute_type]:
                return True
        return False

    def _get_attr_type(self,df):
        """
        input format:
        [(attribute value, attribute type, source, score, start_idx, end_idx, line_idx),...]
        """
        attr_counter=set()
        def _get_attr_type(col):
            for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(col):
                attr_counter.add(attribute_type) 
        df["extraction_right_value_list_all"].map(_get_attr_type)
        return list(attr_counter)
    
    @staticmethod
    def statistic_attr_type2value2cnt(df):
        type2value_counter=collections.defaultdict(dict)
        for one in df['title_tagged']:
            for indexed_text, attribute_type, start_idx, end_idx in one:
                if indexed_text not in type2value_counter[attribute_type]:
                    type2value_counter[attribute_type][indexed_text]=1
                else:
                    type2value_counter[attribute_type][indexed_text]+=1
        return type2value_counter
    
    @staticmethod
    def get_attr_type2cnt(df):
        type2value_counter=collections.defaultdict(int)
        for one in df['title_tagged']:
            for _, attribute_type, _, _ in one:
                type2value_counter[attribute_type]+=1
        return type2value_counter
    
    
    def _replace_space_to_spe_token(self,attr_type,spe_token="##"):
        return attr_type.strip().replace(" ",spe_token)

    def _clean_text(self,description_text,title_text,taggings,handle_attr,line_break="\x01"):
        """
        text: text
        tagging:(attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx) lists.
        """

        # tagging all occurs
        # title_entity=set()
        # desp_entity=set()
        # title_pos2type={}
        # desp_pos2type={}
        # description_lists=description_text.split(line_break)
        # for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(taggings):
        #     if source=="title":
        #         title_entity.add((title_text[start_idx:end_idx],attribute_type))
        #     elif source=="description":
        #         desp_entity.add((description_lists[line_idx][start_idx:end_idx],attribute_type))
        # for title,attribute_type in title_entity:
        #     for p in re.finditer(r"\b(%s)\b"%(title),title_text):
        #         title_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
        # for desp,attribute_type in desp_entity:
        #     for p in re.finditer(r"\b(%s)\b"%(desp),title_text):
        #         desp_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
                
        title_pos2type={}
        desp_pos2type={}
        last_end_title=(-1,-1,"")#(line index,end index,attr_type)
        last_end_desp=(-1,-1,"")
        title_tagging=[]
        desp_tagging=[]
        try:
            description_text=ast.literal_eval(description_text.lower())
        except:
            if isinstance(description_text,str):
                description_text=description_text.split(line_break)
        # tagging only index
        for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in sorted(set(ast.literal_eval(taggings)),key=lambda x:(x[6],x[4],x[5])):
            if start_idx<0 or end_idx<0 or attribute_type not in handle_attr:
                continue
            if source=="title":
                if attribute_value!=title_text[start_idx:end_idx]:
                    error=True
                    tmp=title_text[start_idx:end_idx]
                    index_search=re.search(r"\b(%s)\b"%tmp,title_text)
                    attr_search=re.search(r"\b(%s)\b"%attribute_value,title_text)
                    # # if attribute_value in tmp:
                    # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
                    # #     error=False
                    # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
                    # #     error=False
                    if attr_search:#attr的文本可以全字匹配到
                        start_idx,end_idx=attr_search.span()
                        error=False
                    elif index_search:#attr的文本不可以全字匹配到，考虑用index的全字匹配文本
                        start_idx,end_idx=index_search.span()
                        error=False
                    if error:
                        print(f"Error attribute:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                        continue
                    
                indexed_text=title_text[start_idx:end_idx]
                if len(indexed_text)==0:
                    print(f"Empty content:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                    continue
                if last_end_title[0]<line_idx or (last_end_title[0]==line_idx and start_idx>= last_end_title[1]):
                    title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                else:# must be last_end_title[0]==line_idx
                    if last_end_title[2]==attribute_type:
                        if last_end_title[1]>=end_idx:# current belong to last
                            print(f"check same type but including position case, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]})")
                            last_end_title=(line_idx,last_end_title[1],attribute_type)
                            continue
                        else:
                            title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                            print(f"check same type but different position case, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]})")
                    else:
                        title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                        print(f"check different type overlap, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]}")
                    # raise Exception

                last_end_title=(line_idx,end_idx,attribute_type)
            # elif source=="description":
            #     if line_idx>= len(description_text):
            #         print(f"error line index:{str((attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx,description_text))}")
            #         continue
            #     tmp=description_text[line_idx][start_idx:end_idx]
                
            #     if attribute_value!=tmp:
            #         error=True
            #         index_search=re.search(r"\b(%s)\b"%tmp,description_text[line_idx])
            #         attr_search=re.search(r"\b(%s)\b"%attribute_value,description_text[line_idx])
                    
            #         # # if attribute_value in tmp:
            #         # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
            #         # #     error=False
            #         # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
            #         # #     error=False
            #         if attr_search:#index的文本可以全字匹配到
            #             start_idx,end_idx=attr_search.span()
            #             error=False
            #         elif index_search:#index的文本不可以全字匹配到，考虑用attr的全字匹配文本
            #             start_idx,end_idx=index_search.span()
            #             error=False

            #         if error:
            #             print(f"Error attribute:{attribute_value},index:{description_text[line_idx][start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},desp_text:{description_text}")
            #             continue
                    
            #     if last_end_desp[0]<line_idx or (last_end_desp[0]==line_idx and start_idx>= last_end_desp[1]):
            #         desp_pos2type[(line_idx,start_idx,end_idx)]=attribute_type
            #     else:
            #         print(f"check overlapping entity:{description_text[line_idx][start_idx:end_idx]},type:{attribute_type},pos:({line_idx},{start_idx},{end_idx})")
            #         # raise Exception
            #     last_end_desp=(line_idx,end_idx)
        # return title_tagged_str,desp_tagged_str
        return list(set(title_tagging))
    
    def _select_attr_until_threshold(self,statistic_df,df,test_ratio=0.2,select_threshold=40,column_name="title_tagged"):
        # less means 
        new_attrtype=statistic_df.sort_values(["count"],ascending=[1],inplace=False).groupby("attr_type").sum()
        no_new_words_attrtype=new_attrtype[new_attrtype["count"]<4*select_threshold].index.tolist()
        statistic_df=statistic_df.loc[~statistic_df["attr_type"].isin(no_new_words_attrtype)]

        num_instances=sum(df[column_name].apply(lambda x:len(x))!=0)
        all_attrs_num=statistic_df["count"].sum()
        # assert sum(df[column_name].apply(lambda x:len(x)))==statistic_df["count"].sum()
        attrs_per_instance=all_attrs_num/num_instances
        
        select_attrs_num=int(num_instances*test_ratio*attrs_per_instance)
        
        attr_group=statistic_df.sort_values(["count"],ascending=[1],inplace=False).groupby("attr_type")
        # statistic_df.groupby("attr_type").apply(lambda x: x.sort_values(["count"],ascending=[1],inplace=False))
        cur_attrs_num=attr_group["count"].head(1).sum()
        last_attrs_num=-1
        k=2
        
        # TODO:If I have spare time, change to binary search
        while cur_attrs_num<select_attrs_num and cur_attrs_num!=last_attrs_num:
            last_attrs_num=cur_attrs_num
            cur_attrs_num=attr_group["count"].head(k).loc[attr_group["count"].head(k)<select_threshold].sum()
            k+=1
        k-=1
        print(f"select top {k} attributes and less than {select_threshold},attr num:{cur_attrs_num}")
        # select_values
        all_value_occur_count=statistic_df.sort_values(["count"],ascending=[1],inplace=False).groupby("attr_type")['value_content'].apply(list).apply(len)
        test_value_occur_count=attr_group.head(k).loc[attr_group["count"].head(k)<select_threshold][["attr_type","value_content"]].groupby("attr_type")['value_content'].apply(list).apply(len)
        
        type2new_values=attr_group.head(k).loc[attr_group["count"].head(k)<select_threshold][["attr_type","value_content"]].groupby("attr_type")['value_content'].apply(list).to_dict()
        # type2new_value_count=attr_group.head(k).loc[attr_group["count"].head(k)<select_threshold]
        return type2new_values,no_new_words_attrtype,(all_value_occur_count,test_value_occur_count)


    def _tagging_for_multirows(self,label_lists,text,attr_type2idx,label_format="BIO"):
        
        """_summary_
        text_snippet = []
        # title not use line break
        # text_lists=texts.split(line_break)
        last_line=0
        last_end=0
        for line_idx,start_idx,end_idx in sorted(label_lists.keys(),key=lambda x:(x[0],x[1],x[2])):
            attribute_type=label_lists[(line_idx,start_idx,end_idx)]
            if line_idx>last_line:
                # handling the rest of the last line
                if last_end < len(text_lists[last_line]):
                    snippet1 = [[word, 'O'] for word in nltk.word_tokenize(text_lists[last_line][last_end:])]
                    text_snippet.extend(snippet1)
                last_line=line_idx
                last_end=0
            if line_idx<last_line:
                print("Error in multiple tagging!")
                raise Exception

            if line_idx==last_line:#attention:not elif
                snippet1 = [[word, 'O'] for word in nltk.word_tokenize(text_lists[line_idx][last_end:start_idx])]
                text_snippet.extend(snippet1)
                snippet2 = nltk.word_tokenize(text_lists[line_idx][start_idx:end_idx])#?难道有词不是空格分的？
                if label_format == "BIO":
                    if len(snippet2) == 1:
                        text_snippet.append([snippet2[0], "B-" + str(attr_type2idx[attribute_type])])
                    else:
                        for index, word in enumerate(snippet2):
                            if index == 0:
                                text_snippet.append([word, "B-" + str(attr_type2idx[attribute_type])])
                            else:
                                text_snippet.append([word, "I-" + str(attr_type2idx[attribute_type])])
                elif label_format == "BIOSE":
                    if len(snippet2) == 1:
                        text_snippet.append([snippet2[0], "S-" + str(attr_type2idx[attribute_type])])
                    else:
                        for index, word in enumerate(snippet2):
                            if index == 0:
                                text_snippet.append([word, "B-" + str(attr_type2idx[attribute_type])])
                            elif index == len(snippet2) - 1:
                                text_snippet.append([word, "E-" + str(attr_type2idx[attribute_type])])
                            else:
                                text_snippet.append([word, "I-" + str(attr_type2idx[attribute_type])])
                else:
                    raise "label format illegal,should use BIO or BIOSE"
                last_end=end_idx

        if last_end < len(text_lists[last_line]):
            snippet1 = [[word, 'O'] for word in nltk.word_tokenize(text_lists[last_line][last_end:])]
            text_snippet.extend(snippet1)
        if last_line< len(text_lists):
            for line_num in range(last_line,len(text_lists)):
                snippet1 = [[word, 'O'] for word in nltk.word_tokenize(text_lists[line_num])]
                text_snippet.extend(snippet1)
        return text_snippet
        """
        # labels list:(indexed_text, attribute_type, start_idx, end_idx)
        text_snippet = []
        # title not use line break
        # text_lists=texts.split(line_break)
        last_end=0
        overlap_counter=0
        
        for indexed_text, attribute_type, start_idx, end_idx in sorted(label_lists,key=lambda x:(x[1],x[2],-x[3])):
        
            if start_idx>= last_end:
                snippet1 = [[word, 'O'] for word in nltk.word_tokenize(text[last_end:start_idx])]
                text_snippet.extend(snippet1)
                snippet2 = nltk.word_tokenize(text[start_idx:end_idx])#?难道有词不是空格分的？
                if label_format == "BIO":
                    if len(snippet2) == 1:
                        text_snippet.append([snippet2[0], "B-" + str(attr_type2idx[attribute_type])])
                    else:
                        for index, word in enumerate(snippet2):
                            if index == 0:
                                text_snippet.append([word, "B-" + str(attr_type2idx[attribute_type])])
                            else:
                                text_snippet.append([word, "I-" + str(attr_type2idx[attribute_type])])
                elif label_format == "BIOSE":
                    if len(snippet2) == 1:
                        text_snippet.append([snippet2[0], "S-" + str(attr_type2idx[attribute_type])])
                    else:
                        for index, word in enumerate(snippet2):
                            if index == 0:
                                text_snippet.append([word, "B-" + str(attr_type2idx[attribute_type])])
                            elif index == len(snippet2) - 1:
                                text_snippet.append([word, "E-" + str(attr_type2idx[attribute_type])])
                            else:
                                text_snippet.append([word, "I-" + str(attr_type2idx[attribute_type])])
                else:
                    raise "label format illegal,should use BIO or BIOSE"
                last_end=end_idx
            else:
                overlap_counter+=1
                continue

        if last_end < len(text):
            snippet1 = [[word, 'O'] for word in nltk.word_tokenize(text[last_end:])]
            text_snippet.extend(snippet1)
        NER_BIO=[word+" "+tagging  for word,tagging in text_snippet]
        BIO_str="\n".join(NER_BIO)
        return BIO_str,overlap_counter


    

def get_attr_type(df):
    
    """
    input format:
    [(attribute value, attribute type, source, score, start_idx, end_idx, line_idx),...]
    """
    attr_counter=set()
    def _get_attr_type(col):
        for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(col):
            attr_counter.add(attribute_type) 
    df["extraction_right_value_list_all"].map(_get_attr_type)
    return list(attr_counter)

def get_value_dict_of_local_write(value):
    if value == np.nan:
        return value
    value_dict = {}
    wrong_value_list = re.split('[;；]', value)
    if len(wrong_value_list) > 0:
        for value in wrong_value_list:
            value_list = value.split(':')
            if len(value_list) == 2:
                new_value_list = re.split('\|\|\||,', value_list[1].strip().lower())
                new_value_list = [temp_value.strip() for temp_value in new_value_list]
                value_dict[value_list[0].strip().lower()] = new_value_list
    return value_dict

def repair_contain_case(attr_value,index_text,start_index):
    return (re.search(attr_value, index_text).span()[0]+start_index,re.search(attr_value, index_text).span()[1]+start_index)

def split_origin_desp(description_text,line_break="\x01"):
    try:
        description_text=ast.literal_eval(description_text.lower())
    except:
        if isinstance(description_text,str):
            description_text=description_text.split(line_break)
    return description_text

def text2tagged_text(description_text,title_text,taggings,attr_type2idx,handle_attr,type_counter,lowercase=True,tagging_way="BIO",line_break="\x01"):
    """
    text: text
    tagging:(attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx) lists.
    """

    # tagging all occurs
    # title_entity=set()
    # desp_entity=set()
    # title_pos2type={}
    # desp_pos2type={}
    # description_lists=description_text.split(line_break)
    # for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(taggings):
    #     if source=="title":
    #         title_entity.add((title_text[start_idx:end_idx],attribute_type))
    #     elif source=="description":
    #         desp_entity.add((description_lists[line_idx][start_idx:end_idx],attribute_type))
    # for title,attribute_type in title_entity:
    #     for p in re.finditer(r"\b(%s)\b"%(title),title_text):
    #         title_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
    # for desp,attribute_type in desp_entity:
    #     for p in re.finditer(r"\b(%s)\b"%(desp),title_text):
    #         desp_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
            
    title_pos2type={}
    desp_pos2type={}
    last_end_title=(-1,-1)#(line index,end index)
    last_end_desp=(-1,-1)
    try:
        description_text=ast.literal_eval(description_text.lower())
    except:
        if isinstance(description_text,str):
            description_text=description_text.split(line_break)
    # tagging only index
    for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in sorted(set(ast.literal_eval(taggings)),key=lambda x:(x[6],x[4],x[5])):
        if start_idx<0 or attribute_type not in handle_attr:
            continue
        if source=="title":
            if attribute_value!=title_text[start_idx:end_idx]:
                error=True
                tmp=title_text[start_idx:end_idx]
                index_search=re.search(r"\b(%s)\b"%tmp,title_text)
                attr_search=re.search(r"\b(%s)\b"%attribute_value,title_text)
                
                # # if attribute_value in tmp:
                # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
                # #     error=False
                # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
                # #     error=False
                if attr_search:#index的文本可以全字匹配到
                    start_idx,end_idx=attr_search.span()
                    error=False
                elif index_search:#index的文本不可以全字匹配到，考虑用attr的全字匹配文本
                    start_idx,end_idx=index_search.span()
                    error=False

                if error:
                    print(f"Error attribute:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                    continue
            if last_end_title[0]<line_idx or (last_end_title[0]==line_idx and start_idx>= last_end_title[1]):
                title_pos2type[(line_idx,start_idx,end_idx)]=attribute_type
            else:
                print(f"check overlapping entity:{title_text[start_idx:end_idx]},type:{attribute_type},pos:({line_idx},{start_idx},{end_idx})")
                # raise Exception
            last_end_title=(line_idx,end_idx)
        # elif source=="description":
        #     if line_idx>= len(description_text):
        #         print(f"error line index:{str((attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx,description_text))}")
        #         continue
        #     tmp=description_text[line_idx][start_idx:end_idx]
            
        #     if attribute_value!=tmp:
        #         error=True
        #         index_search=re.search(r"\b(%s)\b"%tmp,description_text[line_idx])
        #         attr_search=re.search(r"\b(%s)\b"%attribute_value,description_text[line_idx])
                
        #         # # if attribute_value in tmp:
        #         # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
        #         # #     error=False
        #         # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
        #         # #     error=False
        #         if attr_search:#index的文本可以全字匹配到
        #             start_idx,end_idx=attr_search.span()
        #             error=False
        #         elif index_search:#index的文本不可以全字匹配到，考虑用attr的全字匹配文本
        #             start_idx,end_idx=index_search.span()
        #             error=False

        #         if error:
        #             print(f"Error attribute:{attribute_value},index:{description_text[line_idx][start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},desp_text:{description_text}")
        #             continue
                
        #     if last_end_desp[0]<line_idx or (last_end_desp[0]==line_idx and start_idx>= last_end_desp[1]):
        #         desp_pos2type[(line_idx,start_idx,end_idx)]=attribute_type
        #     else:
        #         print(f"check overlapping entity:{description_text[line_idx][start_idx:end_idx]},type:{attribute_type},pos:({line_idx},{start_idx},{end_idx})")
        #         # raise Exception
        #     last_end_desp=(line_idx,end_idx)
        type_counter[source+"-"+attribute_type]+=1
    # if len(desp_pos2type):
    #     NER_data_desp=tagging_for_multirows(desp_pos2type,description_text,attr_type2idx,label_format="BIO",line_break="\x01")
    #     NER_data_desp=[word+" "+tagging  for word,tagging in NER_data_desp]
    #     desp_tagged_str="\n".join(NER_data_desp)
    # else:
    #     desp_tagged_str=None
    if len(title_pos2type):
        NER_data_title=tagging_for_multirows(title_pos2type,[title_text.lower()],attr_type2idx,label_format="BIO",line_break="\x01")
        NER_data_title=[word+" "+tagging  for word,tagging in NER_data_title]
        title_tagged_str="\n".join(NER_data_title)
    else:
        title_tagged_str=None

    # return title_tagged_str,desp_tagged_str
    return title_tagged_str



def get_attr_val(col):
    attr_counter={}
    """
    input format:
    [(attribute value, attribute type, source, score, start_idx, end_idx, line_idx),...]
    """
    for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(col):
        if attribute_type not in attr_counter:
            attr_counter[attribute_type]=[attribute_value]
        else:
            attr_counter[attribute_type].append(attribute_value)
    return attr_counter

def replace_space_to_spe_token(attr_type,spe_token="##"):
    return attr_type.strip().replace(" ",spe_token)

def convert_BIO_process(data_path,handle_attr=None):
    """
    input:data_path:csv file.
    ['Category ID', 'attr_type_value_dict', 'Local Confirm', 'Wrong Reason',
       'Inconsistent Attribute', 'item_id', 'shop_id', 'model_id',
       'norm_attribute_value', 'title', 'specification', 'description',
       'model_name', 'attribute_value_with_index',
       'extraction_right_value_list_all', 'integration_right_value_list_all',
       'extraction_error_value_list_all', 'integration_error_value_list_all',
       'norm_error_value_list_all', 'instance_source', 'type2value']
    output:standard format:('Category ID', 'item_id', 'shop_id', 'model_id',
        'title_training',  'description_training',) 
    """
    df=pd.read_csv(data_path)
    df=df[['Category ID', 'item_id', 'shop_id', 'model_id','title',  'description','extraction_right_value_list_all']]
    # Use extraction_right_value_list_all to generate training data for title and description
    attribute_types=get_attr_type(df)
    if handle_attr is None:
        handle_attr=attribute_types
    # handle_attr2idx={t:i for i,t in enumerate(handle_attr)}
    handle_attr2spe_attr_type={t:replace_space_to_spe_token(t) for t in handle_attr}
    
    # title_tagged_str,desp_tagged_str=df.apply(lambda x:text2tagged_text(description_text=x['description'],title_text=x['title'],taggings=x['extraction_right_value_list_all'],attr_type2idx=attr_type2idx,lowercase=True,tagging_way="BIO",line_break="\x01"),axis=1)
    # df["title_tagged_str"],df["desp_tagged_str"]=title_tagged_str,desp_tagged_str
    type_counter=collections.defaultdict(int)
    title_tagged_str=df.apply(lambda x:text2tagged_text(description_text=x['description'],title_text=x['title'],taggings=x['extraction_right_value_list_all'],type_counter=type_counter,attr_type2idx=handle_attr2spe_attr_type,lowercase=True,tagging_way="BIO",line_break="\x01",handle_attr=handle_attr),axis=1)
    df["title_tagged_str"]=title_tagged_str
    
    # split dataset
    test_ratio_of_all_data=0.2
    train_dataset_df,test_dataset_df=sklearn.model_selection.train_test_split(df,train_size=1-test_ratio_of_all_data, test_size=test_ratio_of_all_data)
    train_data=train_dataset_df["title_tagged_str"].loc[pd.notnull(train_dataset_df["title_tagged_str"])].tolist()
    test_data=test_dataset_df["title_tagged_str"].loc[pd.notnull(test_dataset_df["title_tagged_str"])].tolist()
    print(f"train amount:{len(train_data)},test amount:{len(test_data)},type_count:{type_counter}")
    train_data="\n\n".join(train_data)
    test_data="\n\n".join(test_data)
    
    output_path="./output/single-value"
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path,"train.txt"),"w",encoding="utf-8") as f:
        f.write(train_data)
    with open(os.path.join(output_path,"test.txt"),"w",encoding="utf-8") as f:
        f.write(test_data)
    with open(os.path.join(output_path,"key_attr.pkl"),"wb") as f:
        pickle.dump(key_attr,f) 
    with open(os.path.join(output_path,"handle_attr2spe_attr_type.pkl"),"wb") as f:
        pickle.dump(handle_attr2spe_attr_type,f)
    # return attr_type2idx

def text2indexed_tag(description_text,title_text,taggings,attr_type2idx,handle_attr,type_counter,lowercase=True,tagging_way="BIO",line_break="\x01"):
    """
    text: text
    tagging:(attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx) lists.
    """

    # tagging all occurs
    # title_entity=set()
    # desp_entity=set()
    # title_pos2type={}
    # desp_pos2type={}
    # description_lists=description_text.split(line_break)
    # for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(taggings):
    #     if source=="title":
    #         title_entity.add((title_text[start_idx:end_idx],attribute_type))
    #     elif source=="description":
    #         desp_entity.add((description_lists[line_idx][start_idx:end_idx],attribute_type))
    # for title,attribute_type in title_entity:
    #     for p in re.finditer(r"\b(%s)\b"%(title),title_text):
    #         title_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
    # for desp,attribute_type in desp_entity:
    #     for p in re.finditer(r"\b(%s)\b"%(desp),title_text):
    #         desp_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
            
    title_pos2type={}
    desp_pos2type={}
    last_end_title=(-1,-1,"")#(line index,end index,attr_type)
    last_end_desp=(-1,-1,"")
    title_tagging=[]
    desp_tagging=[]
    try:
        description_text=ast.literal_eval(description_text.lower())
    except:
        if isinstance(description_text,str):
            description_text=description_text.split(line_break)
    # tagging only index
    for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in sorted(set(ast.literal_eval(taggings)),key=lambda x:(x[6],x[4],x[5])):
        if start_idx<0 or end_idx<0 or attribute_type not in handle_attr:
            continue
        if source=="title":
            if attribute_value!=title_text[start_idx:end_idx]:
                error=True
                tmp=title_text[start_idx:end_idx]
                index_search=re.search(r"\b(%s)\b"%tmp,title_text)
                attr_search=re.search(r"\b(%s)\b"%attribute_value,title_text)
                
                # # if attribute_value in tmp:
                # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
                # #     error=False
                # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
                # #     error=False
                if attr_search:#index的文本可以全字匹配到
                    start_idx,end_idx=attr_search.span()
                    error=False
                elif index_search:#index的文本不可以全字匹配到，考虑用attr的全字匹配文本
                    start_idx,end_idx=index_search.span()
                    error=False

                if error:
                    print(f"Error attribute:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                    continue


            indexed_text=title_text[start_idx:end_idx]
            if len(indexed_text)==0:
                print(f"Empty content:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                continue
            if last_end_title[0]<line_idx or (last_end_title[0]==line_idx and start_idx>= last_end_title[1]):
                title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
            else:# must be last_end_title[0]==line_idx
                if last_end_title[2]==attribute_type:
                    if last_end_title[1]>=end_idx:# current belong to last
                        print(f"check same type but including position case, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]})")
                        continue
                    else:
                        title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                        print(f"check same type but different position case, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]})")
                else:
                    title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                    print(f"check different type, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]}")
                # raise Exception

            last_end_title=(line_idx,end_idx,attribute_type)
        # elif source=="description":
        #     if line_idx>= len(description_text):
        #         print(f"error line index:{str((attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx,description_text))}")
        #         continue
        #     tmp=description_text[line_idx][start_idx:end_idx]
            
        #     if attribute_value!=tmp:
        #         error=True
        #         index_search=re.search(r"\b(%s)\b"%tmp,description_text[line_idx])
        #         attr_search=re.search(r"\b(%s)\b"%attribute_value,description_text[line_idx])
                
        #         # # if attribute_value in tmp:
        #         # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
        #         # #     error=False
        #         # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
        #         # #     error=False
        #         if attr_search:#index的文本可以全字匹配到
        #             start_idx,end_idx=attr_search.span()
        #             error=False
        #         elif index_search:#index的文本不可以全字匹配到，考虑用attr的全字匹配文本
        #             start_idx,end_idx=index_search.span()
        #             error=False

        #         if error:
        #             print(f"Error attribute:{attribute_value},index:{description_text[line_idx][start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},desp_text:{description_text}")
        #             continue
                
        #     if last_end_desp[0]<line_idx or (last_end_desp[0]==line_idx and start_idx>= last_end_desp[1]):
        #         desp_pos2type[(line_idx,start_idx,end_idx)]=attribute_type
        #     else:
        #         print(f"check overlapping entity:{description_text[line_idx][start_idx:end_idx]},type:{attribute_type},pos:({line_idx},{start_idx},{end_idx})")
        #         # raise Exception
        #     last_end_desp=(line_idx,end_idx)
        type_counter[source+"-"+attribute_type]+=1
    # if len(desp_pos2type):
    #     NER_data_desp=tagging_for_multirows(desp_pos2type,description_text,attr_type2idx,label_format="BIO",line_break="\x01")
    #     NER_data_desp=[word+" "+tagging  for word,tagging in NER_data_desp]
    #     desp_tagged_str="\n".join(NER_data_desp)
    # else:
    #     desp_tagged_str=None
    # if len(title_pos2type):
    #     NER_data_title=tagging_for_multirows(title_pos2type,[title_text.lower()],attr_type2idx,label_format="BIO",line_break="\x01")
    #     NER_data_title=[word+" "+tagging  for word,tagging in NER_data_title]
    #     title_tagged_str="\n".join(NER_data_title)
    # else:
    #     title_tagged_str=None
    # return title_tagged_str,desp_tagged_str
    return title_tagging

def get_type2value_counter(description_text,title_text,taggings,attr_type2idx,handle_attr,type2value_counter,line_break="\x01"):
    """
    text: text
    tagging:(attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx) lists.
    """

    # tagging all occurs
    # title_entity=set()
    # desp_entity=set()
    # title_pos2type={}
    # desp_pos2type={}
    # description_lists=description_text.split(line_break)
    # for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in ast.literal_eval(taggings):
    #     if source=="title":
    #         title_entity.add((title_text[start_idx:end_idx],attribute_type))
    #     elif source=="description":
    #         desp_entity.add((description_lists[line_idx][start_idx:end_idx],attribute_type))
    # for title,attribute_type in title_entity:
    #     for p in re.finditer(r"\b(%s)\b"%(title),title_text):
    #         title_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
    # for desp,attribute_type in desp_entity:
    #     for p in re.finditer(r"\b(%s)\b"%(desp),title_text):
    #         desp_pos2type[(0,p.span()[0],p.span()[1])]=attribute_type
            
    title_pos2type={}
    desp_pos2type={}
    last_end_title=(-1,-1,"")#(line index,end index,attr_type)
    last_end_desp=(-1,-1,"")
    title_tagging=[]
    desp_tagging=[]
    try:
        description_text=ast.literal_eval(description_text.lower())
    except:
        if isinstance(description_text,str):
            description_text=description_text.split(line_break)
    # tagging only index
    for attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx in sorted(set(ast.literal_eval(taggings)),key=lambda x:(x[6],x[4],x[5])):
        if start_idx<0 or end_idx<0 or attribute_type not in handle_attr:
            continue
        if source=="title":
            if attribute_value!=title_text[start_idx:end_idx]:
                error=True
                tmp=title_text[start_idx:end_idx]
                index_search=re.search(r"\b(%s)\b"%tmp,title_text)
                attr_search=re.search(r"\b(%s)\b"%attribute_value,title_text)
                # # if attribute_value in tmp:
                # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
                # #     error=False
                # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
                # #     error=False
                if attr_search:#attr的文本可以全字匹配到
                    start_idx,end_idx=attr_search.span()
                    error=False
                elif index_search:#attr的文本不可以全字匹配到，考虑用index的全字匹配文本
                    start_idx,end_idx=index_search.span()
                    error=False
                if error:
                    print(f"Error attribute:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                    continue
                
            indexed_text=title_text[start_idx:end_idx]
            if len(indexed_text)==0:
                print(f"Empty content:{attribute_value},index:{title_text[start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},title_text:{title_text}")
                continue
            if last_end_title[0]<line_idx or (last_end_title[0]==line_idx and start_idx>= last_end_title[1]):
                title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
            else:# must be last_end_title[0]==line_idx
                if last_end_title[2]==attribute_type:
                    if last_end_title[1]>=end_idx:# current belong to last
                        print(f"check same type but including position case, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]})")
                        continue
                    else:
                        title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                        print(f"check same type but different position case, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]})")
                else:
                    title_tagging.append((indexed_text, attribute_type, start_idx, end_idx))
                    print(f"check different type, entity:{indexed_text},type:{attribute_type},pos:({start_idx},{end_idx}),last_type:{last_end_title[2]},last_end:{last_end_title[1]}")
                # raise Exception
            if indexed_text not in type2value_counter[attribute_type]:
                type2value_counter[attribute_type][indexed_text]=1
            else:
                type2value_counter[attribute_type][indexed_text]+=1
            last_end_title=(line_idx,end_idx,attribute_type)
        # elif source=="description":
        #     if line_idx>= len(description_text):
        #         print(f"error line index:{str((attribute_value, attribute_type, source, score, start_idx, end_idx, line_idx,description_text))}")
        #         continue
        #     tmp=description_text[line_idx][start_idx:end_idx]
            
        #     if attribute_value!=tmp:
        #         error=True
        #         index_search=re.search(r"\b(%s)\b"%tmp,description_text[line_idx])
        #         attr_search=re.search(r"\b(%s)\b"%attribute_value,description_text[line_idx])
                
        #         # # if attribute_value in tmp:
        #         # #     start_idx,end_idx=repair_contain_case(attribute_value,tmp,start_idx)
        #         # #     error=False
        #         # # elif tmp in attribute_value or (set(attribute_value.split(" "))==set(tmp.split(" "))) or attribute_value.replace(" ","")==tmp.replace(" ",""):
        #         # #     error=False
        #         if attr_search:#index的文本可以全字匹配到
        #             start_idx,end_idx=attr_search.span()
        #             error=False
        #         elif index_search:#index的文本不可以全字匹配到，考虑用attr的全字匹配文本
        #             start_idx,end_idx=index_search.span()
        #             error=False

        #         if error:
        #             print(f"Error attribute:{attribute_value},index:{description_text[line_idx][start_idx:end_idx]},position:{(start_idx,end_idx)},attribute type:{attribute_type},desp_text:{description_text}")
        #             continue
                
        #     if last_end_desp[0]<line_idx or (last_end_desp[0]==line_idx and start_idx>= last_end_desp[1]):
        #         desp_pos2type[(line_idx,start_idx,end_idx)]=attribute_type
        #     else:
        #         print(f"check overlapping entity:{description_text[line_idx][start_idx:end_idx]},type:{attribute_type},pos:({line_idx},{start_idx},{end_idx})")
        #         # raise Exception
        #     last_end_desp=(line_idx,end_idx)
    # return title_tagged_str,desp_tagged_str
    return title_tagging


def convert_multivalue_process(data_path,handle_attr=None,convert_to_BIO=True):
    """
    input:data_path:csv file.
    ['Category ID', 'attr_type_value_dict', 'Local Confirm', 'Wrong Reason',
       'Inconsistent Attribute', 'item_id', 'shop_id', 'model_id',
       'norm_attribute_value', 'title', 'specification', 'description',
       'model_name', 'attribute_value_with_index',
       'extraction_right_value_list_all', 'integration_right_value_list_all',
       'extraction_error_value_list_all', 'integration_error_value_list_all',
       'norm_error_value_list_all', 'instance_source', 'type2value']
    output:standard format:('Category ID', 'item_id', 'shop_id', 'model_id',
        'title_training',  'description_training',) 
    """
    df=pd.read_csv(data_path)
    df=df[['Category ID', 'item_id', 'shop_id', 'model_id','title',  'description','extraction_right_value_list_all']]
    # Use extraction_right_value_list_all to generate training data for title and description
    attribute_types=get_attr_type(df)
    if handle_attr is None:
        handle_attr=attribute_types
    # handle_attr2idx={t:i for i,t in enumerate(handle_attr)}
    handle_attr2spe_attr_type={t:replace_space_to_spe_token(t) for t in handle_attr}
    
    type_counter=collections.defaultdict(int)
    title_tagged_str=df.apply(lambda x:text2indexed_tag(description_text=x['description'],title_text=x['title'],taggings=x['extraction_right_value_list_all'],type_counter=type_counter,attr_type2idx=handle_attr2spe_attr_type,lowercase=True,tagging_way="BIO",line_break="\x01",handle_attr=handle_attr),axis=1)
    df["title_tagged"]=title_tagged_str
    
    
    path = os.getcwd()
    output_path=os.path.join(path,"output/Normal-value")
    os.makedirs(output_path, exist_ok=True)
    
    if convert_to_BIO:
        df["title_tagged_str"],df["num_overlap"]=zip(*df.apply(lambda x:tagging_for_multirows(x["title_tagged"],x['title'],attr_type2idx=handle_attr2spe_attr_type,label_format="BIO"),axis=1))


    df= df.loc[df.apply(lambda x:len(x['title_tagged'])!=0,axis=1)]#去除空值
    # split dataset
    test_ratio_of_all_data=0.2
    train_dataset_df,test_dataset_df=sklearn.model_selection.train_test_split(df,train_size=1-test_ratio_of_all_data, test_size=test_ratio_of_all_data)
    
    if convert_to_BIO:
        train_data=train_dataset_df["title_tagged_str"].loc[pd.notnull(train_dataset_df["title_tagged_str"])].tolist()
        test_data=test_dataset_df["title_tagged_str"].loc[pd.notnull(test_dataset_df["title_tagged_str"])].tolist()
        print(f"train amount:{len(train_data)},test amount:{len(test_data)}")
        train_data="\n\n".join(train_data)
        test_data="\n\n".join(test_data)
        with open(os.path.join(output_path,"train.txt"),"w",encoding="utf-8") as f:
            f.write(train_data)
        with open(os.path.join(output_path,"test.txt"),"w",encoding="utf-8") as f:
            f.write(test_data)
        with open(os.path.join(output_path,"num_overlap.txt"),"w",encoding="utf-8") as f:
            f.write(str(train_dataset_df["num_overlap"].sum())+"#"+str(test_dataset_df["num_overlap"].sum()))
        
    train_data=train_dataset_df
    test_data=test_dataset_df
    assert train_data.apply(lambda x:len(x['title_tagged'])==0,axis=1).sum()==0
    assert test_data.apply(lambda x:len(x['title_tagged'])==0,axis=1).sum()==0
    print(f"train amount:{len(train_data)},test amount:{len(test_data)}")

    
    train_data.to_csv(os.path.join(output_path,"train.csv"))
    test_data.to_csv(os.path.join(output_path,"test.csv"))
    with open(os.path.join(output_path,"key_attr.pkl"),"wb") as f:
        pickle.dump(key_attr,f) 
    with open(os.path.join(output_path,"handle_attr2spe_attr_type.pkl"),"wb") as f:
        pickle.dump(handle_attr2spe_attr_type,f)

def select_attr_until_threshold(statistic_df,df,test_ratio=0.2,column_name="title_tagged"):
    select_threshold=50# less means 
    new_attrtype=statistic_df.sort_values(["count"],ascending=[1],inplace=False).groupby("attr_type").sum()
    no_new_words_attrtype=new_attrtype[new_attrtype["count"]<4*select_threshold].index.tolist()
    statistic_df=statistic_df.loc[~statistic_df["attr_type"].isin(no_new_words_attrtype)]

    num_instances=sum(df[column_name].apply(lambda x:len(x))!=0)
    all_attrs_num=statistic_df["count"].sum()
    # assert sum(df[column_name].apply(lambda x:len(x)))==statistic_df["count"].sum()
    attrs_per_instance=all_attrs_num/num_instances
    
    select_attrs_num=int(num_instances*test_ratio*attrs_per_instance)
    
    attr_group=statistic_df.sort_values(["count"],ascending=[1],inplace=False).groupby("attr_type")
    # statistic_df.groupby("attr_type").apply(lambda x: x.sort_values(["count"],ascending=[1],inplace=False))
    cur_attrs_num=attr_group["count"].head(1).sum()
    last_attrs_num=-1
    k=2
    
    # TODO:If I have spare time, change to binary search
    while cur_attrs_num<select_attrs_num and cur_attrs_num!=last_attrs_num:
        last_attrs_num=cur_attrs_num
        cur_attrs_num=attr_group["count"].head(k).loc[attr_group["count"].head(k)<select_threshold].sum()
        k+=1
    k-=1
    print(f"select top {k} attributes and less than {select_threshold},attr num:{cur_attrs_num}")
    # select_values
    all_value_occur_count=statistic_df.sort_values(["count"],ascending=[1],inplace=False).groupby("attr_type")['value_content'].apply(list).apply(len)
    test_value_occur_count=attr_group.head(k).loc[attr_group["count"].head(k)<select_threshold][["attr_type","value_content"]].groupby("attr_type")['value_content'].apply(list).apply(len)
    
    type2new_values=attr_group.head(k).loc[attr_group["count"].head(k)<select_threshold][["attr_type","value_content"]].groupby("attr_type")['value_content'].apply(list).to_dict()
    # type2new_value_count=attr_group.head(k).loc[attr_group["count"].head(k)<select_threshold]
    return type2new_values,no_new_words_attrtype,(all_value_occur_count,test_value_occur_count)
    


def sample_no_occur_data(data_path,handle_attr=None,convert_to_BIO=True):
    # get every attr2value_count, then sample the least value in each attribute, when every attributes are sampled at least one value, check the amount of training data. 
    """
    input:data_path:csv file.
    ['Category ID', 'attr_type_value_dict', 'Local Confirm', 'Wrong Reason',
       'Inconsistent Attribute', 'item_id', 'shop_id', 'model_id',
       'norm_attribute_value', 'title', 'specification', 'description',
       'model_name', 'attribute_value_with_index',
       'extraction_right_value_list_all', 'integration_right_value_list_all',
       'extraction_error_value_list_all', 'integration_error_value_list_all',
       'norm_error_value_list_all', 'instance_source', 'type2value']
    output:standard format:('Category ID', 'item_id', 'shop_id', 'model_id',
        'title_training',  'description_training',) 
    """
    #读取和清理，生成(indexed_text, attribute_type, start_idx, end_idx)集合
    df=pd.read_csv(data_path)
    df=df[['Category ID', 'item_id', 'shop_id', 'model_id','title',  'description','extraction_right_value_list_all']]
    # Use extraction_right_value_list_all to generate training data for title and description
    attribute_types=get_attr_type(df)
    if handle_attr is None:
        handle_attr=attribute_types
    # handle_attr2idx={t:i for i,t in enumerate(handle_attr)}
    handle_attr2spe_attr_type={t:replace_space_to_spe_token(t) for t in handle_attr}
    
    type2value_counter=collections.defaultdict(dict)
    title_tagged=df.apply(lambda x:get_type2value_counter(description_text=x['description'],title_text=x['title'],taggings=x['extraction_right_value_list_all'],type2value_counter=type2value_counter,attr_type2idx=handle_attr2spe_attr_type,line_break="\x01",handle_attr=handle_attr),axis=1)
    df["title_tagged"]=title_tagged
    # To dataframe
    dict2dataframe=collections.defaultdict(list)
    for attr_type in type2value_counter.keys():
        for value_content,cnt in type2value_counter[attr_type].items():
            dict2dataframe["attr_type"].append(attr_type)
            dict2dataframe["value_content"].append(value_content)
            dict2dataframe["count"].append(cnt)
    statistic_df=pd.DataFrame.from_dict(dict2dataframe)#"attr_type","value_content","count"
    type2test_value,no_new_words_attrtype,(all_value_occur_count,test_value_occur_count)=select_attr_until_threshold(statistic_df,df,test_ratio=0.2,column_name="title_tagged")
    # split dataset 
    df["is_test"]=df.apply(lambda x:filter_new_value(x["title_tagged"],type2test_value,no_new_words_attrtype),axis=1)
    
    
    path = os.getcwd()
    output_path=os.path.join(path,"output/new-value")
    os.makedirs(output_path, exist_ok=True)
    
    
    if convert_to_BIO:
        df["title_tagged_str"],df["num_overlap"]=zip(*df.apply(lambda x:tagging_for_multirows(x["title_tagged"],x['title'],attr_type2idx=handle_attr2spe_attr_type,label_format="BIO"),axis=1))


    df= df.loc[df.apply(lambda x:len(x['title_tagged'])!=0,axis=1)]#去除空值
    train_dataset_df,test_dataset_df=df.loc[df["is_test"]==False ],df.loc[df["is_test"]==True]
    
    if convert_to_BIO:
        train_data=train_dataset_df["title_tagged_str"].loc[pd.notnull(train_dataset_df["title_tagged_str"])].tolist()
        test_data=test_dataset_df["title_tagged_str"].loc[pd.notnull(test_dataset_df["title_tagged_str"])].tolist()
        print(f"train amount:{len(train_data)},test amount:{len(test_data)}")
        train_data="\n\n".join(train_data)
        test_data="\n\n".join(test_data)
        with open(os.path.join(output_path,"train.txt"),"w",encoding="utf-8") as f:
            f.write(train_data)
        with open(os.path.join(output_path,"test.txt"),"w",encoding="utf-8") as f:
            f.write(test_data)
        with open(os.path.join(output_path,"num_overlap.txt"),"w",encoding="utf-8") as f:
            f.write(str(train_dataset_df["num_overlap"].sum())+"#"+str(test_dataset_df["num_overlap"].sum()))
        
    train_data=train_dataset_df
    test_data=test_dataset_df
    assert train_data.apply(lambda x:len(x['title_tagged'])==0,axis=1).sum()==0
    assert test_data.apply(lambda x:len(x['title_tagged'])==0,axis=1).sum()==0
    print(f"train amount:{len(train_data)},test amount:{len(test_data)},no_new_words_attrtype:{no_new_words_attrtype}")
    print(f"all value occur count:{all_value_occur_count},test value occur count:{test_value_occur_count}")
    
    
    train_data.to_csv(os.path.join(output_path,"train.csv"))
    test_data.to_csv(os.path.join(output_path,"test.csv"))

    with open(os.path.join(output_path,"key_attr.pkl"),"wb") as f:
        pickle.dump(key_attr,f) 
    with open(os.path.join(output_path,"handle_attr2spe_attr_type.pkl"),"wb") as f:
        pickle.dump(handle_attr2spe_attr_type,f)

if __name__=="__main__":
    # all data get its attrtype2idx
    
    # sample ratio as dev,test data
    
    # just training data to BIO
    df1=pd.read_csv("/ldap_home/charles.hu/git/NER/DataContruction/test_attr.csv")
    key_attr=df1["Attribute Type"].unique().tolist()
    df2=pd.read_csv("/ldap_home/charles.hu/git/NER/DataContruction/hard_attr.csv")
    hard_attr=df2["Attribute Type"].unique().tolist()
    # convert_BIO_process("/ldap_home/charles.hu/git/NER/DataContruction/test.csv",handle_attr=key_attr)
    # convert_multivalue_process("/ldap_home/charles.hu/git/NER/DataContruction/test.csv",handle_attr=hard_attr)
    # sample_no_occur_data("/ldap_home/charles.hu/git/NER/DataContruction/hard.csv",handle_attr=hard_attr)
    # print(df)
    QCClean(qc_data_path="/ldap_home/charles.hu/git/NER/DataContruction/test.csv",handle_attr=key_attr,convert_to_BIO=True,split_way=["random"])
    QCClean(qc_data_path="/ldap_home/charles.hu/git/NER/DataContruction/hard.csv",handle_attr=hard_attr,convert_to_BIO=True,split_way=["new-value"])