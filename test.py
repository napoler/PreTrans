
#encoding=utf-8
from __future__ import unicode_literals
import sys
import os 
import tkitJson
import tqdm
# 切换到上级目录
sys.path.append("../")
# 引入本地库
from PreTrans import PreTrans

from transformers import BertTokenizer
# /kaggle/input/mcbert/mc_bert_base
tokenizer = BertTokenizer.from_pretrained('/home/terry/dev/model/base')

# teacher_model= AutoModel.from_pretrained('/home/terry/dev/model/base')
file="../data"






# 
# 


g = os.walk(file)  
fileList=[]
for path,dir_list,file_list in g:  
    for file_name in file_list:  
#         print(os.path.join(path, file_name) )
        fileList.append(os.path.join(path, file_name))

    

fileList=fileList[:5]
# fileList



P=PreTrans(tokenizer,max_length=512) 
# lw=P.autoCut("借助python 脚本，可以轻松实现，原理就是：字符串的按照固定长度拆分这个模块提供了正则表达式匹配操作，正则表达式是一个特殊的字符序列，它能帮助你方便的检查一个字符串是否与某种模式匹配。")
# lw[:2]



for fileName in tqdm.tqdm(fileList):
    Tjson=tkitJson.Json(fileName)
    for it in Tjson.auto_load():
        P.autoCut(it['text'])
    #每个文件保存一次,便于获取
    P.getTok()
    P.save()



for it in P.load():
    print(it["data"]["input_ids"].size())

