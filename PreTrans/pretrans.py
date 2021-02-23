# -*- coding: utf-8 -*-

import tqdm
import re
import pickle


# data=[]
# target=[]
class PreTrans:
    """预处理文本,方便bert类的自然语言使用"""
    def __init__(self,tokenizer,max_length=512,slide_len=2,slide=True):
        self.LiWords=[]
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.slide_len=slide_len
        self.slide=slide
        
        pass
    def autoCut(self,text):
        """自动分割文本,支持划窗模式 支持自动累加
        自动累加
        返回切割后的文本列表数组
        """
        words=self.tokenizer.tokenize(text)
        seq_len=self.max_length
        #         print(words)
        if self.slide_len>seq_len:
            return []
        if self.slide:
            wordList=[words[i:i+seq_len] for i in range(0, len(words), seq_len-self.slide_len)]
            

            for i,it in enumerate(wordList):
                if i>0:
                    self.LiWords.append(wordList[i-1][-self.slide_len:]+it)
                else:
                    self.LiWords.append(it)
    #             print(i,it)
            return self.LiWords
            pass
        else:
            #处理非划窗
            self.LiWords= self.LiWords+[words[i:i+seq_len] for i in range(0, len(words), seq_len)]
            pass
        return self.LiWords
    def getList(self):
        """获取合并后的"""
        data=[]
        for it in self.LiWords:
            data.append(" ".join(it))
        
        return data
    def getTok(self,return_tensors="pt", padding="max_length",truncation=True):
        """获取处理后的数据 """
        data=self.getList()
        self.tokdatas=self.tokenizer(data, return_tensors=return_tensors, padding=padding,max_length=self.max_length,truncation=truncation)
        return self.tokdatas
    
    def save(self,data=None,path="data.plk"):
        """保存data,可选择self.tokdatas,self.LiWords 支持增量
        可以方便合并群组输入 lables
        data={
        "data":[],
        "lables":lables
        
        }
        """
        df2=open(path,'ab')# 注意一定要写明是wb 而不是w.
        # df2=open(path,'wb')# 注意一定要写明是wb 而不是w.
        if data==None:
            pickle.dump({"data":self.tokdatas},df2)
        else:
            pickle.dump(data,df2)
        df2.close()
    def load(self,path="data.plk"):
        """[summary]
        自动加载数据
        每次保存数据会逐一返回yield

        Args:
            path (str, optional): [description]. Defaults to "data.plk".

        Yields:
            [type]: [description]
        """
        
        with open(path,'rb') as f:
            while True:
                try:
                    yield pickle.load(f)

                except EOFError:
                    break
#         #读取文件中的内容。注意和通常读取数据的区别之处
#         df=open(path,'rb')#注意此处是rb
#         #此处使用的是load(目标文件)
#         data3=pickle.load(df)
#         df.close()
#         return data3
# P=PreTrans(tokenizer,max_length=20) 
# lw=P.autoCut("借助python 脚本，可以轻松实现，原理就是：字符串的按照固定长度拆分这个模块提供了正则表达式匹配操作，正则表达式是一个特殊的字符序列，它能帮助你方便的检查一个字符串是否与某种模式匹配。")
# lw[:2]

# for fileName in fileList:

#     Tjson=tkitJson.Json(fileName)
#     for it in tqdm.tqdm(Tjson.auto_load()):
# #         print(it['text'][:500])
#         data.append(it['text'][:500])
#         data_ids = tokenizer(data, return_tensors="pt", padding="max_length",max_length=seq_len,truncation=True)["input_ids"]
# #     target.append(it[1])