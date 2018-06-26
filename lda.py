import pandas as pd
import numpy as np
filename1='C:\\Users\\guojia\\Desktop\\comment.csv'
filename2='C:\\Users\\guojia\\Desktop\\comment_1.csv'
data1=pd.read_csv(filename1,encoding='gbk',header=None)
data2=pd.read_csv(filename2,encoding='gbk',header=None)
data=pd.concat([data1,data2])
data=data.drop_duplicates()#删除重复项
data.dropna()#删除空值
import re
data_1=re.compile(r'[\u4e00-\u9fa5]+')#正则表达式匹配汉字，删除评论中的各种符号
data[0]=data[0].map(lambda x:''.join(re.findall(data_1,x)))
data_2=[]
for i in data[0]:
    line=i.replace('举报楼打赏回复评论赞','').replace('举报','').replace('评论','').replace('客户端','').replace('来自天涯社区','').replace('楼主','').replace('回复','').replace('作者','')
    data_2.append(line)
data_2=pd.DataFrame(data_2)
import jieba
mycut=lambda s:' '.join(jieba.cut(s,cut_all = False))
data3=data_2[0].apply(mycut)
outputfile='C:\\Users\\guojia\\Desktop\\output.txt'
data3.to_csv(outputfile,index=False,header=False,encoding='utf-8')
negfile='C:\\Users\\guojia\\Desktop\\output.txt'
stoplist='G:\\python_work\\stoplist.txt'
data_n=pd.read_csv(negfile,encoding='utf-8',header=None)
stop=pd.read_csv(stoplist,encoding='utf-8',header=None,sep='tipdm')
data_n[1]=data_n[0].apply(lambda s:s.split(' '))
data_n[2]=data_n[1].apply(lambda y: [i for i in y if i not in stop])
from gensim import corpora,models
dict_1=corpora.Dictionary(data_n[2])
dict_corpora=[dict_1.doc2bow(i) for i in data_n[2]]
data_lda=models.LdaModel(dict_corpora,num_topics=5,id2word=dict_1)
for j in range(5):
    print(data_lda.print_topic(j))










