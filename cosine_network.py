#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as  pd
import sys
from time import clock
import time
print "开始。。。%s"%str(time.asctime(time.localtime(time.time())))
start_time=time.time()
question_data=pd.read_table('/usr/python/toutiao/question_info.txt',sep='\t',names=['qid','qtype','qstr1','qstr2','like_num','ans_num','good_ans_num'])
print "数据导入完成"
sys.stdout.flush()
qid_map=dict()#问题id与行号相对应
for i in range(len(question_data)):
	q=question_data.ix[i,'qid']
	if not qid_map.has_key(q):
		qid_map[q]=i
print "问题id和行号、用户id和行号对应完成，分别存在qid_map和uid_map中"
sys.stdout.flush()
question_set=set()
user_set=set()
for i in range(len(question_data)):
	line=question_data.ix[i,'qstr1']
	words=line.split('/')
	question_set.update(words)
finish_time=time.time()
print "问题词语采集完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
question_word_list=[i for i in question_set]
print "问题词语长度"+str(len(question_set))
sys.stdout.flush()
question_feature_list=[]

for i in range(len(question_data)):
	line=question_data.ix[i,'qstr1']
	words=line.split('/')
	each_feature=[]
	for j in range(len(question_set)):
		num=words.count(question_word_list[j])
		each_feature.append(num)
	question_feature_list.append(each_feature)
finish_time=time.time()
print "问题向量构建完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()

question_word_list=[]
len_question_set=len(question_set)
question_set=set()
q_split_point=int(0.5*len(question_data))
question_feature_train=question_feature_list[:q_split_point]
question_feature_test=question_feature_list[q_split_point:]
print "数据集划分完成"
sys.stdout.flush()
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer,SigmoidLayer
dsTrain = SupervisedDataSet(len_question_set, len_question_set)

# 将输入输出数据加入Pybrain的数据集合
for i in range(len(question_feature_train)):
	dsTrain.addSample(question_feature_train[i], question_feature_train[i])
# 建立测试数据集
dsTest = SupervisedDataSet(len_question_set,len_question_set)

# 将输入输出数据加入Pybrain的数据集合
for i in range(len(question_feature_test)):
	dsTest.addSample(question_feature_test[i], question_feature_test[i])

finish_time=time.time()
print "问题模型的pybrain 训练、测试数据加入完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
fnn=buildNetwork(len_question_set,50,20,50,len_question_set,hiddenclass=SigmoidLayer,outclass=SoftmaxLayer,bias=True)
trainer = BackpropTrainer(fnn, dsTrain, momentum=0.1, verbose=True, weightdecay=0.01)
sys.stdout.flush()
trainer.trainUntilConvergence(maxEpochs=2)

finish_time=time.time()
print "训练完成。。。%s"%str(finish_time-start_time)
sys.stdout.flush()
test_err=percentError(trainer.testOnClassData(dataset=dsTest), dsTest['target'])
print "测试集错误"+str(test_err)
sys.stdout.flush()
print "开始生成编码"
sys.stdout.flush()
question_code=[]
for i in range(len(question_feature_list)):
	fnn.activate(question_feature_list[i])
	code=fnn['hidden1'].outputbuffer[fnn['hidden1'].offset]
	question_code.append(code)
print "问题编码完成"
sys.stdout.flush()
#清空没用数据
question_feature_list=[]
question_feature_train=[]
question_feature_test=[]
question_data=0


user_data=pd.read_table('/usr/python/toutiao/user_info.txt',sep='\t',names=['uid','utype','ustr1','ustr2'])
uid_map=dict()
for i in range(len(user_data)):
	u=user_data.ix[i,'uid']
	if not uid_map.has_key(u):
		uid_map[u]=i
print "问题id和行号、用户id和行号对应完成，分别存在qid_map和uid_map中"
sys.stdout.flush()
for i in range(len(user_data)):
	line=user_data.ix[i,'ustr1']
	words=line.split('/')
	user_set.update(words)
print "用户词语长度"+str(len(user_set))
finish_time=time.time()
print "用户词语采集完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
user_word_list=[i for i in user_set]
user_feature_list=[]
for i in range(len(user_data)):
	line=user_data.ix[i,'ustr1']
	words=line.split('/')
	each_feature=[]
	for j in range(len(user_set)):
		num=words.count(user_word_list[j])
		each_feature.append(num)
	user_feature_list.append(each_feature)
finish_time=time.time()
print "用户向量构建完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
user_word_list=[]
len_user_set=len(user_set)
user_set=set()
u_split_point=int(0.5*len(user_data))
user_feature_train=user_feature_list[:u_split_point]
user_feature_test=user_feature_list[u_split_point:]
print "数据集划分完成"
sys.stdout.flush()
dsTrain = SupervisedDataSet(len_user_set, len_user_set)

# 将输入输出数据加入Pybrain的数据集合
for i in range(len(user_feature_train)):
	dsTrain.addSample(user_feature_train[i], user_feature_train[i])
# 建立测试数据集
dsTest = SupervisedDataSet(len_user_set,len_user_set)

# 将输入输出数据加入Pybrain的数据集合
for i in range(len(user_feature_test)):
	dsTest.addSample(user_feature_test[i], user_feature_test[i])

finish_time=time.time()
print "用户模型的pybrain 训练、测试数据加入完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
fnn=buildNetwork(len_user_set,100,20,100,len_user_set,hiddenclass=SigmoidLayer,outclass=SoftmaxLayer,bias=True)
trainer = BackpropTrainer(fnn, dsTrain, momentum=0.1, verbose=True, weightdecay=0.01)
sys.stdout.flush()
trainer.trainUntilConvergence(maxEpochs=2)

finish_time=time.time()
print "训练完成。。。time:%s"%str(finish_time-start_time)
sys.stdout.flush()
test_err=percentError(trainer.testOnClassData(dataset=dsTest), dsTest['target'])
print "测试集错误"+str(test_err)
sys.stdout.flush()
print "开始生成编码"
sys.stdout.flush()
user_code=[]
for i in range(len(user_feature_list)):
	fnn.activate(user_feature_list[i])
	code=fnn['hidden1'].outputbuffer[fnn['hidden1'].offset]
	user_code.append(code)
print "用户编码完成"
sys.stdout.flush()
#清空没用数据
user_feature_list=[]
user_feature_train=[]
user_feature_test=[]
user_data=0
finish_time=time.time()
print "开始导入邀请表数据 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
invited_data=pd.read_table('/usr/python/toutiao/invited_info_train.txt',sep='\t',names=['qid','uid','is_ans'])
feature=[]
labels=[]
for i in range(len(invited_data)):
	qid=invited_data.ix[i,'qid']
	uid=invited_data.ix[i,'uid']
	label=invited_data.ix[i,'is_ans']
	# 用两位来表示，第一位是回答的概率，第二位是不回答 的概率
	if label==1:
		label=[1,0]
		pass
	elif label==0:
		label=[0,1]
	labels.append(label)
	q_feature=question_code[qid_map[qid]]
	u_feature=user_code[uid_map[uid]]
	each_feature.extend(q_feature)
	each_feature.extend(u_feature)
	feature.append(each_feature)

finish_time=time.time()
print "特征和标签保存完成 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
#清空没用数据
question_code=[]
user_code=[]
split_point=int(0.5*len(feature))
train_f=feature[:split_point]
test_f=feature[split_point:]
train_l=labels[:split_point]
test_l=labels[split_point:]
dsTrain = SupervisedDataSet(40, 2)
# 将输入输出数据加入Pybrain的数据集合
for i in range(len(train_f)):
	dsTrain.addSample(train_f[i], train_l[i])
# 建立测试数据集
dsTest = SupervisedDataSet(40,2)
# 将输入输出数据加入Pybrain的数据集合
for i in range(len(test_l)):
	dsTest.addSample(test_f[i], test_l[i])

finish_time=time.time()
print "训练数据和测试数据准备完毕 time:%s"%str(finish_time-start_time)
sys.stdout.flush()
fnn=buildNetwork(40,20,2,hiddenclass=SigmoidLayer,outclass=SoftmaxLayer,bias=True)
trainer = BackpropTrainer(fnn, dsTrain, momentum=0.1, verbose=True, weightdecay=0.01)

finish_time=time.time()
print "start training.... time:%s"%str(finish_time-start_time)
sys.stdout.flush()
trainer.trainUntilConvergence(maxEpochs=10)
test_err=percentError(trainer.testOnClassData(dataset=dsTest), dsTest['target'])
print "测试集错误"+str(test_err)

for i in range(5,20):
	print test_l[i]
	result=fnn.activate(test_f[i])
	print result
finish_time=time.time()
print "over...time:"%str(finish_time-start_time)
print "%s"%str(time.asctime(time.localtime(time.time())))
sys.stdout.flush()
