import torch
import torchvision
import numpy as np
import csv
from torchvision import datasets, transforms
import random
import time
from maml9_16 import MAML
from train9_16 import adaptation
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict
import torch.nn.functional as F


# 乱数の種を設定
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

# list からランダムに kosuu の数値をとってきて list を作る。
def select( list, kosuu ):
    while True:
        idx = torch.randint( 0, len(list), (kosuu,))
        if kosuu == len( torch.unique(idx)):
            break
    #print( "idx:", idx )
    return list[idx]

# ( outer_batch の次元のある)taskset から outer_batch_size の outer_batch データを作る。    
def create_batch_of_tasks(taskset, is_shuffle = True, outer_batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    idxs = torch.tensor( idxs )
    idxss = select( idxs, outer_batch_size )
    output = []

    for i in idxss:
        output.append( taskset[i])
    
    return output 

# [num_task, inner_batch, num_class * k, n_class = 3, img_size = 32, img_size =32] 画像タスクデータ
# [num_task, inter_batch, num_class * k ]　ラベルタスクデータを作る。
def build_task_dataset( img, target, num_all_task, num_task, k_support, k_query, num_class, inner_batch, is_val = False ):
    
    #画像データとラベルデータをシャフル    
    idx = range( 0, len(img), 1 )
    idx2 = random.sample( idx, len(img ))
    img = img[ idx2 ]
    target = target[ idx2 ]
    
    
    k_support = k_support
    k_query = k_query

    supports_img = []  # support image set
    supports_target = []  # support label set
    queries_img = []  # query image set
    queries_target = []  # query label set
        
    exam_support3_img = []
    exam_support3_target = []
    exam_query3_img = []
    exam_query3_target = []
    for b in range(num_task):  # タスクのループ
        # タスクは 0 ～ 19 の20個、ランダムに選択
        if is_val == False:
            current_task_support = torch.randint( 0, num_all_task - 3, size=(1,))
        else:
            current_task_support = torch.randint( num_all_task - 3, num_all_task, size=(1,))
        #current_task_query = torch.randint( 0, num_all_task, size=(1,))
        # support データセットと query データセットのタスクは同じで良いようです。
        current_task_query = current_task_support
        exam_support2_img = []
        exam_support2_target = []
        exam_query2_img = []
        exam_query2_target = []
        for k in range( num_class ): # N-way K-shot のループ N 部分
            # タスク3 だったら、 3 * (num_class = 5) + 0～4 と ラベルが等しい添え字を True False で取得。
            idx_support = target ==  current_task_support * num_class + k
            idx_query = target ==  current_task_query * num_class + k
            # 0 ～ データ数の数字を生成。
            idx2 = torch.arange( 0, len(img) , 1 ).long()
            # 上で True False の添え字を番号添え字に変換
            idx3_support = idx2[idx_support]
            idx3_query = idx2[idx_query]
            # 番号添え字の中からランダムに k_support あるいは k_query 個選ぶ。
            idx4_support = select( idx3_support, k_support )
            idx4_query = select( idx3_query, k_query )
            # 選択したデータを追加。
            for l in range( k_support ): # suppor データの N-way K-shot のループ k 部分
                exam_support2_img.append(img[idx4_support[l]])
                exam_support2_target.append( k )
            for l in range( k_query ): # query データの N-way K-shot のループ k 部分
                exam_query2_img.append(img[idx4_query[l]])
                exam_query2_target.append( k )
                
        exam_support2_img = torch.stack( exam_support2_img, dim = 0 )
        exam_support2_target = torch.tensor( exam_support2_target )
        exam_query2_img = torch.stack( exam_query2_img, dim = 0 )
        exam_query2_target = torch.tensor( exam_query2_target )

        exam_support3_img.append( exam_support2_img )
        exam_support3_target.append( exam_support2_target )
        exam_query3_img.append( exam_query2_img )
        exam_query3_target.append( exam_query2_target )
            
    exam_support3_img = torch.stack( exam_support3_img, dim = 0 )
    exam_support3_target = torch.stack( exam_support3_target, dim = 0 )
    exam_query3_img = torch.stack( exam_query3_img, dim = 0 )
    exam_query3_target = torch.stack( exam_query3_target, dim = 0 )

    supports_img = exam_support3_img
    supports_target = exam_support3_target
    queries_img = exam_query3_img
    queries_target = exam_query3_target

    return supports_img, supports_target, queries_img, queries_target
