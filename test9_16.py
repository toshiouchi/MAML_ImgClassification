import torch
import torchvision
import numpy as np
import csv
from torchvision import datasets, transforms
import random
import time
from maml9_16 import MAML
from train9_16 import test_model
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict
import torch.nn.functional as F
from build_task_dataset9_16 import build_task_dataset, create_batch_of_tasks
import os
    
def main():

    os.makedirs( "model/", exist_ok=True)
    os.makedirs( "log/", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = MAML().to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load("model/model.pth")
    else:
        checkpoint = torch.load("model/model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    result_path = "./log/test"

    # dataset

    # Transform を作成する。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset を作成する。
    download_dir = "./data"  # ダウンロード先は適宜変更してください
    testset = datasets.CIFAR100(download_dir, train=False, transform=transform, download=True)

    #test用の画像データとラベルデータ
    img = []
    target = []
    for i, (im, tar) in enumerate( testset ):
        if 4999 < i and i < 10000:
            img.append( im )
            target.append( tar )
        elif i > 10000:
            break
        
    img = torch.stack( img, dim = 0 )
    target = torch.tensor( target )

    test_loss_log = []
    test_acc_log = []

    all_class = 100 # 全データの全クラス数
    num_class = 5 # N-way K-shot の N
    n_class = 3 # 画像がカラー
    img_size = 32 #画僧のサイズ 32 * 32

    print( "len of test_img:", len( img ) )
    print( "len of test_target:", len( target ) )

    print( "10 loop" )

    outer_batch0 = 20

    ob_test = []
    # test 用の taskset を作り outer_batch の次元を加える。
    for i in range( outer_batch0 ):
        #print( "i:", i )
        test = build_task_dataset( img, target, num_all_task = all_class // num_class,  num_task = 20, k_support=20, k_query=20, num_class = 5, inner_batch = 1, is_val = True )
        ob_test.append( test )

    # test
    db_test = create_batch_of_tasks(ob_test, is_shuffle = False, outer_batch_size = 10)
    acc_all_test = []
    loss_all_test = []

    f = open('log.txt', 'a')
    for loop, test_task in enumerate( db_test ):
        #random_seed(123)
        loss, acc = test_model(model, test_task, loss_fn, train_step = 10, lr1 = 1e-3, device=device, n_class = n_class, img_size = img_size)
        acc_all_test.append(acc)
        loss_all_test.append( loss )
        #random_seed(int(time.time() % 10))

        print('loop:', loop, 'Test loss:', np.mean(loss_all_test),'\tacc:', np.mean(acc_all_test))
        #test_loss_log.append( np.mean(loss_all_test) )
        #test_acc_log.append( np.mean(acc_all_test ) )
        f.write('Test' + str(np.mean(acc_all_test)) + '\n')
            
        
    all_result = { 'test_loss': loss_all_test, 'test_acc': acc_all_test}

    with open(result_path + '.pkl', 'wb') as f:
        pickle.dump(all_result, f)
    
if __name__ == "__main__":
    main()

