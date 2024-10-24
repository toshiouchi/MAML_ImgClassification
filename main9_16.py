import torch
import torchvision
import numpy as np
import csv
from torchvision import datasets, transforms
import random
import time
from maml9_16 import MAML
from train9_16 import adaptation, validation
import pickle
from torch.utils.data import Dataset
from build_task_dataset9_16 import build_task_dataset, create_batch_of_tasks, random_seed
import os

def main():

    os.makedirs( "model/", exist_ok=True)
    os.makedirs( "log/", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    epochs = 300
    model = MAML().to(device)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model_path = "./model/"
    result_path = "./log/train"

    # dataset

    # Transform を作成する。
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset を作成する。
    download_dir = "./data"  # ダウンロード先は適宜変更してください
    trainset = datasets.CIFAR100(download_dir, train=True, transform=transform, download=True)
    evalset = datasets.CIFAR100(download_dir, train=False, transform=transform, download=True)

    #学習用の画像データとラベルデータ
    img = []
    target = []
    for i, (im, tar) in enumerate( trainset ):
        if i < 10000:
            img.append( im )
            target.append( tar )
        else:
            break

    #validation 用の画像データとラベルデータ
    img2 = []
    target2 = []
    for i, (im, tar) in enumerate( evalset ):
        if i < 5000:
            img2.append( im )
            target2.append( tar )
        else:
            break
        
    img = torch.stack( img, dim = 0 )
    target = torch.tensor( target )
    img2 = torch.stack( img2, dim = 0 )
    target2 = torch.tensor( target2 )

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    all_class = 100 # 全データの全クラス数
    num_class = 5 # N-way K-shot の N
    n_class = 3 # 画像がカラー
    img_size = 32 #画僧のサイズ 32 * 32

    print( "len of train_img:", len( img ) )
    print( "len of train_target:", len( target ) )
    print( "len of validation_img:", len( img2 ) )
    print( "len of validation_target:", len( target2 ) )

    print( "epochs:", epochs )

    outer_batch0 = 5 # 実際の outer_batch 数より大きめの値を設定しておく。

    ob_val = []
    # validation 用の taskset を作り outer_batch の次元を加える。
    for i in range( outer_batch0 ):
        val = build_task_dataset( img2, target2, num_task = 20, k_support=20, k_query=20, num_class = 5, inner_batch = 1 )
        ob_val.append( val )

    global_step = 0

    for epoch in range(epochs):
        trainbatch = {}
        evalbatch = {}

        ob_train = []
        # 学習用の taskset を作り outer_batch の次元を加える。
        for i in range( outer_batch0 ):
            train = build_task_dataset(img, target, num_task = 20, k_support=10, k_query=15, num_class = 5, inner_batch = 3 )
            ob_train.append( train )

        # 学習用データセットを作る。
        db_train = create_batch_of_tasks( ob_train, is_shuffle = True, outer_batch_size = 3 )

        for step, train_task in enumerate(db_train):
        
            f = open('log.txt', 'a')
        
            #学習。
            loss, acc = adaptation(model, outer_optimizer, train_task, loss_fn,  train_step=5, train=True, device=device)
            train_loss_log.append( loss )
            train_acc_log.append( acc )   
        
            print('Epoch:', epoch, 'Step:', step, '\ttraining Loss:', loss,'\ttraining Acc:', acc)
            f.write(str(acc) + '\n')
        
            # Validation
            if global_step % 20 == 0:
                random_seed(123)
                print("\n-----------------Validation Mode-----------------\n")
                db_val = create_batch_of_tasks(ob_val, is_shuffle = False, outer_batch_size = 1)
                acc_all_val = []
                loss_all_val = []

                for val_task in db_val:
                    loss, acc = validation(model, val_task, loss_fn, train_step = 10, device=device)
                    acc_all_val.append(acc)
                    loss_all_val.append( loss )


                print('Epoch:', epoch, 'Step:', step, 'Validation F1 loss:', np.mean(loss_all_val),'\tacc:', np.mean(acc_all_val))
                val_loss_log.append( np.mean(loss_all_val) )
                val_acc_log.append( np.mean(acc_all_val ) )
                f.write('Validation' + str(np.mean(acc_all_val)) + '\n')
            
                random_seed(int(time.time() % 10))
        
            global_step += 1
            f.close()
        
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': outer_optimizer.state_dict(),
                'loss': loss,},
               model_path + 'model.pth')

    all_result = {'train_loss': train_loss_log, 'train_acc': train_acc_log, 'val_loss': val_loss_log, 'val_acc': val_acc_log}

    with open(result_path + '.pkl', 'wb') as f:
        pickle.dump(all_result, f)
    
if __name__ == "__main__":
    main()
