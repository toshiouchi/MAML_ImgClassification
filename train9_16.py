import torch
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

n_class = 3   #カラー画像
img_size = 32 # img_size = width = height

def adaptation(model, outer_optimizer, batch, loss_fn, train_step, train, lr1,  device):

    x_train = batch[0] #support 画像データ
    y_train = batch[1] #support ラベルデータ
    x_val = batch[2]   #query 画像データ
    y_val = batch[3]   #query ラベルデータ

    task_accs = []
    num_task = len( x_train )

    outer_loss_item = 0

    if train:
        weights0 = OrderedDict(model.named_parameters()) #今回の基準パラメータ
    for idx in range(x_train.size(0)): # task
        if train:
            weights = weights0
        weights2 = OrderedDict(model.named_parameters())
        # batch 抽出
        input_x = x_train[idx].to(device)
        input_y = y_train[idx].to(device)
        x = input_x
        y = input_y
        
        print('----Task',idx, '----')

        # タスクごとの損失の計算
        loss_item = 0
        for iter in range(train_step): # train_step のループ
            model.train()
            logits = model.adaptation(x, weights2 )
            loss = loss_fn(logits, y)
            loss_item += loss.item()
            #各タスクについて一番目の損失関数からモデルパラメーターを求める。
            #graph を残して、2回めの更新のときにその情報を使う。
            #gradients = torch.autograd.grad(loss, weights2.values())
            gradients = torch.autograd.grad(loss, weights2.values(), create_graph=True)
            weights2 = OrderedDict((name, param - lr1 * grad) for ((name, param), grad) in zip(weights2.items(), gradients))

        loss_item = loss_item / train_step 

        print("Inner Loss: ", loss.item())
        
        # query データからバッチ抽出
        input_x = x_val[idx].to(device)
        input_y = y_val[idx].to(device)
        
        # 訓練時に query データ（inner_batch = 1, query_k * 5クラス )　で二番目の損失関数の各タスクについての総和を求める。
        x = input_x
        y = input_y
        # 各タスクについて、上で求めたモデルパラメーターを使って損失を求める。
        if train:
            model.train()
            logits = model.adaptation( x, weights2 )
        else:
            model.eval()
            with torch.no_grad():
                logits = model.adaptation( x, weights2 )
            
        outer_loss0 = loss_fn( logits, y )
        outer_loss_item += outer_loss0.item()
        if train:
            tmp = torch.autograd.grad( outer_loss0, weights.values() )
            if idx ==0:
                gradients2 = list(tmp)
            else:
                gradients2 = [x + y for x, y in zip(gradients2, list(tmp))]
        pre_label_id = torch.argmax( logits, dim = 1 )
        acc = torch.sum( torch.eq( pre_label_id, y ).float() ) / y.size(0)
        task_accs.append(acc)


    # 訓練時、二番目の損失関数（各タスクの総和）を使って、一番目の損失関数によるモデルパラメータの前を基準に勾配を求める。
    if train:
        outer_optimizer.zero_grad()
        for i, params in enumerate(model.parameters()):
            params.grad = gradients2[i]
        outer_optimizer.step()

    task_accs = torch.stack( task_accs )
    outer_loss_item = outer_loss_item / num_task

    print( "loss:", outer_loss_item )

    return outer_loss_item, torch.mean(task_accs).item()

def test_model(model, batch, loss_fn, train_step, lr1,  device, n_class, img_size):
    #評価用ルーチン
    x_train = batch[0] #support 画像データ
    y_train = batch[1] #support ラベルデータ
    x_val = batch[2]   #query 画像データ
    y_val = batch[3]   #query ラベルデータ
    
    predictions = []
    labels = []
    num_task = len( x_train )

    loss1 = 0
    outer_loss = 0

    task_accs = []

    for idx in range(x_train.size(0)): # task
        weights2 = OrderedDict(model.named_parameters())
        # model.parameter を weights 関数に格納。 idx のループの間、weights は書き換えられるが model.parameter は変わらない。
        weights2 = OrderedDict(model.named_parameters()) #今回の基準パラメータ
        # batch 抽出
        input_x = x_train[idx].to(device)
        input_y = y_train[idx].to(device)
        
        print('----Task',idx, '----')

        # 各タスクについて train_step 回学習をループし、パラメーターを求める。
        for iter in range(train_step):
            x = input_x
                # support_batch [ inner_batch, N * K, 3,32,32] → [ inner_batch * N * K, 3,32,32]
            y = input_y
            logits = model.adaptation(x, weights2)
            loss = loss_fn(logits, y)
            loss1 += loss.item()
            gradients = torch.autograd.grad(loss, weights2.values())
            #gradients = torch.autograd.grad(loss, weights.values(), create_graph=True)
            # validation では、二回目の微分がないため graph はいらない。
            weights2 = OrderedDict((name, param - lr1 * grad) for ((name, param), grad) in zip(weights2.items(), gradients))

        loss1 = loss1 / train_step
        
        print( "Inner loss:", loss1 )

        #各タスクについて上で求めた weights を用い、損失と精度を計算する。
        # query data
        input_x = x_val[idx].to(device)
        input_y = y_val[idx].to(device)
        x = input_x
        y = input_y
        with torch.no_grad():
            model.eval()
            logits = model.adaptation( x, weights2 )
            outer_loss += loss_fn( logits, y ).item()
        pre_label_id = torch.argmax( logits, dim = 1 )
        acc = torch.sum( torch.eq( pre_label_id, y ).float() ) / y.size(0)
        task_accs.append(acc)

    task_accs = torch.stack( task_accs )

    print( "loss:", outer_loss / num_task )

    return outer_loss / num_task, torch.mean(task_accs).item()
