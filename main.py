from HEMAsNet import *
from torch.utils.data import Dataset, ConcatDataset
from new_data import *
from focalloss import *
from torch.optim import lr_scheduler
from data_splitting import *
device = torch.device('cuda')
deal_data()

data1=si_dataset_mt('./1')
data2=si_dataset_mt('./2')
data3=si_dataset_mt('./3')
data4=si_dataset_mt('./4')
data5=si_dataset_mt('./5')
data6=si_dataset_mt('./6')
data7=si_dataset_mt('./7')
data8=si_dataset_mt('./8')
data9=si_dataset_mt('./9')
data10=si_dataset_mt('./10')

learning_rate = 1e-3
alpha = 0.75
gamma = 1

for round in range(10):
    dalist=(data1,data2,data3,data4,data5,data6,data7,data8,data9,data10)
    train_data=ConcatDataset(dalist[:round] + dalist[round+1:])
    val_data=dalist[round]
    val_data_name = [k for k, v in locals().items() if v is val_data][0]
    train_data_size=len(train_data)
    val_data_size=len(val_data)
    print('数据集的长度为:{}'.format(train_data_size))
    print('测试集的长度为:{}'.format(val_data_size))
    train_dataloader=DataLoader(train_data,batch_size=300,shuffle=True)
    val_dataloader=DataLoader(val_data,batch_size=300,shuffle=True)
    net = Global()
    net=net.to(device)
    loss_focal = FocalLoss(alpha=alpha,gamma=gamma)
    loss_focal = loss_focal.to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=8,eta_min=0.2*learning_rate)
    total_train_step=0
    total_val_step=0
    epoch=60
    round_accuracies = []
    previous_accuracy = 0.0

    for ii in range(epoch):
        total_tr_loss=0
        print('第{}轮训练开始'.format(ii+1))
        net.train()
        time_start=time.time()
        counter = 0
        for data in train_dataloader:

            time_start=time.time()
            xl,lable,_=data
            xl=xl.to(device)
            lable=lable.to(device)
            output,att=net(xl)
            ll=lable.long()
            loss=loss_focal(output,lable.long())
            total_tr_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step+=1

        current_lr = scheduler.get_last_lr()
        print('current_lr:',current_lr)
        print('Loss on the overall train set:{}'.format(total_tr_loss))
        scheduler.step()
        net.eval()
        total_acc=0
        total_val_loss=0
        classes=('c','d')
        N_CLASSES = 2
        class_correct = list(0. for i in range(N_CLASSES))
        class_total = list(0. for i in range(N_CLASSES))
        lable_true=[]
        lable_pred=[]

        with torch.no_grad():
            FN=0
            TP=0
            FP=0
            TN=0
            for data in val_dataloader:
                xl,lable,name=data
                xl=xl.to(device)
                lable=lable.to(device)
                output,att=net(xl)
                lable_true=lable_true+lable.tolist()
                lable_pred=lable_pred+output.argmax(1).tolist()
                loss=loss_focal(output,lable.long())
                total_val_loss+=loss.item()
                acc=(output.argmax(1)==lable).sum()
                _, preds = output.max(1)
                c = (preds == lable).squeeze()
                total_acc=total_acc+acc
                for kk in range(len(output.argmax(1))):

                    if output.argmax(1)[kk]==lable[kk]:
                        if output.argmax(1)[kk]==1 :
                            TN+=1
                        else: TP+=1
                    if output.argmax(1)[kk]!=lable[kk]:
                        if output.argmax(1)[kk]==1 :
                            FN+=1
                        else: FP+=1

        time_end=time.time()
        print('time cost:',time_end-time_start,'s')
        print('Loss on the overall test set:{}'.format(total_val_loss))
        print('The accuracy on the overall test set:{}'.format((total_acc/val_data_size)*100))
        round_accuracies.append((total_acc/val_data_size)*100)
        zhun=(total_acc/val_data_size)*100
        print('TP={}'.format(TP))
        print('TN={}'.format(TN))
        print('FN={}'.format(FN))
        print('FP={}'.format(FP))
        print('灵敏度：{}'.format(TP/(TP+FN)))
        print('特异度：{}'.format(TN/(TN+FP)))
        if (TP+FN) == 0 or (TP+FP) == 0 or TP==0 :
            f1 = 0.0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2*(TP/(TP+FN))*(TP/(TP+FP))/((TP/(TP+FN))+(TP/(TP+FP)))
        print('f1：{}'.format(f1))
        if zhun >previous_accuracy and zhun>60:
            torch.save(net.state_dict(), '{}-{}-{}-{}-{}.pth'.format(val_data_name,learning_rate,zhun,(TP/(TP+FN)),(TN/(TN+FP))))
        total_val_step+=1
    print(max(round_accuracies))
average_accuracy = sum(round_accuracies) / 10
print(average_accuracy)
