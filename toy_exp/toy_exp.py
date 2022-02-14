import os
import torch
import torch.nn.modules
import torch.nn
import numpy as np
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(SEED)


class topk_crossEntrophy(nn.Module):
    def __init__(self, top_k=0.75):
        super(topk_crossEntrophy, self).__init__()
        self.top_k = top_k
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        loss = self.loss(input, target)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]), dim=0)    
            return torch.mean(valid_loss)

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss()

    def forward(self, model_output, target):
        log = F.log_softmax(model_output, dim=1)
        loss = self.loss(log, target)
        return loss

class LabelSmoothLoss(nn.Module):
    def __init__(self, lsr_param=0.05, class_number=3, smooth=True, gt_log=False, reduction="mean"):
        """
        use label smoothing regularization to avoid overfitting
        :param lsr_param: parameter for the smoothing(e.g. uniform distribution)
        :param class_number: parameter for the number of classes
        :param smooth: parameter for label smoothing
        :param gt_log: parameter for input ground-truth (True for probability tensor eg. [[0.1,0.7,0.2],], False for class number eg. [1,])
        :param reduction method for the loss
        """
        super(LabelSmoothLoss, self).__init__()
        self.lb_pos = 1 - lsr_param
        self.class_number = class_number
        self.smooth = smooth
        self.gt_log = gt_log
        if reduction not in ["mean", "sum"]:
            raise ValueError("Reduction method only supports mean/sum, but receives {}.".format(reduction))
        self.reduction = reduction

    def forward(self, inputs, labels):
        lb_neg = (1 - self.lb_pos) / (self.class_number - 1)
        logs = F.log_softmax(inputs, dim=1)
        if not self.gt_log:
            # convert into one-hot encoding format
            one_hot_labels = torch.zeros(inputs.shape).scatter_(1, labels.unsqueeze(1), 1)
            # smooth the labels
            if self.smooth:
                labels = self.lb_pos * one_hot_labels + lb_neg * (1 - one_hot_labels)
            else:
                labels = one_hot_labels
        loss_ = torch.neg(torch.sum(logs * labels, dim=1))
        if self.reduction == "mean":
            loss = loss_.mean()
        else:
            loss = loss_.sum()    
        return loss
    

class Net(torch.nn.Module):
    
    def __init__(self,n_features,n_hiddens,n_outputs):
        super(Net,self).__init__()
        if isinstance(n_hiddens,int):
            self.hidden = nn.Linear(n_features,n_hiddens)
            self.predict = nn.Linear(n_hiddens,n_outputs)
        if isinstance(n_hiddens,list) and (len(n_hiddens)==2):
            self.hidden = nn.Sequential(
                nn.Linear(n_features,n_hiddens[0]),
                nn.ReLU(),
                nn.Linear(n_hiddens[0],n_hiddens[1])
                )
            self.predict = nn.Linear(n_hiddens[1],n_outputs)
 
    def forward(self, x):
        x = F.relu(self.hidden(x))
        predict = self.predict(x)
        return predict
 
class MyNet:
    
    def __init__(self,n_features,n_hiddens,n_outputs,times):
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.NeuronalNet=Net(self.n_features, self.n_hiddens, self.n_outputs)
        self.realX=None
        self.realY=None
        self.opitimizer=None
        self.lossFunc=None
        self.times=times

        self.Y=None

    def getData(self, classes=2, num=50, noise=0.0): # positive=negative=num; 0<noise<1
        if classes == 2:
            temp = torch.ones(num, 2)
            B = torch.normal(2 * temp, 1.5)
            By = torch.ones(num)
            A = torch.normal(-2 * temp, 1.5)
            Ay = torch.zeros(num)
     
            self.realX = Variable(torch.cat([A, B], 0))
            self.realY = Variable(torch.cat([Ay, By]).type(torch.LongTensor))

            ## insert noise
            if  noise:
                numb_noise = math.floor(noise * num * 2)
                lst = list(range(num * 2))
                random.shuffle(lst)
                for i in lst[:numb_noise]:
                    self.realY[i] = 1 - self.realY[i]

        if classes == 3:
            B = torch.normal(2 * torch.ones(num, 2) , 1.5) #(2,2)
            By = torch.ones(num)
            A = torch.normal(torch.tensor([[-2.,0.]]*num), 1.5) #(-2,0)
            Ay = torch.zeros(num)
            C = torch.normal(torch.tensor([[0.,-2.]]*num), 1.5) #(-2,0)
            Cy = torch.ones(num)*2
     
            self.realX = Variable(torch.cat([A, B, C], 0))
            self.realY = Variable(torch.cat([Ay, By, Cy]).type(torch.LongTensor))

            ## insert noise
            if  noise:
                num_noise = math.floor(noise * num * 3)
                lst = list(range(num * 2))
                random.shuffle(lst)
                for i in lst[:num_noise]:
                    self.realY[i] = (self.realY[i] + random.randint(1,2))%3

        self.Y = torch.zeros(len(self.realY),self.n_outputs)
        for idx, y in enumerate(self.realY):
            self.Y[idx][int(y)] = 1
            
        
    def run(self):
        self.opitimizer=torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = nn.CrossEntropyLoss()
 
        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.realY)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()

    def run_ohem(self, top_k=0.5):
        self.opitimizer = torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = topk_crossEntrophy(top_k=top_k)
 
        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.realY)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()

    def run_label_smooth(self, lsr_param=0.05, class_number=3):
        self.opitimizer = torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = LabelSmoothLoss(lsr_param, class_number)
 
        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.realY)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()

    def run_slr(self):
        # cycle 1:
        FOLD = 5
        sfolder = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)
        oof_list = []
        val_idx = []
        for train, test in sfolder.split(self.realX, self.realY):
            val_idx.append(test)
            net = Net(self.n_features, self.n_hiddens, self.n_outputs)
            opitimizer = torch.optim.SGD(net.parameters(),lr=LR)
            lossFunc = KLDivLoss() #nn.BCEWithLogitsLoss()
            for i in range(self.times):
                out=net(self.realX[train])
                loss=lossFunc(out,self.Y[train])
                opitimizer.zero_grad()
                loss.backward()
                opitimizer.step()
            oof_list.append(F.softmax(net(self.realX[test]), dim=1).detach().numpy())
        oof = np.zeros([len(self.realY),self.n_outputs])
        for fold in range(FOLD):
            for i,idx in enumerate(val_idx[fold]):
                oof[idx] = oof_list[fold][i]

        self.Y = torch.tensor((2*self.Y.numpy() + oof)/(2+1), dtype=torch.float32)

        self.NeuronalNet=Net(self.n_features, self.n_hiddens, self.n_outputs) ## re init
        self.opitimizer = torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = KLDivLoss()

        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.Y)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()
 
    def run_kd(self):
        self.opitimizer=torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = nn.CrossEntropyLoss()
 
        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.realY)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()

        out=self.NeuronalNet(self.realX).detach().numpy()

        self.Y = torch.tensor((2*self.Y.numpy() + out)/(2+1), dtype=torch.float32)

        self.NeuronalNet=Net(self.n_features, self.n_hiddens, self.n_outputs) ## re init
        self.opitimizer = torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = KLDivLoss()

        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.Y)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()

    def run_lr(self):
        self.opitimizer=torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = nn.CrossEntropyLoss()
 
        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.realY)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()

        out=self.NeuronalNet(self.realX).detach().numpy()

        self.Y = torch.tensor(out, dtype=torch.float32)

        self.NeuronalNet=Net(self.n_features, self.n_hiddens, self.n_outputs) ## re init
        self.opitimizer = torch.optim.SGD(self.NeuronalNet.parameters(),lr=LR)
        self.lossFunc = KLDivLoss()

        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
            loss=self.lossFunc(out,self.Y)
            self.opitimizer.zero_grad()
            loss.backward()
            self.opitimizer.step()
 
    #可视化
    def showBoundary(self, n, m, show_refine=False):
        x_min, x_max = self.realX[:, 0].min() - 0.1, self.realX[:, 0].max() + 0.1
        y_min, y_max = self.realX[:, 1].min() - 0.1, self.realX[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
        cmap = plt.cm.Spectral
 
        X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        y_pred = self.NeuronalNet(X_test)
        _, y_pred = y_pred.max(dim=1)
        y_pred = y_pred.reshape(xx.shape)

        if show_refine:
            _, Y_refine = self.Y.max(dim=1)
            showY = Y_refine
        else:
            showY = self.realY
 
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(self.realX[:, 0], self.realX[:, 1], c=showY, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #plt.title("toy classifier")
        plt.savefig('save/noise%d_%s.png'%(n,m))
        plt.clf()
 
    def predict(self,inputData):
        #inputData should be a 1x2 matrix
        data=torch.from_numpy(np.array(inputData)).int()
        return self.NeuronalNet(data.float())
 
 
 
if __name__=="__main__":

    num_class = 3
    LR = 0.05

    for noise in [0.3]: # noise ratio
        for method in ['baseline','ohem','lr','kd', 'ls', 'slr']:
            seed_everything(SEED)
            myNet =MyNet(2,[18,18],num_class,1000)
            myNet.getData(classes=num_class, num=50, noise=noise)
            show = False
            if method == 'baseline':
                myNet.run()
            elif method == 'ohem':
                myNet.run_ohem(top_k=0.25)
            elif method == 'lr':
                myNet.run_lr()
            elif method == 'kd':
                myNet.run_kd()
            elif method == 'ls':
                myNet.run_label_smooth(0.1, num_class)
            elif method == 'slr':
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'1', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'2', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'3', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'4', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'5', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'6', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'7', show_refine=True)
                myNet.run_slr()
                myNet.showBoundary(noise*10, method +'8', show_refine=True)
                show = False
            else:
                print('None')
            myNet.showBoundary(noise*10, method, show_refine=show)

