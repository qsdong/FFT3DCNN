#!/mnt/sdd/dongqs/venv_gcn/bin/python3
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import glob
import random
import sys
import time
import matplotlib.pyplot as plt
# import matplotlib
import copy
import argparse



class FFT3DCNN_Dataset(Dataset):
    def __init__(self, allFiles, allLabels,device):
        self.device = device
        self.files = allFiles
        self.label = allLabels
        self.n = 8

    def __getitem__(self, idx):
        Nxyz, Lxyz, ph = self.load_ph_ABC_pure(self.files[idx])
        return Nxyz, Lxyz, ph, torch.tensor(self.label[idx]).to(self.device)

    def __len__(self):
        return len(self.files)

    def load_ph_ABC_pure(self,filename):
        Nxyz = np.fromfile(filename, dtype=np.intc, count=3, offset=0) # intc is identical to C int (normally int32 or int64)
        # print(Nxyz[0].itemsize) # the size of intc is 4
        Lxyz = np.fromfile(filename, dtype=np.float64, count=3, offset=3*4)
        # print(Lxyz[0].itemsize) # the size of float64 is 8
        ph = torch.from_numpy(np.fromfile(filename, dtype=np.float64, count=-1, offset=3*(4+8))).to(torch.float32).to(self.device) # read all remaining data
        # NxNyNz = Nxyz[0]*Nxyz[1]*Nxyz[2]
        ph = ph.reshape([3,Nxyz[0],Nxyz[1],Nxyz[2]])

        ph[ph>0.5] = 1
        ph[ph<=0.5] = 0

        return Nxyz, Lxyz, ph


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class FFT3DCNN_Classifier(nn.Module):
    def __init__(self, numStructureClass, convpara, fcNeurons):
        super().__init__() # initialise parent pytorch class
        self.o_channels = 3*10
        # Conv2d input(N, Cin, Hin, Win), N is batch size, C is channel size, H is height in pixels, w is width in pixels
        self.conv1 = nn.Conv3d(in_channels=3, out_channels = self.o_channels, kernel_size=convpara[0], stride=convpara[1])
        self.conv2 = nn.Conv3d(in_channels= self.o_channels, out_channels = self.o_channels, kernel_size=convpara[2], stride=convpara[3])

        self.bn1 = nn.BatchNorm3d(num_features = self.o_channels)
        self.bn2 = nn.BatchNorm3d(num_features = self.o_channels)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(fcNeurons, numStructureClass)

        # create loss function
        self.loss_function = nn.CrossEntropyLoss()

        # create optimiser, using simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(),lr=1e-3,weight_decay=1e-5) #weight decay (L2 penalty) (default: 0)
        pass

    def forward(self, inputs):
        x = F.leaky_relu(self.conv1(inputs))
        x = self.bn1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))

        return F.softmax(x,dim=1)


    def train_model(self, inputs, targets,isTraining):
        # calculate the output of the network
        outputs = self.forward(inputs)
        _, idx_targets = torch.max(targets, 1)

        # calculate loss
        loss = self.loss_function(outputs, idx_targets)
        rLoss = loss.item()

        # zero gradients, perform a backward pass, and update the weights
        if isTraining:
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        return rLoss, outputs


def train_test_split(allFiles,allLabels, percent = 0.8):
    num = len(allFiles)
    train_num = int(num*percent)
    train_list = random.sample(range(num), k=num)

    train_allFiles = [allFiles[i] for i in train_list[0:train_num]]
    train_allLabels = [allLabels[i]  for i in train_list[0:train_num]]

    test_allFiles = [allFiles[i] for i in train_list[train_num:]]
    test_allLabels = [allLabels[i] for i in train_list[train_num:]]
    
    return train_allFiles, train_allLabels, test_allFiles, test_allLabels

def get_file_names(root_path,structureNames,phname):
    
    structureOnehot = F.one_hot(torch.tensor(range(len(structureNames)))) # generate the one_hot tensor of the structure names

    train_allFiles, train_allLabels, test_allFiles, test_allLabels  = [], [], [], []
    for idx, sName in enumerate(structureNames): # get file names and labels
        files = glob.glob(root_path + sName + '/*/'+phname)
        labels = structureOnehot[idx] * torch.ones(len(files),1)# note the broadcast of the tensor, which copies the label len(files) times.

        train_files, train_labels, test_files, test_labels = train_test_split(files, labels.tolist(), percent = 0.8)

        if len(train_files)<10:
            num = 10
        elif len(train_files)<20:
            num = 5
        elif len(train_files)<30:
            num = 3
        elif len(train_files)<40:
            num = 2
        else:
            num = 1
        
        for j in range(num):
            [train_allFiles.append(i) for i in train_files]
            [train_allLabels.append(i) for i in train_labels]
        
        [test_allFiles.append(i) for i in test_files]
        [test_allLabels.append(i) for i in test_labels]
        
        print(sName,'training set count: {}*{}, testing set count: {}'.format(len(train_labels), num, len(test_labels)))
    # the training set should be randomly shuffled
    return  train_allFiles, train_allLabels, test_allFiles, test_allLabels


def test_model(model, dLoader,structureNames):
    model.eval()
    errCount = 0
    confusion_matrix = np.zeros((len(structureNames),len(structureNames)))
    for Nxyz, Lxyz, phs, labels in dLoader:
        cBatchSize = phs.shape[0]
        loss, outputs = model.train_model(phs.view(cBatchSize,3,Nxyz[0][0],Nxyz[0][1],Nxyz[0][2]), labels, isTraining=False)
        maxValue, idx1= torch.max(outputs.detach(), 1) # predicted label
        _, idx2= torch.max(labels, 1) # real label
        for i in range(len(idx1)):
            if idx1[i] != idx2[i]:
                print(structureNames[idx2[i]], idx2[i].item(), '-->', structureNames[idx1[i]], idx1[i].item()) # misclassified as
                errCount += 1
            if maxValue[i]<0.6:
                print('>>>',structureNames[idx1[i]])
            confusion_matrix[idx2[i],idx1[i]] += 1 # Do not transpose in matlab(y label: True，x label: Predicted)
    return errCount, confusion_matrix

def train_model(classifierModel, dLoader, totalEpochs = 1):
    classifierModel.train()
    mean_loss_list = []
    best_loss = 1e5
    for epoch in range(totalEpochs):
        t0 = time.time()
        loss_epoc = []
        dur = []
        for Nxyz, Lxyz, phs, labels in dLoader:
            # print(Nxyz)
            cBatchSize = phs.shape[0]
            if cBatchSize == batchSize:
                loss, _ = classifierModel.train_model(phs.view(cBatchSize,3,Nxyz[0][0],Nxyz[0][1],Nxyz[0][2]), labels, isTraining=True)
                loss_epoc.append(loss)
        
        mean_loss = np.mean(loss_epoc)
        mean_loss_list.append(mean_loss)

        if mean_loss < best_loss:
            best_loss = mean_loss

        dur.append(time.time() - t0)
        # print(loss)
        if (epoch+1) % 1 == 0:
            print("Epoch {:04d}/{:04d} | MeanLoss {:4.4f} | MeanTime(s) {:3.4f}".format(epoch+1, totalEpochs, mean_loss, np.mean(dur) ))
            sys.stdout.flush()
    
    plot_learningProcess(mean_loss_list)
    
    return best_loss
    
def plot_learningProcess(loss_epoc):
    epoc = range(0,len(loss_epoc))
    p1 = plt.plot(epoc,loss_epoc)
    plt.title("")
    plt.xlabel('epoc')
    plt.ylabel('loss')

    plt.savefig("learningProcess.jpg", dpi=300)
    plt.clf()

    with open("learningProcess.txt","w") as fp:
        for i in range(len(loss_epoc)):
            fp.write(str(loss_epoc[i])+"\n")

def my_parse_args():
    parser = argparse.ArgumentParser(description='QingshuDong FFT3DCNN')
    parser.add_argument('-n', type=int, default=8, help='num elements')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--convpara', type=int, default=3221, help='conv kernel_size stride')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    t_start = time.time()
    device = "cuda:0"
    root_path = '/home/dongqs/sdd/AABCn3D/' # linux
    # root_path = 'Z:/sdd/AABCn3D/' # windows

    structureNames = ['sadc','dadc','gagc','CsCl','dg_cs','dgn','fcc_c','bcc_c','bcc_a','g_c','IWP','PP','sagc','sddn','sdgn',
                        'c4c4','c4_cs','calc','L3',
                        'c3c6','c6_c','c6c3','c6_c_cs','c6_cs','smch','splh',
                        'sigmaN']

    args = my_parse_args()

    print(args)

    # conv1 kernel_size=3, stride=2
    # conv2 kernel_size=2, stride=1
    if args.convpara == 5353:
        convpara = [5,3,5,3]
        ph_fc_dict = {  'ph4.bin':np.nan,
                        'ph6.bin':np.nan,
                        'ph8.bin':np.nan,
                        'ph10.bin':30,
                        'ph12.bin':30,
                        'ph14.bin':240,
                        'ph16.bin':240,
                        }
    elif args.convpara == 5332:
        convpara = [5,3,3,2]
        ph_fc_dict = {  'ph4.bin':np.nan,
                        'ph6.bin':30,
                        'ph8.bin':30,
                        'ph10.bin':240,
                        'ph12.bin':810,
                        'ph14.bin':810,
                        'ph16.bin':1920,
                        }
    elif args.convpara == 5321:
        convpara = [5,3,2,1]
        ph_fc_dict = {  'ph4.bin':30,
                        'ph6.bin':240,
                        'ph8.bin':810,
                        'ph10.bin':3750,
                        'ph12.bin':6480,
                        'ph14.bin':10290,
                        'ph16.bin':21870,
                        }
    elif args.convpara == 5232:
        convpara = [5,2,3,2]
        ph_fc_dict = {  'ph4.bin':np.nan,
                        'ph6.bin':30,
                        'ph8.bin':240,
                        'ph10.bin':810,
                        'ph12.bin':1920,
                        'ph14.bin':3750,
                        'ph16.bin':6480,
                        }
    elif args.convpara == 5231:
        convpara = [5,2,3,1]
        ph_fc_dict = {  'ph4.bin':np.nan,
                        'ph6.bin':240,
                        'ph8.bin':1920,
                        'ph10.bin':6480,
                        'ph12.bin':15360,
                        'ph14.bin':30000,
                        'ph16.bin':51840,
                        }
    elif args.convpara == 3232:
        convpara = [3,2,3,2]
        ph_fc_dict = {  'ph4.bin':30,
                        'ph6.bin':240,
                        'ph8.bin':810,
                        'ph10.bin':1920,
                        'ph12.bin':3750,
                        'ph14.bin':6480,
                        'ph16.bin':10290,
                        }
    elif args.convpara == 3231:
        convpara = [3,2,3,1]
        ph_fc_dict = {  'ph4.bin':30,
                        'ph6.bin':810,
                        'ph8.bin':3750,
                        'ph10.bin':10290,
                        'ph12.bin':21870,
                        'ph14.bin':39930,
                        'ph16.bin':65910,
                        }
    elif args.convpara == 3221:
        convpara = [3,2,2,1]
        ph_fc_dict = {  'ph4.bin':240,
                        'ph6.bin':1920,
                        'ph8.bin':6480,
                        'ph10.bin':15360,
                        'ph12.bin':30000,
                        'ph14.bin':51840,
                        'ph16.bin':82320,
                        }
    else:
        print("error convpara")
        exit(1)

    print(convpara)
    print(ph_fc_dict)


    batchSize = args.batch
    model_saved_name = 'model_ver06_convpara{}n{}.pt'.format(args.convpara,args.n)
    phname = 'ph{}.bin'.format(args.n)

    print("batch={}, n={}".format(batchSize,args.n))

    train_allFiles, train_allLabels, test_allFiles, test_allLabels = get_file_names(root_path,structureNames,phname=phname)

    dSet_train = FFT3DCNN_Dataset(train_allFiles,train_allLabels,device) # all the files should have equal size
    dLoader_train = DataLoader(dSet_train,batch_size=batchSize,num_workers=0,shuffle=True) # set to True to have the data reshuffled at every epoch (default: False).

    # create neural network
    numStructureClass = len(structureNames)
    classifierModel = FFT3DCNN_Classifier(numStructureClass, convpara, fcNeurons = ph_fc_dict[phname]).to(device)

    best_loss = 0
    # # train network
    best_loss = train_model(classifierModel, dLoader_train, totalEpochs = 1000)
    # exit(1)

    torch.save(classifierModel, model_saved_name)

    # classifierModel = torch.load(model_saved_name).to(device)

    # test network
    dSet_test = FFT3DCNN_Dataset(test_allFiles,test_allLabels,device) # all the files should have equal size
    dLoader_test = DataLoader(dSet_test,batch_size=1,num_workers=0)

    errCount1, confusion_matrix1 = test_model(classifierModel, dLoader_train, structureNames)
    print('---------------------')
    errCount2, confusion_matrix2 = test_model(classifierModel, dLoader_test, structureNames)

    np.savetxt("confusion_matrix_train.txt", confusion_matrix1, fmt="%d", delimiter=" ", header='', comments="")
    np.savetxt("confusion_matrix_test.txt", confusion_matrix2, fmt="%d", delimiter=" ", header='', comments="")


    print(errCount1,dSet_train.__len__())
    print("Error rate of training set: {:.2f} %".format(errCount1/dSet_train.__len__()*100) )

    print(errCount2,dSet_test.__len__())
    print("Error rate of testing set: {:.2f} %".format(errCount2/dSet_test.__len__()*100) )

    t_end = time.time()
    print("Total time {:.4f}(s) or {:.4e}(min) BestLoss {:4.4f}".format(t_end-t_start,(t_end-t_start)/60, best_loss))

# srun --partition=amd_3090 --gpus=1 --cpus-per-gpu=1 ./FFT3DCNN.py -n=4 --convpara=3231  >aa.txt 2>&1 &