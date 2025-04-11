import sys
sys.path.append('/work3/kniud/Voxel_grid/')
from re import X
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional,neuron,layer
import numpy as np

from torch.utils.data import Dataset
import os
import torchvision.transforms as Tr

from collections import OrderedDict

import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader

import h5py
import argparse
import random

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from eventDataSet_voxel_mean_std import ClassificationDataset

class Flatten(torch.nn.Module):
    def forward(self,x):
        batch_size = x.shape[0]
        return x.view(batch_size,-1)

class CustomLIFNode(neuron.ParametricLIFNode):
    def __init__(self,step_mode='s',surrogate_function=None):
        super().__init__()

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        # self.neuronal_reset(spike)
        return self.v

class MultiStepVGGSNN(nn.Module):
    def __init__(self,num_of_inChannels,num_classes=4,args = None,init_weights=True):
        super(MultiStepVGGSNN,self).__init__() #VGGSNN

        self.out_channels = []
        self.nz, self.numel = {}, {}
        #self.idx_pool = [i for i,v in enumerate(cfg) if v=='M']
        #if norm_layer is None:
        #    norm_layer = nn.Identity
        self.args = args
        self.num_cls = num_classes
        bias = False
        img_size = args.inputRes 
        self.img_size = args.inputRes
        self.representationType = args.representationType

        """self.features = self.make_layers(num_of_inChannels,cfg=cfg,norm_layer=norm_layer,lifNeuron=single_step_neuron,bias=bias,**kwargs)"""

        single_step_neuron = nn.ReLU
        
        self.features = None
        self.classifier = None
        "nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),single_step_neuron(**kwargs),"
        if self.args.architecture == "vgg_11":

            self.features = nn.Sequential(
                OrderedDict(
                [
                ("conv_1 ",nn.Conv2d(num_of_inChannels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_1", nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_1",single_step_neuron()),
                ("pool_1",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_2",nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_2",nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_2",single_step_neuron()),
                ("pool_2",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_3",nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_3",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_3",single_step_neuron()),

                ("conv_4",nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_4",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_4",single_step_neuron()),
                ("pool_3",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                
                ("conv_5",nn. Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_5",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_5",single_step_neuron()),

                ("conv_6",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_6",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_6",single_step_neuron()),
                ("pool_6",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_7",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_7",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_7",single_step_neuron()),

                ("conv_8",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_8",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_8",single_step_neuron()),
                ("pool_8",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                ])
                )
        elif self.args.architecture == "vgg_13":

            self.features = nn.Sequential(
                OrderedDict(
                [
                ("conv_1 ",nn.Conv2d(num_of_inChannels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_1", nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_1",single_step_neuron()),

                ("conv_1_1 ",nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_1_1_1", nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_1_1_1",single_step_neuron()),
                ("pool_1_1",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_2",nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_2",nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_2",single_step_neuron()),

                ("conv_2_2",nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_2_2",nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_2_2",single_step_neuron()),
                ("pool_2",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_3",nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_3",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_3",single_step_neuron()),

                ("conv_4",nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_4",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_4",single_step_neuron()),
                ("pool_3",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                
                ("conv_5",nn. Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_5",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_5",single_step_neuron()),

                ("conv_6",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_6",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_6",single_step_neuron()),
                ("pool_6",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_7",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_7",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_7",single_step_neuron()),

                ("conv_8",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_8",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_8",single_step_neuron()),
                ("pool_8",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                ])
                )

        elif self.args.architecture == "vgg_16":

            self.features = nn.Sequential(
                OrderedDict(
                [
                ("conv_1 ",nn.Conv2d(num_of_inChannels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_1", nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_1",single_step_neuron()),

                ("conv_1_1 ",nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_1_1_1", nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_1_1_1",single_step_neuron()),
                ("pool_1_1",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_2",nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_2",nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_2",single_step_neuron()),

                ("conv_2_2",nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_2_2",nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_2_2",single_step_neuron()),
                ("pool_2",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_3",nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_3",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_3",single_step_neuron()),

                ("conv_3_1",nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_3_1",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_3_1",single_step_neuron()),

                ("conv_4",nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_4",nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_4",single_step_neuron()),
                ("pool_3",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                
                ("conv_5",nn. Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_5",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_5",single_step_neuron()),

                ("conv_5_1",nn. Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_5_1",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_5_1",single_step_neuron()),

                ("conv_6",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_6",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_6",single_step_neuron()),
                ("pool_6",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),

                ("conv_7",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_7",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_7",single_step_neuron()),

                ("conv_7_1",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_7_1",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_7_1",single_step_neuron()),

                ("conv_8",nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                ("bn_8",nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)),
                ("act_8",single_step_neuron()),
                ("pool_8",nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
                ])
            )
        if self.args.fc_classifier:
            self.classifier = nn.Sequential(
                            Flatten(),
                            nn.Linear((self.img_size//32)*(self.img_size//32)*512, 256, bias=False),
                            nn.BatchNorm1d(256, eps=1e-4, momentum=0.1, affine=True),
                            nn.Linear(256, self.num_cls, bias=False),
                            single_step_neuron())
        else:
            self.classifier = nn.Sequential(OrderedDict (
                                [("conv_classi",nn.Conv2d(512,num_classes,kernel_size=1, stride=1, padding=1, bias=bias)),
                                ("bn_classi",nn.BatchNorm2d(num_classes, eps=1e-4, momentum=0.1, affine=True)),
                                ("act_classi",single_step_neuron()),
                                ]
                                ))

        if init_weights:
            self._initialize_weights(num_of_inChannels)
    
    def _initialize_weights(self,num_of_inChannels):
        print("kkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        if self.args.initPretrainedImgWeights is not None:

            isFirstLayer = True

            checkpoint = torch.load(self.args.initPretrainedImgWeights)
            print("llllllllllllllllll ")
            
            for name,param in self.features.named_parameters():
                #if "act_" in name:
                #    continue
                if isFirstLayer:
                    isFirstLayer = False
                    continue
                temp_name = "model.features." + name
                print(temp_name)
                if temp_name in checkpoint['state_dict']:
                    #print(checkpoint['state_dict'][temp_name])
                    print(temp_name,"present in reluWeights")
                    param.data = checkpoint['state_dict'][temp_name]
            
            first_conv_out_channels = self.features[0].out_channels
            self.features[0] = nn.Conv2d(num_of_inChannels, first_conv_out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            torch.nn.init.xavier_uniform(self.features[0].weight,gain=2)
            if self.features[0].bias is not None:
                nn.init.constant_(self.features[0].bias, 0)
            
            for name,param in self.classifier.named_parameters():
                if "act_" in name:
                    continue
                temp_name = "model.classifier." + name
                if temp_name in checkpoint['state_dict']:
                    #print(checkpoint['state_dict'][temp_name])
                    print(temp_name,"present in reluWeights")
                    param.data = checkpoint['state_dict'][temp_name]
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    #nn.init.kaiming_normal_(m.weight)
                    m.threshold = 1.0
                    torch.nn.init.xavier_uniform(m.weight,gain=2)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m,nn.Linear):
                    m.threshold = 1.0
                    torch.nn.init.xavier_uniform(m.weight,gain=2)

    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()
            return hook
        
        self.hooks = {}
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))
                
    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
        
    def get_nz_numel(self):
        return self.nz, self.numel
    
    def event_cube_forward(self,x):

        if x.dim() != 5:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
        
        z_seq_step_by_step = []

        for t in range(self.TimeSteps):
            out = x[t]
            out = self.features(out)
            out = self.classifier(out)
            z_seq_step_by_step.append(out.unsqueeze(0))

        z_seq_step_by_step = torch.cat(z_seq_step_by_step, 0)

        if not self.args.fc_classifier:
            z_seq_step_by_step = z_seq_step_by_step.flatten(start_dim=-2).sum(-1)
            
        z_seq_step_by_step = z_seq_step_by_step.permute(1,0,2)
        #z_seq_step_by_step = z_seq_step_by_step.sum(dim=1)
        #z_seq_step_by_step = self.lastbntt(z_seq_step_by_step)
        
        #functional.reset_net(self.features.reset_net) #this to reset the voltages of all the neurons in spikejelly
        #functional.reset_net(self.classifier)

        return z_seq_step_by_step
    
    def forward(self,x):
        
        self.reset_nz_numel()

   
        if self.representationType == 1 or self.representationType == 3: # voxel grid
            if x.dim() != 4:
                assert 'not valid input , for voxel grid [N,Tbins,H,W]'
            out = x
            out = self.features(out)
            out = self.classifier(out)
            if not self.args.fc_classifier:
                out = out.flatten(start_dim=-2).sum(-1)
            #out = self.lastbntt(out)
            return out
        elif self.representationType == 2: #
            if x.dim() != 6:
                assert 'not valid input , for mean stad represent [N,6,H,W]'
            out = x
            out = self.features(out)
            out = self.classifier(out)
            if not self.args.fc_classifier:
                out = out.flatten(start_dim=-2).sum(-1)
            #out = self.lastbntt(out)

class ClassificationLitModule(pl.LightningModule):
    def __init__(self, model,cfg,epochs=10, lr=5e-3,num_classes=4):
        super().__init__()
        self.save_hyperparameters()
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes
        self.cfg = cfg
        self.model = model
        self.test_step_num = 0
        self.all_nnz, self.all_nnumel = 0, 0

        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes)
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_f1score = torchmetrics.F1Score(num_classes=num_classes)
        self.test_mcc = torchmetrics.MatthewsCorrCoef(num_classes=num_classes)
        self.val_f1score = torchmetrics.F1Score(num_classes=num_classes)
        self.val_mcc = torchmetrics.MatthewsCorrCoef(num_classes=num_classes)
        self.train_f1score = torchmetrics.F1Score(num_classes=num_classes)
        self.train_mcc = torchmetrics.MatthewsCorrCoef(num_classes=num_classes)

    def forward(self, x):
        
        retVal = self.model(x)
        print("input shape for the classLit Forward ",x.shape)
        print("ClassLit return value shape = ",retVal.shape)
        return retVal
    
    def step(self, batch, batch_idx, mode):
        events, target,idx = batch

        print("batch idx printingggggggggggggggg ",batch_idx)
        #print("batch_shape  ",batch[0].shape)
        #print("batch target shape  ",batch[1].shape)
        print("printing target ",target)
        outputs = self(events)
        print("target = ", target.shape)
        print("outputsttttttttt ",outputs)
        
        if mode == "test":
            output_save_f_name = f'./output_tensors/output_tensor-{self.test_step_num}'
            target_save_f_name = f'./target_tensors/target_tensor-{self.test_step_num}'
            torch.save(outputs,output_save_f_name)
            torch.save(target,target_save_f_name)
        
        # Measure sparsity if testing
        #self.process_nz(self.model.get_nz_numel())
        
        loss = nn.functional.cross_entropy(outputs, target)
        #Iweight=www)
        print("loss is ",loss)

        sm_outputs = outputs.softmax(dim=-1)
        print("printing softmax outputs =",sm_outputs)

        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')
        f1score = getattr(self, f'{mode}_f1score')
        mccscore = getattr(self, f'{mode}_mcc')

        #print("sm outputs ",sm_outputs)
        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)
        f1score(sm_outputs,target)
        mccscore(sm_outputs,target)
        
        if mode != "test":
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=(mode == "train"))
        if mode == "test":
            mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
            acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()

            for i,acc_i in enumerate(acc_by_class):
                self.log(f'{mode}_acc_{i}', acc_i)
                self.log(f'{mode}_acc', acc)
            
            print(f"{mode} accuracy: {100*acc:.2f}%")
            print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}%") #- corrosion {100*acc_by_class[3]:.2f}%")
            mode_acc.reset()
            mode_acc_by_class.reset()

        functional.reset_net(self.model)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")
        
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")
    
    def test_step(self, batch, batch_idx):
        self.test_step_num += 1
        return self.step(batch, batch_idx, mode="test")
    
    def on_mode_epoch_end(self, mode):
        print()
        mode_acc, mode_acc_by_class,mode_f1_score,mode_mcc = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class"),getattr(self, f"{mode}_f1score"),getattr(self, f"{mode}_mcc")
        acc, acc_by_class,f1_score,mcc_score = mode_acc.compute(), mode_acc_by_class.compute(), mode_f1_score.compute(),mode_mcc.compute()
        for i,acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{i}', acc_i)
        
        self.log(f'{mode}_f1score',f1_score)
        self.log(f'{mode}_mcc',mcc_score)
        
        self.log(f'{mode}_acc', acc)
        
        print(f"{mode} accuracy: {100*acc:.2f}%")
        print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}%") # - corrosion {100*acc_by_class[3]:.2f}%")
        mode_acc.reset()
        mode_acc_by_class.reset()
        
        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        print(f"{mode}: f1score - {f1_score}%",)
        print(f"{mode}: mccscore - {mcc_score}%",)

        #self.log(f'{mode}_confmat', confmat)
        print(confmat)
        self_confmat.reset()
        mode_f1_score.reset()
        mode_mcc.reset()

        if mode =="test":
            sparsity = self.all_nnz / self.all_nnumel
            per_sparsity = 100 * self.all_nnz / self.all_nnumel
            print(f"{mode}_sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            print(f"{mode}_sparsity: {sparsity}({per_sparsity}%)")
            self.log(f"{mode}_sparsity : ",per_sparsity)
            self.all_nnz, self.all_nnumel = 0, 0
    
    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            #print("aaaaaaaaaaaaaaaaaaaaaaaaa ",module)
            if "act" in module:
                #print("bbbbbbbbbbb  1   1bbbbbbbbbbbb ",)
                nnumel = numel[module]
                if nnumel != 0:
                    #print("ccccccccccccccccccccccccccccccc ")
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel
            print("ddddddddddddddddddddddddddddddd ",self.all_nnumel)

    def on_test_epoch_start(self):
        self.model.add_hooks()

    def on_train_epoch_start(self):
        return#self.model.add_hooks()
    
    def on_train_epoch_end(self):
        return self.on_mode_epoch_end(mode="train")

    def on_test_epoch_end(self):
        return self.on_mode_epoch_end(mode="test")
    
    def on_validation_epoch_end(self):
        return self.on_mode_epoch_end(mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-device',default=0,type=int,help='device')
    parser.add_argument('-no_train',action='store_false',help="once this arg is added train will not run",dest='train')
    parser.add_argument('-test',action='store_true',help="once add this arg test will run")
    parser.add_argument('-pretrained',default=None,type=str,help='path to pretrained model')
    parser.add_argument('-data_build_only',action='store_true',help='for building the dataset only ')
    parser.add_argument("--learning_rate", type=float, default=0.001)   #0.02)#    1e-4)
    parser.add_argument('-epochs',type = int, default=27)
    parser.add_argument("--step_size", type=int, default=15)
    parser.add_argument("-fc_classifier",action='store_true',help="fc clissifier")
    parser.add_argument("-network_output",type=int,default=0,help='type of output voltage')
    parser.add_argument("-no_crack",action='store_true',help="once add use data without crack")
    parser.add_argument("-no_corrosion",action='store_true',help="once add use data without corrosion")
    parser.add_argument("-representationType",type=int,default=1,help="chose voxel grid or cube")
    parser.add_argument("-architecture",type=str,default="vgg_16",help="set the desired backbone for the classifier ")
    parser.add_argument( "-voxel_channels",type=int,default=10,help="number of channels for voxel grid")
    parser.add_argument("-saveNameFinalModel",type=str,default = "final.pth",help = "Save name of the final model")
    parser.add_argument("-inputRes",type=int,default = 128,help = "input spatial dimesion")
    parser.add_argument("-trainBatchSize",type=int,default=8,help="train batch size")
    parser.add_argument("-testBatchSize",type=int,default=4,help="test batch size")
    parser.add_argument("-isLocalRun",action='store_true',help="run locally")
    parser.add_argument("-trainCsvFile",type=str,default="train.csv",help="train csv file name")
    parser.add_argument("-testCsvFile",type=str,default="test.csv",help="test csv file name")
    parser.add_argument("-numOfClasses",type=int,default=4,help="number of classes ")
    parser.add_argument("-notInitWeights",action='store_false',help="call init weight function ")
    parser.add_argument("-initPretrainedImgWeights",default=None,type=str,help='path to pretrained model')

    args = parser.parse_args()
    print(args)
    
    callbacks=[]

    ckpt_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f"ckpt-damage-classifier-vgg/",
            filename=f"damage-classifier-vgg" + "-{epoch:02d}-{val_acc:.4f}",
            save_top_k=3,
            mode='min',
    )
    callbacks.append(ckpt_callback)

    logger = None
    try:
        comet_logger = CometLogger(
        api_key=None,
        project_name=f"classif-damage-classifier-vgg/",
        save_dir="comet_logs",
        log_code=True,
        )
        logger = comet_logger
    except ImportError:
        print("Comet is not installed, Comet logger will not be available.")
        

    trainer = pl.Trainer(
            accelerator='gpu',devices=[0],gradient_clip_val=1., max_epochs=args.epochs,
            limit_train_batches=1., limit_val_batches=1.,
            check_val_every_n_epoch=1,
            deterministic=False,
            precision=16,
            callbacks=callbacks,
            logger=logger,
    )

        # Required arguments for ClassificationDataSet and Models
         
    image_shape = (346,260)

    initWeights = args.notInitWeights
    number_of_classes = args.numOfClasses
    sampleSize = 50000 #in nano seconds
    isCreateSingleFileForDataset = False

    numChannels = args.voxel_channels #for voxel grid

    train_batch_size = args.trainBatchSize
    validation_batch_size = args.testBatchSize
    test_batch_size = args.testBatchSize

    #Access Datasets and Data Loaders
    
    train_transform = Tr.Compose([Tr.RandomRotation(5),Tr.RandomHorizontalFlip(p=0.5),Tr.RandomVerticalFlip(p=0.5)])
    folder_path_for_event_data = "/work3/kniud/CUSTOM_DATASET/EVENTS/" #"/work3/kniud/Voxel_grid/ClassificationSetNew_old_Full/" 
    #folder_path_for_event_data =  "/media/dtu-neurorobotics-desk2/data_2/ClassificationSetNew_old_Full/tunnel_image_events/" #"/work3/kniud/Voxel_grid/tunnel_event_extracted/"
    #folder_path_for_event_data = "/work3/kniud/Voxel_grid/ClassificationSetNew_old_Full/" 
    if args.isLocalRun:
        folder_path_for_event_data  = "/home/udayanga/ResearchWork/SNN/selected_train_test/ClassificationSetNew_old_Full/"

    trainDataSet = ClassificationDataset(args,data_folder_path = folder_path_for_event_data,dataSetName = "train_dataset",image_shape = image_shape,transform = train_transform,numberOfChannels = numChannels ,mode = "train")
    train_dataloader = DataLoader(trainDataSet, batch_size=train_batch_size, num_workers=4,shuffle=True,drop_last=True)

    testDataSet = ClassificationDataset(args,data_folder_path = folder_path_for_event_data,dataSetName = "test_dataset",image_shape = image_shape,transform = None,numberOfChannels = numChannels ,mode = "test")
    test_dataloader = DataLoader(testDataSet, batch_size=validation_batch_size, num_workers=4,drop_last=True)

    #tunnelDataSet = v2EDamageDataSet("test",tunnel_event_data,test_tunnel_csv,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1)
    #tunnel_Dataloader = DataLoader(tunnelDataSet, batch_size=8, num_workers=4,drop_last=True)
        
    #lowIllumFolderPath ="/home/udayanga/ResearchWork/Increased_illumination_Images/high_illum_images" #"/home/udayanga/ResearchWork/Reduced_Illumination_images/low_illum_test/"
    #highIllumFolderPath = ""
    #lowIllumCsvFile = lowIllumFolderPath + "/damageDark.csv"
    #highIllumCsvFile = ""

    #testDataSet_lowIllum = v2EDamageDataSet("test",lowIllumFolderPath,lowIllumCsvFile,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1)
    #test_dataloader_lowIllum = DataLoader(testDataSet, batch_size=8, num_workers=4)

    #testDataSet_lowIllum = v2EDamageDataSet("test",highIllumFolderPath,highIllumCsvFile,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1)
    #test_dataloader_lowIllum = DataLoader(testDataSet, batch_size=8, num_workers=4,drop_last=True)

    #testDataSet3 = v2EDamageDataSet("test",folder_path_2,testCSVFile3,representationType,image_shape,"v",numOfTimeSteps = numOfTimeSteps1,sampleSize = 50000,numOfTbins = numOfTbins1)
    #test_dataloader3 = DataLoader(testDataSet3, batch_size=8, num_workers=4,drop_last=True)


    model = MultiStepVGGSNN(numChannels,num_classes=number_of_classes,args=args,init_weights=initWeights)
    
    module = ClassificationLitModule(model,cfg = args,epochs=args.epochs, lr=args.learning_rate,num_classes=number_of_classes)
    if args.pretrained is not None: # loading a pretrained module
        print("mmmmmmmmmmmmmmmmmmmmmmmmmm ")
        ckpt_path = args.pretrained
        module = module.load_from_checkpoint(checkpoint_path=ckpt_path,strict=False) 
        #checkpoint = torch.load(ckpt_path)
        #module.model.load_state_dict(torch.load(ckpt_path))

    if args.train:
        trainer.fit(module,train_dataloader,test_dataloader)
        torch.save(model.state_dict(),"final_model.pth")
    if args.test:
        test_dataloader = DataLoader(testDataSet, batch_size=test_batch_size, num_workers=4)
        trainer.test(module,test_dataloader)


if __name__ == '__main__':
    main()
