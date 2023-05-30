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

cfgs = {'A' : [64,'M',128,'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
         'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}


class VGGSNN(nn.Module):
    def __init__(self,num_of_inChannels,cfg,norm_layer=None,num_classes=4,init_weights=True,single_step_neuron:callable = None,**kwargs):
        super(VGGSNN,self).__init__()

        self.out_channels = []
        self.idx_pool = [i for i,v in enumerate(cfg) if v=='M']
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer,nn.Identity)
        """self.features = self.make_layers(num_of_inChannels,cfg=cfg,norm_layer=norm_layer,lifNeuron=single_step_neuron,bias=bias,**kwargs)"""

        affine_flag = True
        self.conv1 = nn.Conv2d(num_of_inChannels, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt1 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif1 = single_step_neuron(**kwargs)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt2 = nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif2 = single_step_neuron(**kwargs)

        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt3 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif3 = single_step_neuron(**kwargs)

        self.conv4 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt4 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif4 = single_step_neuron(**kwargs)

        self.pool4 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt5 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif5 = single_step_neuron(**kwargs)
        
        self.conv6 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt6 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif6 = single_step_neuron(**kwargs)

        self.pool6 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv7 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt7 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif7 = single_step_neuron(**kwargs)

        self.conv8 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt8 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif8 = single_step_neuron(**kwargs)

        self.pool8 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv9 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt9 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif9 = single_step_neuron(**kwargs)

        self.conv10 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt10 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif10 = single_step_neuron(**kwargs)

        self.pool10 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv11 = nn.Conv2d(512,num_classes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bntt11 = nn.BatchNorm2d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag)
        self.lif11 = single_step_neuron(**kwargs)

        self.conv_list = [self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8, self.conv9,self.conv10,self.conv11]
        self.bntt_list = [self.bntt1,self.bntt2,self.bntt3,self.bntt4,self.bntt5,self.bntt6,self.bntt7,self.bntt8,self.bntt9,self.bntt10,self.bntt11]
        self.lif_list = [self.lif1,self.lif2,self.lif3,self.lif4,self.lif5,self.lif6,self.lif7,self.lif8,self.lif9,self.lif10,self.lif11]
        self.pool_list = [False,self.pool2,False,self.pool4,False,self.pool6,False,self.pool8,False,self.pool10,False]

        self.lastbntt = nn.BatchNorm1d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag)

        if init_weights:
            self._initialize_weights()
    
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def make_layers(self,num_of_inChannels,cfg,norm_layer,lifNeuron,bias,**kwargs):
        layers = []
        channel_in_info = num_of_inChannels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
                self.out_channels.append(channel_in_info)
            else:
                #this may be unwrapped later        
                layers.append(nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0))
                layers.append(nn.Conv2d(channel_in_info, v, kernel_size=(3, 3), stride=(1, 1), bias=False))
                layers.append(norm_layer(v))
                layers.append(lifNeuron(**kwargs))
                #if (v > 64):
                #    layers.append(nn.Dropout(0.1))
                channel_in_info = v
                
        self.out_channels = self.out_channels[2:]
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
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
            """elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)"""
    
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


class MultiStepVGGSNN(VGGSNN):
    def __init__(self, num_of_input_channels, cfg, norm_layer=None, num_classes=4, init_weights=True, timeSteps : int = None,
                 multi_step_neuron: callable = None, **kwargs):
        self.TimeSteps = timeSteps
        super().__init__(num_of_input_channels, cfg, norm_layer, num_classes, init_weights,
                 multi_step_neuron, **kwargs)
                 
    """def forward(self,x,classify = True):
        x_seq = x
        if x.dim() != 5:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
        
        if classify:
            #x_seq = functional.seq_to_ann_forward(x, self.features[0])
            z_seq_step_by_step = []
            for t in range(self.TimeSteps):
                x = x_seq[t]
                y = self.features(x)
                z = self.classifier(y)
                z_seq_step_by_step.append(z.unsqueeze(0))
            z_seq_step_by_step = torch.cat(z_seq_step_by_step, 0)
            z_seq_step_by_step = z_seq_step_by_step.flatten(start_dim=-2).sum(-1)
            return z_seq_step_by_step"""
        
    def forward(self,x):
        x_seq = X
        if x.dim() != 5:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
        z_seq_step_by_step = []
        for t in range(self.TimeSteps):

            out = x[t]
            ll = [i for i in range(0,5)] # need to change for different time steps
            random.shuffle(ll)

            for i in range(len(self.conv_list)):

                out = self.conv_list[i](out)

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out)
                
                if (self.bntt_list[i] is not False):
                    out = self.bntt_list[i](out)

                if self.lif_list[i] is not False:
                    out = self.lif_list[i](out)

            z_seq_step_by_step.append(out.unsqueeze(0))

        z_seq_step_by_step = torch.cat(z_seq_step_by_step, 0)
        print("output of forward method shape ",z_seq_step_by_step.shape)
        z_seq_step_by_step = z_seq_step_by_step.flatten(start_dim=-2).sum(-1)
        z_seq_step_by_step = z_seq_step_by_step.permute(1,0,2)
        z_seq_step_by_step = z_seq_step_by_step.sum(dim=1)
        #z_seq_step_by_step = self.lastlin(z_seq_step_by_step)
        z_seq_step_by_step = self.lastbntt(z_seq_step_by_step)

        return z_seq_step_by_step


class ClassificationDataset(Dataset):
    def __init__(self, mode,image_shape,dataSet_name,file_path,numOfTimeSteps = 5,sampleSize = 97620,numOfTbins = 2):
        self.mode = mode #whether test or train
        self.tbin = numOfTbins
        self.C, self.T = self.tbin * 2, numOfTimeSteps
        self.sample_size = sampleSize
        self.quantization_size = [self.sample_size//self.T,1,1] #sample length for one time step
        self.w, self.h = image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]
        
        save_file_name = f"{dataSet_name}_{self.mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        save_file = os.path.join(file_path, save_file_name)
        
        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(file_path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}.")
            
    def __getitem__(self, index):

        coords, feats, target = self.samples[index]
        sample = torch.sparse_coo_tensor(coords.t(), feats.to(torch.float32)).coalesce()
        sample = sample.sparse_resize_(
            (self.T, sample.size(1), sample.size(2), self.C), 3, 1
        ).to_dense().permute(0,3,1,2)                     #(T,C,W,H)
        
        sample = Tr.Resize((64,64), Tr.InterpolationMode.NEAREST)(sample) 
        
        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, data_dir, save_file):
        raise NotImplementedError("The method build_dataset has not been implemented.")


import h5py

def load_v2e_event_data(filename,augmentedType,isTrainMode=True):
    """Load V2E Events, all HDF5 records."""
    event_segment_list = []
    assert os.path.isfile(filename)

    v2e_data = h5py.File(filename, "r")
    events = v2e_data["events"][()]
    events_1 = events[events[:,0] < 50000]
    event_segment_list.append(events_1)
    if (isTrainMode):
        if augmentedType == 1: #horizontal flip second segment
            events_2 = events[events[:,0] <= 50000]
            events_2[:,1] = 345 - events_2[:,1]
            event_segment_list.append(events_2)
        elif augmentedType == 3: #horizontal flip of first segment and vertical flip second
            events_2 = events[events[:,0] < 50000]
            events_2[:,1] = 345 - events_2[:,1]
            events_3 = events[events[:,0] < 50000]
        
            events_3[:,2] = 259 - events_3[:,2]
            event_segment_list.append(events_2)
            event_segment_list.append(events_3)

        


    return event_segment_list

class v2EDamageDataSet(ClassificationDataset):
    def __init__(self,mode,data_folder_path,image_shape = (346,260),dataSet_name = "v",numOfTimeSteps = 5,sampleSize = 97620,numOfTbins=2):
        super().__init__(mode,image_shape,dataSet_name,data_folder_path,numOfTimeSteps,sampleSize)
        self.mode = mode
    def build_dataset(self,data_dir,save_file):
        print("building v2e damage dataset ....")
        
        classIDvsClassNameDict = {}
        classes_dir = [os.path.join(data_dir,class_name) for class_name in os.listdir(data_dir)]
        samples = []
        current_max = 0
        print("class dir ",classes_dir)
        for class_id,class_dir in enumerate(classes_dir):
            self.files = [os.path.join(class_dir,time_seq_name) for time_seq_name in os.listdir(class_dir)]
            target = class_id
            print(f'Building the class id {class_id} and class dir {class_dir}')

            for file_name in self.files:
                #print("processing ",file_name)
                augmentedType = 0
                isTrainMode = True
                if(self.mode == "test"):
                    isTrainMode = False

                event_data_list = load_v2e_event_data(file_name,class_id,isTrainMode)
                
                for events in event_data_list:
                    
                    #print("printing events ...",events)
                    #print("events shape ",events.shape)
                    print("number of events",events.size)
                    if(events.size == 0):
                        print("Empty sample ....", file_name, "for class",class_id)
                        continue

                    # Bin the events on T timesteps
                    coords = events[:,0:3]
                    #print("coords 1 0 ",coords)
                    coords = torch.floor(coords/torch.tensor(self.quantization_size)) 
                    coords[:, 1].clamp_(min=0, max=self.quantized_w-1)
                    coords[:, 2].clamp_(min=0, max=self.quantized_h-1)
                    #print("coords 2 = ",coords)
                    if (coords[:,0].max() > 4):
                        print("exceeding time bins to ",coords[:,0].max()," ",file_name)
                    # TBIN computations
                    #else:
                    #    print("max time ",coords[:,0].max())
                    tbin_size = self.quantization_size[0] / self.tbin
                    tbin_coords = (events[:,0] % self.quantization_size[0]) // tbin_size
                    tbin_feats = (2* tbin_coords) + events[:,3]
                    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*self.tbin).to(bool)
                    
                    samples.append([coords.to(torch.int16), feats.to(bool), target])
        return samples

class ClassificationLitModule(pl.LightningModule):
    def __init__(self, model, epochs=10, lr=5e-3, num_classes=4):
        super().__init__()
        self.save_hyperparameters()
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes

        self.model = model

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
    
    def forward(self, x):
        
        x = x.permute(1,0,2,3,4)
        
        retVal = self.model(x)
        print("input shape for the classLit Forward ",x.shape)
        print("ClassLit return value shape = ",retVal.shape)
        #retVal = self.model(x).sum(dim=1)
        #retVal = retVal.permute(1,0,2)
        #retVal = retVal.sum(dim=1)
        return retVal
    
    def step(self, batch, batch_idx, mode):
        events, target = batch
        #print("batch_shape  ",batch[0].shape)
        #print("batch target shape  ",batch[1].shape)
        #print("printing target ",target)
        outputs = self(events)
        #print("eventssssssss = ", events.shape)
        print("outputsttttttttt ",outputs.shape)
        
        loss = nn.functional.cross_entropy(outputs, target)
        
        outputs.softmax(dim=-1).argmax(dim=-1)[target == 2]

                
        # Measure sparsity if testing
        #if mode=="test":
        #    self.process_nz(self.model.get_nz_numel())

        # Metrics computation
        sm_outputs = outputs.softmax(dim=-1)
        print("printing softmax outputs =",sm_outputs)

        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')

        print("sm outputs ",sm_outputs)
        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)

        if mode != "test":
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=(mode == "train"))
        if mode == "test":
            mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
            acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
            for i,acc_i in enumerate(acc_by_class):
                self.log(f'{mode}_acc_{i}', acc_i)
                self.log(f'{mode}_acc', acc)

            print(f"{mode} accuracy: {100*acc:.2f}%")
            print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}% - corrosion {100*acc_by_class[3]:.2f}% -  efflores {100*acc_by_class[4]:.2f}%")
            mode_acc.reset()
            mode_acc_by_class.reset()
            

        functional.reset_net(self.model)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")

    def on_mode_epoch_end(self, mode):
        print()
        mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
        for i,acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{i}', acc_i)
        self.log(f'{mode}_acc', acc)

        print(f"{mode} accuracy: {100*acc:.2f}%")
        print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}% - corrosion {100*acc_by_class[3]:.2f}% -  efflores {100*acc_by_class[4]:.2f}%")
        mode_acc.reset()
        mode_acc_by_class.reset()
        """
        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        self.log(f'{mode}_confmat', confmat.mean())
        print(confmat.mean())
        self_confmat.reset()"""

        if mode=="test":
            print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0

    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            if "act" in module:
                nnumel = numel[module]
                if nnumel != 0:
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def on_train_epoch_end(self):
        return self.on_mode_epoch_end(mode="train")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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
    args = parser.parse_args()
    print(args)

    folder_path_for_event_data  = "/home/udayanga/ResearchWork/SNN/selected_train_test/" #"/home/dtu-neurorobotics-desk2/Research_Work/Synthesized_event_dataSets/selected_event_train_test/"
    trainDataSet = v2EDamageDataSet("train",folder_path_for_event_data,numOfTimeSteps = 5,sampleSize = 50000,numOfTbins = 2)
    train_dataloader = DataLoader(trainDataSet, batch_size=16, num_workers=4, shuffle=True)
    testDataSet = v2EDamageDataSet("test",folder_path_for_event_data,numOfTimeSteps = 5,sampleSize = 50000,numOfTbins = 2)
    test_dataloader = DataLoader(testDataSet, batch_size=8, num_workers=4)

    if args.data_build_only:
        return
    
    from pytorch_lightning.callbacks import ModelCheckpoint

    from pytorch_lightning.loggers import CometLogger

    callbacks=[]
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"ckpt-damage-classifier-vgg/",
        filename=f"damage-classifier-vgg" + "-{epoch:02d}-{train_acc:.4f}",
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
        accelerator='gpu', devices=[0], gradient_clip_val=1., max_epochs= 10,
        limit_train_batches=1., limit_val_batches=1.,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=16,
        callbacks=callbacks,
        logger=logger,
    )

    norm_lyr = nn.BatchNorm2d
    ms_neuron = neuron.ParametricLIFNode #step_mode='m',backend='cupy') #accelerate the processing in GPU with cupy
    channels_per_timeStep = 4
    number_of_classes = 5
    model = MultiStepVGGSNN(channels_per_timeStep,cfg = cfgs['A'],norm_layer=norm_lyr,num_classes=number_of_classes,init_weights=True,timeSteps=5,multi_step_neuron = ms_neuron,step_mode='s')
    module = ClassificationLitModule(model, epochs=5, lr=5e-3,num_classes=number_of_classes)

    if args.pretrained is not None:
        ckpt_path = args.pretrained
        module = module.load_from_checkpoint(checkpoint_path=ckpt_path,strict=False)
        #checkpoint = torch.load(ckpt_path)
        #model.load_state_dict(checkpoint["state_dict"])

    if args.train:
        trainer.fit(module,train_dataloader,test_dataloader)
    if args.test:
        test_dataloader2 = DataLoader(testDataSet, batch_size=500, num_workers=4)
        trainer.test(module,test_dataloader2)

if __name__ == '__main__':
    main()
    print("dddd eee ")