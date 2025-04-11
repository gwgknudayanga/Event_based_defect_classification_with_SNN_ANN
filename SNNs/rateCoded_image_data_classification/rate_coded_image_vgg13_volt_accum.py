from re import X
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional,neuron,layer
import numpy as np
import torchvision
import torchvision.models as models

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
from dataSet_RGB import DamageImagesDataSet

def poissonGenerator(inp,timeSteps = 10,rescale_fac = 2.0):
    #rand_in = torch.rand_like(inp)
    return torch.mul(torch.le(torch.rand_like(inp) * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))
    """tStepList = []
    
    for t in range(timeSteps):
        rand_in = torch.rand_like(inp,device='cuda')
        tStepList.append(torch.mul(torch.le(rand_in * rescale_fac, torch.abs(inp)).float(), torch.sign(inp)).unsqueeze(0))"""

    return torch.cat(tStepList,0)

class Flatten(torch.nn.Module):
    def forward(self,x):
        batch_size = x.shape[0]
        return x.view(batch_size,-1)

class CustomLIFNode(neuron.LIFNode):
    def __init__(self,step_mode='s',surrogate_function=None):
        super().__init__()
    
    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        # self.neuronal_reset(spike)
        return self.v

class VGGSNN(nn.Module):
    def __init__(self,num_of_inChannels,num_classes=4,args = None,init_weights=True,single_step_neuron : callable = None,**kwargs):
        super(VGGSNN,self).__init__()

        self.nz, self.numel = {}, {}
        self.args = args
        single_step_neuron = neuron.LIFNode

        bias = False
        affine_flag = True
        self.img_size = 128      
        self.num_cls = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn. Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        if self.args.fc_classifier:
            self.classifier = nn.Sequential(
                            Flatten(),
                            nn.Linear((self.img_size//32)*(self.img_size//32)*512, 256, bias=False),
                            nn.BatchNorm1d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
                            nn.Linear(256, self.num_cls, bias=False),
                            CustomLIFNode())
        else:
            self.classifier = nn.Sequential(nn.Conv2d(512,num_classes,kernel_size=1, stride=1, padding=1, bias=bias),
                                nn.BatchNorm2d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag),
                                CustomLIFNode())

        """self.features = nn.Sequential(
            nn.Conv2d(num_of_inChannels, 64, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),
        
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),

            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag),
            single_step_neuron(**kwargs),
            nn.MaxPool2d(kernel_size=2,stride=2))"""

        self.lastbntt = nn.BatchNorm1d(num_classes, eps=1e-4, momentum=0.1, affine=affine_flag)
    
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
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

    def __init__(self, num_of_input_channels, num_classes=4, args = None, timeSteps : int = None,init_weights=True,multi_step_neuron: callable = None, **kwargs):
        self.TimeSteps = timeSteps
        self.args = args
        super().__init__(num_of_input_channels, num_classes,args,init_weights,multi_step_neuron, **kwargs)
    
    def forward(self,x):
      
        z_seq_step_by_step = []

        print("shape of the input to forward ",x.shape)
        for t in range(self.TimeSteps):

            out = poissonGenerator(x)
            out = self.features(out)
            out = self.classifier(out)

            z_seq_step_by_step.append(out.unsqueeze(0))

        z_seq_step_by_step = torch.cat(z_seq_step_by_step, 0)

        #comment following line for fc layers
        if not self.args.fc_classifier:
            z_seq_step_by_step = z_seq_step_by_step.flatten(start_dim=-2).sum(-1)
        z_seq_step_by_step = z_seq_step_by_step.permute(1,0,2)
        #z_seq_step_by_step = z_seq_step_by_step[:,(self.TimeSteps - 1),:] #z_seq_step_by_step.sum(dim=1)/self.TimeSteps #z_seq_step_by_step[:,(self.TimeSteps - 1),:]
        #z_seq_step_by_step = self.lastbntt(z_seq_step_by_step)

        return z_seq_step_by_step

class ClassificationLitModule(pl.LightningModule):
    def __init__(self, model, cfg, epochs=10, lr=5e-3, num_classes=4,timeSteps = 10,focalLossWeightArray = []):
        super().__init__()
        self.save_hyperparameters()
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes
        self.timeSteps = timeSteps
        self.cfg = cfg
        self.model = model
        self.all_nnz, self.all_nnumel = 0, 0
        self.test_step_num = 0

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)

        self.epochCount = 0

        self.focal_loss = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
                model='focal_loss',
                alpha=focalLossWeightArray,
                gamma=2,
                reduction='mean',
                #device='cpu',
                dtype=torch.float32,
                force_reload=False)
    def forward(self, x):
        
        #x = poissonGenerator(x,self.timeSteps)        
        retVal = self.model(x)
        print("input shape for the classLit Forward ",x.shape)
        print("ClassLit return value shape = ",retVal.shape)
        return retVal
    
    def step(self, batch, batch_idx, mode):
        events, target = batch
      
        print("printing target ",target)
        outputs = self(events)
        if mode == "test":

            output_save_f_name = f'./output_tensors/output_tensor-{self.test_step_num}'
            target_save_f_name = f'./target_tensors/target_tensor-{self.test_step_num}'
            torch.save(outputs,output_save_f_name)
            torch.save(target,target_save_f_name)

        #if self.cfg.network_output == 0:
        outputs = outputs[:,(self.timeSteps - 1),:]
        #elif self.cfg.network_output == 1:
        #outputs = outputs.sum(dim=1)/self.timeSteps

        #outputs = outputs.sum(dim=1)/self.timeSteps
        #loss = nn.functional.cross_entropy(outputs, target)
        #print("loss 1 is ",loss)

        loss = self.focal_loss(outputs,target)
        print("loss 2 is ",loss)
        # Measure sparsity if testing
        if mode=="test":
            self.process_nz(self.model.get_nz_numel())

        # Metrics computation
        sm_outputs = outputs.softmax(dim=-1)
        print("printing softmax outputs =",sm_outputs)
    
        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')

        #print("sm outputs ",sm_outputs)
        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)

        #if mode != "test":
        self.log(f'{mode}_loss', loss, on_epoch=True) #, prog_bar=(mode == "train"))
        
        """if mode == "test":
            mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
            acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
            for i,acc_i in enumerate(acc_by_class):
                self.log(f'{mode}_acc_{i}', acc_i)
                self.log(f'{mode}_acc', acc)

            print(f"{mode} accuracy: {100*acc:.2f}%")
            print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}% - corrosion {100*acc_by_class[3]:.2f}%")
            mode_acc.reset()
            mode_acc_by_class.reset()"""
         
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
        mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
        for i,acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{i}', acc_i)
        self.log(f'{mode}_acc', acc)

        print(f"{mode} accuracy: {100*acc:.2f}%")
        print(f"spalling {100*acc_by_class[0]:.2f}% - healthy {100*acc_by_class[1]:.2f}% - crack {100*acc_by_class[2]:.2f}% - corrosion {100*acc_by_class[3]:.2f}% ")
        mode_acc.reset()
        mode_acc_by_class.reset()
        
        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        #self.log(f'{mode}_confmat', confmat.mean())
        print(confmat)
        self_confmat.reset()
        """if mode=="test":
            print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0"""
        
        self.epochCount += 1

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
    
    def on_validation_epoch_end(self):
        return self.on_mode_epoch_end(mode="val")

    def on_test_epoch_end(self):

        return self.on_mode_epoch_end(mode="test")

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),lr=self.cfg.learning_rate)
        if self.cfg.optimizer == 1:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate,momentum=0.9,weight_decay=1e-4) #torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.cfg.epochs,
        )
        if self.cfg.scheduler == 1:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.step_size, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
"""def writeListToFile(outputFileName,desiredList):
    with open(outputFileName, 'w') as fp:
        for item in desiredList:
            # write each item on a new line
            fp.write("%s\n" % item)"""

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)   

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-device',default=0,type=int,help='device')
    parser.add_argument('-no_train',action='store_false',help="once this arg is added train will not run",dest='train')
    parser.add_argument('-test',action='store_true',help="once add this arg test will run")
    parser.add_argument('-pretrained',default=None,type=str,help='path to pretrained model')
    parser.add_argument('-data_build_only',action='store_true',help='for building the dataset only ')
    parser.add_argument('-imageNet_pretrained',action='store_true',help = "import original vgg13 model with pretrained weights")
    parser.add_argument('-epochs',type = int, default=150)
    parser.add_argument("-learning_rate",type=float,default=0.01)   #I0.01)     #5.5e-3)
    parser.add_argument("-step_size",type=int,default = 40)  #40)    #25)
    parser.add_argument("-scheduler",type=int,default = 0)
    parser.add_argument("-optimizer",type=int, default = 0)
    parser.add_argument("-fc_classifier",action='store_true',help="fc clissifier")
    parser.add_argument("-network_output",type=int,default=0,help='neuronal output of last layer')

    args = parser.parse_args()
    print(args)


    train_transform = Tr.Compose([Tr.Resize((128,128)),Tr.ToTensor(),Tr.RandomHorizontalFlip(),Tr.RandomVerticalFlip(),Tr.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
    test_transform = Tr.Compose([Tr.Resize((128,128)),Tr.ToTensor(),Tr.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

    #folder_path_for_data  = "/work3/kniud/rate_Coded/RGBSelected/"
    #folder_path_for_data  = "/work3/kniud/rate_Coded/RGBSelected_New_With_Channel"
    folder_path_for_data = "/home/udayanga/ResearchWork/SNN/selected_train_test/RGBSelected"
    train_annotation_csv = folder_path_for_data + "/train/damageClassesTrainImgData.csv"
    test_annotation_csv = folder_path_for_data + "/test/damageClassesTestImgData.csv"

    g = torch.Generator()
    g.manual_seed(0)

    trainDataSet = DamageImagesDataSet("train","imgs_train_dataset",folder_path_for_data,train_annotation_csv,train_transform)
    train_dataloader = DataLoader(trainDataSet,batch_size=16, num_workers=4, shuffle=True,drop_last=True,worker_init_fn=seed_worker,generator=g)

    testDataSet = DamageImagesDataSet("test","imgs_test_dataset",folder_path_for_data,test_annotation_csv,test_transform)
    test_dataloader = DataLoader(testDataSet, batch_size=8,num_workers=4,shuffle = False,drop_last=True)

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
        accelerator='gpu',devices=[0],gradient_clip_val=1.,max_epochs=args.epochs,
     #   limit_train_batches=1., limit_val_batches=1.,
        check_val_every_n_epoch=10,
        deterministic=False,
        precision=16,
        callbacks=callbacks,
        logger=logger,
    )

    number_of_classes = 4
    num_of_input_channels = 3
    num_of_tSteps = 20   
    ms_neuron = neuron.LIFNode #step_mode='m',backend='cupy') #accelerate the processing in GPU with cupy
    model = MultiStepVGGSNN(num_of_input_channels,number_of_classes,args,num_of_tSteps,init_weights=True,multi_step_neuron = ms_neuron,step_mode='s')
    weight_array_for_focal_loss = trainDataSet.getWeightsForEachLabel()
    module = ClassificationLitModule(model, args, epochs=args.epochs,lr=args.learning_rate,num_classes=number_of_classes,timeSteps = num_of_tSteps,focalLossWeightArray = weight_array_for_focal_loss)
           
    if args.pretrained is not None:
        ckpt_path = args.pretrained
        module = module.load_from_checkpoint(checkpoint_path=ckpt_path,strict=False)
        #checkpoint = torch.load(ckpt_path)
        #model.load_state_dict(checkpoint["state_dict"])
    if args.train:
        trainer.fit(module,train_dataloader,test_dataloader)
        torch.save(model.state_dict(),"./final_model_15_0004_anneal_224_16.pth")
        #print(module.getTrainLossArray())
        #writeListToFile("trainloss.txt",module.getTrainLossArray())
        #writeListToFile("testloss.txt",module.getValLossArray())
    if args.test:
        test_dataloader2 = DataLoader(testDataSet, batch_size=16, num_workers=4,shuffle=False)
        trainer.test(module,test_dataloader2)

if __name__ == '__main__':
    print("run started ...")
    main()
