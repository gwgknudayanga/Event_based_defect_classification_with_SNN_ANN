import os
import random
import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image

import numpy as np
#from numpy.lib.recfunctions import structured_to_unstructured


class DamageImagesDataSet(Dataset):
    def __init__(self,mode,dataSet_name,folder_path,annotations_file,transform = None):

        self.root = folder_path + '/' + mode
        self.dataSet_name = dataSet_name
        self.mode = mode
        #self.dataSetSize = sum([len(files) for r,d,files in os.walk(self.root)])
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
            
    def __getitem__(self, index):
        
        image_full_path = os.path.join(self.root, self.img_labels.iloc[index, 0])
        image = Image.open(image_full_path).convert("RGB") #read image convert the image to a tensor
        label = self.img_labels.iloc[index,1]
        if self.transform:
            image = self.transform(image)
        return image, label

    def getWeightsForEachLabel(self):
        labels_pd = self.img_labels.iloc[:,1]
        label_array = labels_pd.to_numpy()
        weights = np.bincount(label_array)
        print("weights ... ",weights)
        weights = 1/weights
        weights = weights/weights.sum()
        return weights

    def __len__(self):
        #return len(self.samples)
        return len(self.img_labels)

"""def main():

    train_transform = T.Compose([T.Resize((128, 128)),T.ToTensor(),T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
    test_transform = T.Compose([T.Resize((128, 128)),T.ToTensor(),T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])

    mode = "train"
    dataSetName = "img_test_dataset"
    img_parent_folder = "/media/dtu-neurorobotics-desk2/data_2/RGBSelected"
    annotation_file = img_parent_folder + "/train/damageClassesTrainImgData.csv"
    
    test_dataset = DamageImagesDataSet(mode,dataSetName,img_parent_folder,annotation_file,test_transform)
    test_dataset.getWeightsForEachLabel()
    print(test_dataset.__len__())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,shuffle=True)
    test_dataset[0]
    #imgs,labels = next(iter(test_dataloader))

    
    #imgs,labels = test_dataloader[0]
if __name__ == '__main__':
    main()"""
