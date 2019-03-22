import numpy as np

import time
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import BoundingBoxDataset

class ModifiedCNN(nn.Module):
    def __init__(self, class_sizes, attributes, model_pretrained):
        super(ModifiedCNN, self).__init__()
        self.class_sizes = class_sizes
        self.attributes = attributes
        self.model_pretrained = model_pretrained
        self.num_ftrs = model_pretrained.fc.out_features
        # Create N classifiers
        for c, attribute in zip(range(len(self.class_sizes)), attributes):
            self.__setattr__(attribute,
                             nn.Linear(self.num_ftrs, self.class_sizes[c]))
    
    def forward(self, x):
        out = self.model_pretrained(x)
        # returns a dictionary where key indicates attribute c
        return {attribute: self.__getattr__(attribute)(out)
                for c, attribute in zip(range(len(self.class_sizes)), self.attributes)}
    
    def set_multiple_gpus(self):
        # uses mulitple gpu
        if torch.cuda.device_count() > 1:
            self.model_pretrained = nn.DataParallel(self.model_pretrained).cuda()
        else:
            print('One or no GPU detected, so model will not be parallelized')
            
class ModifiedCNNv2(nn.Module):
    def __init__(self, class_sizes, attributes, model_pretrained):
        super(ModifiedCNNv2, self).__init__()
        self.class_sizes = class_sizes
        self.attributes = attributes
        self.model_pretrained = model_pretrained
        self.num_ftrs = model_pretrained.fc.out_features
        # Create N classifiers
        for c, attribute in zip(range(len(self.class_sizes)), attributes):
            self.__setattr__(attribute,
                             nn.Linear(self.num_ftrs, self.class_sizes[c]))
    
    def forward(self, x):
        out = self.model_pretrained(x)
        # returns a dictionary where key indicates attribute c
        return {attribute: F.softmax(self.__getattr__(attribute)(out))
                for c, attribute in zip(range(len(self.class_sizes)), self.attributes)}
    
    def set_multiple_gpus(self):
        # uses mulitple gpu
        if torch.cuda.device_count() > 1:
            self.model_pretrained = nn.DataParallel(self.model_pretrained).cuda()
        else:
            print('One or no GPU detected, so model will not be parallelized')
            
class BoundingBoxCNN(nn.Module):
    def __init__(self, model_pretrained, dropout_rate=None):
        super(BoundingBoxCNN, self).__init__()
        self.model_pretrained = model_pretrained
        self.dropout_rate = dropout_rate
        self.num_ftrs = model_pretrained.fc.out_features
#         self.fc_size = 100
#         self.output_attributes = ['x0', 'y0', 'x1', 'y1']
        # Create N regressors
#         for attribute in self.output_attributes:
#             self.__setattr__(attribute,
#                              nn.Linear(self.fc_size, 1))
    
    def forward(self, x):
        x = self.model_pretrained(x)
        x = nn.Linear(self.num_ftrs, 100)(x)
        x = nn.Relu()(x)
        if dropout_rate:
            x = nn.Dropout(p=dropout_rate)(x)
            x = nn.Relu()(x)
        x = nn.Linear(100, 4)(x)
        return x
        # returns a dictionary where key indicates output name
#         return {attribute: self.__getattr__(attribute)(x)
#                 for attribute in self.output_attributes}
    
    def set_multiple_gpus(self):
        # uses mulitple gpu
        if torch.cuda.device_count() > 1:
            self.model_pretrained = nn.DataParallel(self.model_pretrained).cuda()
        else:
            print('One or no GPU detected, so model will not be parallelized')
    
def train_model(model, dataset, criterion, optimizer, device, num_epochs=25,  batch_size=32, num_workers=4):
    since = time.time()
    
#     model.to(device)
    
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        # Set model to training mode
        model.train()
        
        running_loss = 0.0
#         running_corrects = 0
        
        num_batch = len(dataloader)
        pbar = tqdm(total=num_batch)
        # Iterate over data
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            # track history
            output_dict = model(inputs)
            loss = torch.tensor(0).to(device)
            for output, label in zip(output_dict.values(), labels):
                label = label.to(device)
                loss = torch.add(loss, criterion(output, label))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            pbar.update()
            
#             num_batch += 1
#             if num_batch % 1000 == 0:
#                 print('{} batches processed'.format(num_batch))
        
        pbar.close()
        epoch_loss = running_loss / dataset_size
        
        print('Training Loss: {:.4f}'.format(epoch_loss))
        
        print()
        
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model

def train_bb_model(model, df_train, df_val, root_dir, criterion, optimizer, bootstrap_size=1000,
                   num_epochs=25,  batch_size=32, num_workers=4):
    since = time.time()
    
    if torch.cuda.is_available():
        model.cuda()
    
    dataset_val = BoundingBoxDataset(df=df_val, root_dir=root_dir, transform=True, horizontal_flip=False, vertical_flip=False)
    dataset_val_size = len(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    training_loss_list, val_loss_list = list(), list()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        running_loss_train = 0.0
        running_loss_val = 0.0
#         running_corrects = 0

        
        # Bootstrap training set
        df_train_bootstrapped = df_train.sample(n=bootstrap_size, replace=True, random_state=epoch).reset_index(drop=True)
        
        dataset_train = BoundingBoxDataset(df=df_train_bootstrapped, root_dir=root_dir, transform=True, horizontal_flip=True,
                                           vertical_flip=True)
        dataset_train_size = len(dataset_train)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        ## Training Phase
        num_batch_train = len(dataloader_train)
        pbar = tqdm(total=num_batch_train)
        # Set model to training mode
        model.train()
        # Iterate over data
        for inputs, values in dataloader_train:
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            # track history
            outputs = model(inputs)
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
            for output, value in zip(outputs, values):
                value = value.float()
                if torch.cuda.is_available():
                    value = value.cuda()
                loss = torch.add(loss, criterion(output, value))
            loss.backward()
            optimizer.step()
            
            running_loss_train += loss.item() * inputs.size(0)
            
            pbar.update()
        pbar.close()
        
        ## Validation Phase
        num_batch_val = len(dataloader_val)
        pbar = tqdm(total=num_batch_val)
        # Set model to evaluation mode
        model.eval()
        # Iterate over data
        for inputs, values in dataloader_val:
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            # track history
            outputs = model(inputs)
            loss = torch.tensor(0)
            if torch.cuda.is_available():
                loss = loss.cuda()
            for output, value in zip(outputs, values):
                value = value.float()
                if torch.cuda.is_available():
                    value = value.cuda()
                loss = torch.add(loss, criterion(output, value))
#             loss.backward()
#             optimizer.step()
            
            running_loss_val += loss.item() * inputs.size(0)
                        
            pbar.update()
        pbar.close()        
        
        epoch_loss_train = running_loss_train / dataset_train_size
        epoch_loss_val = running_loss_val / dataset_val_size
        
        training_loss_list.append(epoch_loss_train)
        val_loss_list.append(epoch_loss_val)
        
        print('Training Loss: {:.4f}'.format(epoch_loss_train))
        print('Vallidation Loss: {:.4f}'.format(epoch_loss_val))
        
        print()
        
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, training_loss_list, val_loss_list