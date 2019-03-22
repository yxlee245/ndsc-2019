import numpy as np
import pandas as pd

from tqdm import tqdm
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def pil_loader(path):
    # making sure that path ends with '.jpg'
    if path[-4:] != '.jpg':
        path += '.jpg'
    
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        img = Image.open(file)
        return img.convert('RGB')

class ProductDataset(Dataset):
    '''Product Images dataset'''
    def __init__(self, df, root_dir, encoded_attributes, transform=None):
        '''
            Args:
                df (Pandas dataframe): Dataframe with image path
                root_dir (string): Directory with all the images
                encoded_attributes (list): Attributes encoded using the LabelEncoder class in scikit-learn
                transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.df = df
        self.root_dir = root_dir
        self.encoded_attributes = encoded_attributes
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df['image_path'][idx])
        image = pil_loader(img_name)
        if self.transform:
            image = self.transform(image)
        labels = [attribute[idx] for attribute in self.encoded_attributes]
        return image, labels
    
class BoundingBoxDataset(Dataset):
    '''Dataset to train bounding box model'''
    def __init__(self, df, root_dir, transform=False, horizontal_flip=False, vertical_flip=False):
        '''
            Args:
                df (Pandas dataframe): Dataframe with image path
                root_dir (string): Directory with all the images
                resize_shape (tuple or list): (Width, height) of the resized image
                transform (boolean, optional): Optional transform to be applied on a sample.
                horizontal_flip (boolean, optional): Optional random horizontal flip to be applied to image.
                vertical_flip (boolean, optional): Optional random vertical flip to be applied to image.
        '''
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df['image_path'][idx])
        image = pil_loader(img_name)
        width, height = image.size
        x0 = int(self.df['x0'][idx])
        y0 = int(self.df['y0'][idx])
        x1 = int(self.df['x1'][idx])
        y1 = int(self.df['y1'][idx])
        
        if self.transform:
            width_factor = float(self.df['width_factor'][0])
            height_factor = float(self.df['height_factor'][0])
            x0_scaled, y0_scaled = int(x0 * width_factor), int(y0 * height_factor)
            x1_scaled, y1_scaled = int(x1 * width_factor), int(y1 * height_factor)
            image = transforms.Resize(224)(image)
            # Random Horizontal Flip (50% chance of horizontal flip)
            if self.horizontal_flip:
                np.random.seed(idx)
                flip_boolean = bool(np.random.binomial(1, 0.5))
                if flip_boolean:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    x0_scaled, x1_scaled = width - x1_scaled, width - x0_scaled
            # Random Vertical Flip (50% chance of vertical flip)
            if self.vertical_flip:
                np.random.seed(2 * idx)
                flip_boolean = bool(np.random.binomial(1, 0.5))
                if flip_boolean:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    y0_scaled, y1_scaled = height - y1_scaled, height - y0_scaled
            image = transforms.ToTensor()(image)
            image =  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
            coordinates = [x0_scaled, y0_scaled, x1_scaled, y1_scaled]
        else:
            coordinates = [x0, y0, x1, y1]
        return image, coordinates
    
def array_filter(array, label_encoder, n_top):
    '''
    array - NumPy array containing softmax probabilities on which sorting is based
    label_encoder - Sci-kit Learn Label Encoder object containing the encoding scheme of the labels
    n_top - the number of top elements to find
    '''
    idx = (-array).argsort()[:n_top]
    labels_array = label_encoder.inverse_transform(idx)
    labels_string = ''
    for label in labels_array:
        labels_string += str(label) + ' '
    labels_string = labels_string.strip()
    return labels_string

def make_submission(df_val, model, transforms, root_dir, le_list):
    model.eval()
    num_items = len(df_val)
    itemid_series = df_val['itemid']
    filename_series = df_val['image_path']
    df_submission = pd.DataFrame()
    pbar = tqdm(total=num_items)
    for itemid, filename in zip(itemid_series, filename_series):
        image_path = os.path.join(root_dir, filename)
        val_image_tensor = transforms(pil_loader(image_path)).unsqueeze(0)
        outputs = model(val_image_tensor)
        
        id_list, tagging_list = list(), list()
        for attribute, label_encoder in zip(outputs.keys(), le_list):
            probs = outputs[attribute].cpu().detach().numpy().flatten()
            labels_string = array_filter(probs, label_encoder, 2)
            id_string = str(itemid) + '_' + attribute
            id_list.append(id_string)
            tagging_list.append(labels_string)
        df_submission = df_submission.append(pd.DataFrame({'id': id_list, 'tagging': tagging_list}), ignore_index=True)
        pbar.update()
    pbar.close()
    return df_submission