import os
import torch
import numpy as np
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
import cv2
import albumentations as album
from tqdm import tqdm
import gc
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import segmentation_models_pytorch as smp
from functools import partial

path = 'final-roads/'

os.chdir(path)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Accessing the custom segmentation dataset 

class roadsDataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform=None):
            
            self.image_paths = [os.path.join(folder, image_id) for image_id in os.listdir(folder) if image_id != ".DS_Store"]
            self.mask_paths = [os.path.join(folder + '_labels', image_id) for image_id in os.listdir(folder) if image_id != ".DS_Store"]
                
            self.transform = transform

    def __getitem__(self, i):


        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
        mask = np.where(mask >= 128, 255, 0) // 255
        
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return transforms.ToTensor()(image.astype('float32')), transforms.ToTensor()(mask.astype('float32'))
    
    def __len__(self):

        return len(self.image_paths)


def transform_divisible(image, mask):

    transform = [
        album.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
    ]
    
    return album.Compose(transform)(image=image, mask=mask)

# Creating the training loop for TPE evaluations

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,val_loader,preprocessing,decoderOnly=False):

    for epoch in range(1, n_epochs + 1):

        print('Epoch {}'.format(epoch))

        model.train()
        loss_train = 0.0
        n_batches_train = len(train_loader)
        train_loop = tqdm(train_loader, desc="Training")

        for imgs, masks in train_loop:

            if decoderOnly:
                imgs =  torch.stack([preprocessing(image.permute(1, 2, 0)).permute(2,0,1).float() for image in imgs])
            else:
                imgs = (imgs - imgs.mean())/imgs.std()
                
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():

            loss_val = 0.0
            n_batches_val = len(val_loader)
            val_loop = tqdm(val_loader, desc="Validation")
            
            for imgs, masks in val_loop:
                
                imgs = (imgs - imgs.mean())/imgs.std()
                imgs, masks = imgs.to(device), masks.to(device)
                
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                loss_val += loss.item()

        val_loss_t = loss_val / n_batches_val
        train_loss_t = loss_train / n_batches_train
        
        print('Training loss {} | Validation loss {}'.format(train_loss_t, val_loss_t))
        
    return val_loss_t,train_loss_t


# Custom loss combining BCE Dice and Dice Loss

class CustomLoss(torch.nn.Module):
    
    def __init__(self,alpha):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.loss1 = torch.nn.BCELoss()
        self.loss2 = smp.losses.DiceLoss(mode="binary",from_logits=False)
        
    def forward(self, output, target):
        
        loss = self.alpha*self.loss1(output,target) + (1-self.alpha)*self.loss2(output,target)
        
        return loss

# Defining the objective function

def objective(params,train_loader,val_loader):

    print(params)
    
    gc.collect()
    torch.cuda.empty_cache()

    encoder = params['encoder']
    encoder_weights = "imagenet"
    
    model = smp.DeepLabV3Plus(
            encoder_name= encoder,
            encoder_weights= encoder_weights,
            activation="sigmoid",
            in_channels=3,
            classes=1,
        ) 

    model.to(device)

    n_epochs = 4
    loss_fn = CustomLoss(0.25)
    
    if not params['decoder-only']:
        optimizer = torch.optim.Adam([ dict(params=model.parameters(), lr=params['learning-rate'])])
    else:
        optimizer = torch.optim.Adam([dict(params=model.decoder.parameters(), lr=params['learning-rate'])])
        
    preprocessing = smp.encoders.get_preprocessing_fn(encoder_name=encoder, pretrained=encoder_weights)

    val_loss,train_loss = training_loop(n_epochs, optimizer, model, loss_fn, train_loader,val_loader,preprocessing,params['decoder-only'])
    
    return {
        'loss': val_loss,
        'status': STATUS_OK,
        'params': params,
        'train_loss': train_loss
        }

# Defining the space of possible training settings

trials = Trials()
space = {
        'encoder': hp.choice('encoder',['resnet101','tu-xception71']),
        'decoder-only': hp.choice('decoder-only',[False,True]),
        'learning-rate': hp.loguniform('learning_rate',
                                                     np.log(0.000001),
                                                     np.log(0.01))
        }

train_dataset = roadsDataset('train',transform=transform_divisible)
valid_dataset = roadsDataset('val',transform=transform_divisible)

train_dataset_lite = Subset(train_dataset, torch.randperm(len(train_dataset))[:8000])
valid_dataset_lite = Subset(valid_dataset, torch.randperm(len(valid_dataset))[:2000])

train_loader = DataLoader(train_dataset_lite, batch_size=10, shuffle=True)
val_loader = DataLoader(valid_dataset_lite, batch_size=5, shuffle=False)

# Running TPE using HyperOpt 

hyperopt_objective = partial(objective, train_loader=train_loader,val_loader=val_loader)

best = fmin(hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

filename = 'hyperopt_trials.pkl'

with open(filename, 'wb') as f:
    pickle.dump(trials, f)