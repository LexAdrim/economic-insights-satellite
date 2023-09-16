import os
import torch
import numpy as np
import cv2
import albumentations as album
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle 

path = 'final-roads/'

os.chdir(path)        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Accessing the custom segmentation dataset 

class roadsDataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform=None):
            
            self.image_paths = [os.path.join(folder, image_id) for image_id in os.listdir(folder) if image_id != ".DS_Store"]
            self.mask_paths = [os.path.join(folder + '_labels', image_id) for image_id in os.listdir(folder) if image_id != ".DS_Store" ]
                
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

# Creating the training loop for the segmentation model

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,val_loader):

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, n_epochs + 1):

        print('Epoch {}'.format(epoch))

        model.train()
        loss_train = 0.0
        n_batches_train = len(train_loader)
        train_loop = tqdm(train_loader, desc="Training")

        for imgs, masks in train_loop:

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
        val_loss_list.append(val_loss_t)
        train_loss_t = loss_train / n_batches_train
        train_loss_list.append(train_loss_t)

        os.chdir('../')
        print('Training loss {} | Validation loss {}'.format(train_loss_t, val_loss_t))
        torch.save(model.state_dict(), 'deeplabv3+-xception-{}-{}.pth'.format(epoch,val_loss_t))
        os.chdir('final-roads/')

    results = {
        'train_loss': train_loss_list ,
        'val_loss': val_loss_list,
    }

    with open('training_res.pkl', 'wb') as f:
        pickle.dump(results, f)

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

# Defining our dataset and instantiating DeepLabv3+

train_dataset = roadsDataset('train',transform=transform_divisible)
valid_dataset = roadsDataset('val',transform=transform_divisible)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=5, shuffle=False)

encoder = "tu-xception71"
encoder_weights = "imagenet"
decoderOnly=False

model = smp.DeepLabV3Plus(
    encoder_name= encoder,
    encoder_weights= encoder_weights,
    activation="sigmoid",
    in_channels=3,
    classes=1,
)

model.to(device)

# Train the model for 25 epochs

n_epochs = 25
loss_fn = CustomLoss(0.25)

optimizer = torch.optim.Adam([ dict(params=model.parameters(), lr=0.0001231689732371627)])

        
training_loop(n_epochs, optimizer, model, loss_fn, train_loader,val_loader)