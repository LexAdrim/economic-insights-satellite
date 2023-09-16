import os
import torch
import numpy as np
import cv2
import timm
import re
import pickle
import pandas as pd
import albumentations as album
from tqdm import tqdm
from torchvision import transforms
from torch import nn
from patchify import patchify

path = 'forecasting-income/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.chdir(path)

# Accessing the data outputted by the infrastructure extraction pipeline and the income per capita 
class forecastingDataset(torch.utils.data.Dataset):

    def __init__(self, folder,path_csv):
            
            self.image_paths = [image_id for image_id in os.listdir(folder + '/segmented') if image_id != ".DS_Store"]
            self.df = pd.read_csv(path_csv)
            self.folder = folder

    def __getitem__(self, i):

        city_name = re.sub(r'[_]+$', '', re.sub(r'[A-Z]|-|_+$', '', self.image_paths[i][:-4]))
        row = self.df[self.df['Cities'] == city_name]
        infra_tens = torch.tensor(row.iloc[:, 2:-1].values.astype('float32'))

        income = self.df[self.df['Cities'] == city_name]['income'].values

        return os.path.join(self.folder + '/segmented', self.image_paths[i]),infra_tens,torch.tensor(income).to(torch.float32)
    
    def __len__(self):

        return len(self.image_paths)

# Training loop for the forecasting pipeline
def training_loop(n_epochs, optimizer, model, loss_fn, train_dataset,valid_dataset):

    train_loss_list = []
    val_loss_list = []
    val_pred = []

    for epoch in range(1, n_epochs + 1):

        print('Epoch {}'.format(epoch))

        model.train()
        loss_train = 0.0
        n_batches_train = len(train_dataset)

        for i in tqdm(range(n_batches_train), desc="Training"):
            
            image_path,vec,income = train_dataset[i]
            image = cv2.imread(image_path)

            outputs = model(image,vec)
            income = income.to(device)


            loss = loss_fn(outputs, income.unsqueeze(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():

            loss_val = 0.0
            n_batches_val = len(valid_dataset)
            
            for i in tqdm(range(n_batches_val), desc="Validation"):
                
                image_path,vec,income = valid_dataset[i]
                image = cv2.imread(image_path)

                outputs = model(image,vec)

                income = income.to(device)

                loss = loss_fn(outputs, income.unsqueeze(0))
                loss_val += loss.item()

                if epoch == 2:
                    val_pred.append((image_path,outputs.cpu()))

        val_loss_t = loss_val / n_batches_val
        val_loss_list.append(val_loss_t)
        train_loss_t = loss_train / n_batches_train
        train_loss_list.append(train_loss_t)

        print('Training loss {} | Validation loss {}'.format(train_loss_t, val_loss_t))
        torch.save(model.state_dict(), 'nnincome-{}-{}.pth'.format(epoch,val_loss_t))
    
    results = {
        'train_loss': train_loss_list ,
        'val_loss': val_loss_list,
        'pred': val_pred
    }

    with open('training_res.pkl', 'wb') as f:
        pickle.dump(results, f)

# Neural network architecture predicting the income per capita for a given city
class incomeModel(nn.Module):
    def __init__(self, backbone_name,custom_bias):
        super(incomeModel, self).__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=True,num_classes=0,in_chans=3, global_pool='catavgmax')
        self.finallayer = nn.Linear(512, 1)
        with torch.no_grad():
            self.finallayer.bias.copy_(custom_bias)

        self.fc_layers = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            self.finallayer 
        )
        
    def forward(self, image,y):

        (h,w,_) = image.shape
        size = 224
        min_h = h + size-h%size; min_w = w + size-w%size

        transform = album.PadIfNeeded(min_height=min_h, min_width=min_w, always_apply=True, border_mode=0)
        image2 = transform(image=image)['image']

        patches = patchify(image2, (size, size, 3), step=size)
        patches = np.squeeze(patches)
        rows, cols, _, _, _ = patches.shape
        x = torch.zeros(1,1024)

        for i in range(rows):
            for j in range(cols):
                
                img_tensor = transforms.ToTensor()(patches[i,j].astype('float32'))

                if img_tensor.std() != 0:
                    img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.std()

                img_tensor = img_tensor.to(device)
                
                temp = self.backbone(img_tensor.unsqueeze(0))
                x += temp.cpu().detach()
        

        x = x.to(device)
        y = y.to(device)
        x = torch.cat((x, y), dim=1)
        x = self.fc_layers(x)

        return x


# Training the architecture for 7 epochs

train_dataset = forecastingDataset('train','infrastructures-income.csv')
valid_dataset = forecastingDataset('val','infrastructures-income.csv')

model = incomeModel('resnet18',49587)

model.to(device)

n_epochs = 7
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.01,weight_decay=1e-4)])

        
training_loop(n_epochs, optimizer, model, loss_fn, train_dataset,valid_dataset)