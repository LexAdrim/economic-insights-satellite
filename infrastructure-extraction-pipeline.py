import os
import torch
import numpy as np
from torchvision import transforms
import cv2
import pandas as pd
import albumentations as album
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
from patchify import patchify
from patchify import unpatchify


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating the pipeline combining the trained segmentation model and the trained detection model 
class SegmentationDetectionPipeline(torch.nn.Module):

    def __init__(self, segmentation_weights,detection_weights,device):
        super().__init__()

        # Loading the weights of the trained models
        self.segmentation_model = smp.DeepLabV3Plus(
                                        encoder_name= "tu-xception71",
                                        activation="sigmoid",
                                        in_channels=3,
                                        classes=1,
                                    )
        self.segmentation_model.load_state_dict(segmentation_weights)
        self.segmentation_model.to(device)
        self.segmentation_model.eval()

        self.detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=detection_weights) 
        self.detection_model.conf = 0.5
        self.detection_model.to(device)

    def forward(self, image,city):

        # Creating the 512x512 grid for the segmentation component
        (h,w,_) = image.shape
        size = 512
        min_h = h + size-h%size; min_w = w + size-w%size

        transform = album.PadIfNeeded(min_height=min_h, min_width=min_w, always_apply=True, border_mode=0)
        image2 = transform(image=image)['image']

        patches = patchify(image2, (512, 512, 3), step=512)
        patches = np.squeeze(patches)
        rows, cols, _, _, _ = patches.shape
        patches_segmentation = patches.copy()

        for i in tqdm(range(rows)):
            for j in tqdm(range(cols)):

                img_tensor = transforms.ToTensor()(patches[i,j].astype('float32'))
                img_tensor = (img_tensor-img_tensor.mean())/img_tensor.std()
                img_tensor = img_tensor.to(device)
                pred_masks = self.segmentation_model(img_tensor.unsqueeze(0))
                pred_binary = torch.where(pred_masks >= 0.5,1,0)
                patches_segmentation[i,j] =  cv2.cvtColor(pred_binary[0].squeeze().numpy(force=True).astype('uint8')*255, cv2.COLOR_GRAY2RGB)
        
        stitched_segment = unpatchify(np.expand_dims(patches_segmentation, axis=2), image2.shape)

        # Segmented road network
        cv2.imwrite(os.path.join('../../segmented', city + '.jpg'), stitched_segment)

        # Creating the 1280x1280 grid for the detection component
        size = 1280
        min_h = h + size-h%size; min_w = w + size-w%size

        transform = album.PadIfNeeded(min_height=min_h, min_width=min_w, always_apply=True, border_mode=0)
        image3 = transform(image=image)['image']

        patches = patchify(image3, (size, size, 3), step=size)
        patches = np.squeeze(patches)
        rows, cols, _, _, _ = patches.shape
        patches_detection = patches.copy()

        columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'] 
        detection_df = pd.DataFrame(columns=columns)

        for i in tqdm(range(rows)):
            for j in tqdm(range(cols)):
                
                detection = self.detection_model(patches[i,j], augment=True)
                patches_detection[i,j] = detection.render()[0]
                detection_df = pd.concat([detection_df , detection.pandas().xyxy[0]], ignore_index=True)

        stitched_detect = unpatchify(np.expand_dims(patches_detection, axis=2), image3.shape)
        
        # Image with the detected infrastrucure marked
        cv2.imwrite(os.path.join('../../final_pipeline', city + '_detect.jpg'), stitched_detect)     
        # CSV file with the detected infrastrucure 
        detection_df.to_csv(os.path.join('../../infrastructures', city + '.csv'), index=False)

        transform2 = album.CenterCrop(h, w)
        stitched_detect = transform2(image=stitched_detect)['image']
        stitched_segment = transform2(image=stitched_segment)['image']

        final = cv2.addWeighted(stitched_detect, 1, stitched_segment, 1, 1)

        # Image overlayed with all the information
        cv2.imwrite(os.path.join('../../final_pipeline', city + '_final.jpg'), final)


segmentation_weights= torch.load('DeepLabV3Plus-best.pth')
detection_weights = 'YOLOv5x6-best.pt'
pipeline = SegmentationDetectionPipeline(segmentation_weights,detection_weights,device)

list_cities = [city for city in os.listdir('Cities/stitched_cities/') if city != '.DS_Store' and city != 'stitched_cities']
os.chdir('Cities/stitched_cities/')

# Running the pipeline on our US Cities dataset
for city in list_cities:
    image = cv2.imread(city)
    pipeline(image,city[:-4])

