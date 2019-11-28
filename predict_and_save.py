import torch
import torchvision

import torch.optim as optim
from torch.utils.data import *
from torchvision import models, transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import glob
import os.path as osp


eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
for param in model.parameters(): # set all pretrained weights are kept & not update that gradients
    param.requires_grad = False

numfts = model.fc.in_features
model.fc = nn.Linear(numfts, 2)
model = model.to(device)
model.load_state_dict(torch.load('best_model.pt'))


from PIL import Image


csv_filename = "submission.csv"
csv_file = open(csv_filename, 'w')
csv_file.write("id,label"+"\n")


root="/home/jingwenlai/data/kaggle/dogs_and_cats/test1"
img_files = glob.glob(osp.join(root,"*.jpg"))
img_files.sort(key=lambda x: int(osp.basename(x)[:-4]))
    
model.eval()
with torch.no_grad():
    for img_file in img_files:
        basename = osp.basename(img_file)
        print('predicting...', basename)
        file_id = basename.split('.')[0]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img_tensor = eval_transform(img)
            img_tensor = img_tensor.unsqueeze(0).to(device)            

            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)

            predict_result = preds.cpu().item()
            csv_file.write(str(file_id)+","+str(predict_result)+"\n")

csv_file.close()