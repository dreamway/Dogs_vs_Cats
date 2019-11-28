import torch
import torchvision

import torch.optim as optim
from torch.utils.data import *
from torchvision import models, transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='runs')


def imshow(inp, title=None):
    """ Imshow for Tensor"""
    print(inp.size())
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = inp*std+mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

def test_show_data():
    # Test Showing, Get a batch of training data
    inputs, classes = next(iter(train_dataloader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])




data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(root="/home/jingwenlai/data/kaggle/dogs_and_cats/train", transform=data_transforms['train'])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dataset = datasets.ImageFolder(root="/home/jingwenlai/data/kaggle/dogs_and_cats/dev", transform=data_transforms['val'])
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

class_names = ['cat', 'dog']
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# Now, because the val set have problem, only use trainset
def train_model(model, criterion, optimizer, lr_scheduler, num_epoches):
    best_model = None
    best_acc = 0.0

    for epoch in range(0, num_epoches):
        print("---------Training {}/{}---------".format(epoch, num_epoches))
        model.train()
        running_loss = 0.0
        running_corrects = 0

        #Iterate over data
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameters gradients
            optimizer.zero_grad()

            #forward 
            outputs = model(inputs)            
            _, preds = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)

            #backward + optimize only in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)

        lr_scheduler.step()

        epoch_train_loss = running_loss/len(train_dataset)
        epoch_train_acc = running_corrects.double()/len(train_dataset)
        print('epoch train loss:{:.4f}, epoch train acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))
        #writer.add_scalar('epoch_train_loss',epoch_train_loss, epoch)
        #writer.add_scalar('epoch_train_acc', epoch_train_acc, epoch)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss/len(val_dataset)
        epoch_val_acc = val_corrects.double()/len(val_dataset)
        print('epoch val loss: {:.4f}, epoch val acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
        #writer.add_scalar('epoch_val_loss', epoch_val_loss, epoch)
        #writer.add_scalar('epoch_val_acc', epoch_val_acc, epoch)

        writer.add_scalars('loss', {'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}, epoch)
        writer.add_scalars('acc', {'train_acc': epoch_train_acc, 'val_acc': epoch_val_acc}, epoch)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            ckpt_state = {
                'epoch': epoch,
                'best_acc' : best_acc,
                'model_state': model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'loss': loss
            }
            torch.save(ckpt_state, 'ckpt_{}_model.pt'.format(epoch));
            best_model = model.state_dict()
        
    return best_model


model = models.resnet18(pretrained=True)
for param in model.parameters(): # set all pretrained weights are kept & not update that gradients
    param.requires_grad = False

numfts = model.fc.in_features
model.fc = nn.Linear(numfts, 2)
model = model.to(device)
print("modified model:", model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum = 0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_model = train_model(model, criterion, optimizer, lr_scheduler, num_epoches=25)

torch.save(best_model, 'best_model.pt')
