from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

from dataset import M1Data, XDAData

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, vis_dataloader, device, num_images=6):
    #was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        #tbar = tqdm(enumerate(vis_dataloader), total=len(vis_dataloader))
        #for idx, (inputs, labels) in tbar:
        for i, (inputs, labels) in enumerate(vis_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('gt: {}  pred: {}'.format(labels[j], preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    #model.train(mode=was_training)
                    return
        #model.train(mode=was_training)

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25, save_freq=6):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            tbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            for _, (inputs, labels) in tbar:
                
                inputs = inputs.to(device)
                
                labels = labels.to(device)
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % save_freq == 0:
                    torch.save(best_model_wts, os.path.join('/home/baker/Work/Models/checkpoints/XDA/', '{}.pth'.format(epoch)))
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

cudnn.benchmark = True
plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(764),
        transforms.ColorJitter(0.8,0.8,0.8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(764),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def main():
    root_dir = '/home/baker/Work/data/raw_data/M1/Z7II'
    new_label_file = '/home/baker/Work/data/raw_data/M1/excels/field_cultivar_phenptypes.json'
    excel_dir = '/home/baker/Work/data/raw_data/M1/excels/'

    batch_size = 48
    num_workers = 8
    num_classes = 116

    val_file = os.path.join(excel_dir, 'val.txt')
    val_dataset = M1Data(val_file, new_label_file, root_dir, data_transforms['val'])

    train_file = os.path.join(excel_dir, 'train.txt')
    train_dataset = M1Data(train_file, new_label_file, root_dir, data_transforms['train'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader 

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_dataset)
    dataset_sizes['val'] = len(val_dataset)

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 116.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=25)

    #visualize_model(model_ft, val_dataset)

def xda_train_main():
    root_dir = '/media/baker/C05A8B528B5B5A2D/data/XDA24040201-1/Rice/data_2022/graph_data/Level1'
    txt_dir = '/home/baker/Work/data/file_lists/XDA'

    val_file = os.path.join(txt_dir, 'test.txt')
    val_dataset = XDAData(val_file, root_dir, data_transforms['val'])

    train_file = os.path.join(txt_dir, 'train.txt')
    train_dataset = XDAData(train_file, root_dir, data_transforms['train'])

    batch_size = 48
    num_workers = 8
    num_classes = 79

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader 

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_dataset)
    dataset_sizes['val'] = len(val_dataset)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 78.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=25)

    return

def test():
    num_classes = 116
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load('/home/baker/Work/Codes/Python/imageClassifier/best_model.pth'))
    model_ft = model_ft.to(device)
    excel_dir = '/home/baker/Work/data/raw_data/M1/excels/'
    root_dir = '/home/baker/Work/data/raw_data/M1/Z7II'
    new_label_file = '/home/baker/Work/data/raw_data/M1/excels/field_cultivar_phenptypes.json'
    test_file = os.path.join(excel_dir, 'test.txt')
    test_dataset = M1Data(test_file, new_label_file, root_dir, data_transforms['val'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)
    
    visualize_model(model_ft, test_dataloader, device)

xda_train_main()
#main()
#test()