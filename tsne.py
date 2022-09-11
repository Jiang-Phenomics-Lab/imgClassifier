import os, sys
from dataset import XDAData
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from tester import Tester
from tqdm import tqdm
import numpy as np

import pandas as pd

#from tsne_torch import TorchTSNE as TSNE
from tsnecuda import TSNE
import seaborn as sns

image_transforms = transforms.Compose([
        transforms.CenterCrop(764),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def embed_single_image(img, whichModel):
    
    data = transforms.ToTensor()(img)
    data = data.unsqueeze(0)
    data = data.cuda()
    fvec = whichModel(data)
    
    modules = list(whichModel.children())[:-2]
    last_conv = nn.Sequential(*modules)
    last_conv = last_conv.cuda()
    fmap = last_conv(data)
    
    return fvec.cpu().detach(), fmap.cpu().detach().numpy()

def embed(dsets,whichModel):

    dataloader = torch.utils.data.DataLoader(dsets, batch_size=4, 
                                             shuffle=False)
    tbar = tqdm(enumerate(dataloader), total=len(dataloader))
    whichModel.eval()

    # fc output
    fc = whichModel.fc.state_dict()
    # last conv output
    modules = list(whichModel.children())[:-1]
    last_conv = nn.Sequential(*modules)
    last_conv = last_conv.cuda()
    # iterate batch
    V,M,L = [],[],[]
    
    
    with torch.no_grad():
        for batch_idx, (data, target) in tbar:
            data = data.cuda()
            fvec = whichModel(data)
            fmap = last_conv(data)
            fmap_ = torch.squeeze(fmap)
            
            V.append(fvec)
            M.append(fmap_)
            L.append(target)

    vectors = torch.cat(V, 0)
    maps = torch.cat(M, 0)
    labels = torch.cat(L, 0)

    return vectors, maps, labels

def generateVectors():

    root_dir = '/media/baker/C05A8B528B5B5A2D/data/XDA24040201-1/Rice/data_2022/graph_data/Level1'
    txt_dir = '/home/baker/Work/data/file_lists/XDA'

    val_file = os.path.join(txt_dir, 'test.txt')
    val_dataset = XDAData(val_file, root_dir, image_transforms)

    train_file = os.path.join(txt_dir, 'train.txt')
    train_dataset = XDAData(train_file, root_dir, image_transforms)

    out_dir = '/home/baker/Work/Codes/Python/imageClassifier/tsne_results/'
    model_path = '/home/baker/Work/Models/checkpoints/XDA/24.pth'
    num_classes = 79
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft = model_ft.to(device)

    vectors, maps, labels = embed(train_dataset,model_ft)
    #tester = Tester(model_ft, train_dataloader)
    #vectors, labels = tester.test()
    task_name = 'xda_train'
    load_result_ep = 24
    out_path = os.path.join(out_dir, '{}_ep_{}.pth'.format(task_name, load_result_ep))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    torch.save({'vectors': vectors,
                'maps': maps,
                'labels': labels,
                },
                out_path)

    return

def getTsneResult():

    result_dict_path = '/home/baker/Work/Codes/Python/imageClassifier/tsne_results/xda_train_ep_24.pth'
    result_dict = torch.load(result_dict_path)
    vectors = result_dict['maps']
    tsne_labels = result_dict['labels'].detach().numpy()
    
    tsne_model = TSNE(n_components=2, perplexity=15, learning_rate=10)
    tsne_Y = tsne_model.fit_transform(vectors.cpu())
    #tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True)
    #tsne_Y = tsne_model.fit_transform(vectors)


    in_tsn_Y = '/home/baker/Work/Codes/Python/imageClassifier/tsne_results/xda_train_tsne_Y_cuda.pth'
    torch.save(tsne_Y, in_tsn_Y)
    tsne_Y = torch.load(in_tsn_Y)

    df = pd.DataFrame()
    df["y"] = tsne_labels
    df["comp-1"] = tsne_Y[:,0]
    df["comp-2"] = tsne_Y[:,1]

    num_classes = np.unique(tsne_labels).shape[0]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", num_classes),
                    data=df).set(title="Iris data T-SNE projection")


    return

#generateVectors()
getTsneResult()