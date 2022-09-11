import numpy as np
import cv2, os
import torch
import glob as glob
import torch.nn as nn
from torchvision import transforms, models
from torch.nn import functional as F
from torch import topk
from dataset import M1Data, XDAData
features_blobs = []
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_transforms = transforms.Compose([
        transforms.CenterCrop(764),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(int(class_idx[i])), (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('CAM', result/255.)
        cv2.waitKey(0)
        cv2.imwrite(save_name, result)

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def test():
    excel_dir = '/home/baker/Work/data/raw_data/M1/excels/'
    root_dir = '/home/baker/Work/data/raw_data/M1/Z7II'
    new_label_file = '/home/baker/Work/data/raw_data/M1/excels/field_cultivar_phenptypes.json'
    out_dir = '/home/baker/Work/data/exp/cam/imgClassifier'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    num_classes = 116
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load('/home/baker/Work/Codes/Python/imageClassifier/20.pth'))
    model_ft = model_ft.eval()
    
    

    model_ft._modules.get('fc').register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(model_ft.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    model_ft = model_ft.to(device)

    test_file = os.path.join(excel_dir, 'test.txt')
    test_dataset = M1Data(test_file, new_label_file, root_dir, data_transforms['val'])
    paths = test_dataset.get_paths()
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    for file_name in paths:
        image = cv2.imread(file_name)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)
        height, width, _ = orig_image.shape
        # apply the image transforms
        image_tensor = data_transforms['val'](image)
        # add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        # forward pass through model
        outputs = model_ft(image_tensor)
        # get the softmax probabilities
        probs = F.softmax(outputs).data.squeeze()
        # get the class indices of top k probabilities
        class_idx = topk(probs, 1)[1].int()
        
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
        base_name = os.path.basename(file_name)
        save_path = os.path.join(out_dir, base_name)
        show_cam(CAMs, 512, 512, orig_image, class_idx, save_path)


    return

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.transforms import Compose

def visualize_model(model, vis_dataloader, device, num_images=6):
    #was_training = model.training
    model.eval()
    target_layer = model.layer4[-1]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(vis_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            cam = GradCAM(model=model, target_layer=target_layer, use_cuda="cuda:0")

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

def preprocess_image_with_transforms(img: np.ndarray, image_transforms) -> torch.Tensor:
    preprocessing = Compose(image_transforms)
    rel = preprocessing(img.copy()).unsqueeze(0)
    return rel

def main():
    excel_dir = '/home/baker/Work/data/raw_data/M1/excels/'
    root_dir = '/home/baker/Work/data/raw_data/M1/Z7II'
    new_label_file = '/home/baker/Work/data/raw_data/M1/excels/field_cultivar_phenptypes.json'
    out_dir = '/home/baker/Work/data/exp/cam/imgClassifier'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    num_classes = 116
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load('/home/baker/Work/Codes/Python/imageClassifier/best_model.pth'))
    model_ft = model_ft.to(device)

    test_file = os.path.join(excel_dir, 'test.txt')
    test_dataset = M1Data(test_file, new_label_file, root_dir, data_transforms['val'])
    
    model_ft.eval()
    target_layer = [model_ft.layer4]
    paths = test_dataset.get_paths()
    labels = test_dataset.get_labels()
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    max_save = 2000
    
    for idx, file_name in enumerate(paths):
        rgb_img = cv2.imread(file_name, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        output = model_ft(input_tensor.to(device))
        out_label = torch.argmax(output).to('cpu').detach()
        if out_label != labels['cultivar'][idx]:
            continue
        cam = GradCAM(model=model_ft, target_layers=target_layer, use_cuda=True)
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        base_name = os.path.basename(file_name)[:-4]
        save_path = os.path.join(out_dir, '{}__{}.png'.format(int(out_label), base_name))
        if idx >max_save:
            break
        
        cam_image = cv2.resize(cam_image, (512,512))
        cv2.imwrite(save_path, cam_image)

def xda_cam_main():

    root_dir = '/media/baker/C05A8B528B5B5A2D/data/XDA24040201-1/Rice/data_2022/graph_data/Level1'
    txt_dir = '/home/baker/Work/data/file_lists/XDA'

    val_file = os.path.join(txt_dir, 'test.txt')
    val_dataset = XDAData(val_file, root_dir, image_transforms)

    train_file = os.path.join(txt_dir, 'train.txt')
    train_dataset = XDAData(train_file, root_dir, image_transforms)

    out_dir = '/home/baker/Work/data/exp/cam/XDATrain'
    model_path = '/home/baker/Work/Models/checkpoints/XDA/24.pth'
    num_classes = 79
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(pretrained=False)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    num_classes = 79
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft = model_ft.to(device)
    
    model_ft.eval()
    target_layer = [model_ft.layer4]
    paths = train_dataset.get_paths()
    labels = train_dataset.get_labels()
    max_save = 2000
    
    for idx, file_name in enumerate(paths):
        rgb_img = cv2.imread(file_name, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image_with_transforms(rgb_img, image_transforms)
        #input_tensor = preprocess_image(rgb_img)
        output = model_ft(input_tensor.to(device))
        out_label = torch.argmax(output).to('cpu').detach()
        if out_label != labels['cultivar'][idx]:
            continue
        cam = GradCAM(model=model_ft, target_layers=target_layer, use_cuda=True)
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        base_name = os.path.basename(file_name)[:-4]
        save_path = os.path.join(out_dir, '{}__{}.png'.format(int(out_label), base_name))
        if idx >max_save:
            break
        
        cam_image = cv2.resize(cam_image, (512,512))
        cv2.imwrite(save_path, cam_image)


    return

xda_cam_main()