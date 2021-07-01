import argparse
import cv2
import numpy as np
import torch
from torchvision import models

# from __future__ import print_function, division
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from efficientnet.model import EfficientNet
import cv2
import numpy as np
import shutil
from PIL import Image

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

# some parameters
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = '../clothes_classify'
batch_size = 1
lr = 0.01
momentum = 0.9
num_epochs = 60
input_size = 224
class_num = 4
net_name = 'efficientnet-b0'
resize_size = int(1440 / 2560 * input_size)

def EffModel(net_name):
    pth_map = {
        'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
        'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
    }
    # 自动下载到本地预训练
    # model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    # 离线加载预训练，需要事先下载好
    model_ft = EfficientNet.from_name(net_name)

    # 修改全连接层
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    net_weight = '../clothes_classify/model/' + net_name + ".pth"
    if use_gpu:
        model_ft = torch.load(net_weight)
    else:
        # what if class_num miss match when load weights???
        model_ft = torch.load(net_weight, map_location=torch.device('cpu'))
    # .load_state_dict(state_dict)
    if use_gpu:
        model_ft = model_ft.cuda()
    return model_ft


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    args.method = "gradcam"

    path = './test'
    files = os.listdir(path)
    for f in files:
        args.image_path = path + '/' + f
        methods = \
            {"gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM}

        model = EffModel(net_name)

        # Choose the target layer you want to compute the visualization for.
        # Usually this will be the last convolutional layer in the model.
        # Some common choices can be:
        # Resnet18 and 50: model.layer4[-1]
        # VGG, densenet161: model.features[-1]
        # mnasnet1_0: model.layers[-1]
        # You can print the model to help chose the layer
        target_layer = model._modules['_blocks'][15]

        cam = methods[args.method](model=model,
                                target_layer=target_layer,
                                use_cuda=args.use_cuda)

        rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        target_category = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=target_category)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        cv2.imwrite(f'{f}_cam.jpg', cam_image)
        cv2.imwrite(f'{f}_gb.jpg', gb)
        cv2.imwrite(f'{f}_cam_gb.jpg', cam_gb)
