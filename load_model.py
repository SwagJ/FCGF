import open3d as o3d  # prevent loading error

import sys
import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from model import load_model

from lib.data_loaders import make_data_loader
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.timer import AverageMeter, Timer

import MinkowskiEngine as ME


config = json.load(open('/disk/FCGF/output_kitti' + '/config.json', 'r'))
config = edict(config)
#print(config.model_n_out)
num_feats = 1

Model = load_model(config.model)
model = Model(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      conv1_kernel_size=config.conv1_kernel_size,
      normalize_feature=config.normalize_feature)
checkpoint = torch.load('/disk/FCGF/output_kitti/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth')
model.load_state_dict(checkpoint['state_dict'])

#model = torch.load("/disk/FCGF/output_kitti/checkpoint.pth")
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
