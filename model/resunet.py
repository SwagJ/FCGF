# -*- coding: future_fstrings -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm

from model.residual_block import get_block


class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out = MEF.relu(out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'

class ResUNetIN2E(ResUNetBN2E):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'

class Detection(nn.Module):
  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,radius=10,device='cpu'):
    super(Detection, self).__init__()
    self.device = device
    #self.len_batch = len_batch


  def forward(self,coords,features,len_batch):
    #point_len = self.len_batch
    device = self.device

    score = (torch.Tensor()).to(device)
    for i in range(len_batch):
      batch_score = self._detection_score(coords[i],features[i])
      #print(batch_score.device)
      score = torch.cat((score,batch_score),0)
      #print(score.shape)
      #score1.append(self._detection_score(coords=batch_C1[i],feature=batch_F1[i],radius=radius))

    return score


  def _detection_score(self,coords=None,feature=None):
    #find all points in a cube whose center is the point
    #get alpha score in feature map k
    feature = F.relu(feature)
    max_local = torch.max(feature,dim=1)[0]
    beta = feature/max_local.unsqueeze(1)
  
    del max_local
    #logging.info(f"Beta Done")

    coords_A = (coords.view(coords.shape[0], 1, 3).repeat(1, coords.shape[0], 1)).short()
    coords_B = (coords.view(1, coords.shape[0], 3).repeat(coords.shape[0], 1, 1)).short()
    coords_confusion = (torch.stack((coords_A, coords_B), dim=2)).short()
    del coords_A,coords_B
    every_dist = (((coords_confusion[:, :, 0, :] - coords_confusion[:, :, 1, :]) ** 2).sum(dim=2) ** 0.5)


    neighbors = (torch.topk(every_dist,1,largest=False,dim=1).indices)
    del every_dist
    neighbor9_feature = (feature[neighbors,:])[0]
    del neighbors
    exp_feature = torch.exp(feature)
    exp_neighbor = torch.sum(torch.exp(neighbor9_feature),dim=0)
    alpha = exp_feature/exp_neighbor
    del exp_feature,exp_neighbor
    #logging.info(f"Alpha Done")

    gamma = torch.max(alpha*beta,dim=1).values
    del alpha,beta
    #logging.info(f"Gamma Done, gamma dimension{gamma.shape}")
    score = gamma/torch.norm(gamma)
    del gamma
    #print(score.device)
    torch.cuda.empty_cache()
    return score

class JointNet(nn.Module):
  def __init__(self,
                device,
                batch_size=4,
                in_channels=3,
                out_channels=32,
                bn_momentum=0.1,
                normalize_feature=None,
                conv1_kernel_size=None,
                backbone_model=ResUNetBN2C,
                D=3):
    super(JointNet, self).__init__()

    self.batch_size = batch_size
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bn_momentum = bn_momentum
    self.normalize_feature = normalize_feature
    self.conv1_kernel_size = conv1_kernel_size
    self.backbone_model = backbone_model
    self.device = device
    #self.batch_len = batch_len
    #self.len_batch = len_batch

    #model = load_model(backbone_model)
    self.feature_extraction0 = backbone_model(
                              in_channels,
                              out_channels,
                              bn_momentum=bn_momentum,
                              normalize_feature=normalize_feature,
                              conv1_kernel_size=conv1_kernel_size,
                              D=3)
    self.feature_extraction1 = backbone_model(
                              in_channels,
                              out_channels,
                              bn_momentum=bn_momentum,
                              normalize_feature=normalize_feature,
                              conv1_kernel_size=conv1_kernel_size,
                              D=3)#.to(device)

    self.detection0 = Detection(device=device)
    self.detection1 = Detection(device=device)

  def forward(self,x0,x1,len_batch):
    #x0 = x0.to(self.device)
    #x1 = x1.to(self.device)
    #logging.info(f"input device:{x0.F.device}")
    #print(len_batch)
    sparse0 = self.feature_extraction0(x0)
    sparse1 = self.feature_extraction1(x1)
    #logging.info(f"Feature Extraction Done")
    #logging.info(f"coord at output device:{sparse1.coordinates_at(0).device}")
    coord0 = (sparse0.C.short()).to(self.device)
    feature0 = sparse0.F
    coord1 = (sparse1.C.short()).to(self.device)
    feature1 = sparse1.F
    del sparse0,sparse1
    torch.cuda.empty_cache()

    batch_C1, batch_F1 = [],[]
    batch_C0, batch_F0 = [],[]
    start_idx = np.zeros((2,),dtype=int)
    for i in range(len(len_batch)):
      end_idx = start_idx + np.array(len_batch[i],dtype=int)
      #print(start_idx,end_idx)
      #logging.info(f"Before append device:{sparse0.C.device}")
      C0 = coord0[start_idx[0]:end_idx[0],1:4]
      C1 = coord1[start_idx[1]:end_idx[1],1:4]
      F0 = feature0[start_idx[0]:end_idx[0],:]
      F1 = feature1[start_idx[1]:end_idx[1],:]
      #print(C0.shape,C1.shape,F0.shape,F1.shape)
      batch_C1.append(C1)
      batch_F1.append(F1)
      batch_C0.append(C0)
      batch_F0.append(F0)
      del C0,C1,F0,F1
      torch.cuda.empty_cache()
      start_idx = end_idx

    #logging.info(f"Coord_seperation Done")
    #logging.info(f"After append device:{batch_C0[i].device}")
    with torch.no_grad():
      score0 = self.detection0(batch_C0,batch_F0,len(len_batch))
      score1 = self.detection1(batch_C1,batch_F1,len(len_batch))

    return{
     'feature0': feature0,
     'feature1': feature1,
     'score0': score0,
     'score1': score1
    }






                            


