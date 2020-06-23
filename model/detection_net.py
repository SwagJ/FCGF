import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm

from model.residual_block import get_block

def knn(x,k):
	inner = -2*torch.matmul(x.transpose(2, 1), x)
	xx = torch.sum(x**2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1)

	idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

	return idx

class DetectionNetHead(nn.Module):
	def __init__(self,
				in_channels=3,
				keypoint_num=192
				):

		super(DetectionNetHead, self).__init__()
		self.keypoint_num = keypoint_num
		#self.normalize_feature = normalize_feature

		self.mlp1 = nn.Conv1d(in_channels,64,kernel_size=1,bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.prelu1 = nn.PReLU(64)
		self.mlp2 = nn.Conv1d(64,128,kernel_size=1,bias=False)
		self.bn2 = nn.BatchNorm1d(128)
		self.prelu2 = nn.PReLU(128)
		self.mlp3 = nn.Conv1d(128,256,kernel_size=1,bias=False)
		self.bn3 = nn.BatchNorm1d(256)
		self.prelu3 = nn.PReLU(256)
		self.mlp4 = nn.Conv1d(256,1024,kernel_size=1,bias=False)
		self.bn4 = nn.BatchNorm1d(1024)
		self.prelu4 = nn.PReLU(1024)


		self.fc1 = nn.Linear(1024,512)
		self.fc_bn1 = nn.BatchNorm1d(512)
		self.fc_prelu1 = nn.PReLU(512)
		self.fc2 = nn.Linear(512,256)
		self.fc_bn2 = nn.BatchNorm1d(256)
		self.fc_prelu2 = nn.PReLU(256)
		self.fc3 = nn.Linear(256,keypoint_num*3)
		self.fc_bn3 = nn.BatchNorm1d(keypoint_num*3)
		self.fc_prelu3 = nn.PReLU(keypoint_num*3)
	

	def forward(self, x):
		x = torch.transpose(x,1,2)
		#print("network input shape:",x.shape)
		#print("input type:",x.type())
		batch_size = x.shape[0]
		num_points = x.shape[2]
		#print("self.in_channels:", self.in_channels)
		out = self.mlp1(x)
		out = self.bn1(out)
		out = self.prelu1(out)
		out = self.mlp2(out)
		out = self.bn2(out)
		out = self.prelu2(out)
		out = self.mlp3(out)
		out = self.bn3(out)
		out = self.prelu3(out)
		out = self.mlp4(out)
		out = self.bn4(out)
		#print("BN out shape:",out.shape)
		out = self.prelu4(out)
		#print("Intermediate shape:",out.shape)
		num_batch = x.shape[1]
		out = F.max_pool1d(out,kernel_size=num_points)
		#out = torch.transpose(out,1,2)
		out = out.squeeze(2)
		#print("After Pooling shape:",out.shape)

		out = self.fc1(out)
		#print("FC1 out:",out.shape)
		out = self.fc_bn1(out)
		out = self.fc_prelu1(out)
		out = self.fc2(out)
		#print("FC2 out:",out.shape)
		out = self.fc_bn2(out)
		out = self.fc_prelu2(out)
		out = self.fc3(out)
		#print("FC3 out:",out.shape)
		out = self.fc_bn3(out)
		out = self.fc_prelu3(out)
		#print("final out:",out.shape)

		#reshape to keypoint M*3
		out = torch.reshape(out,(batch_size,self.keypoint_num,3))
		#print("final ouput shape:",out.shape)

		return 	out
