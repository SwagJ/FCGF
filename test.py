import numpy as np
#from lib.feature_extraction import prepare_feature
from lib.timer import Timer
import ntpath
import os 
import glob

base_dir = '/disk_ssd/threedmatch/'
ntpath.basename(base_dir)
raw = glob.glob(base_dir + '*.npz')

og_name = []
for name in raw:
	abs_dir = os.path.splitext(name)[0]
	og_name.append(ntpath.split(abs_dir)[1])
	pcd = np.load(name,allow_pickle=True)['pcd']
	if pcd.shape[0] == 0:
		print("This file has no pcd:",name)

print(og_name[0])
print(raw[0])

feature_dir = '/disk_ssd/threedmatch/features/'
ntpath.basename(feature_dir)
feature_raw = glob.glob(feature_dir + '*.npy')

feature = np.load(feature_dir + '7-scenes-chess@seq-01_000.npy', allow_pickle=True)
print(feature[0])

feature_name = []
for name in feature_raw:
	abs_dir = os.path.splitext(name)[0]
	feature_name.append(ntpath.split(abs_dir)[1])

print(feature_name[0])
print(feature_raw[0])

print(feature_name[0])
print(og_name[0])

#for name in og_name:
#	if name not in feature_name:
		#print(name)