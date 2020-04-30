import os
import sys
import multiprocessing as multiproc
from copy import deepcopy
import logging
import argparse
#import pickle
import numpy as np
from lib.timer import Timer
#import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
import math
import glob
import ntpath
import errno

def calculate_features(pointcloud, nbrs_index, eigens_, vectors_):
    ### calculate handcraft feature with eigens and statistics data

    # features using eigens
	eig3d = eigens_[:3]
	eig2d = eigens_[3:5]

    # 3d
	C_ = eig3d[2] / (eig3d.sum())
	O_ = np.power((eig3d.prod() / np.power(eig3d.sum(), 3)), 1.0 / 3)
	L_ = (eig3d[0] - eig3d[1]) / eig3d[0]
	E_ = -((eig3d / eig3d.sum()) * np.log(eig3d / eig3d.sum())).sum()
	D_ = 3 * nbrs_index.shape[0] / (4 * math.pi * eig3d.prod())
	# 2d
	S_2 = eig2d.sum()
	L_2 = eig2d[1] / eig2d[0]
	# features using statistics data
	neighborhood = pointcloud[nbrs_index]
	nbr_dz = neighborhood[:, 2] - neighborhood[:, 2].min()
	dZ_ = nbr_dz.max()
	vZ_ = np.var(nbr_dz)
	V_ = vectors_[2][2]

	features = np.asarray([C_, O_, L_, E_,  D_, S_2, L_2, dZ_, vZ_, V_])#([C_,O_,L_,E_,D_,S_2,L_2,dZ_,vZ_,V_])
	return features

def calculate_entropy_array(eigen):
	L_ = (eigen[:,0] - eigen[:,1]) / eigen[:,0]
	P_ = (eigen[:,1] - eigen[:,2]) / eigen[:,0]
	S_ = eigen[:,2] / eigen[:,0]
	Entropy = -L_*np.log(L_)-P_*np.log(P_)-S_*np.log(S_)
	return Entropy

def covariation_eigenvalue(neighborhood_index, pointcloud):
	neighborhoods = pointcloud[neighborhood_index]
	
	# 3D cov and eigen by matrix
	Ex = np.average(neighborhoods, axis=1)
	Ex = np.reshape(np.tile(Ex,[neighborhoods.shape[1]]), neighborhoods.shape)
	P = neighborhoods-Ex
	cov_ = np.matmul(P.transpose((0,2,1)),P)/(neighborhoods.shape[1]-1)
	eigen_, vec_ = np.linalg.eig(cov_)
	indices = np.argsort(eigen_)
	indices = indices[:,::-1]
	pcs_num_ = eigen_.shape[0]
	indx = np.reshape(np.arange(pcs_num_), [-1, 1])
	eig_ind = indices + indx*3
	vec_ind = np.reshape(eig_ind*3, [-1,1]) + np.full((pcs_num_*3,3), [0,1,2])
	vec_ind = np.reshape(vec_ind, [-1,3,3])
	eigen3d_ = np.take(eigen_, eig_ind)
	vectors_ = np.take(vec_, vec_ind)
	entropy_ = calculate_entropy_array(eigen3d_)

	# 2D cov and eigen
	cov2d_ = cov_[:,:2,:2]
	eigen2d, vec_2d = np.linalg.eig(cov2d_)
	indices = np.argsort(eigen2d)
	indices = indices[:, ::-1]
	pcs_num_ = eigen2d.shape[0]
	indx = np.reshape(np.arange(pcs_num_), [-1, 1])
	eig_ind = indices + indx * 2
	eigen2d_ = np.take(eigen2d, eig_ind)

	eigens_ = np.append(eigen3d_,eigen2d_,axis=1)

	return cov_, entropy_, eigens_, vectors_

def build_neighbors_NN(k, pointcloud):
	### using KNN NearestNeighbors cluster according k
	nbrs = NearestNeighbors(n_neighbors=k).fit(pointcloud)
	distances, indices = nbrs.kneighbors(pointcloud)
	covs, entropy, eigens_, vectors_ = covariation_eigenvalue(indices, pointcloud)
	neighbors = {}
	neighbors['k'] = k
	neighbors['indices'] = indices
	neighbors['covs'] = covs
	neighbors['entropy'] = entropy
	neighbors['eigens_'] = eigens_
	neighbors['vectors_'] = vectors_
	#logging.info("KNN:{}".format(k))
	return neighbors

def prepare_file(file, name, args):
	### Parallel process pointcloud files
	print("processing :",file)
	# load pointcloud file
	if args.dataset == "ThreeDMatch":
		pointcloud = np.load(file,allow_pickle=True)['pcd']
	elif args.dataset == "Kitti":
		pointcloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:,0:2]
	else:
		raise NotImplementedError
	#pointcloud = np.reshape(pointcloud, (pointcloud.shape[0]//3, 3))
	#args.pointcloud = pointcloud

	#print(pointcloud.shape)
	# prepare KNN cluster number k
	print(np.shape(pointcloud))
	cluster_number = []
	for ind in range(((args.k_end - args.k_start) // args.k_step) + 1):
		cluster_number.append(args.k_start + ind * args.k_step)

		k_nbrs = []
		for k in cluster_number:
			k_nbr = build_neighbors_NN(k, pointcloud)
			k_nbrs.append(k_nbr)

			#print("Processing pointcloud KNN Done")
	# multiprocessing pool to parallel cluster pointcloud
	#pool = multiproc.Pool(len(cluster_number))
	#build_neighbors_func = partial(build_neighbors, args=deepcopy(args))
	#k_nbrs = pool.map(build_neighbors_func, cluster_number)
	#pool.close()
	#pool.join()

	# get argmin k according E, different points may have different k
	k_entropys = []
	for k_nbr in k_nbrs:
		k_entropys.append(k_nbr['entropy'])
		argmink_ind = np.argmin(np.asarray(k_entropys), axis=0)

	points_feature = []
	for index in range(pointcloud.shape[0]):
		### per point
		neighborhood = k_nbrs[argmink_ind[index]]['indices'][index]
		eigens_ = k_nbrs[argmink_ind[index]]['eigens_'][index]
		vectors_ = k_nbrs[argmink_ind[index]]['vectors_'][index]

		# calculate point feature
		feature = calculate_features(pointcloud, neighborhood, eigens_, vectors_)
		#rint(feature.shape)
		points_feature.append(feature)
	
	points_feature = np.array(points_feature)
	#points_feature = np.expand_dims(points_feature, axis=0)
	write_path = args.fcd_path + name + '.npy'

	if os.path.exists(args.fcd_path):
		np.save(write_path,points_feature)
		f"saved files to {write_path}"
	else:
		f"{write_path} doesn't exist, skipped"

	# save to point feature folders and bin files
	#feature_cloud = np.append(pointcloud, points_feature, axis=1)
	#pointfile_path, pointfile_name = os.path.split(pointcloud_file)
	#filepath = os.path.join(os.path.split(pointfile_path)[0], args.featurecloud_fols, pointfile_name)
	#feature_cloud.tofile(filepath)

	# build KDTree and store fot the knn query
	#kdt = KDTree(pointcloud, leaf_size=50)
	#treepath = os.path.splitext(filepath)[0] + '.pickle'
	#with open(treepath, 'wb') as handel:
	#	pickle.dump(kdt, handel)

	#logging.info("Feature cloud file saved:{}".format(filepath))

def prepare_dataset(args):
	### Parallel process dataset folders
	# Initialize pandas DataFrame

	# creat feature_cloud folder
	if not os.path.exists(args.fcd_path):
		try:
			os.makedirs(args.fcd_path)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
				pass

	if args.dataset == "ThreeDMatch":
		pointcloud_files = glob.glob(args.BASE_DIR + '*.npz')
		base_dir = args.BASE_DIR
		ntpath.basename(base_dir)
	elif args.dataset == "Kitti":
		pointcloud_files = glob.glob(args.BASE_DIR + '%02d/velodyne/*.bin' % args.pointcloud_folder) 
		base_dir = args.BASE_DIR + '%02d/velodyne/'
		ntpath.basename(base_dir)
	else:
		raise NotImplementedError

	# multiprocessing pool to parallel process pointcloud_files
	pool = multiproc.Pool(args.bin_core_num)
	for file in pointcloud_files:
		abs_dir = os.path.splitext(file)[0]
		head, name = ntpath.split(abs_dir)
		#print(head,name)
		#print(name)	
		export_name = args.fcd_path + name + '.npy'	
		if not os.path.exists(export_name):
			#timer = Timer()
			#timer.tic()
		#	prepare_file(file, name, args)
			pool.apply_async(prepare_file,(file, name, deepcopy(args)))
		else:
			f"{export_name} exists, skipped"
		#time = timer.toc()
		#print("Feature Process Time:",time) 
	
	pool.close()
	f"Cloud folder processing:{args.fcd_path}"
	pool.join()
	f"end folder processing"
	print("="*30)

def run_all_processes(all_p):
	try:
		for p in all_p:
			p.start()
			for p in all_p:
				p.join()
	except KeyboardInterrupt:
		for p in all_p:
			if p.is_alive():
				p.terminate()
			p.join()
		exit(-1)

def main(args):
	# prepare dataset folders
	if args.dataset == "ThreeDMatch":
		args.BASE_DIR = '/disk_ssd/threedmatch/'
		args.fcd_path = '/disk_ssd/threedmatch/features/'
		#filenames = glob.glob(BASE_DIR + '*.npz')
		prepare_dataset(args)
	elif args.dataset == "Kitti":
		args.BASE_DIR = '/disk_ssd/kitti/dataset/sequences/'
		args.folders = folders = np.arange(args.max_folder)
		all_p = []
		for folder in folders:
			args.pointcloud_folder = folder
			args.fcd_path = args.BASE_DIR + '%02d/features/' % folder
			all_p.append(multiproc.Process(target=prepare_dataset, args=(deepcopy(args),)))
	
		run_all_processes(all_p)
	else:
		raise NotImplementedError

	#print(np.shape(filenames[0]))
	f"Dataset preparation Finised"

if __name__ == '__main__':
	parse = argparse.ArgumentParser(sys.argv[0])
	parse.add_argument('--k_start', type=int, default=20,
						help="KNN cluster k range start point")
	parse.add_argument('--k_end', type=int, default=100,
						help="KNN cluster k range end point")
	parse.add_argument('--k_step', type=int, default=10,
						help="KNN cluster k range step")
	parse.add_argument('--bin_core_num', type=int, default=10, help="Parallel process file Pool core num")
	parse.add_argument('--dataset', type=str, default="ThreeDMatch", 
						help="Possible Choice:ThreeDMatch,Kitti")
	parse.add_argument('--max_folder',type=int, default=10)

	args = parse.parse_args(sys.argv[1:])
	main(args)

