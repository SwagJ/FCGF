# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib

from util.pointcloud import get_matching_indices, make_open3d_point_cloud, get_neighbor_indices, get_hardest_neigative_indices
import lib.transforms as t

import MinkowskiEngine as ME

import open3d as o3d

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from lib.timer import Timer
from lib.metrics import pdist
#from lib.feature_extraction import prepare_feature
import ntpath
from sklearn.preprocessing import normalize


kitti_cache = {}
kitti_icp_cache = {}
kaist_cache = {}
kaist_icp_cache = {}


def collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, sample_num, neighbor0, neighbor1,num_neighbor0,num_neighbor1 = list(
      zip(*list_data))
  xyz_batch0, xyz_batch1 = [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []
  coords0_in_batch,coords1_in_batch = [],[]
  unique_match = []
  centroid_in_batch = []
  max_in_batch = []
  neighbor0_in_batch = []
  neighbor1_in_batch = []
  num_neighbor0_in_batch = []
  num_neighbor1_in_batch = []
  correspondance_in_batch = []
  #neg_ind_in_batch=[]
  pos1_coord_in_batch = []
  pos0_coord_in_batch = []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  #print(sample_num)
  sample_num = min(sample_num)



  def to_tensor(x):
    if isinstance(x, torch.Tensor):
      return x
    elif isinstance(x, np.ndarray):
      return torch.from_numpy(x)
    else:
      raise ValueError(f'Can not convert to torch tensor, {x}')

  for batch_id, _ in enumerate(coords0):
    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]
    #print("num of points:",N0,N1)
    #print("neighborhood0 size:",neighbor0[batch_id].shape)
    #print("neighborhood1 size:",neighbor1[batch_id].shape)
    neighbor0_in_batch.append(to_tensor(neighbor0[batch_id]))
    neighbor1_in_batch.append(to_tensor(neighbor1[batch_id]))

    num_neighbor0_in_batch.append(to_tensor(num_neighbor0[batch_id]))
    num_neighbor1_in_batch.append(to_tensor(num_neighbor1[batch_id]))

    xyz_batch0.append(to_tensor(xyz0[batch_id]))
    xyz_batch1.append(to_tensor(xyz1[batch_id]))
    #points normalization
    coord0_i = to_tensor(coords0[batch_id]).float()
    coord1_i = to_tensor(coords1[batch_id]).float()
    #print("coord0_i shape:",coord0_i.shape)
    #print("coord1_i shape:",coord1_i.shape)

    batch_corr = np.array(matching_inds[batch_id])

    # normalization
    centroid0 = torch.mean(coord0_i, axis=0)
    centroid1 = torch.mean(coord1_i, axis=0)
    centroid_in_batch.append([centroid0,centroid1])
    #print(centroid0,centroid1)
    centered0 = coord0_i - centroid0
    centered1 = coord1_i - centroid1
    max0 = torch.max(torch.sqrt(torch.sum(abs(centered0)**2,axis=-1)))
    max1 = torch.max(torch.sqrt(torch.sum(abs(centered1)**2,axis=-1)))
    max_in_batch.append([max0,max1])
    normed_coords0 = centered0 / max0
    normed_coords1 = centered1 / max1

    #sample points in batch
    sel0 = np.random.choice(N0, sample_num, replace=False)
    sel1 = np.random.choice(N1, sample_num, replace=False)


    coords0_in_batch.append(normed_coords0[sel0,:].float())
    coords1_in_batch.append(normed_coords1[sel1,:].float())

    #sample positive correspondance
    _,unique_idx = np.unique(batch_corr[:,0],return_index=True)
    #unique_idx = torch.from_numpy(unique_idx)
    corr_match_idx = batch_corr[unique_idx]
    pos0_coord = normed_coords0[corr_match_idx[:,0],:]
    pos1_coord = normed_coords1[corr_match_idx[:,1],:]
    pos0_coord_in_batch.append(to_tensor(pos0_coord))
    pos1_coord_in_batch.append(to_tensor(pos1_coord))


    trans_batch.append(to_tensor(trans[batch_id]))
    #neg_ind_in_batch.append(to_tensor(neg_inds[batch_id]).int())

    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id])).int())

    correspondance_in_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  #print("before sparse_collate:",coords0[0].shape)
  coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
  coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)
  #print("after sparse_collate:",coords_batch0.shape)

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  trans_batch = torch.cat(trans_batch, 0).float()
  #matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  correspondance_in_batch = torch.cat(correspondance_in_batch, 0).int()
  coords0_downsampled = torch.stack(coords0_in_batch,0).float()
  coords1_downsampled = torch.stack(coords1_in_batch,0).float()
  #coords0 = to_tensor(np.array(coords0).astype(np.float32))
  #coords1 = to_tensor(np.array(coords1).astype(np.float32))
  #print(matching_inds_batch[0:100,:])


  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0.float(),
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1.float(),
      'correspondences': correspondance_in_batch,
    #  'matching_inds': matching_inds_batch,
    #  'pos0' : pos0_coord_in_batch,
    #  'pos1' : pos1_coord_in_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch,
      'coords0': coords0_downsampled,
      'coords1': coords1_downsampled,
      'centroid': centroid_in_batch,
      'max': max_in_batch,
      'neighbor0': neighbor0_in_batch,
      'neighbor1': neighbor1_in_batch,
      'num_neighbor0': num_neighbor0_in_batch,
      'num_neighbor1': num_neighbor1_in_batch,
      #'neg_inds': neg_ind_in_batch
  }


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    #self.get_feature = config.get_feature
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)


class IndoorPairDataset(PairDataset):
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.threed_match_dir
    self.get_feature = config.get_feature
    self.sample_num = config.sample_num
    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for name in subset_names:
      fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
      fnames_txt = glob.glob(root + "/" + fname)
      assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
      for fname_txt in fnames_txt:
        with open(fname_txt) as f:
          content = f.readlines()
        fnames = [x.strip().split() for x in content]
        for fname in fnames:
          self.files.append([fname[0], fname[1]])

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    abs_dir0 = os.path.splitext(file0)[0]
    abs_dir1 = os.path.splitext(file1)[0]
    ntpath.basename(self.root)
    featname0 = self.root + 'features/' + ntpath.split(abs_dir0)[1] + '.npy'
    featname1 = self.root + 'features/' + ntpath.split(abs_dir1)[1] + '.npy'
    data0 = np.load(file0,allow_pickle=True)
    data1 = np.load(file1,allow_pickle=True)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    color0 = data0["color"]
    color1 = data1["color"]
    matching_search_voxel_size = self.matching_search_voxel_size


    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    #print("before:", xyz0.shape)
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)


    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    # Get neighborhoods of pcd0 and pcd1
    neighbor0_mask,num_neighbor0 = get_neighbor_indices(pcd0, 0.025*2.5)
    neighbor1_mask,num_neighbor1 = get_neighbor_indices(pcd1, 0.025*2.5)
    #print("pcd1 points shape:",np.shape(pcd1.points))

    #neg_inds = get_hardest_neigative_indices(pcd1,0.125)
    #print("neg_inds shape:",neg_inds.shape)

    #num_neighbor0 = np.count_nonzero(neighbor0_mask != len(pcd0.points),axis=1)
    #num_neighbor1 = np.count_nonzero(neighbor1_mask != len(pcd1.points),axis=1)
    #print(num_neighbor0.shape)
    #print(matches[0:100,:])

    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)
    #print("pcd0 color dim", np.shape(pcd0.colors))
    #print("pcd1 color dim", np.shape(pcd1.colors))
    #print("pcd0 color len:",npts0)
    #print("pcd1 color len:",npts1)
    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((npts0, 1)))
    feats_train1.append(np.ones((npts1, 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)
    #print("after:",xyz0.shape)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)
    #print("Max in pcd0 coord:",np.max(coords0,axis=0))
    #print("Min in pcd0 coord:",np.min(coords0,axis=0))
    #print("Max in pcd1 coord:",np.max(coords1,axis=0))
    #print("Min in pcd1 coord:",np.min(coords1,axis=0))

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    if self.get_feature == True:
      feats0 = np.load(featname0, allow_pickle=True)
      feats1 = np.load(featname1, allow_pickle=True)
      feats0 = feats0[sel0]
      feats1 = feats1[sel1]
      feats0 = normalize(feats0)
      feats1 = normalize(feats1)

    #print("coords shape:",coords0.shape)
    #print("feature shape:", feats0.shape)
    if self.sample_num > min(npts0,npts1):
      sample_num = min(npts0,npts1)
      #print(sample_num)
    else:
      sample_num = self.sample_num


    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans, sample_num, neighbor0_mask, neighbor1_mask,num_neighbor0,num_neighbor1)


class KITTIPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './config/train_kitti.txt',
      'val': './config/val_kitti.txt',
      'test': './config/test_kitti.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method
    if self.IS_ODOMETRY:
      self.root = root = config.kitti_root + '/dataset'
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kitti_date
      self.root = root = os.path.join(config.kitti_root, self.date)

    self.icp_path = os.path.join(config.kitti_root, 'icp')
    self.get_feature = config.get_feature
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the kitti root
    self.max_time_diff = max_time_diff = config.kitti_max_time_diff

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = int(dirname)
      inames = self.get_all_scan_ids(drive_id)
      for start_time in inames:
        for time_diff in range(2, max_time_diff):
          pair_time = time_diff + start_time
          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def get_all_scan_ids(self, drive_id):
    if self.IS_ODOMETRY:
      fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
    else:
      fnames = glob.glob(self.root + '/' + self.date +
                         '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
    assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  @property
  def velo2cam(self):
    try:
      velo2cam = self._velo2cam
    except AttributeError:
      R = np.array([
          7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
          -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
      ]).reshape(3, 3)
      T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
      velo2cam = np.hstack([R, T])
      self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return self._velo2cam

  def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
    if self.IS_ODOMETRY:
      data_path = self.root + '/poses/%02d.txt' % drive
      #print(data_path)
      if data_path not in kitti_cache:
        kitti_cache[data_path] = np.genfromtxt(data_path)
      if return_all:
        return kitti_cache[data_path]
      else:
        return kitti_cache[data_path][indices]
    else:
      data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
      odometry = []
      if indices is None:
        fnames = glob.glob(self.root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
        indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

      for index in indices:
        filename = os.path.join(data_path, '%010d%s' % (index, ext))
        if filename not in kitti_cache:
          kitti_cache[filename] = np.genfromtxt(filename)
        odometry.append(kitti_cache[filename])

      odometry = np.array(odometry)
      return odometry

  def odometry_to_positions(self, odometry):
    if self.IS_ODOMETRY:
      T_w_cam0 = odometry.reshape(3, 4)
      T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
      return T_w_cam0
    else:
      lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

      R = 6378137  # Earth's radius in metres

      # convert to metres
      lat, lon = np.deg2rad(lat), np.deg2rad(lon)
      mx = R * lon * np.cos(lat)
      my = R * lat

      times = odometry.T[-1]
      return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

  def rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m

  def pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT

  def get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)

  def _get_velodyne_fn(self, drive, t):
    if self.IS_ODOMETRY:
      fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
    else:
      fname = self.root + \
          '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
              drive, t)
    return fname

  def _get_feature_fn(self, drive, t):
    if self.IS_ODOMETRY:
      fname = self.root + '/sequences/%02d/features/%06d.npy' % (drive, t)
    else:
      fname = self.root + \
          '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
              drive, t)
    return fname

  def __getitem__(self, idx):
    drive = self.files[idx][0]
    t0, t1 = self.files[idx][1], self.files[idx][2]
    all_odometry = self.get_video_odometry(drive, [t0, t1])
    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
    fname0 = self._get_velodyne_fn(drive, t0)
    fname1 = self._get_velodyne_fn(drive, t1)
    if self.get_feature == True:
      featname0 = self._get_feature_fn(drive,t0)
      featname1 = self._get_feature_fn(drive,t1)

    # XYZ and reflectance
    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

    xyz0 = xyzr0[:, :3]
    xyz1 = xyzr1[:, :3]

    key = '%d_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in kitti_icp_cache:
      if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm
        sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(self.velo2cam)).T
        xyz0_t = self.apply_transform(xyz0[sel0], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1[sel1])
        reg = o3d.registration.registration_icp(
            pcd0, pcd1, 0.2, np.eye(4),
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename,allow_pickle=True)
      kitti_icp_cache[key] = M2
    else:
      M2 = kitti_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    if len(matches) < 1000:
      raise ValueError(f"{drive}, {t0}, {t1}")

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    feats_train0.append(torch.ones((npts0, 1)))
    feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    if self.get_feature == True:
      feats0 = np.load(featname0, allow_pickle=True)
      feats1 = np.load(featname1, allow_pickle=True)
      feats0 = feats0[sel0]
      feats1 = feats1[sel1]
      feats0 = normalize(feats0)
      feats1 = normalize(feats1)

    return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
            coords1.int(), feats0.float(), feats1.float(), matches, trans)


class KITTINMPairDataset(KITTIPairDataset):
  r"""
  Generate KITTI pairs within N meter distance
  """
  MIN_DIST = 10

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    if self.IS_ODOMETRY:
      self.root = root = os.path.join(config.kitti_root, 'dataset')
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kitti_date
      self.root = root = os.path.join(config.kitti_root, self.date)

    self.icp_path = os.path.join(config.kitti_root, 'icp')
    self.get_feature = config.get_feature
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if self.IS_ODOMETRY:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        #print(fnames)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        #print(inames)

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))
        valid_pairs = pdist > self.MIN_DIST
        #print(np.shape(valid_pairs))
        curr_time = inames[0]
        while curr_time in inames:
          # Find the min index
          #print(curr_time)
          next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
          if len(next_time) == 0:
            curr_time += 1
          else:
            # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
            next_time = next_time[0] + curr_time - 1

          if next_time in inames:
            #print("Appending Files")
            self.files.append((drive_id, curr_time, next_time))
            curr_time = next_time + 1
    else:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, 0, :3]

        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))

        for start_time in inames:
          pair_time = np.where(
              pdist[start_time][start_time:start_time + 100] > self.MIN_DIST)[0]
          if len(pair_time) == 0:
            continue
          else:
            pair_time = pair_time[0] + start_time

          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

    if self.IS_ODOMETRY:
      # Remove problematic sequence
      for item in [
          (8, 15, 58),
      ]:
        print("items are ",item)
        if item in self.files:
          self.files.pop(self.files.index(item))


#########################################################
#
#              KAIST_DATASET_Left
#
# added config: kaist_root,kaist_date,kaist_max_time_diff
#       train_kaist.txt,val_kaist.txt,test_kaist.txt
#
#########################################################

class KAISTLPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './config/train_kaist.txt',
      'val': './config/val_kaist.txt',
      'test': './config/test_kaist.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method
    if self.IS_ODOMETRY:
      self.root = root = config.kaist_root
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kaist_date
      self.root = root = os.path.join(config.kaist_root, self.date)

    self.icp_path = os.path.join(config.kaist_root, 'icp')
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the kitti root
    self.max_time_diff = max_time_diff = config.kaist_max_time_diff

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = int(dirname)
      inames = self.get_all_scan_ids(drive_id)
      for start_time in inames:
        for time_diff in range(2, max_time_diff):
          pair_time = time_diff + start_time
          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def get_all_scan_ids(self, drive_id):
    load_path = self.root + '/%02d/' % drive_id + 'left_valid.csv'
    valid_time = np.genfromtxt(load_path,delimiter=',')
    fnames = []
    for time in valid_time:
      if self.IS_ODOMETRY:
        fnames.extend(glob.glob(self.root + '/%02d/VLP_left/' % drive_id + '%06d.bin' % time))
      else:
        fnames.extend(glob.glob(self.root + '/' + self.date +
                         '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id))
    assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"

    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  @property
  def velo2cam(self):
    try:
      velo2cam = self._velo2cam
    except AttributeError:
      R = np.array([
          7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
          -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
      ]).reshape(3, 3)
      T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
      velo2cam = np.hstack([R, T])
      self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return self._velo2cam

  def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
    if self.IS_ODOMETRY:
      #logging.info(f"Drive id is {drive}")
      data_path = self.root + '/%02d/VLP_left_pose.csv' % drive
      #print(data_path)
      if data_path not in kaist_cache:
        kaist_cache[data_path] = np.genfromtxt(data_path,delimiter=',')
        #print(np.shape(kaist_cache[data_path]))
      if return_all:
        return kaist_cache[data_path]
      else:
        return kaist_cache[data_path][indices]
    else: 
      data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
      odometry = []
      if indices is None:
        fnames = glob.glob(self.root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
        indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

      for index in indices:
        filename = os.path.join(data_path, '%010d%s' % (index, ext))
        if filename not in kaist_cache:
          kaist_cache[filename] = np.genfromtxt(filename)
        odometry.append(kaist_cache[filename])

      odometry = np.array(odometry)
      return odometry

  def odometry_to_positions(self, odometry):
    if self.IS_ODOMETRY:
      T_w_cam0 = odometry.reshape(3, 4)
      T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
      return T_w_cam0
    else:
      lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

      R = 6378137  # Earth's radius in metres

      # convert to metres
      lat, lon = np.deg2rad(lat), np.deg2rad(lon)
      mx = R * lon * np.cos(lat)
      my = R * lat

      times = odometry.T[-1]
      return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

  def rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m

  def pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT

  def get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)

  def _get_velodyne_fn(self, drive, t):
    if self.IS_ODOMETRY:
      fname = self.root + '/%02d/VLP_left/%06d.bin' % (drive, t)
    else:
      fname = self.root + \
          '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
              drive, t)
    return fname

  def __getitem__(self, idx):
    drive = self.files[idx][0]
    inames = self.get_all_scan_ids(drive)
    t0, t1 = self.files[idx][1], self.files[idx][2]
    t0_odo,t1_odo = self.files[idx][1] - inames[0],self.files[idx][2] - inames[0]
    all_odometry = self.get_video_odometry(drive, [t0_odo, t1_odo])
    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
    fname0 = self._get_velodyne_fn(drive, t0)
    fname1 = self._get_velodyne_fn(drive, t1)

    # XYZ and reflectance
    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

    xyz0 = xyzr0[:, :3]
    xyz1 = xyzr1[:, :3]

    key = '%d_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in kaist_icp_cache:
      if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm
        sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(self.velo2cam)).T
        xyz0_t = self.apply_transform(xyz0[sel0], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1[sel1])
        reg = o3d.registration.registration_icp(
            pcd0, pcd1, 0.2, np.eye(4),
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename,allow_pickle=True)
      kaist_icp_cache[key] = M2
    else:
      M2 = kaist_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    #if len(matches) < 1000:
      #raise ValueError(f"{drive}, {t0}, {t1}")
      #return

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    feats_train0.append(torch.ones((npts0, 1)))
    feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
            coords1.int(), feats0.float(), feats1.float(), matches, trans)



class KAISTLNMPairDataset(KAISTLPairDataset):
  r"""
  Generate KITTI pairs within N meter distance
  """
  MIN_DIST = 10

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    if self.IS_ODOMETRY:
      self.root = root = config.kaist_root
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kaist_date
      self.root = root = os.path.join(config.kaist_root, self.date)

    self.icp_path = os.path.join(config.kaist_root, 'icp')
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if self.IS_ODOMETRY:
      for dirname in subset_names:
        drive_id = int(dirname)
        logging.info(f"Processing Sequence {drive_id}")
        self.kaist_interpolate(left=True,data_root=root,drive_id=drive_id)
        load_path = root + '/%02d/' % drive_id + 'left_valid.csv'
        valid_time = np.genfromtxt(load_path,delimiter=',')
        fnames = []
        for time in valid_time:
          fnames.extend(glob.glob(self.root + '/%02d/VLP_left/%06d.bin' % (drive_id, time)))

        #print(fnames)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        #print(inames)

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        #print(all_odo.shape)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        logging.info(f"Position Acquisition Done")
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))
        valid_pairs = pdist > self.MIN_DIST
        #print(np.shape(valid_pairs))
        curr_time = inames[0]
        loop_names = np.asarray(inames) - inames[0]
        while curr_time in loop_names:
          #print(curr_time)
          # Find the min index
          next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
          #print(next_time)
          if len(next_time) == 0:
            curr_time += 1
          else:
            # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
            next_time = next_time[0] + curr_time - 1

          if next_time in inames:
            if self.valid_pair(drive_id,curr_time + inames[0],next_time + inames[0], inames[0]):
            #print("Appending Files")
              self.files.append((drive_id, curr_time + inames[0], next_time + inames[0]))
            #logging.info(f"current time_stamp {curr_time}")
            curr_time = next_time + 1
    else:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, 0, :3]

        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))

        for start_time in inames:
          pair_time = np.where(
              pdist[start_time][start_time:start_time + 100] > self.MIN_DIST)[0]
          if len(pair_time) == 0:
            continue
          else:
            pair_time = pair_time[0] + start_time

          if pair_time in inames:
              self.files.append((drive_id, start_time, pair_time))

    logging.info(f"length of files found {len(self.files)}")

  def kaist_interpolate(self,left=None,data_root=None,drive_id=None):
    pose_path = data_root + '/' + '%02d/' % drive_id + 'global_pose.csv'
    pose = np.genfromtxt(pose_path,delimiter=',')
    if left == True:
      stamp_path = data_root + '/' + '%02d/' % drive_id + 'VLP_left_stamp.csv'
      write_VLP_path = data_root + '/' + '%02d/' % drive_id + 'VLP_left_pose.csv'
      write_valid_path = data_root + '/' + '%02d/' % drive_id + 'left_valid.csv'
      time_stamp = np.genfromtxt(stamp_path,delimiter=',')
    else: 
      stamp_path = data_root + '/' + '%02d/' % drive_id + 'VLP_right_stamp.csv'
      write_VLP_path = data_root + '/' + '%02d/' % drive_id + 'VLP_right_pose.csv'
      write_valid_path = data_root + '/' + '%02d/' % drive_id + 'right_valid.csv'
      time_stamp = np.genfromtxt(stamp_path,delimiter=',')

    time = pose[:,0]
    pose = pose[:,1:13]

    pose = pose.reshape(-1,3,4)
    rotate = pose[:,:,0:3]
    translation = pose[:,:,3]

    valid_start = (time_stamp > np.min(time)).astype(int)
    valid_end = (time_stamp < np.max(time)).astype(int)
    valid_idx = np.where(valid_start*valid_end == 1)
    valid_time = time_stamp[valid_idx]

    # interpolate translation matrix
    trans_interpolate = interp1d(time,translation,kind='linear',axis=0)
    trans = trans_interpolate(valid_time).reshape(valid_time.shape[0],3,1)

    # interpolate rotation matrix
    slerp = Slerp(time,R.from_matrix(rotate))
    rotate = slerp(valid_time).as_matrix()

    VLP_pose = np.concatenate((rotate,trans),axis=2).reshape(valid_time.shape[0],12)

    valid_idx = np.asarray(valid_idx).T
    np.savetxt(write_VLP_path,VLP_pose,delimiter=',')
    np.savetxt(write_valid_path,valid_idx,delimiter=',')  
    
    return

  def valid_pair(self,drive=None,t0=None,t1=None,offset=None):
    #logging.info(f"Function called")
    all_odometry = self.get_video_odometry(drive, [t0-offset, t1-offset])
    #print(np.shape(all_odometry))
    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
    fname0 = self.root + '/%02d/VLP_left/%06d.bin' % (drive, t0)
    fname1 = self.root + '/%02d/VLP_left/%06d.bin' % (drive, t1)
    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

    xyz0 = xyzr0[:, :3]
    xyz1 = xyzr1[:, :3]

    key = '%d_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in kitti_icp_cache:
      if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm
        sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(self.velo2cam)).T
        xyz0_t = self.apply_transform(xyz0[sel0], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1[sel1])
        reg = o3d.registration.registration_icp(
            pcd0, pcd1, 0.2, np.eye(4),
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename,allow_pickle=True)
      kitti_icp_cache[key] = M2
    else:
      M2 = kitti_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    if len(matches) < 1000:
      return False
    else:
      return True

    #if self.IS_ODOMETRY:
      # Remove problematic sequence
     # for item in [
     #     (8, 15, 58),
     # ]:
     #   print("items are ",item)
     #   if item in self.files:
     #     self.files.pop(self.files.index(item))

#########################################################
#
#              KAIST_DATASET_Right
#
# added config: kaist_root,kaist_date,kaist_max_time_diff
#       train_kaist.txt,val_kaist.txt,test_kaist.txt
#
#########################################################

class KAISTRPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './config/train_kaist.txt',
      'val': './config/val_kaist.txt',
      'test': './config/test_kaist.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method
    if self.IS_ODOMETRY:
      self.root = root = config.kaist_root
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kaist_date
      self.root = root = os.path.join(config.kaist_root, self.date)

    self.icp_path = os.path.join(config.kaist_root, 'icp')
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the kitti root
    self.max_time_diff = max_time_diff = config.kaist_max_time_diff

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = int(dirname)
      inames = self.get_all_scan_ids(drive_id)
      for start_time in inames:
        for time_diff in range(2, max_time_diff):
          pair_time = time_diff + start_time
          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def get_all_scan_ids(self, drive_id):
    load_path = self.root + '/%02d/' % drive_id + 'right_valid.csv'
    valid_time = np.genfromtxt(load_path,delimiter=',')
    fnames = []
    for time in valid_time:
      if self.IS_ODOMETRY:
        fnames.extend(glob.glob(self.root + '/%02d/VLP_right/' % drive_id + '%06d.bin' % time))
      else:
        fnames.extend(glob.glob(self.root + '/' + self.date +
                         '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id))
      assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"

    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  @property
  def velo2cam(self):
    try:
      velo2cam = self._velo2cam
    except AttributeError:
      R = np.array([
          7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
          -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
      ]).reshape(3, 3)
      T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
      velo2cam = np.hstack([R, T])
      self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return self._velo2cam

  def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
    if self.IS_ODOMETRY:
      data_path = self.root + '/%02d/VLP_right_pose.csv' % drive
      if data_path not in kaist_cache:
        kaist_cache[data_path] = np.genfromtxt(data_path,delimiter=',')
      if return_all:
        return kaist_cache[data_path]
      else:
        return kaist_cache[data_path][indices]
    else:
      data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
      odometry = []
      if indices is None:
        fnames = glob.glob(self.root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
        indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

      for index in indices:
        filename = os.path.join(data_path, '%010d%s' % (index, ext))
        if filename not in kaist_cache:
          kaist_cache[filename] = np.genfromtxt(filename)
        odometry.append(kaist_cache[filename])

      odometry = np.array(odometry)
      return odometry

  def odometry_to_positions(self, odometry):
    if self.IS_ODOMETRY:
      T_w_cam0 = odometry.reshape(3, 4)
      T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
      return T_w_cam0
    else:
      lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

      R = 6378137  # Earth's radius in metres

      # convert to metres
      lat, lon = np.deg2rad(lat), np.deg2rad(lon)
      mx = R * lon * np.cos(lat)
      my = R * lat

      times = odometry.T[-1]
      return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

  def rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m

  def pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT

  def get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)

  def _get_velodyne_fn(self, drive, t):
    if self.IS_ODOMETRY:
      fname = self.root + '/%02d/VLP_right/%06d.bin' % (drive, t)
    else:
      fname = self.root + \
          '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
              drive, t)
    return fname

  def __getitem__(self, idx):
    drive = self.files[idx][0]
    t0, t1 = self.files[idx][1], self.files[idx][2]
    all_odometry = self.get_video_odometry(drive, [t0, t1])
    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
    fname0 = self._get_velodyne_fn(drive, t0)
    fname1 = self._get_velodyne_fn(drive, t1)

    # XYZ and reflectance
    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

    xyz0 = xyzr0[:, :3]
    xyz1 = xyzr1[:, :3]

    key = '%d_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in kaist_icp_cache:
      if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm
        sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(self.velo2cam)).T
        xyz0_t = self.apply_transform(xyz0[sel0], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1[sel1])
        reg = o3d.registration.registration_icp(
            pcd0, pcd1, 0.2, np.eye(4),
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename,allow_pickle=True)
      kaist_icp_cache[key] = M2
    else:
      M2 = kaist_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    if len(matches) < 1000:
      raise ValueError(f"{drive}, {t0}, {t1}")

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    feats_train0.append(torch.ones((npts0, 1)))
    feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
            coords1.int(), feats0.float(), feats1.float(), matches, trans)



class KAISTRNMPairDataset(KAISTRPairDataset):
  r"""
  Generate KITTI pairs within N meter distance
  """
  MIN_DIST = 10

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    if self.IS_ODOMETRY:
      self.root = root = config.kaist_root
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kaist_date
      self.root = root = os.path.join(config.kaist_root, self.date)

    self.icp_path = os.path.join(config.kaist_root, 'icp')
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if self.IS_ODOMETRY:
      for dirname in subset_names:
        drive_id = int(dirname)
        #print("Processing sequence ", drive_id)
        self.kaist_interpolate(left=False,data_root=root,drive_id=drive_id)
        load_path = root + '/%02d/' % drive_id + 'right_valid.csv'
        valid_time = np.genfromtxt(load_path,delimiter=',')
        fnames = []
        for time in valid_time:
          fnames.extend(glob.glob(self.root + '/%02d/VLP_right/' % drive_id + '%06d.bin' % time))
        
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        #print(inames)

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        #print("Odometry and position acquisition done")
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))
        valid_pairs = pdist > self.MIN_DIST
        #print("Valid Pairs Generated")
        curr_time = inames[0] - inames[0]
        while curr_time in inames:
          # Find the min index
          next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
          if len(next_time) == 0:
            curr_time += 1
          else:
            # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
            next_time = next_time[0] + curr_time - 1

          if next_time in inames:
            self.files.append((drive_id, curr_time + inames[0], next_time + inames[0]))
            curr_time = next_time + 1
    else:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, 0, :3]

        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))

        for start_time in inames:
          pair_time = np.where(
              pdist[start_time][start_time:start_time + 100] > self.MIN_DIST)[0]
          if len(pair_time) == 0:
            continue
          else:
            pair_time = pair_time[0] + start_time

          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def kaist_interpolate(self,left=True,data_root=None,drive_id=None):
    pose_path = data_root + '/' + '%02d/' % drive_id + 'global_pose.csv'
    pose = np.genfromtxt(pose_path,delimiter=',')
    if left == True:
      stamp_path = data_root + '/' + '%02d/' % drive_id + 'VLP_left_stamp.csv'
      write_VLP_path = data_root + '/' + '%02d/' % drive_id + 'VLP_left_pose.csv'
      write_valid_path = data_root + '/' + '%02d/' % drive_id + 'left_valid.csv'
      time_stamp = np.genfromtxt(stamp_path,delimiter=',')
    else: 
      stamp_path = data_root + '/' + '%02d/' % drive_id + 'VLP_right_stamp.csv'
      write_VLP_path = data_root + '/' + '%02d/' % drive_id + 'VLP_right_pose.csv'
      write_valid_path = data_root + '/' + '%02d/' % drive_id + 'right_valid.csv'
      time_stamp = np.genfromtxt(stamp_path,delimiter=',')

    time = pose[:,0]
    pose = pose[:,1:13]

    pose = pose.reshape(-1,3,4)
    rotate = pose[:,:,0:3]
    translation = pose[:,:,3]

    valid_start = (time_stamp > np.min(time)).astype(int)
    valid_end = (time_stamp < np.max(time)).astype(int)
    valid_idx = np.where(valid_start*valid_end == 1)
    valid_time = time_stamp[valid_idx]

    # interpolate translation matrix
    trans_interpolate = interp1d(time,translation,kind='linear',axis=0)
    trans = trans_interpolate(valid_time).reshape(valid_time.shape[0],3,1)

    # interpolate rotation matrix
    slerp = Slerp(time,R.from_matrix(rotate))
    rotate = slerp(valid_time).as_matrix()

    VLP_pose = np.concatenate((rotate,trans),axis=2).reshape(valid_time.shape[0],12)

    valid_idx = np.asarray(valid_idx).T
    np.savetxt(write_VLP_path,VLP_pose,delimiter=',')
    np.savetxt(write_valid_path,valid_idx,delimiter=',')  
    
    return 

    #if self.IS_ODOMETRY:
      # Remove problematic sequence
     # for item in [
     #     (8, 15, 58),
     # ]:
     #   print("items are ",item)
     #   if item in self.files:
     #     self.files.pop(self.files.index(item))


class ThreeDMatchPairDataset(IndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './config/train_3dmatch.txt',
      'val': './config/val_3dmatch.txt',
      'test': './config/test_3dmatch.txt'
  }


ALL_DATASETS = [ThreeDMatchPairDataset, KITTIPairDataset, KITTINMPairDataset,KAISTLNMPairDataset
                ,KAISTRNMPairDataset,KAISTLPairDataset,KAISTRPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  dset = Dataset(
      phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config)

  print(len(dset))

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      drop_last=True)

  return loader
