# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash
#from GPUtil import showUtilization as gpu_usage

import MinkowskiEngine as ME
from model.detection_net import DetectionNetHead
from model.resunet import ResUNetBN2C
from model.con_spa_net import ContextNet
import batch_find_neighbors


class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    self.model_n_out = config.model_n_out
    #print("config.get_feature:",config.get_feature)
    self.config = config

    if config.get_feature == True:
      num_feats = 10

    Model = load_model(config.model)
    #print("Config.backbone_model:",config.backbone_model)
    if config.backbone_model != 'None':
      #print("Using Backbone Model")
      self.backbone_model = load_model(config.backbone_model)
      model = Model(
        self.device,
        config.batch_size,
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        backbone_model=self.backbone_model,
        D=3)
      self.model = model
      self.model = self.model.to(self.device)
    else:
      if config.model == 'DetectionNetHead':
        model = DetectionNetHead(in_channels=3,keypoint_num=self.config.keypoint_num)
        self.model = model
        self.model = self.model.to(self.device)
      else:
        model = Model(
          num_feats,
          config.model_n_out,
          bn_momentum=config.bn_momentum,
          normalize_feature=config.normalize_feature,
          conv1_kernel_size=config.conv1_kernel_size,
          D=3)
        self.model = model
        self.model = self.model.to(self.device)

    if config.weights:
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)
    self.keypoint_num = config.keypoint_num
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf
    #self.backbone_model = config.backbone_model

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.optimizer = getattr(optim, config.optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    print(config)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader
    self.sample_num = config.sample_num

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)
    #logging.info(f"num_pos is {config.triplet_num_pos}")

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)


    for epoch in range(self.start_epoch, self.max_epoch + 1):
      ckpt_filename = 'checkpoint_' + str(epoch)
      lr = self.scheduler.get_last_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch,ckpt_filename)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        #print(input_dict['sinput0_F'].shape)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
        F1 = self.model(sinput1).F

        N0, N1 = len(sinput0), len(sinput1)

        pos_pairs = input_dict['correspondences']
        neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        pos_pairs = pos_pairs.long().to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_thresh -
                          ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      #print(input_dict['sinput0_F'].shape)
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
      F0 = self.model(sinput0).F
      output0 = self.model(sinput0)
      #dense_tensor = output0.dense()
      #print("output dense tensor dimension:", dense_tensor[0].shape)
      #print("stride is ",dense_tensor[2])
      #print(F0.shape)

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
      F1 = self.model(sinput1).F
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      #print("xyz0_corr shape:",xyz0_corr.shape)
      #print("xyz1_corr shape:",xyz1_corr.shape)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    #print("CORR F0:",np.shape(F0))
    #print("CORR F1:",np.shape(F1))
    nn_inds,nn_dist = find_nn_gpu(F0, F1,return_distance=True, nn_max_n=self.config.nn_max_n)
    #print("NN_dist:",np.shape(nn_dist))
    #print("NN_inds:",np.shape(nn_inds))
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
    #print("num of positive pair",sample_pos_pairs.shape)

    #print("INPUT POSF0 SHAPE:",np.shape(posF0))
    #print("INPUT SUBF1 SHAPE:",np.shape(subF1))
    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')
    #print("DIST D01 SHAPE:",np.shape(D01))

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)
    #print("pos key num:",pos_keys.shape)


    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)
    #print("neg key num:",neg_keys0.shape,neg_keys1.shape)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2, (posF0-posF1).pow(2).sum(1).mean(), ((D01min + D10min)/2).mean()

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        #print(input_dict['sinput0_F'].shape)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        F0 = self.model(sinput0).F

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)

        F1 = self.model(sinput1).F

        pos_pairs = input_dict['correspondences']
        #print(np.shape(input_dict['correspondences']))
        #print(np.shape(F1))
        #print(np.shape(F0))
        pos_loss, neg_loss, pos_dist, neg_dist = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
            self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f} Pos_dist {:.3f} Neg_dist {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss, pos_dist.item(),neg_dist.item()) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()


class TripletLossTrainer(ContrastiveLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=None,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    #logging.info(f"pos_Pairs: {np.shape(positive_pairs)}")
    #logging.info(f"hash_key_shape: {np.shape(hash_seed)}")
    pos_keys = _hash(positive_pairs, hash_seed)
    #logging.info(f"pos_keys_shape: {np.shape(pos_keys)}")
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    ##logging.info(f"rand_inds dim: {rand_inds.shape}")
    rand_pairs = positive_pairs[rand_inds]
    #ogging.info(f"rand_pairs dim: {rand_pairs.shape}")
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    #logging.info(f"anchors and rand_pairs:{anchors.data == rand_pairs[:,0]}")
   #logging.info(f"anchors size: {anchors.size}")
    #logging.info(f"positives size: {positives.size}")
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()
    #logging.info(f"loss dim:{np.shape(loss)}")

    return loss, pos_dist.mean(), rand_neg_dist.mean()

  def _train_epoch(self, epoch):
    config = self.config

    gc.collect()
    self.model.train()

    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    pos_dist_meter, neg_dist_meter = AverageMeter(), AverageMeter()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_loss = 0
      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        F0 = self.model(sinput0).F
        logging.info(f"coord shape:{np.shape(self.model(sinput0).C)}")

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
        F1 = self.model(sinput1).F

        pos_pairs = input_dict['correspondences']
        logging.info(f"pair max:{torch.max(pos_pairs[:,0])}")
        loss, pos_dist, neg_dist = self.triplet_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=config.triplet_num_pos * config.batch_size,
            num_hn_samples=config.triplet_num_hn * config.batch_size,
            num_rand_triplet=config.triplet_num_rand * config.batch_size)
        loss /= iter_size
        loss.backward()
        batch_loss += loss.item()
        pos_dist_meter.update(pos_dist)
        neg_dist_meter.update(neg_dist)

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        pos_dist_meter.reset()
        neg_dist_meter.reset()
        data_meter.reset()
        total_timer.reset()


class HardestTripletLossTrainer(TripletLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=512,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]


    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(
        torch.cat([
            rand_pos_dist + self.neg_thresh - rand_neg_dist,
            pos_dist[mask0] + self.neg_thresh - D01min[mask0],
            pos_dist[mask1] + self.neg_thresh - D10min[mask1]
        ])).mean()

    return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2

class JointLossTrainer(ContrastiveLossTrainer):

  def joint_loss(self,
                   F0,
                   F1,
                   score0,
                   score1,
                   positive_pairs,
                   len_batch,
                   batch_size,
                   num_pos=1024,
                   num_hn_samples=None,
                   ):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)


    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    # compute score for pairs
    pos_score = score0[pos_ind0]*score1[pos_ind1]
    pos_score_norm = pos_score/torch.sum(pos_score)

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)
    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(np.isin(neg_keys0, pos_keys, assume_unique=False))
    mask1 = torch.from_numpy(np.isin(neg_keys1, pos_keys, assume_unique=False))


    D01min[mask0] = D01min.mean()
    D10min[mask1] = D10min.mean()

    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min).pow(2)
    loss = F.relu(pos_loss + (neg_loss0 + neg_loss1)/2) * pos_score_norm
    return loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2, pos_loss.mean()

  def _train_epoch(self, epoch):
    config = self.config

    gc.collect()
    self.model.train()

    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    pos_dist_meter, neg_dist_meter, mem_meter = AverageMeter(), AverageMeter(), AverageMeter()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_loss = 0
      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        len_batch = input_dict['len_batch']
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)

        out = self.model(sinput0,sinput1,len_batch)
        pos_pairs = input_dict['correspondences']

        loss, neg_dist, pos_dist = self.joint_loss(
            out['feature0'],
            out['feature1'],
            out['score0'],
            out['score1'],
            pos_pairs,
            len_batch = input_dict['len_batch'],
            batch_size = config.batch_size,
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size,
            )
        #logging.info(f" batch {iter_size} Done")
        loss /= iter_size
        loss.backward()
        batch_loss += loss.item()
        pos_dist_meter.update(pos_dist)
        neg_dist_meter.update(neg_dist)

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        pos_dist_meter.reset()
        neg_dist_meter.reset()
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, mem_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()
      len_batch = input_dict['len_batch']
      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
      #F0 = self.model(sinput0).F

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
      out = self.model(sinput0,sinput1,len_batch)
      feat_timer.toc()
      F0 = out['feature0']
      F1 = out['feature1']

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      mem_meter.update(torch.cuda.memory_allocated(self.device))
      torch.cuda.empty_cache()
      

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
            f"Average Meter: {mem_meter.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }


#################################################################################################
# This is the trainer for keypoint exportation and detection 
# Loss type: Contrastive Loss Trainer
# 
#
#
#
#
#
#################################################################################################
class ContrastiveLossDetectionTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T


  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    DetectNet = DetectionNetHead(in_channels=3,keypoint_num=self.config.keypoint_num)
    DetectNet.to(self.device)
    DetectNet.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_sk0_pos0, batch_sk1_pos1 = 0, 0
      batch_sk0_Cpj0, batch_sk1_Cpj1 = 0, 0
      batch_sk0_sk0, batch_sk1_sk1 = 0, 0
      loss,loss0,loss1 = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        coord0 = input_dict['coords0'].to(self.device)
        coord1 = input_dict['coords1'].to(self.device)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C'])
        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C'])

        out0 = DetectNet(coord0)
        out1 = DetectNet(coord1)
        for k in range(self.batch_size):
          kpt0_i = out0[k,:,:]
          kpt1_i = out1[k,:,:]
          batch_corr = torch.tensor(input_dict['matching_inds'][0]).to(self.device)
          coord0_i = sinput0.coordinates_at(k).to(self.device)
          coord1_i = sinput1.coordinates_at(k).to(self.device)
          N0 = coord0_i.shape[0]
          N1 = coord1_i.shape[0]

          centroid0,centroid1 = input_dict['centroid'][0][0].float().to(self.device),input_dict['centroid'][0][1].float().to(self.device)
          max0,max1 = input_dict['max'][0][0].float().to(self.device),input_dict['max'][0][1].float().to(self.device)
          normed_coord0 = (coord0_i - centroid0) / max0
          normed_coord1 = (coord1_i - centroid1) / max1

          #print("max: norm coord:", torch.max(normed_coord0,dim=0))

          #print("normed_coord0 cuda:",normed_coord0.is_cuda)
          dist0 = pdist(kpt0_i,normed_coord0)
          dist1 = pdist(kpt1_i,normed_coord1)
          #print("max dist0:",torch.max(dist0))
          top32_0_idx = torch.argsort(dist0,dim=1)[:,0:32]
          top32_1_idx = torch.argsort(dist1,dim=1)[:,0:32]

          kpt_neighbor_dist0 = dist0[:,top32_0_idx]
          kpt_neighbor_dist1 = dist1[:,top32_1_idx]
          #neighbor0 = coord0_i[top32_0_idx,:]
          #neighbor1 = coord1_i[top32_1_idx,:]
          #print("max kpt_neighbor_dist1: ", torch.max(kpt_neighbor_dist1))
          
          kpt_dist0 = pdist(kpt0_i,kpt0_i)
          kpt_dist0 = torch.sqrt(torch.sum((kpt0_i.unsqueeze(1) - kpt0_i.unsqueeze(0)).pow(2), 2)+1e-7)
          kpt_dist1 = pdist(kpt1_i,kpt1_i)

          sk0_sk0_dist = torch.sum(F.relu(0.05 - kpt_dist0))
          sk1_sk1_dist = torch.sum(F.relu(0.05 - kpt_dist1))

          #kpt_neighbor_dist0 = pdist(kpt0_i,neighbor0)
          #kpt_neighbor_dist0 = torch.sqrt(torch.sum((kpt0_i.unsqueeze(1) - neighbor0.unsqueeze(0)).pow(2), 2)+1e-7)
          #kpt_neighbor_dist1 = pdist(kpt1_i,neighbor1)
          #kpt_neighbor_dist1 = torch.sqrt(torch.sum((kpt1_i.unsqueeze(1) - neighbor1.unsqueeze(0)).pow(2), 2)+1e-7)
          #print("kpt0 to neighbor distance:",kpt_neighbor_dist0.shape)
          #print("kpt1 to neighbor distance:",kpt_neighbor_dist1.shape)
          sk0_Cpj0 = F.relu(kpt_neighbor_dist0-0.05).sum() 
          sk1_Cpj1 = F.relu(kpt_neighbor_dist1-0.05).sum()
          #unique_pos0 = batch_corr[:,0].unique(dim=0)
          #print("num of unique_pos0",batch_corr[:,0].shape)
          pos0 = input_dict['pos0'][k].to(self.device)
          pos1 = input_dict['pos1'][k].to(self.device)

          sk0_pos0,_ = pdist(kpt0_i,pos0).min(1)
          sk1_pos1,_ = pdist(kpt1_i,pos0).min(1)

          loss0 += (sk0_Cpj0.mean() + sk0_pos0.mean()) / sk0_sk0_dist.mean()
          loss1 += (sk1_Cpj1.mean() + sk1_pos1.mean()) / sk1_sk1_dist.mean()
          loss = (loss0 + loss1) / 2

          batch_sk0_sk0 += sk0_sk0_dist.mean()
          batch_sk1_sk1 += sk1_sk1_dist.mean()
          batch_sk0_Cpj0 += sk0_Cpj0.mean()
          batch_sk1_Cpj1 += sk1_Cpj1.mean()
          batch_sk0_pos0 += sk0_pos0.mean()
          batch_sk1_pos1 += sk1_pos1.mean()

        loss = loss / self.batch_size
        loss.backward()

        batch_loss = loss.item()
        batch_loss0 = (loss0 / self.batch_size).item()
        batch_loss1 = (loss1 / self.batch_size).item()
        batch_sk0_sk0 = (batch_sk0_sk0 / self.batch_size).item()
        batch_sk1_sk1 = (batch_sk1_sk1 / self.batch_size).item()
        batch_sk0_Cpj0 = (batch_sk0_Cpj0 / self.batch_size).item()
        batch_sk1_Cpj1 = (batch_sk1_Cpj1 / self.batch_size).item()
        batch_sk0_pos0 = (batch_sk0_pos0 / self.batch_size).item()
        batch_sk1_pos1 = (batch_sk1_pos1 / self.batch_size).item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        #self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        #self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} loss0: {:.3f} loss1: {:.3f} "
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_loss0, batch_loss1) +
            "sk0_sk0:{:.3f} sk1_sk1:{:.3f} sk0_Cpj0:{:.3f} sk1_Cpj1:{:.3f} sk0_pos0:{:3f} sk1_pos1:{:3f}"
            .format(batch_sk0_sk0,batch_sk1_sk1,batch_sk0_Cpj0,batch_sk1_Cpj1,batch_sk0_pos0,batch_sk1_pos1) + 
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    num_feats = 1
    feat_ckpt = torch.load('./pretrained_model/checkpoint.pth')
    config = feat_ckpt['config']
    feat_net = ResUNetBN2C(
                num_feats,
                config.model_n_out,
                bn_momentum=config.bn_momentum,
                normalize_feature=config.normalize_feature,
                conv1_kernel_size=config.conv1_kernel_size,
                D=3)
    feat_net.to(self.device)
    # Change the network to evaluation mode
    feat_net.eval()
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    feat_net.load_state_dict(feat_ckpt['state_dict'])
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      #print(input_dict['sinput0_C'].shape)

      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)

      F0 = feat_net(sinput0).F
      F1 = feat_net(sinput1).F
      
      # export keypoint
      coord0 = sinput0.coordinates_at(0).to(self.device)
      coord1 = sinput1.coordinates_at(0).to(self.device)
      coord0_i = coord0.unsqueeze(0).float()
      coord1_i = coord1.unsqueeze(0).float()
      N0 = input_dict['len_batch'][0][0]
      N1 = input_dict['len_batch'][0][1]

      out0 = self.model(coord0_i).squeeze(0)
      out1 = self.model(coord1_i).squeeze(0)
      #print("out0 shape:",out0.shape)
      #print("out1 shape:",out1.shape)

      dist0 = pdist(out0,coord0)
      dist1 = pdist(out1,coord1)
      _,sk0_close = torch.min(dist0,dim=1)
      _,sk1_close = torch.min(dist1,dim=1)

      # export feature

      sk0_F0 = F0[sk0_close]
      sk1_F1 = F1[sk1_close]
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, sk0_close, sk1_close)
      #print("xyz0_corr shape:",xyz0_corr.shape)
      #print("xyz1_corr shape:",xyz1_corr.shape)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, sk0_close, sk1_close):
    nn_inds = find_nn_gpu(F0[sk0_close,:], F1[sk1_close,:], nn_max_n=self.config.nn_max_n)
    return xyz0[sk0_close], xyz1[sk1_close[nn_inds]]


  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()


#####################################################################
# D3Feat detector and descriptor trainer. 
# Loss Function will be FCGF's hardest contrastive loss
#
#
#####################################################################
class DetectandDescribeLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        neighbor0,
                                        neighbor1,
                                        num_neighbor0,
                                        num_neighbor1,
                                        positive_pairs,
                                        #neg_inds,
                                        num_pos=5192,
                                        num_hn_samples=1024,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    # generate positive sampling index
    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs
      num_pos = N_pos_pairs

    # generate negative sampling index
    subF0, subF1 = F0[sel0], F1[sel1]
    # sample pos
    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
    # sample negatives
    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')
    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    # search for hardest negatives
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))

    #print("mask0 shape:",np.where(mask0==False))
    #print("mask1 shape:",np.where(mask1==False))
    #print("D01min shape:",D01min.shape)
    #print("D10min shape:",D10min.shape)
    #zeros = torch.zeros((1,32),dtype=torch.float)
    #D01min_mask = D01min
    #D10min_mask = D10min
    #print("D01min shape:", D01min.unsqueeze(1).shape)
    #D01_max,_ = torch.max(D01min,dim=0)
    #D10_max,_ = torch.max(D10min,dim=0)
    #print(D01_max,D10_max)
    #D01min_mask[mask0==False] = D01_max
    #D10min_mask[mask1==False] = D10_max

    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    neg_loss = (neg_loss0.mean() + neg_loss1.mean()) / 2

    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1)) 
    neg0_dist = D01min
    neg1_dist = D10min

    #print("mask0 shape:",np.where(mask0==False))
    #print("mask1 shape:",np.where(mask1==False))
    # calculate alpha score


    shadow_F = torch.zeros(1,32).float().to(self.device)
    F0 = torch.cat((F0,shadow_F),dim=0)
    F1 = torch.cat((F1,shadow_F),dim=0)

    neighbor_feat0 = F0[neighbor0[pos_ind0],:]
    neighbor_feat1 = F1[neighbor1[pos_ind1],:]

    neighbor_avg0 = torch.sum(neighbor_feat0,dim=1) / torch.cat([num_neighbor0[pos_ind0].unsqueeze(1)]*self.model_n_out,dim=1) #sub,32
    neighbor_avg1 = torch.sum(neighbor_feat1,dim=1) / torch.cat([num_neighbor1[pos_ind1].unsqueeze(1)]*self.model_n_out,dim=1) #sub,32
    saliency0 = F0[pos_ind0] - neighbor_avg0
    saliency1 = F1[pos_ind1] - neighbor_avg1

    alpha0 = F.softplus(saliency0) # N0,32
    alpha1 = F.softplus(saliency1) # N1,32
    if torch.isnan(alpha0).any() == True:
      print("Nan at alpha0")

    #beta score
    max0_beta,_ = torch.max(posF0,dim=1)
    max1_beta,_ = torch.max(posF1,dim=1)
    #print("min max0_beta:",max0_beta.min())
    if torch.isnan(max0_beta).any() == True:
      print("Nan at max0_beta")

    if torch.isnan(torch.stack([max0_beta]*self.model_n_out,dim=1)).any() == True:
      print("Nan at stacked_max0_beta")
    #print("stacked max0_beta:",torch.stack([max0_beta]*self.model_n_out,dim=1).shape)
    #print("posF0:",posF0.shape)
    beta0 = posF0 / torch.stack([max0_beta + 1e-7]*self.model_n_out,dim=1)
    beta1 = posF1 / torch.stack([max1_beta + 1e-7]*self.model_n_out,dim=1)
    if torch.isnan(beta0).any() == True:
      print("Nan at beta0")

    score0,_ = torch.max(alpha0 * beta0,dim=1)
    score1,_ = torch.max(alpha1 * beta1,dim=1)


    L_desc = pos_loss.mean() + self.neg_weight * neg_loss.mean() 
    score_loss0 = (pos_dist - neg0_dist) * (score0 + score1)
    score_loss1 = (pos_dist - neg1_dist) * (score0 + score1)
    L_det = (score_loss0.mean() + score_loss1.mean()) / 2

    acc0 = torch.sum(pos_dist < neg0_dist).float() / float(num_pos)
    acc1 = torch.sum(pos_dist < neg1_dist).float() / float(num_pos)
    acc = (acc0 + acc1) /2 
    #print(acc)

    # if torch.isnan(L_det).any() == True:
    #   print("Nan at L_det")
    #   print("current N0 N1:",N0,N1)
    # if torch.isnan(L_desc).any() == True:
    #   print("Nan at L_desc")
    # if torch.isnan(score0).any() == True:
    #   print("Nan at score0")
    # if torch.isnan(score1).any() == True:
    #   print("Nan at score1")
    # if torch.isnan(pos_loss).any() == True:
    #   print("Nan at pos_loss")
    # if torch.isnan(neg_loss).any() == True:
    #   print("Nan at neg_loss")

    return pos_dist.mean(), (neg0_dist.mean() + neg1_dist.mean()), L_desc, L_det ,score0.mean(),score1.mean(), acc

  def check_neighbor(self, points, batch_len, neighbor_in):
    neighbor_in = neighbor_in.detach().numpy()
    neighbors = batch_find_neighbors.compute(points, points, torch.from_numpy(batch_len).int(), torch.from_numpy(batch_len).int(), 0.025 * 2.5)
    neighbors = neighbors.reshape([points.shape[0], -1]).detach().numpy()
    neighbors = neighbors[:, :40]
    sample_neighbors = neighbors[:,1]
    sample_neighbor_in = neighbor_in[:,1]
    non_the_same = np.where(neighbors != neighbor_in)
    #print("difference index:",non_the_same)
    print("current index:",neighbors[non_the_same[0],0])
    print("neighbor:", neighbors[neighbor_in != neighbors])
    print("neighbor_in:", neighbor_in[neighbor_in != neighbors])
    #print("neighbors:",neighbors[:20,:])


    return 

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    torch.autograd.set_detect_anomaly(True)

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
      batch_L_desc, batch_L_det, batch_score0, batch_score1 = 0,0,0,0
      batch_acc = 0
      loss = 0

      data_time = 0
      total_timer.tic()

      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)
        #print("Running points:",input_dict['N0'],input_dict['N1'])

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        F0 = self.model(sinput0)

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
        F1 = self.model(sinput1)
        self.check_neighbor(input_dict['pcd0'], np.array(input_dict['len_batch'])[:, 0],input_dict['neighbor0'][0])

        #print("running points:",sinput0.coordinates_at(0).shape,sinput1.coordinates_at(0).shape)
        #alpha_score()
        if (torch.abs(F0.F) > 1).any() == True:
          print("F0 max > 1")
        if (torch.abs(F1.F) > 1).any() == True:
          print("F1 max > 1")

        for i in range(self.batch_size):
          F0_in_batch = F0.features_at(i)
          F1_in_batch = F1.features_at(i)

          pos_loss, neg_loss, L_desc, L_det, score0, score1, acc = self.contrastive_hardest_negative_loss(
                                        F0=F0_in_batch,
                                        F1=F1_in_batch,
                                        neighbor0=input_dict['neighbor0'][i],
                                        neighbor1=input_dict['neighbor1'][i],
                                        num_neighbor0=input_dict['num_neighbor0'][i].to(self.device),
                                        num_neighbor1=input_dict['num_neighbor1'][i].to(self.device),
                                        positive_pairs=input_dict['matching_inds'][i],
                                        #neg_inds=input_dict['neg_inds'][i].to(self.device),
                                        num_pos=self.config.num_pos_per_batch // self.batch_size,
                                        num_hn_samples = self.config.num_hn_samples_per_batch,
                                        thresh=None
            )
          loss += (L_desc + L_det) / 2
          batch_pos_loss += pos_loss
          batch_neg_loss += neg_loss
          batch_L_desc += L_desc
          batch_L_det += L_det
          batch_score0 += score0
          batch_score1 += score1
          batch_acc += acc

        loss = loss / self.batch_size
        loss.backward()  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss = loss.item()
        batch_pos_loss = (batch_pos_loss / self.batch_size).item()
        batch_neg_loss = (batch_neg_loss / self.batch_size).item()
        batch_L_desc = (batch_L_desc / self.batch_size).item()
        batch_L_det = (batch_L_det / self.batch_size).item()
        batch_score0 = (batch_score0 / self.batch_size).item()
        batch_score1 = (batch_score1 / self.batch_size).item()
        batch_acc = (batch_acc / self.batch_size).item()


      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.6f} Neg: {:.6f} L_desc: {:.3f} L_det: {:.3f}, score0: {:.3f}, score1: {:.3f}, acc: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss, 
                    batch_L_desc, batch_L_det, batch_score0, batch_score1, batch_acc) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      #print(input_dict['sinput0_F'].shape)
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
      F0 = self.model(sinput0).F
      #output0 = self.model(sinput0)
      #dense_tensor = output0.dense()
      #print("output dense tensor dimension:", dense_tensor[0].shape)
      #print("stride is ",dense_tensor[2])
      #print(F0.shape)

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
      F1 = self.model(sinput1).F
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      #print("xyz0_corr shape:",xyz0_corr.shape)
      #print("xyz1_corr shape:",xyz1_corr.shape)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    #print("CORR F0:",np.shape(F0))
    #print("CORR F1:",np.shape(F1))
    nn_inds,nn_dist = find_nn_gpu(F0, F1,return_distance=True, nn_max_n=self.config.nn_max_n)
    #print("NN_dist:",np.shape(nn_dist))
    #print("NN_inds:",np.shape(nn_inds))
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()

####################################################################################
# D3Feat Joint Detection and Description Trainer
# loss: D3feat hardest mining method
#
#
#
####################################################################################
#####################################################################
# D3Feat detector and descriptor trainer. 
#
#
#
#####################################################################
class D3FeatLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def contrastive_hardest_negative_loss(self,
                                        C1,
                                        F0,
                                        F1,
                                        neighbor0,
                                        neighbor1,
                                        num_neighbor0,
                                        num_neighbor1,
                                        positive_pairs,
                                        safe_radius,
                                        num_pos=5192,
                                        num_hn_samples=1024,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    #hash_seed = max(N0, N1)
    #sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    #sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    # generate positive sampling index
    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs
      num_pos = N_pos_pairs

    # generate negative sampling index
    #subF0, subF1 = F0[sel0], F1[sel1]
    # sample pos

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posC1 = C1[pos_ind1]
    dist_C1 = pdist(posC1,posC1).to(self.device)
    identity = torch.eye(num_pos,dtype=torch.float).to(self.device)
    neg_mask = ((dist_C1 > safe_radius).float() * (1 - identity)).to(self.device)
    #logging.info(f"negative amount:{dist_C1.mean()}")


    
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
    # negative hardest mining
    #neg_mask = neg_inds[pos_ind1,:]
    pos_neg = pdist(posF0, posF1, dist_type='L2')
    pos_dist,_ = (pos_neg * identity).max(1)
    neg_dist = pos_neg * neg_mask + 1e8 * (1 - neg_mask)
    neg_dist, neg_dist_inds = neg_dist.min(1)

    # if (neg_mask == 1e6).any() == True:
    #   print("Neg_mask working")
    # calculate loss
    pos_loss = F.relu(pos_dist - self.pos_thresh)
    neg_loss = F.relu(self.neg_thresh - neg_dist)
    
    # calculate alpha score
    shadow_F = torch.zeros(1,32).float().to(self.device)
    F0 = torch.cat((F0,shadow_F),dim=0)
    F1 = torch.cat((F1,shadow_F),dim=0)

    neighbor_feat0 = F0[neighbor0[pos_ind0],:]
    neighbor_feat1 = F1[neighbor1[pos_ind1],:]

    neighbor_avg0 = torch.sum(neighbor_feat0,dim=1) / torch.cat([num_neighbor0[pos_ind0].unsqueeze(1)]*self.model_n_out,dim=1) #sub,32
    neighbor_avg1 = torch.sum(neighbor_feat1,dim=1) / torch.cat([num_neighbor1[pos_ind1].unsqueeze(1)]*self.model_n_out,dim=1) #sub,32
    saliency0 = F0[pos_ind0] - neighbor_avg0
    saliency1 = F1[pos_ind1] - neighbor_avg1
    #print("Saliency shape:",saliency0.shape)

    alpha0 = F.softplus(saliency0) # N0,32
    alpha1 = F.softplus(saliency1) # N1,32
    if torch.isnan(alpha0).any() == True:
      print("Nan at alpha0")

    #beta score
    max0_beta,_ = torch.max(posF0,dim=1)
    max1_beta,_ = torch.max(posF1,dim=1)
    #print("min max0_beta:",max0_beta.min())
    if torch.isnan(max0_beta).any() == True:
      print("Nan at max0_beta")

    if torch.isnan(torch.stack([max0_beta]*self.model_n_out,dim=1)).any() == True:
      print("Nan at stacked_max0_beta")
    #print("stacked max0_beta:",torch.stack([max0_beta]*self.model_n_out,dim=1).shape)
    #print("posF0:",posF0.shape)
    beta0 = posF0 / torch.stack([max0_beta + 1e-7]*self.model_n_out,dim=1)
    beta1 = posF1 / torch.stack([max1_beta + 1e-7]*self.model_n_out,dim=1)
    if torch.isnan(beta0).any() == True:
      print("Nan at beta0")

    score0,_ = torch.max(alpha0 * beta0,dim=1)
    score1,_ = torch.max(alpha1 * beta1,dim=1)


    L_desc = pos_loss.mean() + self.neg_weight * neg_loss.mean() 
    L_det = ((pos_dist - neg_dist)*(score0 + score1)).mean()

    acc = torch.sum(pos_dist < neg_dist).float() / float(num_pos)
    #print(acc)

    # if torch.isnan(L_det).any() == True:
    #   print("Nan at L_det")
    #   print("current N0 N1:",N0,N1)
    # if torch.isnan(L_desc).any() == True:
    #   print("Nan at L_desc")
    # if torch.isnan(score0).any() == True:
    #   print("Nan at score0")
    # if torch.isnan(score1).any() == True:
    #   print("Nan at score1")
    # if torch.isnan(pos_loss).any() == True:
    #   print("Nan at pos_loss")
    # if torch.isnan(neg_loss).any() == True:
    #   print("Nan at neg_loss")

    return pos_dist.mean(), neg_dist.mean(), L_desc, L_det ,score0.mean(),score1.mean(), acc

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    torch.autograd.set_detect_anomaly(True)

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
      batch_L_desc, batch_L_det, batch_score0, batch_score1 = 0,0,0,0
      batch_acc = 0
      loss = 0

      data_time = 0
      total_timer.tic()

      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)
        #print("Running points:",input_dict['N0'],input_dict['N1'])

        # pairs consist of (xyz1 index, xyz0 index)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
        out0 = self.model(sinput0)

        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
        out1 = self.model(sinput1)

        #print("running points:",sinput0.coordinates_at(0).shape,sinput1.coordinates_at(0).shape)
        #alpha_score()
        if (torch.abs(out0.F) > 1).any() == True:
          print("F0 max > 1")
        if (torch.abs(out1.F) > 1).any() == True:
          print("F1 max > 1")

        for i in range(self.batch_size):
          F0_in_batch = out0.features_at(i)
          F1_in_batch = out1.features_at(i)
          #C0_in_batch = out0.coordinates_at(i)
          C1_in_batch = out1.coordinates_at(i)

          pos_loss, neg_loss, L_desc, L_det, score0, score1, acc = self.contrastive_hardest_negative_loss(
                                        C1=C1_in_batch,
                                        F0=F0_in_batch,
                                        F1=F1_in_batch,
                                        neighbor0=input_dict['neighbor0'][i],
                                        neighbor1=input_dict['neighbor1'][i],
                                        num_neighbor0=input_dict['num_neighbor0'][i].to(self.device),
                                        num_neighbor1=input_dict['num_neighbor1'][i].to(self.device),
                                        positive_pairs=input_dict['matching_inds'][i],
                                        safe_radius=self.config.safe_radius,
                                        num_pos=self.config.num_pos_per_batch,
                                        num_hn_samples = self.config.num_hn_samples_per_batch,
                                        thresh=None
            )
          loss += (L_desc + L_det) / 2
          batch_pos_loss += pos_loss
          batch_neg_loss += neg_loss
          batch_L_desc += L_desc
          batch_L_det += L_det
          batch_score0 += score0
          batch_score1 += score1
          batch_acc += acc

        loss = loss / self.batch_size
        loss.backward()  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss = loss.item()
        batch_pos_loss = (batch_pos_loss / self.batch_size).item()
        batch_neg_loss = (batch_neg_loss / self.batch_size).item()
        batch_L_desc = (batch_L_desc / self.batch_size).item()
        batch_L_det = (batch_L_det / self.batch_size).item()
        batch_score0 = (batch_score0 / self.batch_size).item()
        batch_score1 = (batch_score1 / self.batch_size).item()
        batch_acc = (batch_acc / self.batch_size).item()


      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.6f} Neg: {:.6f} L_desc: {:.3f} L_det: {:.3f}, score0: {:.3f}, score1: {:.3f}, acc: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss, 
                    batch_L_desc, batch_L_det, batch_score0, batch_score1, batch_acc) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      #print(input_dict['sinput0_F'].shape)
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.device)
      F0 = self.model(sinput0).F
      #output0 = self.model(sinput0)
      #dense_tensor = output0.dense()
      #print("output dense tensor dimension:", dense_tensor[0].shape)
      #print("stride is ",dense_tensor[2])
      #print(F0.shape)

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.device)
      F1 = self.model(sinput1).F
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      #print("xyz0_corr shape:",xyz0_corr.shape)
      #print("xyz1_corr shape:",xyz1_corr.shape)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    #print("CORR F0:",np.shape(F0))
    #print("CORR F1:",np.shape(F1))
    nn_inds,nn_dist = find_nn_gpu(F0, F1,return_distance=True, nn_max_n=self.config.nn_max_n)
    #print("NN_dist:",np.shape(nn_dist))
    #print("NN_inds:",np.shape(nn_inds))
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()