import sys
import argparse
import yaml
import math
import random
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class AimCLR_Processor(PT_Processor):
    """
        Processor for AimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        self.model.embedding_callback.clean_storage()

        for [data1, data2, data3], label, skeletons in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            q_skeleton = skeletons['q'][0]
            k_skeleton = skeletons['k'][0] # Skeletons should be the same for the entire stream

            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)
                motion3 = torch.zeros_like(data3)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
                motion3[:, :, :-1, :, :] = data3[:, :, 1:, :, :] - data3[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
                data3 = motion3
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
                Bone = {'ntu-rgb+d': [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                            (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                            (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)],
                     'smpl_24': [(1, 4), (4, 7), (7, 10), (13, 10), (16, 13),
                            (2, 1), (5, 2), (8, 5), (11, 8), (3, 1), (6, 3), (9, 6), (12, 9),
                            (14, 10), (17, 14), (19, 17), (21, 19), (23, 21),
                            (15, 10), (18, 15), (20, 18), (22, 20), (24, 22)],
                     'smplx_42': [(7, 10), (4, 7), (1, 4),
                            (2, 1), (5, 2), (8, 5), (35, 8), (11, 8), (33, 11), (34, 11), # Left leg
                            (3, 1), (6, 3), (9, 6), (38, 9), (12, 9), (36, 12), (37, 12), # Right Leg
                            (14, 10), (17, 14), (19, 17), (21, 19), (26, 21), (39, 21), (40, 21), # Left Arm
                            (15, 10), (18, 15), (20, 18), (22, 20), (27, 22), (41, 22), (42, 22), # Right Arm
                            (13, 10), (23, 13), (16, 23), (28, 16), (30, 28), (32, 30),
                            (29, 28), (31, 29)],
                     'berkeley_mhad_43':[(1, 2), (1, 3), (4, 5), (4, 36), (5, 6), (5, 7), 
                            (7, 28), (8, 9), (8, 29), (9, 10), (9, 11), (11, 37), 
                            (12, 13), (12, 20), (13, 14), (14, 15), (15, 16), 
                            (16, 17), (16, 18), (16, 19), (20, 21), (21, 22), 
                            (22, 23), (23, 24), (24, 25), (24, 26), (24, 27), 
                            (28, 30), (28, 36), (29, 30), (29, 31), (31, 32), 
                            (32, 33), (33, 34), (34, 35), (36, 38), (37, 38), 
                            (37, 39), (39, 40), (40, 41), (41, 42), (42, 43)]}

                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)
                bone3 = torch.zeros_like(data3)

                for v1, v2 in Bone[q_skeleton]:
                    #  Data1 and data2 go to the query encoders --> Belong to the same skeleton
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                for v1, v2 in Bone[k_skeleton]:
                    bone3[:, :, :, v1 - 1, :] = data3[:, :, :, v1 - 1, :] - data3[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
                data3 = bone3
            else:
                raise ValueError

            # forward
            if epoch <= self.arg.mining_epoch:
                output1, target1, output2, output3, target2 = self.model(data1, data2, data3, q_skeleton=q_skeleton, k_skeleton=k_skeleton)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = self.loss(output1, target1)
                loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                loss3 = -torch.mean(torch.sum(torch.log(output3) * target2, dim=1))  # DDM loss
                loss = loss1 + (loss2 + loss3) / 2.
            else:
                output1, mask, output2, output3, target2 = self.model(data1, data2, data3, nnm=True, topk=self.arg.topk, q_skeleton=q_skeleton, k_skeleton=k_skeleton)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = - (F.log_softmax(output1, dim=1) * mask).sum(1) / mask.sum(1)
                loss1 = loss1.mean()
                loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                loss3 = -torch.mean(torch.sum(torch.log(output3) * target2, dim=1))  # DDM loss
                loss = loss1 + (loss2 + loss3) / 2.

            # Embedding Callback
            self.model.embedding_callback.store_labels(label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.wandb.log({'loss':self.epoch_info['train_mean_loss'], 'epoch': epoch})

        self.show_epoch_info()

        if epoch % self.model.embedding_callback.epoch_plot_interval == 0:
            self.model.embedding_callback.plot_tsne(epoch, self.wandb, stage="train", view='Joint')

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        return parser
