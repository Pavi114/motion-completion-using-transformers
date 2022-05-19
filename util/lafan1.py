from constants import DEVICE
from util import quaternions, extract
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("..")


class LaFan1(Dataset):

    def __init__(self, dataset_directory, train=False, seq_len=50, offset=10, files_to_read=-1):
        """
        Args:
            dataset_directory (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """
        if train:
            self.actors = ['subject1', 'subject2', 'subject3', 'subject4']
        else:
            self.actors = ['subject5']
        self.train = train
        self.seq_len = seq_len
        self.offset = offset
        self.files_to_read = files_to_read
        self.data = self.load_data(dataset_directory)
        self.cur_seq_length = 5

    def load_data(self, dataset_directory):

        print('Building the data set...')
        X, Q, parents = extract.get_lafan1_set(
            dataset_directory, self.actors, window=self.seq_len, offset=self.offset, files_to_read=self.files_to_read)

        Q = torch.Tensor(Q).cpu()
        X = torch.Tensor(X).cpu()

        # Global representation:
        q_glbl, x_glbl = quaternions.quat_fk_tensor(Q, X, parents)

        # Global positions stats:
        # self.x_mean = torch.mean(x_glbl.reshape(
        #     [x_glbl.shape[0], x_glbl.shape[1], -1]).permute([0, 2, 1]), dim=(0, 2), keepdim=True)
        # self.x_std = torch.std(x_glbl.reshape(
        #     [x_glbl.shape[0], x_glbl.shape[1], -1]).permute([0, 2, 1]), dim=(0, 2), keepdim=True)

        input_ = {}
        # The following features are inputs:
        # 1. local quaternion vector (J * 4d)
        input_['local_q'] = Q

        # 2. global root velocity vector (3d)
        input_['root_v'] = x_glbl[:, 1:, 0, :] - x_glbl[:, :-1, 0, :]

        # Add zero velocity vector for last frame
        input_['root_v'] = torch.cat(
            (input_['root_v'], torch.zeros((input_['root_v'].shape[0], 1, 3))), dim=-2)

        # 3. contact information vector (4d)
        # input_['contact'] = torch.cat([contacts_l, contacts_r], dim=-1)

        # 4. global root position offset (?d)
        input_['root_p_offset'] = x_glbl[:, -1, 0, :]

        # 5. local quaternion offset (?d)
        input_['local_q_offset'] = Q[:, -1, :, :]

        # 6. target
        input_['target'] = Q[:, -1, :, :]

        # 7. root pos
        input_['root_p'] = x_glbl[:, :, 0, :]

        # 8. X
        input_['X'] = x_glbl[:, :, :, :]

        # 9. local_p
        input_['local_p'] = X

        # print('Nb of sequences : {}\n'.format(X.shape[0]))
        # print(input_['X'].shape, input_['local_q'].shape)
        # print(input_['X'][0][0])

        return input_

    def __len__(self):
        return len(self.data['local_q'])

    def __getitem__(self, idx):
        sample = {}
        sample['local_q'] = self.data['local_q'][idx]
        sample['root_v'] = self.data['root_v'][idx]
        # sample['contact'] = self.data['contact'][idx]
        sample['root_p_offset'] = self.data['root_p_offset'][idx]
        sample['local_q_offset'] = self.data['local_q_offset'][idx]
        sample['target'] = self.data['target'][idx]
        sample['root_p'] = self.data['root_p'][idx]
        sample['X'] = self.data['X'][idx]
        sample['local_p'] = self.data['local_p'][idx]
        return sample
