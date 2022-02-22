from util import quaternions, extract
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("..")


class LaFan1(Dataset):

    def __init__(self, dataset_directory, train=False, seq_len=50, offset=10):
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
        self.data = self.load_data(dataset_directory)
        self.cur_seq_length = 5

    def load_data(self, dataset_directory):

        print('Building the data set...')
        X, Q, parents, contacts_l, contacts_r = extract.get_lafan1_set(
            dataset_directory, self.actors, window=self.seq_len, offset=self.offset)
        # print(X.shape)
        # print(X[0][0])
        # Global representation:
        q_glbl, x_glbl = quaternions.quat_fk(Q, X, parents)

        # Global positions stats:
        x_mean = np.mean(x_glbl.reshape(
            [x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        x_std = np.std(x_glbl.reshape(
            [x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        self.x_mean = torch.from_numpy(x_mean)
        self.x_std = torch.from_numpy(x_std)

        input_ = {}
        # The following features are inputs:
        # 1. local quaternion vector (J * 4d)
        input_['local_q'] = Q

        # 2. global root velocity vector (3d)
        input_['root_v'] = x_glbl[:, 1:, 0, :] - x_glbl[:, :-1, 0, :]

        # Add zero velocity vector for last frame
        input_['root_v'] = np.concatenate(
            (input_['root_v'], np.zeros((input_['root_v'].shape[0], 1, 3))), axis=-2)

        # 3. contact information vector (4d)
        input_['contact'] = np.concatenate([contacts_l, contacts_r], -1)

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
        sample['local_q'] = self.data['local_q'][idx].astype(np.float32)
        sample['root_v'] = self.data['root_v'][idx].astype(np.float32)
        sample['contact'] = self.data['contact'][idx].astype(np.float32)
        sample['root_p_offset'] = self.data['root_p_offset'][idx].astype(
            np.float32)
        sample['local_q_offset'] = self.data['local_q_offset'][idx].astype(
            np.float32)
        sample['target'] = self.data['target'][idx].astype(np.float32)
        sample['root_p'] = self.data['root_p'][idx].astype(np.float32)
        sample['X'] = self.data['X'][idx].astype(np.float32)
        sample['local_p'] = self.data['local_p'][idx].astype(np.float32)
        return sample
