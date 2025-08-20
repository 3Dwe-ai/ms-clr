import random
import numpy as np
import pickle, torch
from . import tools
import torch.nn.functional as F


num_nodes = {"ntu-rgb+d": 25,
             "smpl_24": 24,
             "smplx_42": 42,
             "berkeley_mhad_43": 43}


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
    
class Feeder_ensemble(torch.utils.data.Dataset):
    """ Feeder for ensemble classification """

    def __init__(self, data_path, label_path, mmap=True, factor=0.001, shear_amplitude=-1, temperal_padding_ratio=-1):
        self.data_path = data_path # Should be a dict with skeletons as keys and paths as values
        self.data = {}
        self.skeletons = list(data_path.keys())
        self.label_path = label_path # TODO: Make sure labels are the same, they should but just in case

        self.factor = factor
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            for skeleton in self.skeletons:
                self.data[skeleton] = np.load(self.data_path[skeleton], mmap_mode='r')
        else:
            for skeleton in self.skeletons:
                self.data[skeleton] = np.load(self.data_path[skeleton])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = {}
        for skeleton in self.skeletons:
            data[skeleton] = np.array(self.data[skeleton][index])
            # Hacky but in ensemble, ntu is ground truth and therefore already scaled properly
            if skeleton != 'ntu-rgb+d':
                data[skeleton] *= self.factor

        label = self.label[index]

        return data, label


class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._aug(data_numpy)
        data3 = self._aug(data_numpy)
        return [data1, data2, data3], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy

class Feeder_mixed_triple(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345', factor=1.0, factor_ntu=True, full_random_skeleton=True):
        self.data_path = data_path
        self.data = {}
        self.skeletons = list(data_path.keys())
        self.batch_skeletons = {'q': None, 'k': None} # Holds the query and key skeletons for the current batch
        
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.factor = factor
        self.factor_ntu = factor_ntu

        self.max_num_node = 0
        for skeleton in self.skeletons:
            if num_nodes[skeleton] > self.max_num_node:
                self.max_num_node = num_nodes[skeleton]

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            for skeleton in self.skeletons:
                self.data[skeleton] = np.load(self.data_path[skeleton], mmap_mode='r')
        else:
            for skeleton in self.skeletons:
                self.data[skeleton] = np.load(self.data_path[skeleton])

    def __len__(self):
        # Increase dataset size by number of skeletons
        return len(self.label)*len(self.skeletons)

    def __getitem__(self, index):
        # get data
        data_q = np.array(self.data[self.batch_skeletons['q']][index%len(self.label)])*self.factor
        data_k = np.array(self.data[self.batch_skeletons['k']][index%len(self.label)])*self.factor

        _, _, V_q, _ = data_q.shape
        _, _, V_k, _ = data_k.shape

        if not self.factor_ntu:
            if self.batch_skeletons['q'] == 'ntu-rgb+d':
                data_q = np.array(self.data[self.batch_skeletons['q']][index%len(self.label)])
            if self.batch_skeletons['k'] == 'ntu-rgb+d':
                data_k = np.array(self.data[self.batch_skeletons['k']][index%len(self.label)])

        label = self.label[index%len(self.label)]
        #print(f"Before augmentation: dataq shape: {data_q.shape}, datak shape: {data_k.shape}")

        # processing
        data1 = self._strong_aug(data_q)
        data2 = self._aug(data_q)
        data3 = self._aug(data_k)

        if V_q < self.max_num_node:
            pad_size = self.max_num_node - V_q
            data1 = np.pad(data1, ((0, 0), (0, 0), (0, pad_size), (0, 0)), mode="constant", constant_values=0)
            data2 = np.pad(data2, ((0, 0), (0, 0), (0, pad_size), (0, 0)), mode="constant", constant_values=0)

        if V_k < self.max_num_node:
            pad_size = self.max_num_node - V_k
            data3 = np.pad(data3, ((0, 0), (0, 0), (0, pad_size), (0, 0)), mode="constant", constant_values=0)

        
        assert data1.shape == data2.shape == data3.shape, (
                f"Shape mismatch at index {index}: data1 shape {data1.shape}, data2 shape {data2.shape}, data3 shape {data3.shape}"
            )
        return [data1, data2, data3], label, self.batch_skeletons

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy
    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy
    
    def set_batch_skeletons(self, query_skeleton, key_skeleton):
        self.batch_skeletons['q'] = query_skeleton
        self.batch_skeletons['k'] = key_skeleton
    

class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
