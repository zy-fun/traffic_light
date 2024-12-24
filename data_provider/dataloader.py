import os
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
import pickle
import numpy as np
from typing import Tuple, List


class Car_Passing_Data(Dataset):
    def __init__(self):
        self.__read_data__()

    def __read_data__(self, flag:str ='train'):
        self.lane_indices = [9]
        self.car_passing_matrix = load_npz('processed_matrix/car_passing_matrix.npz')
        self.signal_matrix = load_npz('processed_matrix/signal_matrix.npz')
        with open('processed_matrix/lane_type_vector.pkl', 'rb') as f:
            self.lane_type_vector = pickle.load(f)

        self.car_passing_vector = self.car_passing_matrix[self.lane_indices].toarray().squeeze()
        self.signal_vector = self.signal_matrix[self.lane_indices].toarray().squeeze()
        self.signal_vector, self.signal_intervals, _ = self.fix_discontinuity(self.signal_vector)

        self.__generate_samples__()
        # self.signal_matrix

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        ratios = [0.7, 0.8, 1.0]
        border1s = [0, int(ratios[0] * len(self)), int(ratios[1] * len(self))]
        border2s = [int(ratios[0] * len(self)), int(ratios[1] * len(self)), len(self)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.car_passing_samples = self.car_passing_samples[border1: border2]

    def __generate_samples__(self, car_threshold=6, L=30):
        self.valid_signal_intervals = []
        self.car_passing_samples = []
        # max_len = 0
        for l_border, r_border in self.signal_intervals:
            car_passing = self.car_passing_vector[l_border: r_border]

            # max_len = max(max_len, r_border - l_border + 1)
            if not np.count_nonzero(car_passing[:L]) < car_threshold:
                self.valid_signal_intervals.append((l_border, r_border))
                self.car_passing_samples.append(car_passing[:L])
        
        for i, sample in enumerate(self.car_passing_samples):
            # self.car_passing_samples[i] = np.pad(sample, (max_len - len(sample), 0))
            self.car_passing_samples[i] = np.pad(sample, (L - len(sample), 0))
        self.car_passing_samples = np.array(self.car_passing_samples)
        
        # print(len(self))
        # print(self.car_passing_samples)
        # print(np.count_nonzero(self.car_passing_samples) / np.size(self.car_passing_samples))

    def __len__(self):
        return self.car_passing_samples.shape[0]
    
    def __getitem__(self, index: int):
        return self.car_passing_samples[index]
    
    def fix_discontinuity(self, signal_vector: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int]], List[int]]:
        fixed_signal_vector = np.zeros_like(signal_vector)

        signal_time = np.where(signal_vector)[0]
        db = DBSCAN(eps=5, min_samples=2)
        labels = db.fit_predict(signal_time.reshape(-1, 1))
        signal_intervals = []
        signal_intervals_length = []
        for label in np.unique(labels):
            if label == -1:
                continue
            idx, = np.where(labels == label)    # unpacking
            signal_interval = signal_time[idx].squeeze()
            left_border, right_border = signal_interval[0], signal_interval[-1]
            fixed_signal_vector[left_border: right_border] = 1
            signal_intervals.append((left_border, right_border))
            signal_intervals_length.append(right_border-left_border + 1)
        return fixed_signal_vector, signal_intervals, signal_intervals_length

if __name__ == "__main__":
    Car_Passing_Data()