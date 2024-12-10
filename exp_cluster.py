import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from data_provider import Data_Provider
from datetime import datetime, timedelta
import os

class Exp_cluster:
    def __init__(self):
        self.dataset = Data_Provider()
        eps = 30
        avg_passing_lens = []
        avg_signal_lens = []
        for lane_id in range(346):
            print(lane_id, end=' ')
            avg_passing_len, avg_signal_len = self.exp(lane_id, eps=eps)
            print(avg_passing_len, avg_signal_len)
            avg_passing_lens.append(avg_passing_len)
            avg_signal_lens.append(avg_signal_len)
        avg_passing_lens = np.array(avg_passing_lens)
        avg_signal_lens = np.array(avg_signal_lens)
        np.save(f'avg_passing_lens_eps_{eps}.npy', avg_passing_lens)
        np.save(f'avg_signal_lens_eps_{eps}.npy', avg_signal_lens)

    def cluster(self, data, cluster_name, eps):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if cluster_name == 'kmeans':
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(data)
            labels = kmeans.labels_
        elif cluster_name == "dbscan":
            db = DBSCAN(eps=eps, min_samples=2)
            labels = db.fit_predict(data)
        return labels

    def exp(self, 
            lane_id,
            start_datetime = datetime(2023, 8, 1, 11, 0, 0),
            end_datetime = datetime(2023, 8, 1, 11, 11, 0),
            cluster = 'dbscan',
            eps = 20,   # dbscan
            ):
        car_passing_vector, signal_vector, lane_type = self.dataset.get_data(lane_id, start_datetime, end_datetime)
        car_passing_time = np.where(car_passing_vector.toarray().squeeze())[0]
        signal_time = np.where(signal_vector.toarray().squeeze())[0]
        if car_passing_time.size == 0 or signal_time.size == 0:
            return 0, 0

        labels = self.cluster(car_passing_time, cluster, eps)
        passing_intervals = []
        passing_intervals_length = []
        for label in np.unique(labels):
            if label == -1:
                continue
            idx, = np.where(labels == label)    # unpacking
            passing_interval = car_passing_time[idx].squeeze()
            left_border, right_border = passing_interval[0], passing_interval[-1]
            passing_intervals.append((left_border, right_border))
            passing_intervals_length.append(passing_interval[-1] - passing_interval[0])
        # print("passing_intervals:", passing_intervals)
        # print("passing_interval_length:", passing_intervals_length)

        plt.figure(figsize=(12, 6))
        plt.scatter(car_passing_time, np.zeros_like(car_passing_time), c=labels, cmap='viridis',)

        labels = self.cluster(signal_time, cluster, eps)
        signal_intervals = []
        signal_intervals_length = []
        for label in np.unique(labels):
            if label == -1:
                continue
            idx, = np.where(labels == label)    # unpacking
            signal_interval = signal_time[idx].squeeze()
            left_border, right_border = signal_interval[0], signal_interval[-1]
            signal_intervals.append((left_border, right_border))
            signal_intervals_length.append(right_border-left_border)
        # print("signal_intervals:", signal_intervals)
        # print("signal_intervals_length:", signal_intervals_length)
        plt.scatter(signal_time, np.ones(len(signal_time)), color='green', label='green light')
        title = f'cluster {cluster} eps {eps} result of {lane_type} lane {lane_id}'
        plt.title(title)
        plt.ylim(-0.2, 1.2)

        # x_axis
        total_seconds = int((end_datetime - start_datetime).total_seconds())
        new_ticks = np.linspace(0, total_seconds, 5, dtype=np.int32)
        relative_time_to_datetime = np.vectorize(lambda t: start_datetime + timedelta(seconds=int(t)))
        plt.xticks(new_ticks, relative_time_to_datetime(new_ticks))
        
        # # legend
        # plt.plot([], [], c=-1, label='-1', cmap='viridis')
        plt.legend()
        
        if not os.path.exists(f'figures/exp_cluster/{cluster}_{eps}'):
            os.mkdir(f'figures/exp_cluster/{cluster}_{eps}')
        save_path = f'figures/exp_cluster/{cluster}_{eps}/lane_{lane_id}_{lane_type}.png'
        plt.savefig(save_path)
        plt.close()
        
        avg_passing_interval_length = np.mean(passing_intervals_length)
        avg_signal_interval_length = np.mean(signal_intervals_length)
        return avg_passing_interval_length, avg_signal_interval_length

Exp_cluster()