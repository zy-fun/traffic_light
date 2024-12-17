import matplotlib.pyplot as plt
from data_provider import Data_Provider
from datetime import datetime, timedelta
import numpy as np

class Visualizer:
    def __init__(self,):
        # self.original_start_datetime = datetime(2023, 7, 9, 0, 0, 0)
        # self.original_end_datetime = datetime(2023, 9, 13, 0, 1, 19)
        # self.dataset = Data_Provider()
        # for lane in range(346):
        #     self.car_passing_visualize(lane)
        self.car_passing_interval_visualize()


    def car_passing_interval_visualize(
            self,
            eps = 20,
            lane_num = 100,
            ):
        passing_lens_path=f"exp_cluster/avg_passing_lens_eps_{eps}.npy"
        avg_truth_passing_lens_path=f"exp_cluster/avg_truth_passing_lens_eps_{eps}.npy"
        max_truth_passing_lens_path=f"exp_cluster/max_truth_passing_lens_eps_{eps}.npy"
        signal_lens_path=f"exp_cluster/avg_signal_lens_eps_{eps}.npy"

        car_passing = np.load(passing_lens_path)       
        car_passing_1 = np.load(avg_truth_passing_lens_path)
        car_passing_2 = np.load(max_truth_passing_lens_path)
        signal = np.load(signal_lens_path)

        car_passing = np.nan_to_num(car_passing)
        car_passing_1 = np.nan_to_num(car_passing_1)
        car_passing_2 = np.nan_to_num(car_passing_2)
        signal = np.nan_to_num(signal)

        remove_nan_elements = lambda arr: arr[(~np.isnan(arr)) & (arr != 0)]
        acc0 = remove_nan_elements(car_passing / signal)
        acc1 = remove_nan_elements(car_passing_1 / signal)
        acc2 = remove_nan_elements(car_passing_2 / signal)
        print(len(acc0), np.mean(acc0), np.median(acc0))
        print(len(acc1), np.mean(acc1), np.median(acc1))
        print(len(acc2), np.mean(acc2), np.median(acc2))
        # for c_pass in [car_passing, car_passing_1, car_passing_2]:
        x = np.arange(lane_num)
        plt.bar(x, car_passing[:lane_num], alpha=1, label='car passing', color='orange', zorder=4)  
        plt.bar(x, car_passing_1[:lane_num], alpha=1, label='car passing 1', color='red', zorder=3) 
        plt.bar(x, car_passing_2[:lane_num], alpha=1, label='car passing 2', color='blue', zorder=2) 
        plt.bar(x, signal[:lane_num], alpha=1, label='signal', color='black', zorder=1) 

        # 添加图例和标签
        plt.legend()
        plt.title('bar of Two Arrays with Overlap')
        plt.xlabel('lane')
        plt.ylabel('interval length')

        # 显示图形
        plt.savefig(f"figures/car_passing_interval_visualize/fig_eps_{eps}.png")

    def car_passing_visualize(
            self,
            lane_id = 0, 
            start_datetime = datetime(2023, 8, 1, 11, 0, 0),
            end_datetime = datetime(2023, 8, 1, 11, 10, 0)
            ):
        car_passing_color = 'blue'
        signal_color = 'green' 
        total_seconds = int((end_datetime - start_datetime).total_seconds())

        # load data
        car_passing_vector, signal_vector, lane_type = self.dataset.get_data(lane_id, start_datetime, end_datetime)
        car_passing_time = np.where(car_passing_vector.toarray().squeeze())[0]
        signal_time = np.where(signal_vector.toarray().squeeze())[0]

        # plot
        plt.figure(figsize=(12, 6))
        plt.scatter(car_passing_time, np.zeros(len(car_passing_time)), color='blue', label='car passing')
        plt.scatter(signal_time, np.ones(len(signal_time)), color='green', label='green light')
        plt.legend()
        plt.xlabel('relative time')
        plt.ylim(-0.2, 1.2)
        plt.title(f'car passing of {lane_type} lane {lane_id}\nfrom {start_datetime} to {end_datetime}')

        # change xticks
        new_ticks = np.linspace(0, total_seconds, 5, dtype=np.int32)
        relative_time_to_datetime = np.vectorize(lambda t: start_datetime + timedelta(seconds=int(t)))
        plt.xticks(new_ticks, relative_time_to_datetime(new_ticks))

        # save
        save_path = f'figures/car_passing_visualize/lane_{lane_id}_{lane_type}.png'
        plt.savefig(save_path)
        plt.close()
        
if __name__ == "__main__":
    Visualizer()