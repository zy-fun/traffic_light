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
            passing_interval_lens_path="avg_passing_lens_eps_20.npy",
            signal_interval_lens_path="avg_signal_lens_eps_20.npy",
            lane_num = 100,
            ):
        car_passing = np.load(passing_interval_lens_path)[:lane_num]       
        signal = np.load(signal_interval_lens_path)[:lane_num]
        x = np.arange(len(car_passing))
        plt.bar(x, car_passing, alpha=0.5, label='car passing', color='blue')  # 第一个数组
        plt.bar(x, signal, alpha=0.5, label='signal', color='orange')  # 第二个数组

        # 添加图例和标签
        plt.legend()
        plt.title('bar of Two Arrays with Overlap')
        plt.xlabel('lane')
        plt.ylabel('interval length')

        # 显示图形
        plt.savefig("figures/car_passing_interval_visualize/fig.png")

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