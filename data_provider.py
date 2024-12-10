from datetime import datetime
from scipy.sparse import load_npz
import pickle
import numpy as np

class Data_Provider:
    def __init__(self):
        self.lane_type_dict = {
            1: 'straight', 
            2: 'left', 
            3: 'right', 
            4: 'reverse; left',
            5: 'straight; right'
        }

        self.original_start_datetime = datetime(2023, 7, 9, 0, 0, 0)
        # original_end_datetime = datetime(2023, 9, 13, 0, 1, 19)

        # load data from files
        self.car_passing_matrix = load_npz('processed_matrix/car_passing_matrix.npz')
        self.signal_matrix = load_npz('processed_matrix/signal_matrix.npz')
        with open('processed_matrix/lane_type_vector.pkl', 'rb') as f:
            self.lane_type_vector = pickle.load(f)
            # left lane or straight lane
            # target_lane_indices = np.where((lane_type_vector == 1) | (lane_type_vector == 2))[0]

    def get_data(
            self,
            lane_id = 0, 
            start_datetime = datetime(2023, 7, 9, 11, 0, 0),
            end_datetime = datetime(2023, 7, 9, 12, 10, 0)
        ):
        start_seconds = int((start_datetime - self.original_start_datetime).total_seconds())
        end_seconds = int((end_datetime - self.original_start_datetime).total_seconds())

        car_passing_vector = self.car_passing_matrix[lane_id, start_seconds: end_seconds]
        signal_vector = self.signal_matrix[lane_id, start_seconds: end_seconds]

        lane_type = self.lane_type_dict[self.lane_type_vector[lane_id]]
        return car_passing_vector, signal_vector, lane_type