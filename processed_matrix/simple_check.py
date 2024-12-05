import pickle
import numpy as np
from scipy.sparse import load_npz

# (lane, time)
car_passing_matrix = load_npz('data_provider/data/traffic_light/processed_matrix/car_passing_matrix.npz')
print(car_passing_matrix.shape)

# with open('data_provider/data/traffic_light/processed_matrix/lane_type_vector.pkl', 'rb') as f:
#     # type: list[int]
#     lane_type_vector = pickle.load(f)
#     print('lane_type_vector\tlist[int]\tlength:', len(lane_type_vector))

signal_matrix = load_npz('data_provider/data/traffic_light/processed_matrix/signal_matrix.npz')
print(np.unique(signal_matrix.data))
