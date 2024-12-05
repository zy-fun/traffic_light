import matplotlib.pyplot as plt
from data_provider import Data_Provider
from datetime import datetime

lane_id = 0

dataset = Data_Provider()
car_passing_vector, signal_vector, lane_type = dataset.get_data(lane_id)

print(signal_vector)

