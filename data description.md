### 1. 车辆通过矩阵
- **vehicle_matrix**: 一个 [N, L] 的 0/1 矩阵，表示各车道的车辆通过情况。
  - N: 车道总数。
  - L: 时间范围从 **2023年7月9日 00:00:00** 至 **2023年9月13日 00:01:19**，共计 5,702,479 秒。
  - `veh_matrix[i, j] = 1`: 第 i 个车道在第 j 秒有车辆通过。
  - `veh_matrix[i, j] = 0`: 第 i 个车道在第 j 秒无车辆通过。

### 2. 信号灯状态矩阵
- **signal_matrix**: 一个 [N, L] 的 0/1 矩阵，表示各车道的信号灯状态。
  - `signal_matrix[i, j] = 1`: 第 i 个车道在第 j 秒为绿灯（可通行）。
  - `signal_matrix[i, j] = 0`: 第 i 个车道在第 j 秒为非绿灯（不可通行）。

### 3. 车道类型向量

- **定义**: 一个长度为 N 的向量 `lane_type_vector`，表示每个车道的类型。具体编号如下：
  - `lane_type_vector[i] = 1`: 直行车道（through）
  - `lane_type_vector[i] = 2`: 左转车道（left）
  - `lane_type_vector[i] = 3`: 右转车道（right）
  - `lane_type_vector[i] = 4`: 掉头及左转车道（reverse;left）
  - `lane_type_vector[i] = 5`: 直行及右转车道（through;right）