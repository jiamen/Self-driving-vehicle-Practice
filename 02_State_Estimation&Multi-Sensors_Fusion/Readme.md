
### data_synthetic.txt是需要读取的传感器输出数据集。
    该数据集包含了追踪目标的激光雷达和毫米波雷达的测量值，以及测量的时间点，
    同时为了验证追踪目标的精度，该数据还包含了追踪目标的真实坐标。

第一列（L和R）表示测量数据是来自激光雷达（Lidar）还是毫米波雷达（Radar）；
    如果第一列是L，则第2、3列表示测量的目标（x,y），第4列表示以设备为参考的测量时间点，
                   第5、6、7、8列表示真实的（x, y, vx, vy）；
    如果第一列是R，则前三咧分别是（p, Ψ, ρ）, 其余列的数据意义与第一列为L时一样。




