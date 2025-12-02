import numpy as np
from keras.models import load_model
import time  # 导入时间模块
#随机种子
# 加载预训练模型11111ssaaa
net6 = load_model('./saved_model/6_1_sc_distinguisher.h5')

def test_single_ciphertext_pair_time(net):
    """
    测试神经网络模型对单个密文对的打分时间
    :param net: 加载的神经网络模型
    :return: 单个密文对的预测时间（秒）
    """
    # 1. 生成一个随机的密文对（模拟输入）
    # 假设密文对是4个16位整数（c0a, c1a, c0b, c1b）
    random_ciphertext_pair = np.random.randint(0, 2**16, size=(1, 4), dtype=np.uint16)
    
    # 2. 转换为二进制特征（假设speck.convert_to_binary函数可用）
    # 如果没有speck模块，可以手动实现二进制转换
    def convert_to_binary(ciphertext_pair):
        """
        将密文对转换为二进制特征向量
        :param ciphertext_pair: 密文对，形状为 (1, 4)
        :return: 二进制特征向量，形状为 (1, 64)
        """
        binary_features = np.unpackbits(ciphertext_pair.view(np.uint8)).reshape(1, -1)
        return binary_features
    
    X = convert_to_binary(random_ciphertext_pair)
    
    # 3. 记录神经网络预测时间
    start_time = time.time()  # 记录开始时间
    Z = net.predict(X, batch_size=1).flatten()  # 神经网络预测
    end_time = time.time()    # 记录结束时间
    
    # 4. 计算单个密文对的预测时间
    elapsed_time = end_time - start_time
    return elapsed_time


def test_average_time(net, num_tests=100):
    total_time = 0
    for _ in range(num_tests):
        total_time += test_single_ciphertext_pair_time(net)
    return total_time / num_tests

avg_time = test_average_time(net6, num_tests=100)

print(f"Average time for single ciphertext pair prediction: {avg_time:.6f} seconds")











