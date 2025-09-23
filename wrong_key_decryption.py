# 错误密钥响应
import speck as sp
import numpy as np
from keras.models import model_from_json
from os import urandom
import os
import tensorflow as tf
from keras.models import model_from_json, load_model


# 模型网络结构定义
net6 = load_model('./saved_model/6r_ND1_sc_distinguisher.h5')

def wrong_key_decryption(n, diff=(0x0040,0x0), nr=6, net = net6):
  means = np.zeros(2**16); sig = np.zeros(2**16);
  for i in range(2**16):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);#生成密钥
    ks = sp.expand_key(keys, nr+1); #把密钥扩展至n+1轮
    pt0a = np.frombuffer(urandom(2*n),dtype=np.uint16);#n个密文对
    pt1a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks);
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks);
    rsubkeys = i ^ ks[nr];#用当前的i去异或该轮的密钥形成错误密钥
  #rsubkeys = rdiff ^ 0;
    c0a, c1a = sp.dec_one_round((ct0a, ct1a),rsubkeys);
    c0b, c1b = sp.dec_one_round((ct0b, ct1b),rsubkeys);
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
    Z = net.predict(X,batch_size=10000);
    Z = Z.flatten();
    means[i] = np.mean(Z);#算均值
    sig[i] = np.std(Z);#算标准差
  return(means, sig);


if __name__ == "__main__":
    m6, s6 = wrong_key_decryption(n=3000, nr=6, net=net6)
    np.save("speck_wrong_key_mean_6r.npy", m6)
    np.save("speck_wrong_key_std_6r.npy", s6)
