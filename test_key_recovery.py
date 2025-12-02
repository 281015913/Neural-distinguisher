#Proof of concept implementation of 11-round key recovery attack

import speck as sp
import numpy as np
import tensorflow as tf
from keras.models import model_from_json,load_model
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
#修改参数12211

WORD_SIZE = sp.WORD_SIZE();

neutral13 = [22, 21, 20, 14, 15,  7, 23, 30,  0, 24,  8, 31,  1];

#load distinguishers
#json_file = open('single_block_resnet.json','r');
#json_model = json_file.read();


net7 = load_model('./saved_model/7r_ND1_sc_distinguisher.h5')
net6 = load_model('./saved_model/6r_ND1_sc_distinguisher.h5')


#m8 = np.load('data_wrong_key_8r_mean_1e6.npy');
#s8 = np.load('data_wrong_key_8r_std_1e6.npy'); s8 = 1.0/s8;
m7 = np.load('speck_wrong_key_mean_7r.npy');
s7 = np.load('speck_wrong_key_std_7r.npy'); s7 = 1.0/s7;
m6 = np.load('speck_wrong_key_mean_6r.npy');
s6 = np.load('speck_wrong_key_std_6r.npy'); s6 = 1.0/s6;

#binarize a given ciphertext sample
#ciphertext is given as a sequence of arrays
#each array entry contains one word of ciphertext for all ciphertexts given
def convert_to_binary(l):#将输入的密文样本转换为二进制表示，同时形成密文对的形式输入
  n = len(l);
  k = WORD_SIZE * n;
  X = np.zeros((k, len(l[0])),dtype=np.uint8);
  for i in range(k):
    index = i // WORD_SIZE;
    offset = WORD_SIZE - 1 - i%WORD_SIZE;
    X[i] = (l[index] >> offset) & 1;
  X = X.transpose();
  return(X);

def hw(v):#计算输入向量的汉明重量（即二进制表示中的非零位数）。
  res = np.zeros(v.shape,dtype=np.uint8);
  for i in range(16):
    res = res + ((v >> i) & 1)
  return(res);

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16);#定义了一个low_weight数组，从0-2**16
low_weight = low_weight[hw(low_weight) <= 2];#选出low_weight数组中汉明重量小于等于2的，也就是汉明重量小于等于2的数组

#make a plaintext structure
#takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits
def make_structure(pt0, pt1, diff=(0x211,0xa04),neutral_bits = [20,21,22,14,15]):#利用中性比特位对于单个明文对进行拓展，从一个拓展到64个
  p0 = np.copy(pt0); p1 = np.copy(pt1);
  p0 = p0.reshape(-1,1); p1 = p1.reshape(-1,1);
  for i in neutral_bits:
    d = 1 << i; d0 = d >> 16; d1 = d & 0xffff
    p0 = np.concatenate([p0,p0^d0],axis=1);
    p1 = np.concatenate([p1,p1^d1],axis=1);
  p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1];
  return(p0,p1,p0b,p1b);

#generate a Speck key, return expanded key
def gen_key(nr):#用于生成一个Speck加密算法的随机密钥，并返回扩展后的密钥。
  key = np.frombuffer(urandom(8),dtype=np.uint16);
  ks = sp.expand_key(key, nr);
  return(ks);

def gen_plain(n):#用于生成n位随机明文。
  pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  return(pt0, pt1);

#用于加密明文结构，并返回加密后的密文结构和密钥。
def gen_challenge(n, nr, diff=(0x211, 0xa04), neutral_bits = [20,21,22,14,15,23], keyschedule='real'):
  pt0, pt1 = gen_plain(n);
  pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits);#得到明文结构
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a),0);#用密钥0解密一轮，也就是无消耗扩展一轮
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b),0);
  key = gen_key(nr);
  if (keyschedule is 'free'): key = np.frombuffer(urandom(2*nr),dtype=np.uint16);#生成密钥
  ct0a, ct1a = sp.encrypt((pt0a, pt1a), key);#明文结构加密为密文结构，这里加密11轮
  ct0b, ct1b = sp.encrypt((pt0b, pt1b), key);
  return([ct0a, ct1a, ct0b, ct1b], key);

#将密文结构解密八轮，得到三轮的密文寻找其中差分符合的3轮密文
def find_good(cts, key, nr=3, target_diff = (0x0040,0x0)):
  pt0a, pt1a = sp.decrypt((cts[0], cts[1]), key[nr:]);#解密8轮
  pt0b, pt1b = sp.decrypt((cts[2], cts[3]), key[nr:]);
  diff0 = pt0a ^ pt0b; diff1 = pt1a ^ pt1b;
  d0 = (diff0 == target_diff[0]); d1 = (diff1 == target_diff[1]);#判断diff0是否完全与target_diff[0]相同,d0的形状是[100,64]
  d = d0 * d1;#d0与d1相应位置上均为ture时才被认为是对的，d的形状也是[100,64]
  v = np.sum(d,axis=1);#按行求和，也就是算出每一行有多少个满足差分值的密文对，v的形状是[100,1]
  return(v);

#having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
#函数作用：寻找与猜测候选密钥汉明重量为2以内的密钥，看看里面有没有更高分数的
def verifier_search(cts, best_guess, use_n = 64, net = net6):
  #print(best_guess);#这里传进来的cts形状为（4，64）
  ck1 = best_guess[0] ^ low_weight;#首先传入的best_guess为(0,0),因此第一次异或low_weight不变,形状为（137，）
  ck2 = best_guess[1] ^ low_weight;
  n = len(ck1);#n=137
  ck1 = np.repeat(ck1, n); keys1 = np.copy(ck1);#将ck1重复137次，然后赋值给keys1
  ck2 = np.tile(ck2, n); keys2 = np.copy(ck2);#注意这里是堆叠
  ck1 = np.repeat(ck1, use_n);#将ck1再次复制64份，形状为(1201216,)
  ck2 = np.repeat(ck2, use_n);
  ct0a = np.tile(cts[0][0:use_n], n*n);#将cts的第一行由0取值到64，也就是全部的第一行，然后将其复制137*137次，形状为(1201216,)
  ct1a = np.tile(cts[1][0:use_n], n*n);
  ct0b = np.tile(cts[2][0:use_n], n*n);
  ct1b = np.tile(cts[3][0:use_n], n*n);
  pt0a, pt1a = sp.dec_one_round((ct0a, ct1a), ck1);
  pt0b, pt1b = sp.dec_one_round((ct0b, ct1b), ck1);
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), ck2);
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), ck2);
  X = sp.convert_to_binary([pt0a, pt1a, pt0b, pt1b]);
  Z = net.predict(X, batch_size=10000);#六轮区分器进行辅助预测打分，因为解密了两轮
  Z = Z / (1 - Z);
  Z = np.log2(Z);
  Z = Z.reshape(-1, use_n);#Z重塑为（64，18769）
  v = np.mean(Z, axis=1) * len(cts[0]);#按行求均值后内部元素乘以64，v的形状为（64，）
  m = np.argmax(v); val = v[m];#val为数组v中最大的值
  key1 = keys1[m]; key2 = keys2[m];
  return(key1, key2, val);


#test wrong-key decryption
#本函数就是计算区分器对于所有错误密钥解密的密文打分的情况
#3000个密文用错误的密钥解密（错误的密钥就是正确的密钥与0-2^16进行异或），然后对区分器输出的分数取均值和标准差，然后把这两个称之为区分器的m和s，得到的m和s都是大小为2^16的数组
def wrong_key_decryption(n, diff=(0x0040,0x0), nr=7, net = net7):
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

#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here
tmp_br = np.arange(2**14, dtype=np.uint16);#(16384,)
tmp_br = np.repeat(tmp_br, 32).reshape(-1,32);#tmp_br用于存储32个候选密钥与2^14的所有可能位的异或结果，(16384, 32)

#用于算欧氏距离，cand候选密钥，emp_mean平均值
def bayesian_rank_kr(cand, emp_mean, m=m7, s=s7):#cand、emp_mean形状均为(32,),m与y的形状都是大小为2^16的数组
  global tmp_br;
  n = len(cand);#n=32
  if (tmp_br.shape[1] != n):#采样并扩展到于候选密钥相同
      tmp_br = np.arange(2**14, dtype=np.uint16);
      tmp_br = np.repeat(tmp_br, n).reshape(-1,n);
  tmp = tmp_br ^ cand;#tmp是与候选密钥汉明距离为tmp_br（0-2^14）的密钥，(16384, 32)
  v = (emp_mean - m[tmp]) * s[tmp];#(16384, 32)
  v = v.reshape(-1, n);#没用，因为n=32
  scores = np.linalg.norm(v, axis=1);#计算二范式
  return(scores);

#贝叶斯密钥搜索，迭代五次，第一次是随机选取32个密钥，后面四次是根据上次的密钥算出与之差分为（0-2^14）的密钥的密钥的欧氏距离，选取其中最小的32个作为下次的密钥
def bayesian_key_recovery(cts, net=net7, m = m7, s = s7, num_cand = 32, num_iter=5, seed = None):
  n = len(cts[0]);#形状为（64，）
  keys = np.random.choice(2**(WORD_SIZE-2),num_cand,replace=False); scores = 0; best = 0;#num_cand = 32个随机选取的候选密钥cand,(32,)
  if (not seed is None):
    keys = np.copy(seed);
  ct0a, ct1a, ct0b, ct1b = np.tile(cts[0],num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand);#将选中的密文结构横向平铺32次
  scores = np.zeros(2**(WORD_SIZE-2));
  used = np.zeros(2**(WORD_SIZE-2));#未使用
  all_keys = np.zeros(num_cand * num_iter,dtype=np.uint16);#密钥
  all_v = np.zeros(num_cand * num_iter);#分数
  for i in range(num_iter):
    k = np.repeat(keys, n);#把密钥复制64次，注意这里是复制而不是堆叠
    c0a, c1a = sp.dec_one_round((ct0a, ct1a),k); c0b, c1b = sp.dec_one_round((ct0b, ct1b),k);#32个候选密钥都能对密文结构进行解密
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
    Z = net.predict(X,batch_size=10000);
    Z = Z.reshape(num_cand, -1);#将分数转化为(32,64)格式
    means = np.mean(Z, axis=1);#按行取均值，因为每一行时用同一个密钥得到的,(32,)
    Z = Z/(1-Z); Z = np.log2(Z); v =np.sum(Z, axis=1); #对于得分代入公式进行计算,v是32个密钥分别的得分（32，）
    all_v[i * num_cand:(i+1)*num_cand] = v;#把分数记录进数组all_v中
    all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys);#把本次用过的32个密钥放入数组
    scores = bayesian_rank_kr(keys, means, m=m, s=s);#这一步是计算欧氏距离
    tmp = np.argpartition(scores+used, num_cand)#used没用，也就是把scores数组中最小的num_cand个选出来,就是从0-16384内的所有可能的密钥中选取32个欧氏距离最小的
    keys = tmp[0:num_cand];#把上面选出的32个欧氏距离最小的密钥拿出来，作为下一次观测的目标
    r = np.random.randint(0,4,num_cand,dtype=np.uint16); r = r << 14; keys = keys ^ r;#之前因为是14位的贝叶斯优化寻找，这一步是把剩下两位补充进来
  return(all_keys, scores, all_v);


def test_bayes(cts,it=1, cutoff1=10, cutoff2=10, net=net7, net_help=net6, m_main=m7, m_help=m6, s_main=s7, s_help=s6, verify_breadth=None):
  n = len(cts[0]);#n=100,ct[0]的格式为（100，64）
  if (verify_breadth is None): verify_breadth=len(cts[0][0]);#cts[0][0]格式为（64，）
  alpha = sqrt(n);#alpha =10
  best_val = -100.0; best_key = (0,0); best_pod = 0; bp = 0; bv = -100.0;#记录猜测key
  keys = np.random.choice(2**WORD_SIZE, 32, replace=False);
  eps = 0.001; local_best = np.full(n,-10); num_visits = np.full(n,eps);#local_best记录过去奖励，num_visits访问次数，eps是为了赋一个很小的初值设置的
  guess_count = np.zeros(2**16,dtype=np.uint16);
  for j in range(it):#迭代it次,这里it为500
      priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits);#priority:优先得分，即置信区间上界,priority形状为[100,]
      i = np.argmax(priority);#i为priority中的最大值的索引
      num_visits[i] = num_visits[i] + 1;
      if (best_val > cutoff2):#最佳值大于阈值2时，也就是回复成功了，执行短验证搜索，就是每次恢复结束后121那个进度条
        improvement = (verify_breadth > 0);#当verify_breadth > 0时，improvement赋值ture
        while improvement:#当improvement为ture时，寻找与猜测候选密钥汉明重量为2以内的密钥，看看里面有没有更高分数的
          k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key,net=net_help, use_n = verify_breadth);#先传入的时密文结构中的第一个密文对
          improvement = (val > best_val);
          if (improvement):#如果有那么就更新密钥和best_val
            best_key = (k1, k2); best_val = val;
        return(best_key, j);
      keys, scores, v = bayesian_key_recovery([cts[0][i], cts[1][i], cts[2][i], cts[3][i]], num_cand=32, num_iter=5,net=net, m=m_main, s=s_main);#使用评分最高的密文结构，迭代5次，得到的是160个密钥及其分数
      vtmp = np.max(v);#选取密钥中分数最高
      if (vtmp > local_best[i]): local_best[i] = vtmp;
      if (vtmp > bv):
        bv = vtmp; bp = i;
      if (vtmp > cutoff1):
        l2 = [i for i in range(len(keys)) if v[i] > cutoff1];
        for i2 in l2:
          c0a, c1a = sp.dec_one_round((cts[0][i],cts[1][i]),keys[i2]);
          c0b, c1b = sp.dec_one_round((cts[2][i],cts[3][i]),keys[i2]);         
          keys2,scores2,v2 = bayesian_key_recovery([c0a, c1a, c0b, c1b],num_cand=32, num_iter=5, m=m6,s=s6,net=net_help);
          vtmp2 = np.max(v2);
          if (vtmp2 > best_val):
            best_val = vtmp2; best_key = (keys[i2], keys2[np.argmax(v2)]); best_pod=i;
  improvement = (verify_breadth > 0);
  while improvement:
    k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key, net=net_help, use_n = verify_breadth);
    improvement = (val > best_val);
    if (improvement):
      best_key = (k1, k2); best_val = val;
  return(best_key, it);


def test(n, nr=11, num_structures=100, it=500, cutoff1=5.0, cutoff2=10.0, neutral_bits=[20,21,22,14,15,23], keyschedule='real',net=net7, net_help=net6, m_main=m7, s_main=s7,  m_help=m6, s_help=s6, verify_breadth=None):
  print("Checking Speck32/64 implementation.");
  if (not sp.check_testvector()):
    print("Error. Aborting.");
    return(0);
  arr1 = np.zeros(n, dtype=np.uint16); arr2 = np.zeros(n, dtype=np.uint16);#生成两个长度为100的全是0的np数组，用于存储回复的密钥
  t0 = time();
  data = 0; av=0.0; good = np.zeros(n, dtype=np.uint8);
  zkey = np.zeros(nr,dtype=np.uint16);#生成1个长度为11的全是0的np数组
  for i in range(n):#循环100次
    print("Test:",i);
    ct, key = gen_challenge(num_structures,nr, neutral_bits=neutral_bits, keyschedule=keyschedule);#通过gen_challenge函数得到加密后的100个密文结构和加密用的11轮密钥，此密文结构是密文通过中性比特位翻转后加密得到的,ct格式为（4,100,64)
    g = find_good(ct, key); g = np.max(g);good[i] = g;#g代表100个密文结构里在解密到第三轮时满足差分值最多的密文结构中有多少个满足差分的密文对，并且每次循环时都将这个值存储于数组good中
    guess, num_used = test_bayes(ct,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help, verify_breadth=verify_breadth);#返回猜测的密钥和迭代的次数，每次迭代中都是选取密文结构，进行贝叶斯优化得到优秀密钥，然后返回密钥
    num_used = min(num_structures, num_used); data = data + 2 * (2 ** len(neutral_bits)) * num_used;#num_used是迭代次数，data是攻击n次使用的数据量
    arr1[i] = guess[0] ^ key[nr-1]; arr2[i] = guess[1] ^ key[nr-2];
    print("Difference between real key and key guess: ", hex(arr1[i]), hex(arr2[i]));
  t1 = time();
  print("Done.");
  d1 = [hex(x) for x in arr1]; d2 = [hex(x) for x in arr2];
  print("Differences between guessed and last key:", d1);
  print("Differences between guessed and second-to-last key:", d2);
  print("Wall time per attack (average in seconds):", (t1 - t0)/n);
  print("Data blocks used (average, log2): ", log2(data) - log2(n));
  return(arr1, arr2, good);


arr1, arr2, good = test(20);
np.save(open('run_sols1.npy','wb'),arr1);
np.save(open('run_sols2.npy','wb'),arr2);
np.save(open('run_good.npy','wb'),good);

#arr1, arr2, good = test(20, nr=12, num_structures=500, it=2000, cutoff1=20.0, cutoff2=500, neutral_bits=neutral13,keyschedule='free',net=net8, net_help=net7, m_main=m8, s_main=s8, m_help=m7, s_help=s7, verify_breadth=128);

#np.save(open('run_sols1_12r.npy', 'wb'), arr1);
#np.save(open('run_sols2_12r.npy', 'wb'), arr2);
#np.save(open('run_good_12r.npy', 'wb'), good);

