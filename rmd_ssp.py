# coding:utf-8
from semi_supervised_pagerank import semi_supervised_pagerank as ssp

import sys, os

v_path = './data/recommend/V数据挑选/'
file_names = os.listdir(v_path)

# for file_name in file_names:
    # try:
    #     ssp('./data/recommend/sum_weight_fin.csv', v_path+file_name, fake_B)
    # except:
    #     continue