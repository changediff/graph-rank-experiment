# coding:utf-8
import sys, os
from weighted_pagerank import wpr

# 点特征文件所在文件夹
v_path = './data/recommend/V数据挑选/'
file_names = os.listdir(v_path)

for file_name in file_names:
    print(file_name, 'begin')
    try:
        wpr('./data/recommend/sum_weight_fin.csv', os.path.join(v_path,file_name), 
            output='./result/recommend/'+file_name)
    except:
        continue
    print(file_name, 'done')

print('ALL DONE')