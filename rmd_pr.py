
import sys, os
from weighted_pagerank import wpr

v_path = './data/recommend/V数据挑选/'
file_names = os.listdir(v_path)

for file_name in file_names:
    try:
        wpr('./data/recommend/sum_weight_fin.csv', v_path+file_name, output='./result/recommend/'+file_name)
    except:
        continue