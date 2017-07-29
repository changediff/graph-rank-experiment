# graph-rank-experiment

围绕PageRank的图排序实验代码
主要使用networkx库，将PageRank用到关键词提取和电影推荐中

## 图评分功能

### weighted_pagerank.py

该文件可以直接通过命令行运行或者`from weighted_pagerank import wpr`后使用。

+ 命令行调用示例
```
weighted_pagerank.py -e <edge_path> -n <node_path> -d <0-1 num> -m <输出结果中点的数量，默认全部输出> --omega=<list> --phi=<list>

python3 weighted_pagerank.py -e './xx/edge.csv' -n './xx/node.csv' --omega=[1,2,3] --phi=[0.1,0.2,0.3]
```
+ import调用示例见`recommend_pr.py`

### semi_supervised_pagerank.py

未完善，`from semi_supervised_pagerank import ssp`后使用

## 关键词提取

数据集为`./data/embedding/`部分，主要代码为`ke_*.py`.

1. `ke_preprocess.py`负责预处理部分，`from ke_preprocess import filter_text`后使用。
2. `ke_feature_extract.py`负责抽取特征，现已经完成边特征的抽取，特征抽取后存成csv表格，方便`weighted_pagerank.py`的使用
3. `ke_postprocess.py`负责多元词组的生成，其从原文中找出可能的关键词，根据PageRank结果给定每个词组分数，

## 电影推荐

只作为推荐实验的一部分，负责用PageRank算法对电影类型评分.
数据集为`./data/recommend/`部分，主要代码为`recommend_*.py`