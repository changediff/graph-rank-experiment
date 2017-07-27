# graph-rank-experiment

围绕PageRank的图排序实验代码
主要使用networkx库，将PageRank用到关键词提取和电影推荐中

## 关键词提取

数据集为`./data/embedding/`部分，主要代码为`ke_*.py`.

## 电影推荐

只作为推荐实验的一部分，负责用PageRank算法对电影类型评分.
数据集为`./data/recommend/`部分，主要代码为`recommend_*.py`