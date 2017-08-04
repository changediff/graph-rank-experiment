# graph-rank-experiment

pagerank

## weighted_pagerank.py

该文件可以直接通过命令行运行或者`from weighted_pagerank import wpr`后使用。

+ 命令行调用示例
```
weighted_pagerank.py -e <edge_path> -n <node_path> -d <0-1 num> -m <输出结果中点的数量，默认全部输出> --omega=<list> --phi=<list>

python3 weighted_pagerank.py -e './xx/edge.csv' -n './xx/node.csv' --omega=[1,2,3] --phi=[0.1,0.2,0.3]
```
+ import调用示例见`recommend_pr.py`

## semi_supervised_pagerank.py

未完善，`from semi_supervised_pagerank import ssp`后使用