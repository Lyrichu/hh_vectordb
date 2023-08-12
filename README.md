## 一个用 GO 语言实现的简单向量数据库,支持ANN 查询、索引构建、存储等基本功能
### 以下是已经实现的基本功能
1. 通用的查询接口
- Insert 数据插入接口
- Nearest 最近邻接口
- KNearest KNN 接口
- Vectors() 当前向量查询接口
- Delete() 向量删除接口

2. 已经实现的数据结构
- brute-force 暴力查询
- kd-tree
- ball-tree
- vp-tree
- cover-tree
- lsh

3. TODO
- 各种 ANN 算法，如 PQ、VQ 等等