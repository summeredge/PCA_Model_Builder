# 模型包契约

## 当前状态

当前模型包使用：

-   manifest.json
-   arrays.npz

不包含原始过程数据，不使用 pickle。

当前支持：

-   exploratory / draft
-   normal_state / candidate
-   normal_state / validated

冻结模型包需要后续 schema 演进实现。

## 冻结模型字段

必须包含：

-   schema版本；
-   模型标识；
-   冻结信息；
-   输入Tag及固定顺序；
-   预处理参数；
-   动态特征顺序；
-   均值和标准差；
-   PCA载荷；
-   特征值；
-   主元数量；
-   T²/SPE控制限；
-   状态规则；
-   贡献规则；
-   训练和验证摘要。

字段顺序属于计算契约，不允许部署端重新推断。
