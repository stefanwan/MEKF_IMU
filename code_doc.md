# MEKF 代码实现

## 变量声明

| 变量名 | 解释 | 数据类型 |
| ----- | ---- | ------- |
| $dt$ |滤波步长 | float |
| $Q\_angle$ | 角度状态的过程噪声方差 | float |
| $R\_measure$ | 量测噪声方差 | float |
| $Q$ | 过程噪声协方差矩阵 | float$_{3\times 3}$ |
| $R$ | 量测噪声协方差矩阵 | float$_{3\times 3}$ |
| $p\_kk、p$ | 状态误差的条件协方差矩阵 | float$_{3\times 3}$ |
| $q\_est$ | 四元数的估计值 | float$_{4}$ |
| $u$ | 陀螺仪角速度，预测方程的输入项 | float$_{3}$ |
| $Phi$ | 预测方程的状态转移矩阵 | float$_{3\times 3}$ |
| $K$ | 滤波方程的增益 | float$_{3\times 3}$ |
| $q\_mea$ | 四元数的观测值 | float$_{4}$ |

## 函数说明

### Euler2Quaternion()

* 函数功能：将欧拉角转换成单位四元数表示。
* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $euler\_angle$ |按照 Z-Y-X 的解算顺序表示姿态旋转的欧拉角 | float$_{3}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q$ | 输入欧拉角对应的四元数 | float$_{4}$ |

### Quaternion2Euler()

* 函数功能：将四元数按照 Z-Y-X 的解算顺序转换成欧拉角表示。
* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q$ | 单位四元数 | float$_{4}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $euler\_angle$ |单位四元数对应的Z-Y-X顺序解算出的欧拉角 | float$_{3}$ |

### EQuaternion()

* 函数功能：计算四元数的指数映射，即将旋转矢映射到四元数所在的群。
  * 如果输入的三维矢量模长为0，那么返回[1.0, 0.0, 0.0, 0.0]
  * 如果输入的如果输入的三维矢量模长不为0，则按照映射关系进行计算

* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $v$ | 非标准旋转矢量 | float$_{3}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q$ | 单位四元数 | float$_{4}$ |

### MultiplyQuaternion()

* 函数功能：四元数的乘法
* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q1$ | 单位四元数 | float$_{4}$ |
  | $q2$ | 单位四元数 | float$_{4}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q$ | 单位四元数 | float$_{4}$ |

### ERotateMatrix()

* 函数功能：计算旋转矩阵的指数映射，即将旋转矢量映射到旋转矩阵所在的群。
  * 如果输入的三维向量模长为0，返回单位矩阵
  * 如果模长不为0，则按照映射关系进行计算
* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $v$ | 非标准旋转矢量 | float$_{3}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $RotM$ | 旋转矩阵 | float$_{3\times 3}$ |

### InvQuaternion()

* 函数功能：计算四元数的逆
* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q$ | 单位四元数 | float$_{4}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q_inv$ | 单位四元数, 对应输入的逆 | float$_{4}$ |

### LogQuaternion()

* 函数功能：计算四元数的对数映射，相当于指数映射的反运算。
  * 如果输入的四元数实部为1，那么返回0.0*[1.0, 0.0, 0.0]
  * 如果输入的四元数实部不为1，那么按照对数映射关系来计算
* 函数输入
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $q$ | 单位四元数 | float$_{4}$ |
* 函数输出
  | 变量名 | 解释 | 数据类型 |
  | ----- | ---- | ------- |
  | $v$ | 标准旋转矢量 | float$_{3}$ |
  
## 代码逻辑

***
**初始化**
设置协方差矩阵 $Q、R、p\_kk$ 和估计状态初始值 $q\_est$
***
**状态预测**
$u \gets$ 陀螺仪测得的三轴角速度
$q\_est \gets$ 根据四元数动力学预测下一步状态
$Phi \gets$ 误差状态的状态转移矩阵
$p \gets$ 更新误差状态的协方差矩阵，计算$p(k+1,k)$
***
**状态滤波**
$K \gets$ 计算滤波增益
$z \gets$ 根据加速度计测量得到的加速度计算角度观测值
$q\_mea \gets$ 将角度观测值转换成四元数
$dv \gets$ 根据测量值和预测值计算误差旋转矢量
$q\_est \gets$ 根据观测值更新状态量
$p\_kk \gets$ 更新误差状态的协方差矩阵，计算$p(k+1,k+1)$