# A swift and brutal introduction to linear algebra

主要用来熟悉一些术语。。

## 向量(Vector)

写作$\vec{a}$，是一条有向线段，$\overrightarrow{AB} = B - A$

要素是方向和长度。

定义$||\vec{a}||$为$a$的长度，那么定义
$$
\hat{a} = \frac{\vec{a}}{||\vec{a}||}
$$
称为单位向量(unit vector)，这一操作为归一化(normalization)

向量基本操作：

- addition ，有平行四边形法则(parallelogram law)和三角形法则(triangle law)
- dot product
- cross product

在笛卡尔坐标系(cartesian coordinates)中，
$$
A = \binom{x}{y}, A^T = (x, y), ||A|| = \sqrt{x^2+y^2}
$$
默认向量为列向量。

定义点乘运算
$$
\vec{a} \cdot \vec{b} = ||\vec{a}|| \cdot ||\vec{b}|| \cos \theta
$$
点乘结果是一个数量。在图形学中，常变换为
$$
\cos \theta = \frac{\vec{a}\cdot \vec{b}}{||\vec{a}|| \ ||\vec{b}||}
$$
特别的，对unit vector，
$$
\cos \theta = \vec{a} \cdot \vec{b}
$$
dot product满足一些性质：

- 交换律
- 分配律
- 对数乘的交换律

在Cartesian coordinates中，
$$
\vec{a} \cdot \vec{b} = x_ay_a + x_by_b + \cdots x_ny_n
$$
此外，我们经常需要找到向量的投影(projection)。

我们常用$\vec{b}_{\perp}$(读作b perp)表示$\vec{b}$在$\vec{a}$上的投影，那么
$$
\vec{b}_{\perp} = ||\vec{b}||\cos \theta \cdot \hat{a}
$$
此时，相当于将$\vec{b}$分解为垂直方向和平行方向的分解(decompose)。

下面给出一些典型应用：

- 判断前后关系。如果dot product大于0，说明二者方向是相同的；否则相反。这里的forward和backward指的是，以当前向量的方向为y轴建立直角坐标系，如果位于1、2平面就称为forward，否则称为backward。
- 判断方向的接近程度。由于$\cos$的单调性，随着角度的接近程度将会呈现始终减小的变化趋势。

定义cross product运算，$\vec{a}\times \vec{b}$得到一个正交(orthogonal)且符合右手系的新向量。常见的，
$$
\vec{z} = \vec{x} \times \vec{y}, \vec{y} = \vec{z} \times \vec{x}, \vec{x} = \vec{y} \times \vec{z}
$$
cross product有如下性质

- 反交换律
- 分配律
- 对数乘的交换律
- $\vec{x} \times \vec{x} = 0$

对cross product，
$$
\vec{a} \times \vec{b} = \begin{pmatrix}y_az_b - y_bz_a \\ z_ax_b - x_az_b \\ x_ay_b - y_ax_b\end{pmatrix}
$$
可以用矩阵形式进行表达
$$
A^* = \pmatrix{0 & -z_a & y_a \\ z_a & 0 & -x_a \\ -y_a & x_a & 0}
$$
那么
$$
\vec{a} \times \vec{b} = A^* \vec{b}
$$
cross product可以判定inside/outside，或者说left/right：

不妨假设$\vec{a}$与$\vec{b}$都处于$xOy$平面上，那么如果$\vec{a}\times\vec{b}$指向$\hat{z}$方向就说明$\vec{b}$在$\vec{a}$左侧。

下面考虑三角形$ABC$中一点$P$，那么如果$P$在三角形内，分别求$AB,AP$、$BC,BP$、$CA,CP$的叉积，发现后者分别在前者的左边。事实上，只要三者叉积方向相同，那么就表明位于三角形内。这就是三角形光栅化的基础。对于一些corner case，没什么所谓..

以此为基础，我们定义直角坐标系(orthonormal coordinate frame)。此时，
$$
\vec{p} = (\vec{p} \cdot \vec{u})\vec{u} + (\vec{p} \cdot \vec{v})\vec{v} + (\vec{p} \cdot \vec{w})\vec{w}
$$

## 矩阵(matrix)

在CG中，矩阵一般用于表示transform。

矩阵最常用的操作是矩阵乘法。如果$A_{m \times n}$，$B_{n \times p}$，那么$C = A\times B$存在，且$C_{ij} = A(i,:) \cdot B(:,j)$

矩阵乘法 **不符合交换律**，但是符合结合律和分配律。

而如果用矩阵乘列向量，就可以对向量进行变换。

例如，
$$
\pmatrix{-1 & 0 \\ 0 & 1}\binom{x}{y} = \binom{-x}{y} 
$$
除此之外，矩阵还有转置(transpose)操作，且
$$
(AB)^T = B^TA^T
$$
我们还需要定义单位矩阵(Identity Matrix)，从而定义逆矩阵(Inverses)
$$
AA^{-1} = I
$$
逆运算满足$(AB)^{-1} = B^{-1}A^{-1}$

此时，我们可以重新定义点乘
$$
\vec{a} \cdot \vec{b} = \vec{a}^T\vec{b}
$$
对于向量，我们定义dual matrix
$$
A^* = \pmatrix{0 & -z_a & y_a \\ z_a & 0 & -x_a \\ -y_a & x_a & 0}
$$
那么之前叉积的式子也可以得到了。

## 2D Transforms

变换分成两种：modeling和viewing。前者是模型空间的操作，比如物体的拉伸。从3D世界到2D image，这种投影就是一个viewing。

变换主要是为了从三维世界到二维世界。这种变换是非常重要的。

### 缩放

称变换
$$
x' = sx
$$

$$
y' = sy
$$

为缩放变换。其矩阵形式为
$$
\binom{x'}{y'} = \pmatrix{s & 0 \\ 0 & s} \binom{x}{y}
$$
对于不均匀(non-uniform)的缩放变换，
$$
\binom{x'}{y'} = \pmatrix{s_x & 0 \\ 0 & s_y} \binom{x}{y}
$$
特别的，定义反射(reflection)操作
$$
\binom{x'}{y'} = \pmatrix{-1 & 0 \\ 0 & 1} \binom{x}{y}
$$
这一反射是相对于$y$轴的。

### 切变

切变(shear matrix)比较复杂。

称变换
$$
x' = x + ay
$$

$$
y' = y
$$

为切变。

那么
$$
\binom{x'}{y'} = \pmatrix{1 & a \\ 0 & 1} \binom{x}{y}
$$

### 旋转

旋转(rotate)是很常见的变换。

默认情况下，旋转是绕$(0,0)$逆时针旋转。

直接求不好求，不妨考虑基向量的变换情况。对于基向量$(1, 0)^T$，旋转之后变为$(\cos \theta, \sin \theta)^T$；对于基向量$(0,1)^T$，旋转之后变成$(-\sin \theta, \cos \theta)^T$。

因此，对于一组基向量
$$
\pmatrix{\cos \theta & -\sin \theta \\ \sin \theta & \cos \theta} = \pmatrix {1 & 0 \\ 0 & 1} P
$$
得
$$
P = \pmatrix{\cos \theta & -\sin \theta \\ \sin \theta & \cos \theta}
$$
这就是旋转变换矩阵。

旋转矩阵有一个很优雅的性质。容易发现，$R(\theta)^{-1} = R(-\theta)$，而$R(-\theta) = R(\theta)^T$。这本质上是由于正交矩阵的性质。

上面的一些矩阵，归根结底都具有如下形式
$$
x' = ax + by
$$

$$
y' = cx + dy
$$

具有这样性质的变换成为矩阵的线性变换(linear transform)，那么
$$
\binom{x'}{y'} = \pmatrix{a & b \\ c & d} \binom{x}{y}
$$

## 齐次坐标(Homogeneous coordinates)

### translation

考虑平移(translation)变换$(x, y) \to (x+t_x, y + t_y)$，只能写成
$$
\binom{x'}{y'} = \pmatrix{1 & 0 \\ 0 & 1} \binom{x}{y} + \binom{t_x}{t_y}
$$
因此，平移变换不是线性变换。

为此，我们引入homogeneous coordinates来维护这些所有变换。

引入一个第三维$w$，对点，用$(x, y ,1)^T$；对向量，用$(x, y, 0)^T$。那么
$$
\pmatrix{x'\\y'\\w'} = \pmatrix{1&0&t_x\\0&1&t_y\\0&0&1} \pmatrix{x\\y\\1} = \pmatrix{x+t_x\\y+t_y\\1}
$$
之所以规定点是1、向量是0，是因为向量不具有平移性。因此，在经过平移变换之后，向量本身应当是不变的。

更深一层，我们观察w-coordinate的取值：

- vector(0) + vector(0) = vector(1)
- point(1) - point(1) = vector(0)
- point(1) + vector(0) = vector(1)
- point(1) + point(1) = 中点

为什么是中点？这里，我们定义
$$
\pmatrix{x\\y\\w} = \pmatrix{x/w\\y/w\\1}, w \ne 0
$$
此时，两点之和就被归一化成中点。

接下来，我们把任意变换推广到齐次坐标下
$$
\pmatrix{x'\\y'\\w'} = \pmatrix{a&b&t_x\\c&d&t_y\\0&0&1} \pmatrix{x\\y\\1} = \pmatrix{x+t_x\\y+t_y\\1}
$$
称这种变换为仿射变换(Affine Transformations)

换言之， **Affine map = Linear map + Translation**

接下来，我们重写几个仿射变换矩阵
$$
\mathrm{S}(s_x, s_y) =\pmatrix{s_x&0&0\\0&s_y&0\\0&0&1} (Scale\ \  Matrix)
$$

$$
\mathrm{R}(\alpha) = \pmatrix{\cos \alpha&-\sin \alpha&0\\\sin \alpha&\cos \alpha&0\\0&0&1} (Rotation\ \ Matrix)
$$

$$
\mathrm{T}(t_x, t_y) = \pmatrix{1&0&t_x\\0&1&t_y\\0&0&1} (Translation \ \ Matrix)
$$

（1）逆变换(Inverse Transform)

若$A \stackrel{P}{\longrightarrow}B$，那么称$B\stackrel{P^{-1}}{\longrightarrow}A$为$P$的逆变换。

在数学上，这种逆变换恰好是逆矩阵。

（2）复合变换

考虑先平移再旋转和先旋转再平移的图形，显然后者更符合人类需求。

也就是说，一般**先线性变换，再进行平移**。

由于矩阵乘法不满足交换律，所以变换之间的复合需要非常考虑顺序。由于矩阵乘法的左结合性，可以写成统一形式
$$
A_n(\cdots A_2(A_1(x))) = \mathrm{A}_n\cdots\mathrm{A}_{2}\mathrm{A}_{1}\cdot(x\ \  y\ \ 1)^T
$$
接下来，利用结合律，令
$$
B = \mathrm{A}_n\cdots\mathrm{A}_2\mathrm{A}_1
$$
就将变换统合(compose)为一个合成矩阵。

同时，合成的变换也可以进行分解(decompose)。举个例子，我们需要让矩阵绕$C(x,y)$进行旋转，那么可以将变换表示为
$$
\mathrm{T}(c)\cdot\mathrm{R}(\alpha)\cdot\mathrm{T}(-c)
$$
也就是分解为了三个操作，其中$C$表示从$(0,0)$到$(x,y)$的平移变换。

## 3D Transformations

将上面的变换过程推广到三维上，就变成了三维变换。同样的，我们定义homogeneous coordinates：

- point $(x,y,z,1)^T$
- vector $(x,y,z,0)^T$ 

那么定义三维仿射变换
$$
\pmatrix{x'\\y'\\z'\\1} = \pmatrix{a&b&c&t_x\\d&e&f&t_y\\g&h&i&t_z\\0&0&0&1} \pmatrix{x\\y\\z\\1}
$$

从而给出常用变换:
$$
\pmatrix{s_x&0&0&0\\0&s_y&0&0\\0&0&s_z&0\\0&0&0&1}   (\mathrm{S}(s_x, s_y,s_z))
$$

$$
\pmatrix{1&0&0&t_x\\0&1&0&t_y\\0&0&1&t_z\\0&0&0&1}   (\mathrm{T}(t_x,t_y,t_z))
$$

旋转变换比较复杂，我们先讨论按固定轴旋转。
$$
\pmatrix{1&0&0&0\\0&\cos \alpha&-\sin \alpha&0\\0&\sin \alpha&\cos \alpha&0\\0&0&0&1}   (\mathrm{R_x}(\alpha))
$$

$$
\pmatrix{\cos \alpha&0&\sin \alpha&0\\0&1&0&0\\-\sin \alpha&0&\cos \alpha&0\\0&0&0&1}   (\mathrm{R_y}(\alpha))
$$

$$
\pmatrix{\cos \alpha&-\sin \alpha&0&0\\\sin \alpha&\cos \alpha&0&0\\0&0&1&0\\0&0&0&1}   (\mathrm{R_z}(\alpha))
$$

之所以$R_y$是反的，这是因为$\vec{y} = \vec{z} \times \vec{x}$，这种轮换对称性就赋予了$y$轴以相反的性质。

我们定义Eular Angles
$$
\mathrm{R}_{x,y,z}(\alpha, \beta, \gamma)=\mathrm{R}_x(\alpha)\mathrm{R}_y(\beta)\mathrm{R}_z(\gamma)
$$
那么可以证明，任意一个复杂的旋转都可以分解成$x,y,z$的绕轴旋转。给出下面的公式

> Rodrigues' Rotation Formula(罗德里格斯旋转公式)：
> 
> 假设绕旋转轴$\vec{n}$（默认过原点）旋转$\alpha$角，那么
> $$
> \mathrm{R}(n,\alpha) = \cos \alpha I + (1-\cos \alpha) nn^T + \sin  \alpha \pmatrix{0 & -n_z & n_y \\ n_z & 0 & -n_x\\-n_y & n_x & 0}
> $$
> 后边的矩阵其实就是叉乘的对应矩阵$N^*$。

如果需要按任意轴旋转，可以先进行平移。

## Viewing transformation

图形学中主要需要三步变换

- model transformation，从模型空间变换到视图空间
- view transformation，从视图空间变换到相机空间
- projection transformation，从相机空间投影

三步合称 **MVP变换** 。

### View / Camera Transformation

假设相机有如下属性

- position $\vec{e}$
- Look-at direction $\hat{g}$
- up-direction $\hat{t}$

由于观察结果只和相对位置有关，所以我们定义相机位于原点、正上方为$y$轴、看的方向是$-z$，称为标准位置。

假设这个变换称为$M_{view}$，可以分解成$R_{view}T_{view}$。

首先进行平移变换
$$
\pmatrix{1&0&0&-x_e\\0&1&0&-y_e\\0&0&1&-z_e\\0&0&0&1}
$$
接下来进行旋转变换。这个旋转变换包含了三步

- $\hat{g} \to -z$
- $\hat{t} \to y$
- $\hat{g} \times \hat{t} \to x$

直接旋转是很难求解的，我们不妨考虑其逆变换。而逆变换是很容易写出的：
$$
R_{view}^{-1} = \pmatrix{x_{\hat{g}\times \hat{t}} & x_t & x_{-g} & 0 
\\y_{\hat{g}\times \hat{t}} & y_t & y_{-g} & 0 
\\z_{\hat{g}\times \hat{t}} & z_t & z_{-g} & 0 
\\ 0 & 0 & 0 & 1}
$$
由于旋转矩阵是正交矩阵，所以
$$
R_{view} = (R^{-1}_{view})^T =R_{view}^{-1} = \pmatrix{x_{\hat{g}\times \hat{t}} & y_{\hat{g}\times \hat{t}}&z_{\hat{g}\times \hat{t}}  & 0 
\\x_t  & y_t & z_t& 0 
\\x_{-g} &  y_{-g} & z_{-g} & 0 
\\ 0 & 0 & 0 & 1}
$$
于此同时，也可以将物体随之变换，这就是ModelView Transformation

### Projection transformation

投影又可以分成两种：orthographic projection(正交投影)与perspective projection(透视投影)。

正交投影不会带来“近大远小”的性质，这是二者本质的区别。

**正交投影**

正交投影非常简单，就是扔掉z坐标。之后再把所有的点变换到$[-1,1]$，就完成了变换。

形式化的，首先我们定义一个空间立方体$[l,r]\times[b,t]\times[f,n]$，并将其映射到规范(canonical)立方体$[-1,1]^3$。这里要注意$f,n$是远和近，因此相对而言，近的点$z$值比较大。这也就是OpenGL等API使用了左手系。那么将其变为标准立方体的变换就是
$$
M_{ortho} = \pmatrix{\frac{2}{r-l} & 0 & 0 & 0 \\
0 & \frac{2}{t-b} & 0 & 0 \\ 0 & 0 &  \frac{2}{n-f} & 0 \\ 0 & 0 & 0 & 1} 
\pmatrix{1 & 0 & 0 & -\frac{r+l}{2} \\
0 & 1 & 0 & -\frac{t+b}{2} \\ 0 & 0 & 1 & -\frac{n+f}{2} \\ 0 & 0 & 0 & 1}
$$
**透视投影**

透视投影应用更加广泛。

首先考虑齐次坐标的推广，$(x,y,z,1)\Leftrightarrow (xz,yz,z^2,z)(z\ne 0)$

我们同样定义$n$和$f$，透视投影的关键就在于将聚点到$f$面某点的连线投影在$n$面上对应交点。我们可以将透视投影分成两部分：

- 将点缩放到与底面上的长方形等大小的平面，也就是”挤压“
- 进行正交投影

我们从侧面来看这个锥体，假设原本$x,y,z$的点会移动到$x,y',z$，那么根据相似
$$
y' = \frac{Near}{z}y
$$
同理，
$$
x' = \frac{Near}{z}y
$$
从齐次坐标的角度来看，
$$
\pmatrix{x\\y\\z\\1} \Rightarrow \pmatrix{nx/z\\ny/z\\\text{unknown}\\1}\Rightarrow\pmatrix{nx\\ny\\\text{unknown}\\z}
$$
因此，
$$
M_{persp\to ortho} = \pmatrix{n&0&0&0\\0&n&0&0\\?&?&?&?\\0&0&1&0}
$$
现在，我们引入两条性质

- 近平面上，$z$坐标不改变
- 远平面上，$z$坐标不改变

对于近端来说，
$$
\pmatrix{x\\y\\n\\1} \Rightarrow \pmatrix{x\\y\\n\\1}\Rightarrow\pmatrix{nx\\ny\\n^2\\z}
$$
因此，
$$
\pmatrix{0&0&A&B}\cdot\pmatrix{x&y&n&1}^T=n^2
$$
因为$n^2$与$x,y$无关。

再考虑远端，取$(0,0,f)$，有
$$
\pmatrix{0\\0\\f\\1} \Rightarrow \pmatrix{0\\0\\f\\1}\Rightarrow\pmatrix{0\\0\\f^2\\f}
$$
进而，
$$
\pmatrix{0&0&A&B}\cdot\pmatrix{0&0&f&1}^T=f^2
$$
换言之，我们列出方程组
$$
\cases{An+B=n^2 \\ Af+B=f^2}
$$
解得
$$
A=n+f,B=-nf
$$
因此，
$$
M_{persp\to ortho} = \pmatrix{n&0&0&0\\0&n&0&0\\0&0&n+f&-nf\\0&0&1&0}
$$
最终，
$$
M_{persp} = M_{ortho}M_{persp\to ortho}
$$

## Eular Angle and Quanternion

### Eular Angle

欧拉角的原理非常简单。我们让物体的旋转用数$(heading, pitch, bank)$描述。其中heading是绕$y$轴，而pitch绕$x$轴，bank绕$z$轴。但是这里的旋转有两个很重要的地方：

- 正方形遵循左手法则。也就是，我们直接规定所谓的顺逆，而要用左手法则来判定每一步的方向。
- 旋转的轴为自身坐标系。也就是在旋转的过程中，旋转轴本身会发生改变。

除去heading-pitch-bank约定之外，也有其它旋转顺序。但这个顺序是最具有实用价值的，因为在系统中常常有一个地面，所以需要绕y轴进行旋转。同时，pitch度量水平方向的倾角。

欧拉角直观，简单。但是欧拉角最大的缺陷在于不唯一性，一般我们取如下限制：

- $heading\in[-180^\circ, 180^\circ]$
- $bank \in [-180^\circ, 180^\circ]$
- $pitch\in[-90^\circ, 90^\circ]$

这样，对于任意方位，仅存在一种角度表示。但是还存在一种很特殊的情况，我们称为**万向锁**：

> 先heading $45^\circ$，再pitch $90^\circ$和先pitch $90^\circ$，再bank $45^\circ$结果一样

这是因为，当我们在pitch上选择$\pm 90^\circ$时，bank的轴就变成了竖直轴。所以我们做如下限制：

- $pitch = \pm 90^\circ$，则$bank=0$

具有上述四条规则的欧拉角称为 **限制欧拉角**。

### Quanternion

如果要用限制欧拉角进行插值，容易出现严重的抖动问题。所以我们下面讨论能够解决这些问题的四元数。

我们先从二维出发。假设有一个复数$p=x+yi$和$q=\cos \theta + i\sin \theta$，那么
$$
p' = pq = (x\cos \theta-y\sin \theta)+(x\sin \theta + y\cos \theta)i
$$
而这与运用旋转变换所得到的结果是一样的。那么如何把这种复数推广到3D上呢？两个虚部是不够的，需要三个虚部，并且$i, j, k$满足

- $i^2=j^2=k^2=-1$
- $ij=k, ji=-k$
- $jk=i, kj=-i$
- $ki=j, ik=-j$

这样，我们就定义了四元数$[w,x,y,z]$表示$w+xi+yj+zk$。

接下来，我们对四元数的基本运算进行一些讨论。

（1）四元数与轴-角数对

假如$\vec n$为旋转轴，旋转角度为$\theta$，我们首先给出四元数的构造：
$$
q = [\cos \frac{\theta}{2}, \sin \frac{\theta}{2} n_x, \sin \frac{\theta}{2}n_y, \sin \frac{\theta}{2}n_z]
$$
这一四元数用来描述绕$\vec n$指定的旋转轴旋转$\theta$角。一般我们取$\vec n$为单位向量。

（2）四元数的模

类似于向量，定义四元数的模
$$
||q|| = \sqrt{w^2+x^2+y^2+z^2}
$$
如果按照轴-角数对构造法，容易发现，当$|\vec n| = 1$有$||q||=1$。这样的四元数称为单位四元数。

（3）四元数的共轭和逆

对于$q$，定义其共轭
$$
q^* = [w\  v]^* = [w \ -v] = [w \ (-x\ -y\ -z)]
$$
接下来定义其逆
$$
q^{-1} = \frac{q^*}{||q||}
$$
特别的，对单位四元数，其逆就是共轭。从表示上，这相当于让旋转轴翻转，所以等价于实施一个逆变换。

（4）四元数的乘法

假如让两个四元数进行乘法运算，
$$
\begin{aligned}
&(w_1+x_1 i + y_1 j + z_1 k)(w_2 + x_2 i + y_2 j + z_2 k)\\
=&w_1w_2 - x_1x_2 - y_1y_2-z_1z_2 \\
&+(w_1x_2+x_1w_2+y_1z_2-z_1y_2)i\\
&+(w_1y_2+y_1w_2+z_1x_2-x_1z_2)j\\
&+(w_1z_2+z_1w_2+x_1y_2-y_1x_2)k\\
=&[w_1w_2-v_1\cdot v_2 \ \ w_1v_2+w_2v_1+v_2\times v_1]
\end{aligned}
$$
在数学上，我们一般这样定义四元数。但为了方便，我们采用另一种表示方法
$$
[w_1 \ \ \vec v_1][w_2 \ \ \vec v_2] = [w_1w_2-\vec v_1\cdot \vec v_2 \ \ w_1\vec v_2+w_2\vec v_1+ v_1 \times v_2]
$$
要说明的是，我们只修改了最后一项的叉乘方向。这样的定义虽然不符合形式化推导的直观结果，却能在应用上起到很大的简化作用。接下来，我们进一步对乘法性质讨论。

i) 模的乘法
$$
||q_1q_2|| = ||q_1||\ ||q_2||
$$
证明就是摁推，所以我们略去。

ii) 乘积的逆
$$
(q_1q_2\cdots q_n)^{-1} = q_n^{-1}q_{n-1}^{-1}\cdots q_1^{-1}
$$
iii) 旋转性

对于一个3D点$(x, y, z)$，拓展到四元数
$$
p=[0\ \ x\ \ y\ \ z]
$$
我们发现，
$$
p' = q^{-1}pq
$$
相当于对$p$执行了旋转操作。如果我们这个旋转是一系列旋转的组合，
$$
p' = a^{-1}b^{-1}pba=(ba)^{-1}p(ba)
$$
这样，先旋转$b$再旋转$a$，相当于对$ba$进行旋转。这样，乘法和旋转顺序的一致性被保证了。

（5）四元数的差

对于两个四元数$a,b$，我们定义差为满足
$$
ad=b
$$
的四元数$d$。很显然，其值为$a^{-1}b$。因而，四元数的差更类似于除法。

（6）点乘

定义点乘
$$
q_1\cdot q_2 = w_1w_2 + \vec v_1\cdot \vec v_2
$$
四元数的点乘和向量点乘是等价的。

（7）求幂

四元数的求幂代表了一种旋转的程度。比如，$q$表示顺时针旋转$30^\circ$，那么$q^{-1/3}$表示逆时针旋转$10^\circ$.对于求幂，我们可以用
$$
q'=\exp (t \log q)
$$
来计算。设$\alpha := \frac{\theta}{2}$，那么我们如下定义自然对数和指数运算：
$$
q = [\cos \alpha\ \ \vec n\sin \alpha], \log q = [0\ \  \alpha n], \exp [0\ \  \alpha n] = [\cos \alpha \ \  \vec n\sin \alpha]
$$
所以对于上面的运算，我们可以化简
$$
q' = \exp [0 \ \ t\alpha \vec n] = [\cos t\alpha \ \ \sin t\alpha \vec n]
$$
（8）插值

slerp是球面上的一种插值公式，很好的解决了欧拉角的所有问题。其形式为`slerp(q0, q1, t)`，而我们将一般插值转化为四元数形式：
$$
s(q_0, q_1, t) = q_0 \cdot (\Delta q)^t = q_0 (q_0^{-1}q_1)^t
$$
这就是理论插值公式。但是我们还觉得这种计算复杂度过高，所以我们常常采用下面的写法。

![](https://cdn.pic.hlz.ink/2021/03/02/604fdf64cff04.png)

考虑$v_0, v_1$之间的向量$v_t$，我们的目标是求$k_1,k_2$使得
$$
v_t = k_1v_1+k_2v_2
$$
假设夹角是$\omega_0$，$<v_t,v_0>=t\omega_0$，那么由正弦定理
$$
\frac{k_1}{\sin t\omega_0} = \frac{k_0}{\sin (1-t)\omega_0} = \frac{1}{\sin \omega_0}
$$
从而
$$
k_0 = \frac{\sin (1-t)\omega_0}{\sin \omega_0}, k_1 = \frac{\sin t\omega_0}{\sin \omega_0}
$$
将其推广到四元数，得到
$$
\operatorname{slerp}(q_0, q_1, t) =\frac{\sin (1-t)\omega_0}{\sin \omega_0} q_0 +\frac{\sin t\omega_0}{\sin \omega_0} q_1
$$

## 相互转换

这三种形式之间可以相互转换，而这种转换方式一共有6种。

（1）欧拉角转矩阵

考虑欧拉角的HPB变换，由于欧拉角本质是对坐标轴的转换，所以矩阵对点的变换就相当于这种旋转的逆变换。因此，令
$$
H=R_y(-h), P = R_x(-p), B = R_z(-b)
$$
那么其复合旋转矩阵
$$
R=BPH = \begin{pmatrix}
\cos h \cos b + \sin h \sin p \sin b & \sin b \cos p & -\sin h \cos b + \cos h \sin p \sin b \\
-\cos h \sin b + \sin h \sin p \cos b & \cos b \cos p & \sin b \sin h + \cos h \sin p \cos b\\
\sin h \cos p & -\sin p & \cos h \cos p
\end{pmatrix}
$$
（2）矩阵转限制欧拉角

很显然，给一个矩阵，想要回解，就从这里凑吧……总是能凑出来的

然后这也有万向锁，很麻烦，我们略去了

