# Ray tracing

光线追踪与光栅化是两种不同的成像方法。

传统光栅化存在很多不足：

- 全局效果较难表示。为了支持这一算法，光栅化进行了很多改进，但是效果仍然不够漂亮
  - 点光源限于硬阴影
  - 雾面反射，光线在场景中反射多次
  - 间接光照，存在漫反射等
- 光栅化是一个快速但是质量较低的方法

相比而来，光线追踪是准确的，但是是很慢的。所以光线追踪一般用在一些离线应用中，一帧需要大约1万cpu小时。

 ## Whitted-Style Ray Tracing

在光线追踪之前，我们需要定义：什么是光线？

- 光线沿着直线传播。也就是，不考虑光的波动性。
- 光线与光线不发生碰撞。也就是，不考虑光的干涉。
- 光线从光源发出，最终进入人的眼睛。这里应用了光线的可逆性(reciprocity)，某个意义就是从眼睛处释放一系列的光线。

> If you gaze long into an abyss, the abyss also gazes into you.
>
> ——Friedrich Wilhelm Nietzsche

从光源出发是光线追踪的思路，首先要进行光线投射(Ray casting)。

首先，将near plane划分。从相机出发，沿near plane的每个像素发出光线。打到物体后，假设进行完美的折射和反射，取出光线最近的物体交点。之后考虑这一点会不会被照亮；将这个点与light source照射，称为shadow ray。如果shadow ray没有物体遮挡，就说明这点不在阴影中，可以根据基础的Phong模型等进行着色。

但是光线是能弹射很多次的，所以这一过程需要递归的进行。求解这一问题的算法是Whitted-Style Ray Tracing，也叫Recursive Ray tracing。

考虑一个光打到镜子会发生什么——一部分的能量发生反射，沿反射方向继续传播；一部分的能量则发生折射。Whitted算法的核心就是，在某点处可以继续传播光线。同时，它所计算的着色会把所有多级光线得到的结果都加回到image plane的着色中。

我们对光线定义类型。

- eye-ray/primary ray 眼睛打出的第一条光线
- secondary ray 二级光线
- shadow rays 判断可见性用的光线

上面是光线追踪的核心流程。下面解决细节问题

一、Ray-surface Intersection 交点问题

定义光线：有起点$O$和传播方向$\vec d$的向量

那么光线上任意一点
$$
\vec r(t) = \vec o + t\vec d \ \  (0 \le t < \infty)
$$
先考虑球
$$
\vec p : (\vec p - \vec c)^2 - R^2 = 0
$$
带入直线方程，得到
$$
(\vec o + t\vec d - \vec c)^2 = R^2
$$
进而
$$
\vec d^2 t^2 + 2(\vec o -\vec c)\cdot \vec d t + (\vec o - \vec c)^2 - R^2 = 0
$$
根据判别式
$$
\Delta = b^2 - 4ac
$$
可以判定直线与圆的位置关系。

把这个推广到隐式表面$f(\vec p) = 0$上，那么
$$
f(\vec o + t\vec d) = 0
$$
求解正实根就是目标的$t$。

隐式表面是很简单的，显示表面呢？不妨考虑三角形面，让光线与三角形逐个求交。

三角形确定一个平面。那么我们就可以从两个问题考量：光线与平面求交？交点是否在三角形内？后者可以直接用三个叉积解决。

为此，我们需要定义平面。利用平面点法式
$$
(\vec p - \vec p')\cdot \vec n = 0
$$
其中$\vec p'$为定点。那么，
$$
(\vec o + t\vec d - \vec p')\cdot \vec n = 0
$$
解得
$$
t = \frac{(\vec p' - \vec o)\cdot \vec n}{\vec d \cdot \vec n}
$$
这个方法还是比较复杂。有一个Moller Trumbore Algorithm，由于$\vec O + t\vec D$在三角形上，可以用重心坐标描述
$$
\vec o + t\vec d = (1-\xi -\eta)\vec{p_0} + \xi \vec{p_1} + \eta \vec{p_2}
$$
这是一个三元线性方程组，利用Cramer法则，
$$
\pmatrix{t\\\xi\\\eta} = \frac{1}{\vec{s_1} \cdot \vec{e_1}} \pmatrix{\vec{s_2}\cdot \vec{e_2} \\ \vec{s_1} \cdot \vec{s} \\ \vec{s_2} \cdot \vec d}
$$
其中，$\vec{e_1} = \vec{p_1} - \vec{p_0}$，$\vec{e_2} = \vec{p_2} - \vec{p_0}$，$\vec s = \vec o - \vec{p_0}$，$\vec{s_1} = \vec{d} \times \vec{e_2}$，$\vec{s_2} = \vec{s} \times \vec{s_1}$

只要满足$\xi, \eta, 1-\xi-\eta > 0$，则位于平面内。

二、Accelerating Ray-Surface Intersection

上面直接这样计算，其复杂度为
$$
O(pix \cdot obj \cdot \triangle)
$$
如果三角形非常多，这个复杂度显然是不可接受的。为了加速，先引入Bounding Volumes（包围体积）

我们设想一个盒子把物体包起来。如果光线碰不到这个盒子，那么一定碰不到三角形面。一般使用长方体盒子。

这里，我们把长方体当作三个“对面”形成的交集。如果每对对面都是坐标面平行面，我们就叫这个盒子为Axis-Aligned Bounding Box(AABB)。

先考虑二维坐标。假设这个盒子是$[x_0,x_1]\cdot[y_0,y_1]$，光线是$\vec o + t\vec d$，那么和$x_0,x_1$的交点是$t_1,t_2(t_1 < t_2)$，和$y_0,y_1$的交点是$t_3,t_4(t_3 < t_4)$。取$\{t\} = [t_1,t_2] \cap [t_3,t_4]$就是目标的时间区间。

继续推广。当光线进入所有的对面中，才认为光线进入了盒子；出去所有对面，才认为光线离开了盒子。所以，$t_{enter} = \max(t_1,t_3,t_5)$，$t_{exit} = (t_2,t_4,t_6)$。当且仅当$t_{enter} < t_{exit}$，才能认为光线进入盒子中。

当然，这里有一些限制要取。如果$t_{exit} < 0$，说明盒子为光线的“背后”，那么就不会有交点；如果$t_{exit} \ge 0$但是$t_{enter} < 0$，说明光线起点在盒子里，这样就有交点。

而求解$t$也是非常简单的：
$$
t = \frac{p'_x - o_x}{d_x}
$$

三、Uniform Spatial Partitions(Grids)

首先我们找到一个正确的包围盒。接下来，把包围盒划分为一系列的网格。

标记与物体相交的格子，这个相交主要体现的是与物体表面的相交。光线在实际行进的过程中，判定是否经过有物体的网格。如果存在交点，就判定是否和物体相交，就找到了交点。所以，问题转化为光线和盒子的求交。

一般来说，实际中取格子数量是$C \cdot t$，其中$t$是场景中物体的数量，$C$一般取27，是一个常数。

如果物体在场景中分布非常均匀，那么均匀格子就有很好的效果。但是很多时候，物体分布不均匀，比如“Teapot in stadium”，在空旷的场景中有一个小茶壶，就会带来大量的浪费。

四、Spatial Partitions

我们已经看到，直接均匀的划分是不合理的。在有物体分布的地方多划分一些格子是自然的想法。

常见的划分有这些：

- Oct-Tree 八叉树，把包围盒切成八份，在空间中均分每个子节点8份，进行划分到合理的位置。当格子中有足够少数量的物体，组织成树结构，就是很漂亮的划分结果。
- KD-Tree 划分树，不砍成八份，而是或水平或竖直或铅直依次交替的砍一刀，每个节点只有两个杈，仍然保持二叉树的性质。

- BSP-Tree ，二分的划分空间，每一次选一个方向来砍节点。它不是使用AABB的盒子，而是使用斜着的划分。但是BSP-tree对空间的划分越来越复杂，所以不便于计算。

对于场景，预处理一颗KD-Tree。

预处理这些性质：

- 划分轴，x、y、z轴
- 划分位置，细分图形的位置点
- 子节点指针
- 在叶子节点上储存物体性质

接下来，在查找时候就可以大大简化这一过程。

但KD-Tree的问题在于，很难判定三角形与AABB盒的交点是否存在。同时，一个物体可能与多个盒子都存在交集，一个物体会出现在多个叶子节点中，造成了很大浪费。因此，现在KD-Tree的应用逐渐变少了。

五、Object Partition & Bounding Volume Hierarchy(BVH)

BVH的核心思想是，将物体进行划分。有很多物体，将他们分为很多部分，对这些部分的物体分别求包围盒。

这样，一个物体只可能出现在一个格子中，同时回避了三角形与包围盒求交的问题。

但是BVH引起了一个问题：它的划分并不能完全对空间划分，可能会导致不同部分之间的交叠。

BVH有很多不同的研究，主要集中在划分方式上，比如按照KD-Tree的思想对物体分割。

总的来说，其过程是

- 找到包围盒
- 递归的将物体分成两部分
- 重新计算各个部分的包围盒
- 重复这一过程，直到包含物体数量足够小。一般将物体存储在叶子节点上。

在划分维度上，常见的方式有：按照x-y-z顺序划分；按照最长的轴划分；按照中间的物体，也就是位于中位数的三角形，让树结构更平衡。一般先在$O(n)$找到三角形重心，之后将三角形划分。

如果有新的物体加入，BVH需要重新计算。

伪代码如下：

```
Intersect(Ray ray, BVH node) {
	if (ray misses node.bbox) return;
	
	if (node is a leaf node)
		test intersection with all objs;
		return closet intersection;
		
	hit1 = Intersect(ray, node.child1);
	hit2 = Intersect(ray, node.child2);
	
	return the closer of hit1, hit2;
}
```

这样，我们完成了光线和场景物体的求交。

## Basic radiometry(辐射度量学)

辐射度量学可以精确的定义光的一系列物理量。传统的着色模型是不真实的，所以引入这些新的物理概念可以很好的做出着色效果。

一、Radiant Energy and Flux

Radiant Energy是电磁辐射的能量，用$Q$表示，单位是$J$。定义Radiant Flux(power)是单位时间的能量，也就是
$$
\Phi = \frac{\mathrm{d}Q}{\mathrm{d}t}
$$
，单位是$W$。它实际上反映了功率。光学中常使用$\text{lm(lumen)}$作为单位，反映光源的亮度。Radiant Flux也常常用单位时间内通过单位面积的光子数量来表示。

一个光源可能向四面八方发射能量，叫做Radiant Intensity。接收到的是Irradiance，光矢是Radiance。

二、Radiant Intensity

Radiant Intensity is the power per unit solid angle（立体角） emitted by a point light source。
$$
I(\omega) = \frac{\mathrm{d}\Phi}{\mathrm{d}\omega}
$$
单位是$\dfrac{\text{lm}}{\text{sr}} = \text{cd(candela)}$。

那么什么是立体角呢？在平面内，我们有弧度的概念；将这个角度在三维空间中延申，定义
$$
\Omega = \frac{A}{r^2}
$$
$A$是一个锥所对应的面积。

如果化成极坐标代换，
$$
dA = r^2 \sin \theta \operatorname{d}\theta \operatorname{d}\phi
$$
进而
$$
\operatorname{d}\omega = \sin \theta \operatorname{d}\theta\operatorname{d}\phi
$$
这个就是differential Solid Angles（单位立体角）。那么，
$$
\Omega_0 = \int_{S^2} \operatorname{d} \omega = 4\pi
$$
因此球的立体角是$4\pi$。

特别的，对均匀辐射的点光源，
$$
\Phi = \int_{S^2} I \operatorname{d} \omega = 4\pi I
$$
从而
$$
I = \frac{\Phi}{4\pi}
$$

三、Irradiance

Irradiance is power per unit area，也就是
$$
E(x) = \frac{\operatorname{d} \Phi(x)}{\operatorname{d} A}
$$
其单位为$\dfrac{W}{m^2}$，或$\dfrac{lm}{m^2} = lux$

这个定义实质上要求垂直方向上的面积，对于平面来说
$$
E = \frac{\Phi}{A} \cos \theta
$$
定义点光源的Power是$\Phi$，那么在某个面上，
$$
E = \frac{\Phi}{4\pi r^2} = \frac{E}{r^2}
$$
因此，在光的传播过程中，Intensity没有发生变化，Irradiace发生衰减。

四、Radiance

Radiance is the power per unit solid angle, per projected unit area，也就是
$$
L(p,w) = \frac{\operatorname{d}^2 \Phi(p, \omega)}{\operatorname{d}\omega \operatorname{d}A \cos \theta}
$$
其中$\theta$是立体角和法线的夹角。

从定义上看，Radiance is Irradiance per solid angle, 也是 Intensity per projected unit area

也就是
$$
L(p,\omega) = \frac{\operatorname{d} E(p)}{\operatorname{d}\omega \cos \theta}
$$
相当于对Irradiance的方向约束；
$$
L(p,\omega) = \frac{\operatorname{d} I(p,\omega)}{\operatorname{d}A \cos \theta}
$$
相当于对Intensity的大小约束。

也可以取逆运算
$$
E(p) = \int_{H^2} L_i(p,\omega) \cos \theta \operatorname{d}\omega
$$
换言之，Irradiance是各方向的Radiance之和。其中$H^2$是单位半球。

五、Bidirectional Reflection Distribution Function(双向反射度量函数，BRDF)

假设有一道光线打到某个表面$\operatorname{d}A$上，发生反射。假设从某一个微小立体角，入射的是$\operatorname{d}E(\omega_i) = L(\omega_i)\cos \theta_i \operatorname{d} \omega_i$出射的是$\operatorname{d}L_r(\omega_r)$，那么定义
$$
f_r(\omega_i \to \omega_r) = \frac{\operatorname{d} L_r(\omega_r)}{\operatorname{d} E_i(\omega_i)} = \frac{\operatorname{d} L_r(\omega_r)}{L_i \cos(\omega_i) \cos \theta_i \operatorname{d}\omega_i}
$$
这个函数可以根据不同情况确定，例如反射可以只在反射方向有值。所以，BRDF定义了物体材质，也就是和光作用的方式。

变换一下，
$$
L_r(p, \omega_r) = \int_{H^2}f_r(\omega_i \to \omega_r)L_i \cos(\omega_i) \cos \theta_i \operatorname{d}\omega_i
$$
就是对入射方向贡献的累加。

这样，对于确定的着色模型，我们唯一需要的只有BRDF函数。

这里有一个繁琐的点，这里的光源可能是由其他面反射造成的，也就是光传播不止一次，这样的定义显得有一点Recursive。

下面我们补足一项自发光Emission，得到
$$
L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega^+}f_r(p,\omega_i, \omega_r)L_i(p, \omega_i) (\vec n \cdot \vec \omega_i) \operatorname{d}\omega_i
$$
这就是 **Rendering Equation**。这一方程可以用来描述所有的光线传播，也是现代图形学的基石所在。

如果发生了反射，
$$
L_o(p, \omega_o) = L_e(p, \omega_o) + \int_{\Omega^+}f_r(p,\omega_i, \omega_r)L_r(x', -\omega_i) (\vec n \cdot \vec \omega_i) \operatorname{d}\omega_i
$$
这样问题就真正的变成了Recursive，变得可解了。形式化的，我们可以简化为
$$
L(u) = e(u) + \int L(v) K(u,v)\operatorname{d} V
$$
从算子的角度来看，就是
$$
L = E + KL
$$
进行变换
$$
(I-K)L = E
$$
则
$$
L = (I-K)^{-1}E
$$
化为几何级数，得到
$$
L = E + KE + K^2E + \cdots
$$
我们进行分析，$E$是直接光源，$KE$是直接光照，$K^2 E$是经过一次弹射的结果，$K^3E$是两次弹射的结果，......

这个级数相当于自然而然的一次分解。 

实际上，$E+KE$用光栅化是可以完成的，而后面的项光栅化则难以完成。所以后面的部分一般使用光线追踪来解决。在多次计算之后，最终的光照一定会收敛到某一亮度。

这本质体现了能量守恒。

## Probability & Monte Carlo Integral

回顾一下概率论。

$X$ 随机变量，表示一个可能取值的数

$X \sim p(x)$ 随机变量分布 PDF（probability density function）

$p_i$ 概率，随机变量取值的可能性 $p_i \ge 0, \sum p_i = 1$

$EX$ $X$的期望

$p(x)$ 连续函数的概率密度函数

$\int xp(x) dx$ 连续函数的期望

对于随机变量函数$y=f(X)$，$EY = \int f(x)p(x)dx$

下面引入Monte Carlo Integral。假如我们希望计算$\int_a^b f(x)dx$，这一方法可以给出一个数值方法。

取$\forall x_i \in [a,b]$，用$(b-a)f(x_i)$近似积分值，这一数学期望就是积分值。为此，我们定义
$$
X_i \sim p(x)
$$
那么
$$
\lim_{n\to \infty} F_N = \lim_{n \to \infty} \frac{1}{N} \sum^N_{i=1} \frac{f(X_i)}{p(X_i)} = \int_a^b f(x)dx
$$
特别的，如果$p(x) \sim U(a,b)$，上式化为
$$
F_N = \frac{b-a}{N} \sum^N_{i=1} f(x_i)
$$

## Path Tracing

Witted-style Ray Tracing总是以高光反射的物体为基础，考虑物体的反射和折射。如果打到了漫反射表面，那么就停止继续反射。

可是这一近似，很多时候是不正确而没有物理依据的。

- Glossy材质的reflection。Glossy是不完全的反射，看上去像是糊着的，这种反射反射到的一小片区域，而不是镜面一样的某点。典型的模型就是Utah teapot。
- 全局光照。一个漫反射面可以反射出其他漫反射面的颜色，这种现象叫做Color bleeding；多次反射也难以用Witted-style解决。典型的模型就是Cornell Box。

为此，我们试图将渲染方程引用到光追中。两大难点是求解球面积分和递归问题。

而在数值上，我们可以使用Monto-Carlo方法。

假设有一个面光源对球面发生光照，且自身不发光，只考虑直接光照，
$$
L_o(p, \omega_o) = \int_{\Omega^+}f_r(p,\omega_i, \omega_r)L_i(p, \omega_i) (\vec n \cdot \vec \omega_i) \operatorname{d}\omega_i
$$
令
$$
f(x) = f_r(p,\omega_i, \omega_r)L_i(p, \omega_i) (\vec n \cdot \vec \omega_i)
$$
使用均匀分布，即PDF满足
$$
p(\omega_i) = \frac{1}{2\pi}
$$
所以
$$
L_o(p, \omega_o) = \frac{1}{N} \sum^n_{i=1} \frac{f_r(p,\omega_i, \omega_r)L_i(p, \omega_i) (\vec n \cdot \vec \omega_i)}{p(\omega_i)}
$$
这就是求解Radiance的算法。

```
shade(p, wo)
	Randomly choose N direction wi ~ pdf
	Lo = 0.0
	For each wi
		Trace a ray r(p, wi)
		If ray r hit the light
			Lo += (1 / N) * L_i * f_r * cosine / pdf(wi)
	Return Lo
```

同时，对于间接光照，反射出的Radiance实际上也可以看作一个光源，形成了Recursive。也就是，
$$
L_o(p, \omega_o) = \int_{\Omega^+}f_r(p,\omega_i, \omega_r)L_r(x', -\omega_i) (\vec n \cdot \vec \omega_i) \operatorname{d}\omega_i
$$


接下来，我们在算法中加上

```
shade(p, wo)
	Randomly choose N direction wi ~ pdf
	Lo = 0.0
	For each wi
		Trace a ray r(p, wi)
		If ray r hit the light
			Lo += (1 / N) * L_i * f_r * cosine / pdf(wi)
		Else If ray r hit an object at q
			Lo += (1 / N) * shade(q, -wi) * f_r * cosine / pdf(wi)
	Return Lo
```

这个递归是没有终止条件的，会导致光线弹射过多，因而这一数量是完全不可接受的。

考虑到原本打出的光线数量是$N$，那么其光线数量
$$
r = N^{b}
$$
其中$b$是反射次数。

如果想回避指数级增长，最好的方法就是取$N=1$，那么

```
shade(p, wo)
	Randomly choose ONE direction wi ~ pdf
    Trace a ray r(p, wi)
    If ray r hit the light
    	Return L_i * f_r * cosine / pdf(wi)
    Else If ray r hit an object at q
    	Return shade(q, -wi) * f_r * cosine / pdf(wi)
```

这就叫做Path Tracing。如果$N \ne 1$，就叫做Distributed Ray Tracing（分布式采样追踪）

直接取$N=1$会导致噪声很大，但是从视点到像素有多条路径。可以取多条射线，求平均，得到的就是某个像素的着色结果。

```
ray_generation(camPos, pixel)
	Uniformly choose N sample positions within the pixel
	pixel_radiance = 0.0
	For each sample in the pixel
		Shoot a ray r(camPos, cam_to_sample)
		If ray r hit the scene at p
			pixel_radiance += 1 / N * shade(p, sample_to_cam)
	Return pixel_radiance
```

另一个问题是递归没有终止条件。我们引入Russian Roulette(RR，俄罗斯轮盘赌)

我们定义一个概率$P(P \in (0,1))$，以概率$P$打一条光线，返回结果$Lo / P$；以概率$1-p$打一条光线，返回结果$0$。那么
$$
E = P \cdot (Lo / P) + (1-P) \cdot 0 = Lo
$$
期望不发生改变。

所以，

```
shade(p, wo)
	Manually specify a probability P_RR
	Randomly select xi in a uniform dist. in [0,1]
	If (xi > P_RR) return 0.0
	
	Randomly choose ONE direction wi ~ pdf
    Trace a ray r(p, wi)
    If ray r hit the light
    	Return L_i * f_r * cosine / pdf(wi) / P_RR
    Else If ray r hit an object at q
    	Return shade(q, -wi) * f_r * cosine / pdf(wi) / P_RR
```

到此为止，我们已经完成了Path Tracing的核心过程。

但是，如果采样率（SPP, samples per pixel）比较小，得到的结果是比较noisy的。所以这一算法效率并不高。

为什么呢？

假设场景中有一个很大很大的光源，那么随意做一条光线，很容易就能找到光源。但是如果这个光源是一个很小的点光源，问题就会变得很严重了——有大量光线被浪费掉了。

我们能不能找到一个更优的PDF函数，来让采样更有效？从光源出发考虑。

假设光源的法向是$\vec n'$，与光线夹角$\theta'$；平面法线$\vec n$，与光线夹角$\theta$。

在光源上采样，让$\int pdf \operatorname{d}A = 1$。接下来用变换把$dA$投影到$d\omega$上。

立体角就是$dA$在球面上投影的面积，因此
$$
d\omega = \frac{dA \cos \theta'}{||x' - x||^2}
$$
进而
$$
L_o(p, \omega_o) = \int_{A}f_r(p,\omega_i, \omega_r)L_i(p, \omega_i) \frac{\cos \theta \cos \theta'}{||x'-x||^2} \operatorname{d}A
$$
利用均匀分布的pdf函数，直接利用蒙特卡洛积分即可。在原始算法中，我们只需要考虑：

- 对于光源贡献，如果不被遮挡，直接求解
- 对于非光源贡献，利用RR

从而完成了下面的算法：

```
shade(p, wo)
	
	# Contribution from the light source
	Uniformly sample the light at x' (pdf = 1/A)
	Shoot a ray from p to x'
	If the ray is not blocked in the middle
		L_dir = L_i * f_r * cosθ * cosθ' / |x' - p|^2 / pdf_light

	# Contribution from other reflectors
	L_indir = 0.0
	Test Russian Roulette with probability P_RR
	Uniformly sample the hemisphere toward wi (pdf_hemi = 1/2pi)
	Trace a ray r(p, wi)
	If ray r hit a non-emitting object at q
		L_indir = shade(q, -wi) * fr * cosθ / pdf_hemi / P_RR
	
	Return L_dir + L_indir
```

Path Tracing是里程碑式的，甚至可以做出照片级的真实感。



在传统的图形学中，Ray tracing一般指Whitted-style ray tracing。但是在如今，Ray Tracing已经是一个很广泛的范畴了。path tracing, photon mapping，....这些模型都会进一步的影响Ray tracing。

还有很多问题没有提到：

- sampling，对函数的采样理论
- importance sampling，设计函数来更好的采样
- low discrepancy sequences，做出更好的随机数
- multiple imp. sampling，把对半球和光源的采样结合起来
- pixel reconstruction filter，对像素不同位置计算结果的加权
- gamma correction, curves, color space，从辐射还原到颜色中