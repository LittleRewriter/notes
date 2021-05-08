# Miscellanies of Rendering

## Appearance and materials

### Materials

在不同表面上，光线会发生各种各样的现象。什么是图形学中的材质呢？实际上，BRDF决定了光的反射方式。因此，Material = BRDF。

先考虑Diffuse Material(Lambertian Material)。假设各个方向的入射光Radiance相同，那么由于这个点上的能量守恒，入射和出射的Irradiance应当相同。从而，入射和出射的Radiance相同。进而
$$
L_o(\omega_o) = \iint_{H^2} f_r L_i(\omega_i) \cos \theta_i \mathbb{d}\omega_i  = \pi f_r L_i
$$
故
$$
f_r = \frac{\rho}{\pi}
$$
其中$\rho \in (0, 1)$，是一个单通道系数或三通道颜色值，称为albedo。

Glossy Material是表面有些粗糙，但又偏向镜面反射的材质。

玻璃/水同时存在折射现象，一般称为Ideal reflective / refractive Material(BSDF)。

对于完全反射的公式，称为Perfect Specular Reflection。假设入射是$\omega_i$，出射是$\omega_o$，由于
$$
\omega_o + \omega_i = 2(\omega_i \cdot \vec n) \vec n
$$
故
$$
\omega_o = -\omega_i + 2 (\omega_i \cdot \vec n) \vec n
$$
此时的BRDF可以用$\delta$函数来描述。

下面我们考察折射。由于Snell's Law，
$$
\eta_i \sin \theta_i = \eta_t \sin \theta_t
$$
对于方位角，
$$
\varphi_t = \varphi_i \pm \pi
$$
很容易推得
$$
\cos \theta_t = \sqrt{1 - \left(\dfrac{\eta_i}{\eta_t}\right)^2 (1 - \cos^2 \theta_i)}
$$
那么，当$\eta_i > \eta_t$，有可能发生全反射。

有一个很经典的现象叫做Snell's Window：当人在水底看上面，只能看到一个锥型区域。对于水而言，这个锥的顶角大约是$97.2^\circ$。

这种折射的现象叫做BTDF，BRDF与BTDF统称BSDF，其中S表示散射。

不同角度观察，可能会发生不同程度的反射。Fresnel Term指出了在不同角度上反射的程度，绝缘体的Fresnel Term呈现剧烈的变化，在垂直的时候反射接近1，而角度减小，反射也急速减小。导体（金属）的曲线则一直保持较高的水平。

定义
$$
R_s = \left| \frac{n_1 \cos \theta_i - n_2 \cos \theta_t}{n_1 \cos \theta_i + n_2 \cos \theta_t}\right|^2
$$

$$
R_p = \left| \frac{n_1 \cos \theta_t - n_2 \cos \theta_i}{n_1 \cos \theta_t + n_2 \cos \theta_i}\right|^2
$$

$$
R_{eff} = \frac{1}{2} (R_s + R_p)
$$

$R_s$是非极化项，$R_p$是极化项，$R_{eff}$就是Fresnel项。

上述公式较为复杂，Schlick提出了一个很好的近似，即
$$
R(\theta) = R_0 + (1 - R_0) (1-\cos \theta)^5
$$
其中
$$
R_0 := \left( \frac{n_1-n_2}{n_1+n_2}\right)^2
$$
容易发现，$R(0) = R_0, R(\pi/2) = 1$。这一近似易于计算，得到了广泛应用。

### Microfacet Material

当我们离得物体足够远，很多微小物体并不会看到，只能看到这些物体对光的作用形成的最终结果。

从远处看，看到的是Macroscale ： flat & rough 粗糙平面

从近处看，看到的是Microscale ： bumpy & specular 几何体

这就是说，“材质”和“几何体”之间会相互过渡，当距离足够远，几何体就可以近似成一种材质。

两个特例是，如果表面足够精细，那么这种模型就是glossy；如果表面足够粗糙，这种模型就是diffuse。所以微表面的法线分布是一个非常重要的参数。定义
$$
f(\vec i, \vec o) = \frac{F(\vec i, \vec h) \cdot G(\vec i, \vec o, \vec h)\cdot D(\vec h)}{4(\vec n \cdot \vec i)(\vec n \cdot \vec o)}
$$
其中$\vec h$是半长向量，也就是$(\vec i + \vec o)/2$，$F$是Fresnel项，$D$是给定方向上法线分布的值，$f$是BRDF。当$\vec h$与$\vec n$重合，这样的光线才能发生较好的反射。

$G$比较复杂。假如从某个角度看，由于表面之间相互遮挡，可能存在某些微表面失去了自身的作用。如果光线以非常小的角度入射，称为Grazing Angle，这种遮挡就会比较严重。

可以看到，这一模型的决定性因素还是$D$。

Isotrophic / Anisotrophic Materials（各向同性与各向异性材质）也是很重要的。

各向同性的微表面是，各个方向法线分布均匀。各向异性则不均匀。

具体到BRDF上，各向异性表面有
$$
f_r (\theta_i, \phi_i ; \theta_r, \phi_r) \ne f_r (\theta_i, \theta_r ; \phi_r, -\phi_i)
$$
也就是在方位上旋转，其函数发生变化，就是各向异性的。

### BRDF's Property

总的来说，BRDF有下面的属性

- 非负性 

  - $$
    f_r(\omega_i \to \omega_r) \ge 0
    $$
  
- 线性性，各个部分可以拆开

  - $$
    L_r (p, \omega_r) = \int_{H^2} f_r (p, \omega_i \to \omega_r) L_i(p, \omega_i) \cos \theta_i \mathrm{d}\omega_i
    $$

- 可逆性

  - $$
    f_r(\omega_r \to \omega_i) = f_r(\omega_i \to \omega_r)
    $$

- 能量守恒

  - $$
    \forall \omega_r, \int_{H^2} f_r(\omega_i \to \omega_r) \cos \theta_i \mathrm{d}\omega_i \le 1
    $$

- 各向同性介质的特点

  - $$
    f_r(\theta_i, \phi_i; \theta_r, \phi_r) = f_r(\theta_i, \theta_r, \phi_r - \phi_i)
    $$

  - 任意介质都有可逆性，相对方位角与对称性无关

### Measuring BRDF

gonioreflectometer是一个测量机器。

```
foreach outgoing direction wo
	move light to illuminate surface with a thin beam from wo
	for each incoming direction wi
		move sensor to be at direction wi from surfaces
		measure incident radiance
```

如果用各向同性介质和对称性优化，可以少一维的基础上再少一半的计算量。

可以用$(\theta_i, \theta_o, |\phi_i - \phi_o|)$存储BRDF。MERL BRDF Database是一个BRDF库，储存了大量测得的BRDF结果。

### Advanced Light Transport

一、Unbiased Light Transport Methods

依据Monte Carlo积分的无偏性和有偏性，可以将光线追踪分成无偏和有偏两个大类。特别的，如果有偏估计的结果收敛到期望，就称为consistent(一致)。

（1）BDPT，Bidirectional Path Tracing

同时从光源和人眼发出光线，把半路径的端点连接起来，形成的路径追踪就叫做BDPT。

对于某些场景，某个光源发出的光作为间接光会占据很大比例，此时可能难以打到能量集中的区域。如果光源侧的光线容易计算，那么BDPT的效果就会比较好。

它的主要问题是，速度较慢。

（2）MLT，Metropolis Light Transport

用Markov Chain Monte Carlo(MCMC)方法进行计算。在样本周围生成新样本，给定足够时间，这一方法可以生成以任意函数形状为PDF生成的样本。

在Monte Carlo积分时，当PDF与原函数形状一致，达到的采样效果最好。而MCMC恰好可以构造这样的一组函数。

因此，MLT是一个局部优秀的方法，它的核心就是在一个path周围产生其他的path。MLT特别适合做复杂或困难的光路传播，只要有一条光路能找到，就能顺势找到其它光路，比如水中的caustics(焦散)现象，光路经过specular-diffuse-specular（SDS）的路线，传统方法很难形成有效光路。

MLT的问题在于，很难分析其收敛速度。同时，其操作集中在局部上，有些像素收敛快、有些像素收敛慢，图像可能的出来很脏。尤其是在动画渲染中，连续两帧可能差别很大。

二、Biased Light Transport Methods

（1）Photon Mapping（光子映射）

Photon Mapping特别适合渲染Caustics。

这里只介绍其中一个方法。从光源随机打出光线，当光子打到Diffuse 物体就停下来。再从人眼出发打出光线，直到打到Diffuse物体上。

接下来进行局部的density estimation。光子分布越集中的地方越亮，越不集中的地方越暗。对任意的着色点，取周围最近的N个光子，然后考察这些光子占据的面积S。用N/S，就构成了其占据的面积。当N比较大，会得到一个比较干净、但是有点糊的结果。

之所以这一方法是有偏的，是因为$\frac{dN}{dA} \ne \frac{\Delta N}{\Delta A}$。所以估计是有偏的。当光子数量足够大、$\Delta A$足够小的时候，$\Delta A \to dA$，这个估计就会更接近真实结果。所以这一方法就是Consistent的。

所以对应图形学来说，只要存在模糊现象，就是有偏估计；如果样本足够大，能收敛到无偏的结果，就是一致估计。

（2）Vertex Connection and Merging（VCM）

把双向路径追踪和光子映射结合起来。例如对于一个Path，光源有$x_0 \to x_1 \to x_2$，相机有$x_3 \to x_2^*$，且$Pr(||x_2-x_2^*||<r)$，不浪费这种比较接近的路径，就是VCM。

三、Instant Radiosity

分析光线传播的时候，我们不区分光线是从光源而来还是传播而来。所以我们可以将光源上的路径的subpath的停驻点构成新光源，相当于进行一个光线弹射。

所以其过程就是先生成一系列VPL，再当作直接光照进行观察。

这一方法容易在某些缝隙中产生发光，这是因为在缝隙处，作为光源的Radiance以平方进行衰减，而两点之间距离又非常接近，导致最后做出的结果很大。同时，VPL不能处理带Glossy的例子。

### Advanced Appearance Modeling

一、Non-surface Models

（1）Participating Media

云、雾这种定义在空间中的介质，叫做散射介质。光线会随机的被介质中的晶体打到各个方向上去，有可能接收到其他方向而来的光。

定义这种散射方式的就是Phase Function，规定了不同位置上散射的程度。

在渲染的时候，我们线随机的发出一条光线，然后对每个接触的着色点与光源连接进行渲染。这一思想与BRDF其实是相当类似的。

某些看上去是表面材质的东西，光线其实是可以进入的，比如人的皮肤。

（2）Hair Appearance

毛发是很复杂的一个东西。仅仅考虑光线和面的作用是不行的，还需要考虑和头发这个整体的立体相互作用。头发有两种高光，一种是普通的，而另一种有颜色。

人们一般使用Kajiya-Kay Model。假如光线打到圆柱上，发生散射，$\vec i \to \vec r$，$\vec r$的范围构成一个圆锥。这样的着色效果并不好。

Marshner Model考虑了光线进入头发，发生折射。光线存在TT（两次折射）、TRT（折射、内部反射、折射）、R（直接反射）等，将头发当成一个玻璃一样的圆柱，外层是cutickes，内层是cortex。头发内部有色素，光线经过就会产生颜色。这一模型考虑TT、TRT、R，得到了很漂亮的效果。同时，我们还需要考虑多次散射，才能得到最多的结果。

（3）Fur Appearance

如果用人的头发模型套在动物毛发上，结果并不好。这是因为，动物毛发有Cuticle-Cortex-Medulla（髓质）三层结构，光线在髓质会被打到四面八方。而动物的髓质是特别大的，所以光线在髓质中更容易散射。

Double Cylinder Model(Yan Model)模拟了双层圆柱的作用，既有R、TT、TRT，也存在在穿过Medulla的过程中发生散射，形成TT^s^、TRT^s^结构。这一结构也得到了很广泛的应用。

（4）Granular Material

类似香料、盐等一粒一粒的材质，计算量是非常大的。

二、Surface Models

（1）Translucent Material

光线可以从某个地方进入表面，再从某个地方穿出表面。这样的材质就叫做Translucent(半透明) Material。玉石、水母等都是这样的材质。

为了描述这样的散射方式，我们需要定义次表面散射，也就是对BRDF的延申。定义
$$
L(x_o, w_o)=\iint_A \int_{H^2} S(x_i, w_i, x_o, w_o) L_i(x_i, w_i) \cos \theta\  \mathrm{d}w_i\  \mathrm{d}A
$$
来描述各个方向和位置的出光。Jensen提出了Dipole Approximation，光线打到物体表面就像是在表面内和表面上方存在两个光源，分别照亮周围的位置。

（2）Cloth

布料是一系列缠绕的纤维是构成的。纤维相互缠绕形成股，股缠绕形成线，线织成衣物。

根据编织的结果，就可能构成BRDF。但是对于天鹅绒这种表面，用BRDF是不合理的。

将织布看成细小的格子，讨论纤维的朝向分布等，转化为光线的吸收与散射，从而转化为对散射介质的渲染。

甚至，我们可以渲染每根纤维，得到的结果是非常真实的——计算量也是非常庞大的。

（3）Detailed Material

图形中渲染的结果都太完美了。例如，车上颗粒和清漆的相互作用导致的划痕等。

例如，我们统计的NDF往往是标准正态。可是实际的分布往往有很多噪声。比如取一个很大的法线贴图，来模拟物体表面的细度结构，就可能得到比较真实的结果。

之所以这样非常复杂，是因为我们把微表面考虑成了一种镜面。如果考虑一个范围内微表面对一个像素的分布，然后讨论到模型中，那么在这个微表面就会显现出一些比较独特的性质。

当我们引入了这些细节，有可能物体和光线波长可比。此时我们就有必要考虑到光的波动性，需要引入波动光学，在复数域上做积分。此时，波动光学的BRDF可能是不连续的。

三、Procedural Appearance

假如一个花瓶打碎了，那么内部结构如何看呢？想要存储三维材质是不现实的。

Noise材质就是一种神奇的材质，通过噪声材质可以计算出空间某点处的材质。这些材质都是随用随取的，对噪声材质进行处理，就可以得到很好的效果。

这种随用随取的材质，就是Procedural Appearance。

## Cameras & Lenses

### Cameras

不论是光栅化还是光追，都是对真实世界中不存在的东西的建构。用相机成像则是另一种方法，也就是捕捉。这就是计算成像学主要研究的范畴。

如果直接把感光元件放在人的面前，那么有可能收到各个方向的光线，各个方向的能量都会被收集到一起，得到的就是Irradiance。

最古老的相机是基于小孔成像的针孔相机，通过后面传感器的捕捉，取得成像结果。针孔相机拍出来的结果是没有深度可言的，得到的每个地方都是清晰锐利的，不存在景深。

接下来引入Field of View(FOV, 视场)。以针孔摄像机为例，Sensor的宽度是$h$，到小孔的距离Focal Length长度是$f$，那么
$$
FOV = 2\arctan \left(\frac{h}{2f}\right)
$$
对于使用透镜的相机，利用焦距就可以定义FOV。焦距越小，FOV就越大。一般焦距都是对35mm胶片下的定义，对手机等来说需要同比例缩小。相机越大，镜头越长，得到的结果就越好。

曝光$H = T \times E$，$T$是时间，$E$是Irradiance。所以$H$是描述能量的。快门控制$T$，当$T$越大，曝光就越大，感知的光就越大。光圈大小会影响$E$，由f-stop控制。光圈是仿照人的瞳孔设计的，暗处瞳孔变大，明处瞳孔变小。此外，ISO增益（感光度），也就是对接收到光的后期处理，一般是直接做个乘法，也会产生影响。

F-stop是光圈直径的倒数。N越大，直径就越小，那么$H$就越小。

快门速度（Shutter Speed）除了曝光之外，还会影响运动模糊。在快门打开的时间内，物体已经发生了一部分运动，传感器有平均作用，导致运动模糊。当然运动模糊不一定是坏事，本质上是反采样。对于机械快门，由于它是渐渐打开的，就会导致Rolling Shutter。这是因为不同位置可能记录不同时间的光，对于高速运动的物体，就会产生扭曲。

### Lens

相机往往使用透镜组来成像。实际的透镜可能很复杂，所以我们只讨论理想化薄透镜。平行光会汇聚到焦点上，这个距离就是焦距。同时，我们假设这个透镜可以任意改变焦距。

假如定义物距$z_0$，像距$z_i$，那么
$$
\frac{1}{f} = \frac{1}{z_i} + \frac{1}{z_o}
$$
（1）Defocus Blur

距离远的物体并无法直接聚焦在Sensor Plane上，而是离Sensor Plane更远的地方。在接收到的时候，这个点就变成了一个圆，称为Circle of Confusion(CoC)。

COC和光圈大小成正比，如果光圈很大，看到的效果就会更模糊，而小光圈看到的就会更清晰。

我们可以给出一个光圈大小的标准定义：
$$
N = \frac{f}{D}
$$
（2）Ray-Tracing Ideal Thin Lenses

如果模拟薄透镜，给一个焦距，我们就可以模拟出摄像机的效果。

假设定义Sensor有一定大小，定义透镜的光圈大小$D$和焦距$f$，然后定义透镜距离场景平面的距离$z_o$。那么，我们就可以计算出像距$z_i$。

接下来从Sensor上选一点$x'$，再从透镜上选一点$x''$。进行连线，计算过棱镜的$x'''$。然后计算折线方向即可。

（3）Depth of Field

大小光圈会影响成像范围。我们会形成一个范围，在这个范围内，CoC都是足够小的。某个意义上，当CoC比像素大小相当或更小，那么可以看作成像是清晰的。

所以，Depth of Field本质就是清晰的范围。

经过不复杂的推导，
$$
DOF = \frac{D_sf^2}{f^2-NC(D_s-f)} - \frac{D_sf^2}{f^2+NC(D_s-f)}
$$

### Light Field / Lumigraph

如果把所有信息都记录在平面上，被人能够看到，那么这和人看到的真实情景是一样的。人只能看到光线，而看不到这个光线是从哪来的——这和VR其实是有些类似的。

假如我们用Plenoptic Function(全光函数)描述人的视野，那么$P(\theta, \phi)$就可以描述在任意一点看到某个方向的内容。由于颜色是由波长引起的，我们引入参量$\lambda$，那么$P(\theta, \phi, \lambda)$就是一个彩色世界。再引入$t$，$P(\theta, \phi,\lambda, t)$就成为了电影。如果我们允许三维空间中的任意移动，形成的就是Holographic Movie(全息电影)，用$P(\theta, \phi, \lambda, t, V_x, V_y, V_z)$描述。

**世界的本质是七维函数。**

首先考虑光线的定义，光线需要一个起点和方向。假如我们把物体放到包围盒里，描述看到这个物体，这条光线可以用摄像机和包围盒上的点描述。反过来，从包围盒上看，记录从物体出发的光线强度，形成的就是光场。

换言之，光场就是从包围盒出发的四维函数。如果我们已经记录过光场，那么我们就可以直接从光场提取信息，也就是任意一点对物体的观测结果。

我们无需知道物体是什么，而只关注这个包围盒外围每一点的能量情况。对一个平面，用4D的位置+方向定义光线；更进一步，我们取两个平行平面，任取两点，就定义了一条光线。这就是一个四维函数$F(u,v,s,t)$，分别指代两个平面上点$(u,v)$，$(s,t)$。

苍蝇的眼睛是复眼。考虑相机的镜头记录是Irradiance，和方向无关。如果我们把一个像素变成透镜，将不同方向的光进行分解，记录的就变成了不同方向上的radiance。依次原理，可以制作光场照相机。这一照相机可以先拍照，再调焦。

## Color

Spectral Power Distribution(谱功率密度，SPD)可以描述不同波长的分布。SPD具有线性性。两光同时照亮，得到的结果是SPD之和。

颜色本质是人的感知，而不是光线的性质。人是如何感知颜色的？视网膜上，人眼存在感光细胞。感光细胞分成两类，视杆细胞感知强度，视锥细胞感知颜色，而视锥细胞又分成S、M、L三类细胞，分别对低、中、高三种波长进行响应。

不同的人三种细胞的分布是大相径庭的，有严重的个体差异。所以，颜色是人感知的结果。

假如某种细胞对$\lambda$的响应是$r(\lambda)$，那么
$$
S = \int r_s(\lambda)s(\lambda) d\lambda
$$
其中$s$是SPD。同理可以求出$M, L$，那么人们看到的就是三元组$(S, M, L)$。

因此，Meramers(同色异谱)是有可能存在的。利用这一点，我们就可以造出某种颜色，也就是color matching。

由此，我们可以构建出一个RGB系统。假设
$$
Rs_R(\lambda) + Gs_G(\lambda)+Bs_B(\lambda)
$$
所看到的恰好是三元组$(R, G, B)$。那么我们就可以用线性组合的方式匹配所有颜色。

有时，什么颜色都没办法混出来。这个时候对这种颜色加上可以得到某种能混出来的颜色，那么就认为是颜色上的减法。

RGB三种基础颜色本质是三个$\delta$函数，建构出三条曲线，也就是匹配函数。从而定义CIERGB值，对任意一个SPD，系统需要的总量是
$$
R_{CIERGB} = \int \bar r(\lambda) s(\lambda) d\lambda
$$
$G, B$同理。

CIE XYZ系统是也是一套颜色匹配函数。这一系统是人造的，$Y$一定程度上可以表示颜色亮度。同时，这一系统不存在负数，并且覆盖了可见光。首先对$X,Y,Z$进行归一，
$$
x = \frac{X}{X+Y+Z},y = \frac{Y}{X+Y+Z},z = \frac{Z}{X+Y+Z}
$$
假设$Y$是固定的，那么只会影响亮度，而$x, z$就可以确定任意一个颜色。$x-z$图构成Gamut（色域），形成一张扇形区域。

PhotoShop常用HSV空间，也就是所谓颜色拾取器。用色调、饱和度、亮度拾取颜色。饱和度是颜色更倾向于白色还是单色，越大说明越倾向于单色。

CIELAB SPACE用L, a, b表示颜色。极限两端是互补色。L是白/黑，a是红/绿，b是黄/蓝。人类无法想象到偏红的绿色，这是与人脑密切相关的。

CMYK是减色系统，应用更加广泛。