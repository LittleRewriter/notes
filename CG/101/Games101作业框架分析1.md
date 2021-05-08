# Games101作业框架阅读（1）

## Asign1 Rasterization

本次作业的框架，需要完成的是main部分，给出的是rasterization和triangle。triangle只是一个存数据的类，没什么特别的，所以不再展开。

框架使用了opencv来绘图。

先分析main的过程：

main函数分成了几种方法来调用，分别讨论了`Rasterize -r `操作和`Rasterize`的情况

```c++
int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);
	// 观察坐标
    Eigen::Vector3f eye_pos = {0, 0, 5};
	// 三角形的顶点坐标
    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};
	// ID
    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};
	
    /**
    load_positions(pos)我们分析一下
    rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &position) {
    	auto id = get_next_id();
    	pos_buf.emplace(id, positions);
    	return {id};
    }
    这个pos_buf_id是一个int的struct，pos_buf是一个map。get_next_id生成一个静态的id字段，所以这个函数执行之后，让pos_buf[1]指向一个vector。
    那么这里为什么要这样处理呢？
    这个框架中用到了一个函数draw用来画线，这个函数需要两个buffer_id参数。为了类型安全考虑，分别封装了两个struct。
    至于这两个buffer怎么用，在draw中讨论。
    */
    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
	
    int key = 0;
    int frame_count = 0;
	
    // 在控制台前提下，直接输出图像。
    if (command_line) {
        // 清除缓冲区
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
		
        // 设置MVP矩阵
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));
		
        // draw
        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }
	
    // 新建界面，实现渲染
    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
```

接下来，我们讨论Rasterizer。先分析类结构：

![](https://cdn.pic.hlz.ink/2021/02/01/3376ed018b35a.png)

大部分的函数是非常简单的，所以我们重点自顶向下的分析draw的过程。

```cpp
void rst::rasterizer::draw(rst::pos_buf_id pos_buffer, rst::ind_buf_id ind_buffer, rst::Primitive type)
{
    if (type != rst::Primitive::Triangle)
    {
        throw std::runtime_error("Drawing primitives other than triangle is not implemented yet!");
    }
    
    // 获取两个buf，即posbuf和indbuf
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];

    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;
	
    //构造mvp矩阵
    Eigen::Matrix4f mvp = projection * view * model;
    
    // 分开处理每一个ind
    for (auto& i : ind)
    {
        Triangle t;
		
        // 逐点变换成mvp矩阵
        Eigen::Vector4f v[] = {
            	// buf[i[j]]，i为index的缓冲，所以i[j]就取出了第j个buf的对应元素。
            	// 开始的时候，建构buf与ind两个缓冲，需要建立一一对应关系。
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
		
        // 归一化
        for (auto& vec : v) {
            vec /= vec.w();
        }
		
        // 变换到屏幕空间中。
        // 由于vert.x()原先范围是(-1,1)，需要变换到(0,1)
        // 而near = 0.1, far = 100，所以需要将(-1,1)变换到(0.1,100) 
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }
		
        // 逐顶点设置坐标，v[i].head<3>()取出头三个元素
        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }
        
        // 设置三个
        t.setColor(0, 255.0,  0.0,  0.0);
        t.setColor(1, 0.0  ,255.0,  0.0);
        t.setColor(2, 0.0  ,  0.0,255.0);
		
        // 光栅化线框
        rasterize_wireframe(t);
    }
}
```

因此，draw函数完成了顶点预处理的操作，并进一步执行光栅化。再分析rasterize_wireframe：

```cpp
void rst::rasterizer::rasterize_wireframe(const Triangle& t)
{
    draw_line(t.c(), t.a());
    draw_line(t.c(), t.b());
    draw_line(t.b(), t.a());
}
```

可以看到，主要操作就是画线...

那么画线是如何实现的呢？这里用到了`Bresenham's line drawing algorithm`。

这一算法是为了优化浮点数判定而存在的。假设$dx > dy$，也就是随着$x=x+1$，$y$只有可能保持不变或+1.

形式化的，我们假设有网格$x_i, x_{i+1}, \cdots, x_{i+n}$和$y_i, y_{i+1},\cdots, y_{i+n}$，直线经过了$(x_i, y_0)$。

那么当$x = x_{i+1}$，$\Delta y = k\Delta x$。这一点距离$y_i$的值是$k x_{i+1}+m-y_i$，距离$y_{i+1}$的距离是$y_{i+1}-kx_{i+1} - m$

现在，我们令二者相减，得到$d_1 - d_2 = 2kx_{i+1}+2m-y_i-y_{i+1}$

所以，当$d_1-d_2 > 0$，应取下面的格点；当$d_1-d_2<0$，应取上面的格点。为了方便起见，取$x_{i+1}-x_i=1$，$y_{i+1}-y_i = \zeta$，那么

定义判别式
$$
\delta = \Delta x \cdot (d_1-d_2) = 2\Delta y (x_{i}+1)-\Delta x(y_i + y_i + \zeta) + 2m\Delta x
$$
化简得到
$$
\delta_i = 2x_i \Delta y - 2y_i \Delta x + C
$$
其中$C = 2\Delta y -\zeta \Delta x + 2m\Delta x$

建立递推公式
$$
\delta_{i+1} = \delta_i +2\Delta y - 2\zeta\Delta x
$$
如果起始点上，$y_1 = kx_1 + m$，恰好在格点上，那么
$$
y_1\Delta x = x_1\Delta y + m\Delta x
$$
带入得到
$$
\delta_1 = 2\Delta y - \zeta \Delta x
$$
特别的，对于正方形网格，有
$$
\delta_1 = 2\Delta y - \Delta x
$$
这就是误差公式的建立。

进而，我们可以构建当$0<k<1,i>0$时算法的伪代码：

```
DRAW_LINE(x1, x2, y1, y2, color)
	x = x1, y = y1
	dy = y2 - y1
	dx = x2 - x1
	p = 2 * dy - dx
	SET_PIXEL(x, y, color)
	for x = x1 to x2
		if p >= 0
			y = y + 1
			p = p + 2 * (dy - dx)
		else
			p = p + 2 * dy
		SET_PIXEL(x, y, color)
```

这一算法需要继续推广，在四个象限和角平分线上分成了8种情况。详细操作办法这里不再赘述，展开就是draw_line函数。

最后一步操作是set_pixel，设定某个像素的color。这一操作和OpenCV库的设定有关，它需要把颜色信息储存到缓冲区中，而对应的映射规则是$(h-y)*w+x$。最后将设定好的pixel扔给opencv，就完成了全部操作。

## Asign2 More Rasterization

作业2的框架比起作业1没有明显变化，不过比较重要的一点是，作业1是对线框进行光栅化，作业2则是对三角形。

新增的函数只有一个：计算重心坐标的函数。

这个函数的实现就是...带公式

没什么特别好说的...

但是在作业里，需要实现一个对$z$的插值。而它没有直接插值，做了一些看上去很魔幻的操作。这是为什么呢？

这里涉及到了Perspective Correct Interpolation(透视投影矫正)。

由于它的原理比较复杂，我们不加证明的给出结论：$1/z$在投影变换中具有线性不变性。

同时，我们关注投影矩阵，会发现一个很奇妙的性质：在进行投影变换之后，$w' = -z$。那么，
$$
\frac{1}{w'} = \frac{\alpha}{t_{1w}} + \frac{\beta}{t_{2w}} + \frac{\gamma}{t_{3w}}
$$

```c++
float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
z_interpolated *= w_reciprocal;
```

换言之，在插值的时候，我们可以先对$1/z$进行插值，然后再取倒数，这样才能得到正确的插值结果。

## Asign3 More and More Rasterization

作业3的框架比起作业2，肥了不是一点半点。

首先我们有必要解释一下这个代码的运行逻辑。

比起之前，这次的代码架构加了一个东西叫做payload。这个payload是给shader注入的参数，其中的某些参数对应的实质上是shader语义。具体来说，此次的作业框架用到的payload大致可以呈现为：

```glsl
struct v2f {
	Vector3f view_pos : SV_POSITION; // 位置
	Vector3f color : COLOR; // 颜色
	Vector3f normal : NORMAL; // 法线
	Vector2f texcoord : TEXCOORD0; // 纹理坐标
	Texture* texture; // 纹理的指针
};
```

这一计算是在`rasterize_triangle()`函数中进行的。具体来说，我们在作业2的基础上，除去z之外，进行颜色、法线、纹理坐标的插值，并将结果通过payload注入到片元着色器中，得到目标的结果。

这里的插值本质是一个近似。由于三角形很小，所以我们可以用变换后的插值结果近似代替原三角形的插值结果。这样，虽然得到的结果是近似的，但是相对而言有比较不错的效果。

理解到这，基本上作业需要的部分就能完成了。下面我们对几个地方更进一步的探讨一下。

（1）框架中的Shader

Shader作为一个函数，框架中使用了一个函数对象来包装。

框架待实现的一共有5个shader，直接更改`active_shader`就可以实现更改操作。

而这里的Shader结构其实非常简单，只有两个结构体，一个是顶点着色器，这个着色器只提供了position；一个是片元着色器，核心就是上面所介绍的payload。由于payload是固定的，所以这些函数具有相同的结构，可以通过同一个函数对象包装，直接注入。

（2）法线变换

假设我们直接使用变换矩阵，就会出现一些问题。具体来说，随着变换之后，法线可能不再垂直于原平面。为此，我们需要引入法线变换。

不妨令$T$为切线，$N$为法线，变换之后为$T', N'$，切线变换矩阵为$R = M\times V$，法线变换矩阵为$G$，那么
$$
N'^T \cdot T' = 0 \Rightarrow (GN)^T RT = 0 \Rightarrow N^TG^TRT = 0
$$
关注到$N^TT = 0$，所以符合上面的一个结论是$G^TR = I$。因此，我们取
$$
G = (R^T)^{-1} = (R^{-1})^T = ((MV)^T)^{-1}
$$
即可作为法线变换矩阵。

（3）TBN矩阵与bump shading

在求解高度纹理的时候，需要用到TBN矩阵。这是因为我们需要改变获取到法线值，让它与变化后的高度相适应。

首先给出切线空间的定义。我们取法线$\vec n$，从此确定切线$\vec t$。

论坛中有一张图片做的很好（http://games-cn.org/wp-content/uploads/2020/08/TNB.png）：

![](https://cdn.pic.hlz.ink/2021/02/04/4d365e1b6368f.webp)

再用$\vec b = \vec t \times \vec n$，就可以建立起坐标系了。

（所以他给的注释的t向量是错误的）

并且，还可以得到从$xyz$向$tbn$的过渡矩阵
$$
TBN = \pmatrix{t_x & b_x & n_x \\ t_y & b_y & n_y \\ t_z & b_z & n_Z}
$$


接下来，我们考虑高度贴图。对于当前位置而言，假设法线是$(0,0,1)$，从当前位置到$(x+1,y,z)$，其高度变化是$\Delta h_x$，从当前位置到$(x, y+1, z)$，其高度变化是$\Delta h_y$，那么变换之后的法向量
$$
L_n = (-h_x, -h_y, 1)^0
$$
而
$$
\Delta h_x = k_h \cdot k_n \cdot (h(x+\Delta x, y) - h(x, y)), \Delta h_y = k_h \cdot k_n \cdot (h(x, y+\Delta y) - h(x, y))
$$
且
$$
\Delta x = \frac{1}{\text{width}}, \Delta y = \frac{1}{\text{height}}
$$
最终，将法向量进行变换
$$
L = TBN \cdot L_n
$$
这里需要注意的一点是，此时的高度是使用三个颜色值的2-范数进行维护的。

## Asign5 Whitted Style Ray Tracing

> 光追比光栅化简单 ——zmy

作业5建了光追的代码框架，慢慢来。先从最简单的开始：

***global.hpp***

这个文件主要给了一些用到的函数。

clamp 用来使v控制在$[lo, hi]$范围内

solveQuadratic 用来解二次方程，无解返回false

MaterialType 枚举类，用来表示材质。三种材质分别是传统的高光和漫反射、反射和折射以及反射。

get_random_float 获取随机数。多提一嘴，这里的随机数是cpp11新引入的，会有比较好的效果。

UpdateProgress 用来显示进度。

***Object.hpp***

Object定义了物体的基本特性。Sphere和MeshTriangle是两个继承类。

![](https://cdn.pic.hlz.ink/2021/02/16/5090f7b6e8da5.png)

主要介绍几个比较重要的函数吧。

`Sphere::intersect` 直接用二次方程算交点。

`MeshTriangle::intersect` 计算三角形组是否相交，并取到最短距离。

`MeshTriangle::getSurfaceProperties` 计算法线和st。这个st我也不知道全称是啥，但是它是一个类似纹理映射的坐标。

之所以需要这个东西，是因为我们地面是橙色和白色相间的色块。假如设$\{x\}$表示小数部分，那么其着色条件实质上就是
$$
[\{5x\}>0.5]\operatorname{xor} [\{5y\}>0.5]
$$
依据这一结果判断着橙色和白色。

***Renderer.cpp***

在Renderer.hpp里定义了payload，之后会提到。这个文件是重中之重，我们逐个函数来分析。

`deg2rad` 角度转弧度，没什么说的

`reflect` 根据$\vec o = \vec i - 2\vec n (\vec i \cdot \vec n)$，可以算出反射角

`refract` 根据Snell's Law，
$$
\frac{\sin i}{\sin r} = \frac{n_1}{n_2}
$$
 故
$$
\cos r = \sqrt{1 - \eta^2 (1 - \cos^2i)},\ \ \  \eta := \frac{n_1}{n_2}
$$
参考https://blog.csdn.net/cui6864520fei000/article/details/86759960，得到最后的结果是
$$
\vec o = \eta\  \vec i + (\eta \cos i-\cos r)\vec n
$$
我们上面的讨论是忽视了方向问题的。需要做如下讨论：

- 如果从光疏介质到光密介质，那么它和法线的夹角是一个钝角。此时，$\cos i$需要取负。
- 如果从光密介质到光疏介质，那么它和法线的夹角是一个锐角。此时，法线需要取另一侧，同时将折射率交换。这是因为我们只知道光线打到了某个平面，而不能知道是从物体外打到还是从物体内打到。
- 如果$\cos r < 0$，说明发生全反射，光线不能打出。

`fresnel` 利用Fresnel定律计算能量比。这部分在课程里有提，所以不赘述。

`trace` 判断光线是否和物体相交。其原理比较简单，分别判断光线是否和每个物体都相交。取出其中的最近交点，并用来更新payload。

其它都比较好理解，这要注意一下，$t$是物体的距离。这是由于$P=\vec o + t\vec i$，求交接出来的t恰好就是要求的t。

这里的payload结构如下：

```
struct v2f{
	hit_obj : Object, //打到的物体
	tNear : float, // o+it中的t
	index : int, // 三角形编号
	uv : float2 // 接触点的重心坐标
};
```

 `castRay` 核心函数。

 这个函数有如下流程：

- 设置hitColor为背景颜色
- 获取表面属性
- 对不同材料属性讨论进行着色
  - 反射材质
    - 计算衰退比例$k_r$
    - 计算反射方向$\vec o$
    - 计算反射光线的出点。如果这点和法线同侧，那么就假设光线发出点是$P + \vec n \cdot \varepsilon$，这是为了防止光线和发出物体发生碰撞。同理，如果在异侧，那么就假设发出点是$P-\vec n\varepsilon$
    - 递归求解其颜色
  - 折射和反射材质
    - 计算折射和反射比例$k_r$，$k_s$
    - 计算折射和反射方向
    - 计算折射和反射光线的出点
    - 递归求解折射和反射光线的颜色
    - 将二者加权作为最终颜色
  - Phong
    - 为了判断阴影，需要计算光线着落点。计算方法也是分同侧和异侧。
    - 这里要注意，我们着色的时候，相当于此时的相机观察方向就是dir.
    - 扫场景中的每个光源。对每个光源进行如下计算：
    - 判断这一点和光源的连线是否与场景中的物体求交。
    - 如果有交，说明这一点位于阴影内。这里交的条件需要满足到光源的距离小于到物体的距离。
    - 不在阴影内，就分别计算漫反射项和高光项，进行累加。
    - 最后，将漫反射和高光进行加权。

`Render` 主接口。

这里作业部分涉及到一个很有趣的事情。可以参考一下https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays，这简单提一下。

在屏幕空间中，坐标范围是$[0,1279]\times [0,959]$，现在目标是映射到世界空间。

首先，我们将其映射到$[0,1]\times [0,1]$，那么
$$
NDCx = \frac{p_x+0.5}{W} \ \ NDCy = \frac{p_y+0.5}{H}
$$
之所以加上0.5作为补正，是为了保证其分布的对称性。接下来，我们将其映射到$[-1,1]$。关注到这里的$y$是上小下大的，所以需要反过来，则
$$
S'_x = 2NDCx -1 \ \ S'_y = 1-2NDC_y
$$
但是这样物体会变形。为此，我们定义
$$
R := \frac{W}{H}
$$
同时，我们假设相机距离屏幕是1.所以，还需要乘上$\tan \frac{FOV}{2}$来变换。最终，
$$
S_x = R\tan \frac{FOV}{2} \cdot (\frac{2p_x+1}{W}-1), \ \ S_y = \tan \frac{FOV}{2} \cdot (1-\frac{2p_y+1}{H})
$$

## Asign6 BVH Acceleration

作业6啃起来还是挺费劲的。

比起作业5，新增了一些东西。

***AreaLight.hpp*** 区域光，好像用不到

***Intersection.hpp*** 这个是用来描述相交结果的一个结构体。

```cpp
struct interSection {
    happened : bool, // 描述是否发生
    coords : Vector3f, // 描述发生位置
    normal : Vector3f, // 描述法线
    distance : double, // 描述光线原点到接触点的距离
    obj : Object*, // 物体指针
    m : Material* // 材质
}
```

***Material.hpp*** 把原先的材质抽象了一个类出来。

其实没什么实质变化。

***Scene.cpp, Scene.hpp*** 把之前Renderer中的操作做了分离。

在场景中，包含一系列的物体，和一个指向BVH树的东西。

比较重要的函数有这么几个：

`buildBVH()` 调用场景中buildBVH的函数。

`intersect()` 求场景中的BVH与光线求交的结果。

`trace(ray, objects, tNear, index, hitObject)`  对每个物体分别进行求交，取出最近的。

`castRay(ray, depth)` 递归进行求交。depth是光线反射次数，在Scene中给出了最大次数。

***Triangle.hpp*** 这个真的很麻烦。

原先，TriangleMesh和Sphere分别继承了Object。现在，逻辑上，把Triangle重新作为一个类。

![](https://cdn.pic.hlz.ink/2021/02/20/c4d403fa12fdb.png)

先看构造函数。

`MeshTriangle()` 有如下修改：

```c++
bounding_box = Bounds3(min_vert, max_vert);
std::vector<Object*> ptrs;
for (auto& tri : triangles)
	ptrs.push_back(&tri);
bvh = new BVHAccel(ptrs);
```

也就是，MeshTriangle建构了自己专用的bvh。这是非常关键的。

`Triangle`的构造函数没什么特别的。

接下来，比起原来的框架，一大变化是把getIntersection和intersect进行了分别处理。

我们分别进行分析：

`MeshTriangle::intersect`

```cpp
bool intersect(const Ray& ray, float& tnear, uint32_t& index) const
{
    bool intersect = false;
    for (uint32_t k = 0; k < numTriangles; ++k) {
        const Vector3f& v0 = vertices[vertexIndex[k * 3]];
        const Vector3f& v1 = vertices[vertexIndex[k * 3 + 1]];
        const Vector3f& v2 = vertices[vertexIndex[k * 3 + 2]];
        float t, u, v;
        if (rayTriangleIntersect(v0, v1, v2, ray.origin, ray.direction, t, u, v) && t < tnear) {
            tnear = t;
            index = k;
            intersect |= true;
        }
    }

    return intersect;
}
```

可以看到，这一intersect函数是对每三个顶点构成的三角形逐个求交，判断是否相交。这一求交函数就是之前我们实现的。

`MeshTriangle::getIntersection`

```cpp
Intersection getIntersection(Ray ray)
{
    Intersection intersec;

    if (bvh) {
        intersec = bvh->Intersect(ray);
    }

    return intersec;
}
```

直接调了自身BVH的求交函数，等我们分析BVH再说。

`Triangle::intersect` 这个东西根本不会被用到

`Triangle::getIntersection` 算交，之后返回intersection即可。

接下来分析两个重头戏。

***Bounds3.hpp***

Bounds3基本上是一个工具类。这个类描述一个AABB，只需要一个`pMin`和一个`pMax`就可以唯一确定。

`Diagonal` 获取偏移

`maxExtent` 判断哪个方向偏离最多

`SurfaceArea` 返回表面积。这个可以用于SAH，但我没写（

`Centroid` 返回中点

`Intersect` 判断是否与另一个AABB相交

`Offset` 判断偏移的比例

`Overlap` 判断两个AABB是否相交

`Inside` 判断两个AABB是否包含

`IntersectP` 判断AABB是否与直线相交

`Union` 将两个AABB合并

***BVH.cpp BVH.hpp***

涉及到两个类，一个是BVHAccel，用来维护整个树；树的叶子节点是BVHBuildNode，有一个bounds，两个儿子，和物体的指针。

这个类的定义没啥，主要还是得吃透build函数是怎么实现的

`BVHAccel::BVHAccel` 构造函数，调用Build顺便记个时间。

`BVHAccel::recursiveBuild` build函数。

```
recursiveBuild(objects) {
	node <- new BVHBuildNode();
	bounds <- union of objects
	if (objects.size == 1) {
		将node的左右指针置空，包围盒为物体包围盒
	}
	else if (objects.size == 2) {
		node -> ls = recursiveBuild(objects[0])
		node -> rs = recursiveBuild(objects[1])
		node -> bounds为左右两物体union
	}
	else {
		centroidBounds <- 所有包围盒中心构成的最小与最大值之间的包围盒
		判断centroidBounds哪个偏移方向偏移最多
			x方向 对中心x排序
			y方向 对中心y排序
			z方向 对中心z排序
		beginning <- obj.begin
		middling <- obj.middle
		ending <- obj.end
		leftshape <- objects从beginning至middling
		rightshape <- objects从middling至ending
		node->ls <- recursiveBuild(leftshape)
		node->rs <- recursiveBuild(rightshape)
		node->bounds <- bounds
	}
	return node
}
```

剩下的就是在树上查找了，因为是作业所以略。

## Asign7 Path Tracing

最后的一部分了。Path Tracing应该是最难的一部分，从Asign6到7也有一定变化。

先去分析核心流程。整体来说，它经过了下面这些环节：

- main调用Renderer::Render
- 对于场景的每一个像素，打出spp条光线调用castRay函数进行采样
- castRay是我们的核心函数。但是这个函数是作业需要完成的，所以不能细嗦（

总体来说，这个代码的难点就在于实现采样上。如果能理解整个采样流程，就没什么难度了。因此，我们还是逐个文件进行分析。

***BVH.cpp***

在BVHAccel类中，添加了采样函数。

需要说明的是，这个采样是基于一个“概率比较”的。比如，现在有10个物体，每个物体面积一样，都是1。那么我们在总面积10范围内随机生成一个值，比如5.5.这个时候，因为他是位于[5,6]的，我们就取第6个物体。

`Sample` 这个函数是执行采样的函数。他首先生成了一个随机数，这个随机数是[0,1]内的，接下来进行开根号。这样实际上是为了保证在分布上，更多的点靠近1。接下来再让这个值乘上面积进行采样，调用getSample。

`getSample`  这个函数其实就执行了二分查找。通过在BVH树中进行二分，就可以定位到真正对应的物体的采样。

这里可能有一点很疑惑，这个pdf是如何处理的呢？我们不妨举个例子：场景中有A、B两个物体，而MeshTriangle B有C、D两个子物体。现在进行采样，我们找到了B，而这个时候递归的调用了B自身的BVH的Sample函数。所以此时采样D的时候，pdf被除成了$1/S_B$，但对于整个场景来说，希望的pdf是$1/S_{sum}$。因此pdf要在getSample中先做一次乘法，变成原本的样子，再除以根的面积。

所以下面的问题就是，各个物体的采样是如何实现的？

**球的采样(Sphere.hpp)**

实际上就是基于球坐标来采样。因为
$$
\vec x = \vec x_o + R\cdot (\cos \varphi \hat i + \sin \varphi \cos \theta \hat j + \sin \varphi \sin \theta \hat k) (\theta \in [0, 2\pi], \varphi \in [0, \pi])
$$
所以我们只需要生成出来随机的两个角度即可。

**三角形的采样(Triangle.hpp)**

```cpp
void Sample(Intersection &pos, float &pdf){
    float x = std::sqrt(get_random_float()), y = get_random_float();
    pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
    pos.normal = this->normal;
    pdf = 1.0f / area;
}
```

虽然我不知道为什么采样这种采样方式，但是我们能够分析处理这种采样得到的是正确的结果。因为$v_0, v_1, v_2$的三个系数之和是1，并且三个值都是[0,1]之内的，所以三者构成了重心坐标，一定能得到三角形内的点。

**三角形网格的采样(Triangle.hpp)**

直接调用他自己的bvh里的sample。

***Material.cpp***

这玩意也值得好好掰饬掰饬。

首先是两个比较简单的函数。pdf根据出射光线和法线的方向返回，如果在一个平面上就返回$1/2\pi$。eval返回$f_r$，也是根据是否在同侧返回$k_d/\pi$。

接下来是比较复杂的sample。这里边只考虑了漫反射，使用的是极坐标：
$$
r =\sqrt {1-z^2}, x = r\cos \varphi, y = r \sin \varphi (z \in [-1,1], \varphi \in [0,2\pi])
$$
但是这里的采样是在法线坐标系内的，因为采样发生在相对于法线的球内。我们还需要通过TBN矩阵的逆矩阵把它变换到世界空间中也就是toWorld函数，具体可以见前文，不再赘述。

因此，*这个sample函数的作用是在材质上采样一个出射方向出来*。这是和前面的一系列Sample不在同一个体系内的。

***Scene.cpp***

这里边还有一个灯光采样函数。

sampleLight实际上就是按光源面积采样。它过程分成两步，先算出来放射的光源总面积，接下来在这个面积内取一个随机数。再一次遍历所有的灯，一个一个往上加，直到此时的累计面积比随机数大为止，就采样到了一个光源。

这里本质和BVH的采样是没有区别的，只不过BVH用树状结构进行了加速，而这里直接用线性结构遍历。

