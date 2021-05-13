## HL-Note
### Detection
#### Detection History
![目标检测发展史](https://user-images.githubusercontent.com/49737867/113255745-c59fde00-92fa-11eb-977c-7d3864960c85.png)

```markdown
Syntax highlighted code block
message ConcatParameter {
  //指定拼接的维度，默认为1即以channel通道进行拼接;支持负索引，即-1表示最后一个维度
  optional int32 axis = 2 [default = 1];
  // 以后会被弃用，作用同axis一样，但不能指定为负数
  optional uint32 concat_dim = 1 [default = 1];
}
- Bulleted
- List
1. Numbered
2. List
**Bold** and _Italic_ and `Code` text
[Link](url) and ![Image](src)
```
# 深度学习 1
## 残差网络 2
### 目标检测 3

#### basic
1.backbone：主干网络，用来做特征提取的网络，代表网络的一部分，一般是用于前端提取图片信息，生成特征图feature map,供后面的网络使用。通常用VGGNet还有你说的Resnet，因为这些backbone特征提取能力是很强，并且可以加载官方在大型数据集(Pascal 、Imagenet)上训练好的模型参数，然后接自己的网络，进行微调finetune即可

2.DPM模型：Deformable part model 为可形变部件模型，简称DPM模型。这种模型非常地直观，它将目标对象建模成几个部件的组合。
特征金字塔

3.ground truth：真实标签，人为为每个目标标记的标签

4.Upsample 上采样 和 downsample下采样
上采样是为了将特征图采样到指定分辨率大小,比如一张(416,416,3)的图片经过一系列卷积池化操作后,得到一个特征图,维度(13,13,16), 为了把这个特征图和原图进行比较,需要将这个特征图变成(416,416,3)大小.这个就称为上采样，上采样的过程类似于一个卷积的过程,只不过在卷积之前将输入特征值插到一个更大的特征图然后进行卷积
upsampling(上采样)的三种方式：
Resize，如双线性插值直接缩放，类似于图像缩放;
反卷积(deconvolution & transposed convolution);
反池化(unpooling)
目标检测中上采样和下采样：
目标检测任务中，上采样和下采样又有了不同的定义。
上采样的目的是减少感受野，一般用来检测小物体，方法是采用高分辨率的图片。下采样的目的是增大感受野，用来检测大物体，方法是采用低分辨率的图片。
上采样有3种常见的方法：双线性插值(bilinear)，反卷积(Transposed Convolution)，反池化(Unpooling)
5.downsample下采样 ：
缩小图像(或称为下采样(subsampled)或降采样(downsampled)的主要目的有两个:1.使得图像符合显示区域的大小,2生成对应图像的缩略图

6. COCO API
构建coco对象， coco = pycocotools.coco.COCO(json_file)

coco.getImgIds(self, imgIds=[], catIds=[]) 返回满足条件的图像id

coco.imgs.keys() 数据集中所有样本的id号

coco.imgToAnns.keys() 数据集中有GT（真实的框）对应的图像样本的id号（用来过滤没有标签的样本）

coco.getCatIds 返回含有某一类或者几类的类别id号

coco.loadImgs()根据id号，导入对应的图像信息

coco.getAnnIds() 根据id号，获得该图像对应的GT的id号

coco.loadAnns() 根据 Annotation id号，导入标签信息

7.Few-shot检测器,称为SRR-FSD

#### NMS（non-maximum suppression）

对于一个预测边界框B，模型最终会输出会计算它属于每个类别的概率值，其中概率值最大对应的类别就是预测边界框的类别。在同一副图像上，把所有预测边界框(不区分类别)的预测概率从大到小进行排列，然后取出最大概率的预测边界框B1作为基准，然后计算剩余的预测边界框与B1的交并比，如果大于给定的某个阈值，则将这个预测边界框移除。这样的话保留了概率最大的预测边界框并移除了其他与其相似的边界框。接下来要做的就是从剩余的预测边界框中选出概率值最大的预测边界框B2计算过程重复上述的过程

测试阶段：
首先在图像中生成多个anchor box，然后根据训练好的模型参数去预测这些anchor box的类别和偏移量，进而得到预测的边界框。由于阈值和anchor box数量选择的问题，同一个目标可能会输出多个相似的预测边界框，这样不仅不简洁，而且会增加计算量，为了解决这个问题，常用的措施是使用非极大值抑制(non-maximum suppression，NMS)。
扩展：在解决多尺度问题时主要采用一种思想--金字塔，或者是例如DPM模型中经典的特征金字塔。在不同分辨率的特征图下检测不同尺寸的目标。但是这样存在一个问题，就是大大的增加了计算量

#### RCNN 
分三步
第一步 通过Selective Search算法生成候选框

第二步 对每个候选框用深度网络提取信息（候选框进入网络之前先进性reset处理，统一缩放到227*227），缩放后的候选框（其实是图片）输入到网络（图像分类网络），得到特征向量（网络的全连接层去掉了）

第三步 特征送入每一个svm类别的分类器 对svm结果矩阵每一列（也是每一类）非极大值抑制 只留下最准确的边界框，删除了重叠的候选框

第四步 使用回归器（最小二乘法）也是算IoU精细修正候选框的位置 （在上一步的基础上）

RCNN存在的问题，候选框大量重叠冗余，训练速度慢（要训练分类网络（去掉全连接层用于提取特征），svm分类器，bbox回归器），每一个候选框都要写入磁盘，浪费空间

![image](https://user-images.githubusercontent.com/49737867/116064994-1f58b580-a6b9-11eb-9f12-8639a6279621.png)

#### FastRCNN
用VGG16作为backbone，比RCNN快了9倍，准确率从62%提升至66%
第一步 通过ss算法生成1k到2k个候选区域

第二步 图像通过CNN生成feature map，将候选区域投影到featuremap中 获得相应的特征矩阵 （参考了spp net）

第三步 将每个特征矩阵通过RoI（region of interest） Pooling层缩放到7*7大小的特征图，接着将特征图展平，通过一些列全连接层得到预测的结果
分类和回归都在一个网络中，因此不需要单独训练svm分类器和回归器

候选区域的特征矩阵直接映射得到，也就不需要重复计算了

并且训练的时候并不使用ss算法提供的所有区域，而且采用了采样sampling
训练数据的采样，（正样本，负样本）
RoI最大池化下采样，把任何图片转成7*7，因此不限制输入图像的尺寸

当然Fast RCNN的主要缺点在于region proposal的提取使用selective search，目标检测时间大多消耗在这上面（提region proposal 2~3s，而提特征分类只需0.32s），这也是后续Faster RCNN的改进方向之一。
边界框公式里面的d是超参

我的问题：
回归的全连接层有（n+1）*4个节点，那么是不是一个候选框的结果框出两个目标

我的理解：虽然bbox回归器中有4*（n+1）个节点，但是我们最后根据分类器概率最大的找出对应的类，只看bbox中对应该类的（x,y,w,h）
![image](https://user-images.githubusercontent.com/49737867/116075425-83817680-a6c5-11eb-99e3-20fac24e8407.png)

#### Faster RCNN
<font color=red>缺点：
  1.检测速度慢（因为分两步）
  2.很难检测小物体</font>

检测速度5fps
1.提出anchor box的原因：
一个窗口只能检测一个目标
无法解决多尺度问题。
2.什么时候触发anchorbox？
训练阶段：
在经过一系列卷积和池化之后，在feature map层使用anchor box，如上图所示，经过一系列的特征提取，最后针对 [公式] 的网格会得到一个 [公式] 的特征层，其中2是anchor box的个数，以《deep learning》课程为例选择两个anchor box，8代表每个anchor box包含的变量数，分别是4个位置偏移量、3个类别(one-hot标注方式)、1个anchor box标注(如果anchor box与真实边框的交并比最大则为1，否则为0)。

（个人认为四个位置偏移量是针对整张图的，而不是针对特征图里的小框框）

到了特征层之后对每个cell映射到原图中，找到预先标注的anchor box，然后计算这个anchor box与ground truth之间的损失，训练的主要目的就是训练出用anchor box去拟合真实边框的模型参数。

![image](https://user-images.githubusercontent.com/49737867/116078973-db21e100-a6c9-11eb-81fa-48f4fbd4079d.png)

计算ZF网络感受野的方法
![image](https://user-images.githubusercontent.com/49737867/116086529-7f0f8a80-a6d2-11eb-9352-b952aead5cdc.png)
backbone提取得到的最后的特征图其过程经过了多次卷积，关键在于卷积本身就会造成分辨率的不停降低，不适合做小目标的检测
#### YOLOv1
优点：
1.Yolo很快，因为用回归的方法，并且不用复杂的框架。
2.Yolo会基于整张图片信息进行预测，而其他滑窗式的检测框架，只能基于局部图片信息进行推理。
3.Yolo学到的图片特征更为通用。作者尝试了用自然图片数据集进行训练，用艺术画作品进行预测，Yolo的检测效果更佳。
缺点：
如上述原文中提及，在强行施加了格点限制以后，每个格点只能输出一个预测结果，所以该算法最大的不足，就是对一些邻近小物体的识别效果不是太好，例如成群结队的小鸟。

#### YOLOv2
改进：
1.batch Normalization（批归一化）
检测系列的网络结构中，BN逐渐变成了标配。在Yolo的每个卷积层中加入BN之后，mAP提升了2%，并且去除了Dropout。

2. High Resolution Classifier（分类网络高分辨率预训练）
在Yolov1中，网络的backbone部分会在ImageNet数据集上进行预训练，训练时网络输入图像的分辨率为224*224。在v2中，将分类网络在输入图片分辨率为448*448的ImageNet数据集上训练10个epoch，再使用检测数据集（例如coco）进行微调。高分辨率预训练使mAP提高了大约4%。

3.Convolutional With Anchor Boxes（Anchor Box替换全连接层）
每个格点指定n个Anchor框。在训练时，最接近ground truth的框产生loss，其余框不产生loss。在引入Anchor Box操作后，mAP由69.5下降至69.2，原因在于，每个格点预测的物体变多之后，召回率大幅上升，准确率有所下降，总体mAP略有下降。
v2中移除了v1最后的两层全连接层，全连接层计算量大，耗时久。

4. Dimension Clusters（Anchor Box的宽高由聚类产生

5.Direct location prediction（绝对位置预测）

6.Fine-Grained Features（细粒度特征）
在26*26的特征图，经过卷积层等，变为13*13的特征图后，作者认为损失了很多细粒度的特征，导致小尺寸物体的识别效果不佳，所以在此加入了passthrough层。

本质其实就是特征重排，26*26*512的feature map分别按行和列隔点采样，可以得到4幅13*13*512的特征，把这4张特征按channel串联起来，就是最后的13*13*2048的feature map.还有就是，passthrough layer本身是不学习参数的，直接用前面的层的特征重排后拼接到后面的层，越在网络前面的层，感受野越小，有利于小目标的检测

7.Multi-Scale Training（多尺寸训练）
很关键的一点是，Yolo v2中只有卷积层与池化层，所以对于网络的输入大小，并没有限制，整个网络的降采样倍数为32，只要输入的特征图尺寸为32的倍数即可，如果网络中有全连接层，就不是这样了。所以Yolo v2可以使用不同尺寸的输入图片训练

8.最大亮点！！图像分类（只有图像分类标签）与目标检测检测（既有图像分类标签又有目标检测标签）联合训练
因为coco数据集类别少，因此想办法引入imagenet的类别

联合训练方法思路简单清晰，Yolo v2中物体矩形框生成，不依赖于物理类别预测，二者同时独立进行（如何实现独立进行？）。当输入是目标检测数据集时，标注信息有类别、有位置，那么对整个loss函数计算loss，进行反向传播；当输入图片只包含分类信息时，loss函数只计算分类loss，其余部分loss为零。当然，一般的训练策略为，先在检测数据集上训练一定的epoch，待预测框的loss基本稳定后，再联合分类数据集、检测数据集进行交替训练，同时为了分类、检测数据量平衡，作者对coco数据集进行了上采样，使得coco数据总数和ImageNet大致相同

这里会遇到一个问题，类别之间并不一定是互斥关系，可能是包含（例如人与男人）、相交（运动员与男人），那么在网络中，该怎么对类别进行预测和训练呢？），在文中，作者使用WordTree，解决了ImageNet与coco之间的类别问题
#### YOLOv3

1.Rectangular training/inference 矩形训练/推理

https://blog.csdn.net/zicai_jiayou/article/details/109623578

参考代码1：

https://github.com/eriklindernoren/PyTorch-YOLOv3

#### SSD算法

在不同特征图上匹配不同的目标
 default box：相当于在每个特征图上的anchor box

#### 论文笔记 2021CVPR

将小样本学习应用在目标检测上。要点是解决视觉与语义上的鸿沟（用transformer进行关系推理）
由于小样本目标检测的性能对显性和隐性的样本数量非常敏感，当数据有限时，性能也会急剧下降，很大程度上受到新类数据稀缺的影响。新目标的学习只通过图像，即视觉信息，并且各类之间的学习是独立的，不存在知识传播。然而因为图像数据的稀缺，视觉信息变得有限。但是无论数据的可用性如何，新类和基本类之间的语义关系都是不变的，结合视觉信息一起学习有助于标注标注完成。当视觉信息难以获得时，显性的关系推理会更有用。

总体训练方法如下：
1.首先训练基本类数据集（不含新类）
2.训练基本类和新类的并集（由于新类样本小，防止训练结果侧重于训练类，因此在训练类中采样与新类一起训练）
![image](https://user-images.githubusercontent.com/49737867/117391073-d65af980-af21-11eb-85b4-18982a18e1b1.png)

#### 目标检测评价指标
1.置信度：置信度即模型认为检测框中存在目标的确信度。对于一个检测框，会首先使用置信度阈值进行过滤，当检测框的置信度大于该阈值时才认为检测框中存在目标（即为正样本，positive），否则认为不存在目标（即为负样本，negative）。

2.IoU：IoU阈值，所谓IoU，就是交并比，是用来衡量两个矩形框重合程度的一种度量指标，当模型给出的检测框和真实的目标框之间的IoU大于该阈值时，才认为该检测框是正确的，即为True Positive，否则为False Positive。只有当检测框的置信度和与真实标定框的IoU分别都大于置信度阈值和IoU阈值时，才认为该检测框为正确的。

3.recall：召回率，记做R，表示你预测的结果中有多少正样本被正确检测出来，当R=100%的时候，表示没有漏检。

4.precision也叫精确率，记做P，表示你预测的结果中有多少样本是正确的，当P=100%的时候，表示没有误检。
例子：对于Precision值，其代表的是你所预测出来准确结果占所有预测结果的准确性，对于Recall值，其代表的是你所预测出来准确结果占总体正样本的准确性。这样说有点难理解，举个例子吧。现在你手上有10个鸡蛋，里面有6个是好的，另外4个是坏的，你训练出一个模型，检测出8个鸡蛋是好的，但实际上只有5个是好的，另外3个是坏的。那么模型的Precision值为5/8=0.625，即表示你所预测出来的8个鸡蛋中只有5个是好的，其值只在你预测结果中计算得到，Recall值为5/6=0.833，即表示总共有6个正样本，你预测出来5个，表示的是你预测出来的正样本占总正样本的比例。(正样本理解为你要检测的目标)

5.Precision x Recall曲线（PR曲线）:
判断法1：观察某一个目标检测模型关于某一类别的PR曲线，如果随着recall的增高，其precision仍旧保持较高的值（无论如何设置confidence的阈值，precision和recall都能保持较高的值），那么我们就可以认为对于该类别来说，该模型具有比较好的性能。

判断法2：判断目标检测模型性能好坏的另外一种方式是：看该模型是否只会识别出真实的目标（False Positives的个数为0，即高precision），同时能够检测出所有的真实目标（False Negatives的个数为0，即高recall）。

而一个性能比较差的模型要想检测出所有的真实目标（高recall），就需要增加其检测出的目标的个数（提高False Positive），这会导致模型的precision降低。在实际测试中，我们会发现PR曲线在一开始就有较高的precision，但随着recall的增高，precision会逐渐降低。

6.Average Precision:
另外一种表征目标检测模型性能的方式是计算PR曲线下的面积（area under the curve, AUC）。因为PR曲线总是呈Z字型上升和下降，因而我们很难将多个模型的PR曲线绘制在一起进行比较（曲线会相互交叉）。这也是我们常使用AP这一具有具体的数值的度量方式的原因。
实际上，可以将AP看作precision以recall为权重的加权平均。

7.mAP的计算 ...

### Colab踩坑记录
1.一定要注意先选取运行环境！！！否则后面的工作白做

2.在colab中使用命令需要注意对空格的转义
奇怪的是，%ls命令输出的是MyDrive/
而%cd命令输出的是My Drive/
我的数据集的路径为MyDrive/也没有报错
!python /content/drive/My\ Drive/BertNer/BERT_NER.py

3.colab运行目录是/content/drive/My Drive
要特别注意当前工作目录，使用以下命令进入当前目录
%cd /content/gdrive/My\ Drive/yourfilename

4.dataset not found错误，数据集的路径没有写对
data.yaml里面的相对路径取决于当前目录（也就是cd），个人觉得换成绝对路径好一些

5.如果跑代码时间过长服务器会断开，因此不适合数据量较大的代码
### Linux命令
1.pwd 可以得知当前工作目录的绝对路径
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

2.cd是change directory 改变当前目录

3.查看当前进程 top -c

4.Linux下还提供了一个killall命令，可以直接使用进程的名字而不是进程标识号，例如：# killall -9 NAME
### C++
Vector 容器 向量类
vector类称作向量类，它实现了动态的数组，用于元素数量变化的对象数组。
构造函数：
vector（）：创建一个空的vector。
vector（int nSize）：创建一个vector，元素个数为nSize。
vector（int nSize， const T& t）：创建一个vector，元素个数为nSize，且值均为t。
vector（const vector&）：拷贝构造函数。
vector<int>a,b(n,0)的意思就是 创建了一个 int 类型的空的vector容器a，和一个 int 类型n个元素，且值均为0的vecotr容器b。
5. sudo install pip requrement（安装环境） 发生错误时，可以sudo chmod -R 777 username

### 最优化

1.启发式算法：启发式算法以仿自然体算法为主，主要有蚁群算法、模拟退火法、神经网络等
2.非基变量：非基变量是运筹学中的一个术语。它的定义是线性规划中除基变量以外的变量称为非基变量。
3.可行解：满足约束条件的解
4.在约束方程组矩阵挑出r个线性无关的列，求出一个x（除了线性无关的列对应的x分量，其他列对应的x都为0）
#### 线性规划
1.单纯形法：建表 计算检验数 找检验数最小的列（ak）进基，计算yi0/yik找到最小的（并且必须大于零）离基，直至检验数为0
2.对偶单纯形法：化标准形式（不要求b大于0）建表，求对偶可行的基本解，若xB中有负的分量把最小的离基，zj-cj/ykj（需ykj<0，yij表示单纯形表中第Bi行的各个元素，k其实是单纯形表中的第k行，也是上一步确定离基矢量时对应的一行） ，直到所有分量都大于0
#### 无约束非线性规划
7.牛顿法，s=负海森阵的逆乘梯度
8.约束优化方法
共轭梯度法在用sn-1时就能找到最优点
DFP算法
#### 约束非线性规划
*******点迭代之后有效集或者二次逼近法里的约束集也会变，约束集也要更新***********************
1.可行性方向法
求可行方向A1d≤0，可行的改进方向▽f（x）d<0
3.外点法可以求等书约束和不等式约束，内点法只能求不等式约束
4.梯度投影法
5.二次规划：
首先找到一个可行点x（1）,x(1)处有效集J1，求解仅含等式约束的二次规划问题（不等式约束也“化”成等式约束了），因为是等式d1，d2一般能直接求出，若d2没约束则可以任取一个值

得到d后，若d=0，分析其乘子，若乘子都大于等于0，已经为最优解，否则取乘子中最小的那个，在有效集中剔除该乘子对应的指标，再次迭代求正定二次规划问题

若d不等于0，x拔=xk+d，若xk+1可行（满足原约束条件），若不满足约束条件，则沿该迭代方向碰第一个边界求交点，计算步长αk=min {-（aiTx^(k)）-bi)/aiTd(k) |i∈I\jk,aiTd(k)>0}
令xk+1=xk+αkd，再次求其有效集
4.二次逼近法：

首先求一个LP规划得到约束集（求ξ），写出原问题对应的正定二次规划，解正定二次规划得到最优解d（迭代方向），若d=0，结束，此时的xk就是最优点。再进行一次罚函数线搜索得到α，得到新点xk+1=xk+αd，求|xk+1-xk|《 ε，迭代结束，否则，用BFGS法更新Bk+1再次求LP问题得到约束集继续解正定二次规划。

### DeepLearning
#### 卷积神经网络

1.卷积神经网络的提出：假设有一张100*100的彩色图片，映射为向量就是3万维，如果用全连接层的话，就需要大量的参数（例如第一层就一千个神经元），因此提出卷积神经网络，它可以滤掉多余的参数。
说白了就是fully connected layer 把一些weight拿掉而已

![卷积原理](https://user-images.githubusercontent.com/49737867/113261213-c425e400-9301-11eb-874b-7b9524bf5449.png)

按理来说，上图中的neuron应该连接36个input，但是卷积操作相当于input只有9个

2.每个neuron（这里是fillter）各司其职，比如一张鸟的图片，其中一个neuron会观察图片中有没有鸟嘴，这就是property2（见图-CNN处理图像的整个过程）。

如图，该fillter职责就是检测有没有主对角线的（1，1，1），最后卷积结果得3（原图像的像素只有0和1，因此卷积运算得3，对应区域主对角线上一定是（1，1，1））

![22](https://user-images.githubusercontent.com/49737867/113258918-1fa2a280-92ff-11eb-958a-341bd5a2ad24.png)

3.sub![Uploading 22.png…]()
sampling子抽样，抽出图片中一个区域，比如把这张图片的奇数行偶数列拿掉，不会影响人对这张image的理解，这也是卷积神经网络池化的思想。

![111](https://user-images.githubusercontent.com/49737867/113255916-00a21180-92fb-11eb-97dd-001229e3ae6c.png)

4.CNN处理图像的整个过程

![cnn](https://user-images.githubusercontent.com/49737867/113255348-4f9b7700-92fa-11eb-8f7c-648c6c732861.png)

5.CNN中每一个filter的每个元素等于Fully Connected layer的每一个neuron，每一个filter做一次卷积操作得到的一个元素（这里不指特征图，该元素指特征图中的一个picxel）对应一个全连接层的neuron（结果），不同之处在于filter共享weight，而全连接层不能。

每一个fillter也是matrix，里面的每个参数如同fully connected layer中的weght和bias一样，是学出来的，卷积运算为内积（对应元素相乘相加）


stride为挪动距离

6.feature map：一个fillter对一个图像进行一次卷积运算后得到的图。如果有100个fillter就会得到100个feature map.

7.colorfully image

不是matrix，是三维张量（立方体）

8.思考：CNN训练方法，现在都是用tookit实现的，理论与全连接层一样，只不过一些weight恒为0，并且用一些方法使weight共享
![dsa](https://user-images.githubusercontent.com/49737867/113260211-92604d80-9300-11eb-8dfa-5f18d2d55004.png)

Maxpooling利用了Max是如何微分的？利用maxout network.

9.CNN处理后的结果是几维取决于fillter的个数

10.inference 推理，就是把训练好的模型拿出来遛一遛

11.1x1 卷积可以压缩信道数。池化可以压缩宽和高

12.卷积在论文中的符号表示  比如一个卷积层conv3-64 表示卷积核的大小为3×3，卷积核的个数为64

13.卷积神经网络一般的结构不是：(卷积—池化) x N —FC—输出  吗？
那么最后一层卷积—池化后输出的结果是一个多通道的3维数据。
但是全连接层输入的不是1维数据吗？那么这中间要怎么处理呢？
是直接ravel吗？还是要作怎样的变换？
用tf.layers.flatten(),可以把3维的向量拍成一维的
14.一维卷积层的作用：
如果卷积的输出输入都是一个平面，那么１X1卷积核并没有什么意义，它是完全不考虑像素与周边其他像素关系。但卷积的输出输入是长方体，所以１X1卷积实际上对每个像素点，在不同的channels上进行线性组合（信息整合），且保留原有平面结构，调控depth,从而完成升维或降维的功能。

15.2*2步距为2图片缩小一半 3*3步距为图片大小不变


调整感受野（多尺度信息）的同时控制分辨率的神器
#### Dropout
当一个复杂的前馈神经网络被训练在小的数据集时，容易造成过拟合
#### 空洞卷积

一句话概括空洞卷积：调整感受野（多尺度信息）的同时控制分辨率的神器

空洞卷积的两大优势：

1.特征图相同情况下，空洞卷积可以得到更大的感受野，从而获得更加密集的数据

2.特征图相同情况下，更大的感受野可以提高在目标检测和语义分割的任务中的小物体识别分割的的效果。

我们可以显而易见的看到，使用空洞卷积代替下采样/上采样可以很好的保留图像的空间特征，也不会损失图像信息。当网络层需要更大的感受野，但是由于计算资源有限无法提高卷积核数量或大小时，可以考虑使用空洞卷积。

在空洞卷积提出以前，大部分的空间尺寸恢复工作都是由上采样或反卷积实现的。前者通常是通过线性或双线性变换进行插值，虽然计算量小，但是效果有时不能满足要求；后者则是通过卷积实现，虽然精度高，但是参数计算量增加了。
17.反卷积
#### Dense Net
DenseNet与ResNet的区别
![image](https://user-images.githubusercontent.com/49737867/116500454-3548bf00-a8e1-11eb-8b29-98291922309c.png)

18.感受野

神经元感受野的值越大表示其能接触到的原始图像范围就越大，也意味着它可能蕴含更为全局，语义层次更高的特征；相反，值越小则表示其所包含的特征越趋向局部和细节。因此感受野的值可以用来大致判断每一层的抽象层次

#### 训练集，测试集，验证集

将数据划分训练集、验证集和测试集。在训练集上训练模型，在验证集上评估模型，一旦找到的最佳的参数，就在测试集上最后测试一次，测试集上的误差作为泛化误差的近似。

### matlab
MATLAB的inline通俗的来说就是用于定义函数，如图所示我们使用inline定义一个函数

 f=inline('a*x+b','a','b','x')
### 概率统计 
1.置信度
置信度就是说，你测得的均值，和总体真实情况的差距小于这个给定的值的概率，应该是1-α，如式换句话说，我们有1-α的信心认为，你测得的这个均值和总体的实际期望很接近了。（说你测得的均值就是总体期望是很草率的，但是说，我有95%的把握认为我测得的均值，非常接近总体的期望了，听起来就靠谱的多

### os命令
1.os.path.exists()

2.os.path.join(Path1,Path2,Path3) 实现路径拼接

3.os.sep
在Windows上，文件的路径分隔符是'\'，在Linux上是'/'。
os.sep实现了命令跨平台的兼容

4.os.path.splitext(x)[-1].lower()

os.path.splitext(“文件路径”) 分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
### tensorboard使用方法

在项目中找到runs文件夹，打开当前代码对应的日期文件夹，可以看到前缀为event.out的文件，在该文件夹下shift+鼠标右键 ，在此目录下打开powershell，此时powershell的路径已经在当前文件夹。
打开powershell输入tensorboard --logdir logs，运行结果会出现localhost网站，复制该网站在浏览器打开，就可以看到tensorboard

### pycharm 踩过的坑
远程调试教程：
https://www.cnblogs.com/xuegqcto/p/8621689.html
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
1.踩过的坑：
远程调试时找不到文件
解决方法：路径错误，run configration里面脚本路径和工作目录都设置为服务器上的路径
！！！远程调试注意设置好三个路径
1.setting里interpreter的本地目录和远程目录
2.run和debug的配置路径（都写远程的）
3.development的配置文件，run/debug的配置文件的路径用“\” development的配置文件路径用“/”（远程开发的情况）

异常问题：运行时停在某个地方不报错（只是一个函数的for循环），但此时debug却能成功运行（不要设置任何断电）。

无线重置 
https://mp.weixin.qq.com/s/4Mxi3A8ZGmAVqRQE3-Pkqw
### git踩过的坑
1.git 先pull然后add再commit最后push
语句分别是：
git pull

git add .

git commit -m '你要填的信息'

git push

push报错 Failed to connect to github.com port 443: Timed out 
解决方法：执行git config --global --unset http.proxy

### pycharm添加库踩过的坑
1.【python】pip指定路径安装文件
（1）在网上下载个tar.gz的安装包，用pip在指定目录安装

pip install --target=路径  文件名

pip install --target=E:\work\zicai\pd_code\AutoTest3.7\shujia\venv\Lib\site-packages xlwings-0.18.0.tar.gz

（2）指定下载pyecharts包1.7.0版本到执行路径

pip3 install -i https://pypi.doubanio.com/simple pyecharts==1.7.0 --target=E:\work\zicai\pd_code\AutoTest3.7\shujia\venv\Lib\site-packages
2.报错numpy.ufunc size changed, may indicate binary incompatibility
numpy版本过低导致，先pip uninstall numpy 然后pip install numpy就可解决问题

### C++踩过的坑

1.定义方法（成员函数）时，return应写在for循环之外，return只能写在函数体里

2.覆盖：在基类中定义了一个非虚拟函数，然后在派生类中又定义了一个同名同参数同返回类型的函数，这就是覆盖了。

3.重载：有两个或多个函数名相同的函数，但是函数的形参列表不同。在调用相同函数名的函数时，根据形参列表确定到底该调用哪一个函数。

4.多态：在基类中定义了一个虚拟函数，然后在派生类中又定义一个同名，同参数表的函数，这就是多态。多态是这3种情况中唯一采用动态绑定技术的一种情况。也就是说，通过一个基类指针来操作对象，如果对象是基类对象，就会调用基类中的那个函数，如果对象实际是派生类对象，就会调用派生类中的那个函数，调用哪个函数并不由函数的参数表决定，而是由函数的实际类型决定。

5.异类收集：
![image](https://user-images.githubusercontent.com/49737867/115204398-071def00-a12b-11eb-960f-0f7a5c12114e.png)
6.vector二维数组的遍历方式
``` mark
#include <iostream>
using namespace std;
int main()
{	
	vector<vector<int>> vec = {
							    {1, 2, 3},
							    {4, 5, 6}
                            };
	for(int i = 0; i < vec.size(); i++)
	{
		for(int j = 0; j < vec[i].size(); j++)
		{
			cout << vec[i][j] << " ";
        }
		cout << endl;
	}
	return 0;
}
```
C++中string为可变，java和python中string为不可变，如果要改变字符串的某一位，则需要新建一个字符串
7.C++使用变量前一定要初始化，不然会报错
terminate called after throwing an instance of 'std::length_error'
例子如下
  s.resize(len+ 2 * count);
  这里的count我没有初始化
  报如下错误
  terminate called after throwing an instance of 'std::length_error'
  what():  basic_string::_M_replace_aux
8.遍历字符串  for(char c : s) 也可以用常见的访问下标的方法
### 数据集
cocoAPI
所以，现在的开源论文项目，都是将COCO API再加工，封装为一个适合模型训练和测试的dataset class。
详情见https://blog.csdn.net/qq_34914551/article/details/103793104

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
### 水刊
ieee Access

idea：
1.用一些模块替换以减少参数数量，避免不必要的计算
https://www.cnblogs.com/andre-ma/p/13424741.html


### jupyter notebook
1.将conda的环境运用到jupyter notebook中
安装
python conda install ipykernel
将conda环境写入kernel中
python -m ipykernel install --user --name 环境名称 --display-name “Python (环境名称)”
在anaconda prompt中激活虚拟环境
cd到项目路径
打开在 ananconda prompt中打开jupyter notebook
2.连接不上kernel
https://blog.csdn.net/rs_gis/article/details/104347481

### java 

#### c++与java区别
1.c++的主函数是单独的一个函数，java的主函数必须放到一个类里面，这个类的名字与其文件名是一致的

2.c++最小的编译单位是函数，java最小的编译单位是类，因此main函数（也称方法）必须定义在某一类里面，类应该成为所保存的源文件的名字

3.java提供8种基本数据类型和1种引用类型（类，数组，接口），java当中所有的类型，他们在内存中所占据的空间，我们称为字节宽度。

4.java的类型在字节宽度和取值范围上是被虚拟机定义好的，c++里内建类型有一个定义的最小的取值范围，但其他部分（字节宽度）可以被映射成具体平台上支持的原生类型。



##### 基本数据类型的相互转换

C++允许一些基本类型之间的隐式转换，也允许程序员对于用户自定义类型相关的隐式转换规则

java中只有基本类型之间，变宽的类型的转换可以是隐式的，其余的转换需要显式的类型转换语法
```
	int a=4;
	if(a=5){...}
	//这里隐式转换，首先将5赋值给a，此时a为5，大于零，转化为了 bool型值为1
```

##### 数据默认值
C++里，数据没有被初始化则得到‘无意义的值’，java里数据没有被显式初始化则得到‘默认值’
|Data type|Default value|
|----|----|
|  byte   |      0      |
|  short  |      0      |
|    int  |      0      |
|    long  |      0      |
|    char  |      '\u0000'|
|    float |  0.0F   |
|  double |  0.0D   |
|  boolean | false|
|reference type| null |









3.


### 傅里叶变换

![image](https://user-images.githubusercontent.com/49737867/116520908-e9f4d780-a905-11eb-95e4-41ab82ea646c.png)

