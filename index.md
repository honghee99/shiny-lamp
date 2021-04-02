## HL-Note
### Detection
#### Detection History
![目标检测发展史](https://user-images.githubusercontent.com/49737867/113255745-c59fde00-92fa-11eb-977c-7d3864960c85.png)

```markdown
Syntax highlighted code block

# 深度学习 1
## 残差网络 2
### 目标检测 3
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
#### basic
1.backbone：主干网络，用来做特征提取的网络，代表网络的一部分，一般是用于前端提取图片信息，生成特征图feature map,供后面的网络使用。通常用VGGNet还有你说的Resnet，因为这些backbone特征提取能力是很强，并且可以加载官方在大型数据集(Pascal 、Imagenet)上训练好的模型参数，然后接自己的网络，进行微调finetune即可

2.DPM模型：Deformable part model 为可形变部件模型，简称DPM模型。这种模型非常地直观，它将目标对象建模成几个部件的组合。
特征金字塔

3.ground truth：真实标签，人为为每个目标标记的标签

#### Faster RCNN
1.提出anchor box的原因：
一个窗口只能检测一个目标
无法解决多尺度问题。
扩展：在解决多尺度问题时主要采用一种思想--金字塔，或者是例如DPM模型中经典的特征金字塔。在不同分辨率的特征图下检测不同尺寸的目标。但是这样存在一个问题，就是大大的增加了计算量
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
  
### 最优化
1.启发式算法：启发式算法以仿自然体算法为主，主要有蚁群算法、模拟退火法、神经网络等
### DeepLearning
1.卷积神经网络的提出：假设有一张100*100的彩色图片，映射为向量就是3万维，如果用全连接层的话，就需要大量的参数（例如第一层就一千个神经元），因此提出卷积神经网络，它可以滤掉多余的参数。
说白了就是fully connected layer 把一些weight拿掉而已
2.Dropout：当一个复杂的前馈神经网络被训练在小的数据集时，容易造成过拟合
证明如下：

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

### matlab
MATLAB的inline通俗的来说就是用于定义函数，如图所示我们使用inline定义一个函数

 f=inline('a*x+b','a','b','x')
### 概率统计 
1.置信度
置信度就是说，你测得的均值，和总体真实情况的差距小于这个给定的值的概率，应该是1-α，如式换句话说，我们有1-α的信心认为，你测得的这个均值和总体的实际期望很接近了。（说你测得的均值就是总体期望是很草率的，但是说，我有95%的把握认为我测得的均值，非常接近总体的期望了，听起来就靠谱的多
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
