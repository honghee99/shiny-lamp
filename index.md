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

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

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

2.每个neuron各司其职，比如一张鸟的图片，其中一个neuron会观察图片中有没有鸟嘴

3.subsampling子抽样，抽出图片中一个区域，比如把这张图片的奇数行偶数列拿掉，不会影响人对这张image的理解，这也是卷积神经网络池化的思想。

![111](https://user-images.githubusercontent.com/49737867/113255916-00a21180-92fb-11eb-97dd-001229e3ae6c.png)


4.CNN处理图像的整个过程

![cnn](https://user-images.githubusercontent.com/49737867/113255348-4f9b7700-92fa-11eb-8f7c-648c6c732861.png)



### 概率统计 
1.置信度
置信度就是说，你测得的均值，和总体真实情况的差距小于这个给定的值的概率，应该是1-α，如式换句话说，我们有1-α的信心认为，你测得的这个均值和总体的实际期望很接近了。（说你测得的均值就是总体期望是很草率的，但是说，我有95%的把握认为我测得的均值，非常接近总体的期望了，听起来就靠谱的多
