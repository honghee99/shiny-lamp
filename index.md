## Deep Learning
### Detection

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

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
![image](https://user-images.githubusercontent.com/49737867/113100535-1eee0b80-922e-11eb-8d11-b165785f2bae.png)

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

4.  Linux下还提供了一个killall命令，可以直接使用进程的名字而不是进程标识号，例如：# killall -9 NAME
### C++
Vector 容器 向量类
vector类称作向量类，它实现了动态的数组，用于元素数量变化的对象数组。
构造函数：
vector（）：创建一个空的vector。
vector（itn nSize）：创建一个vector，元素个数为nSize。
vector（int nSize， const T& t）：创建一个vector，元素个数为nSize，且值均为t。
vector（const vector&）：拷贝构造函数。
vector<int>a,b(n,0)的意思就是 创建了一个 int 类型的空的vector容器a，和一个 int 类型n个元素，且值均为0的vecotr容器b。
