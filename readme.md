### 实时监测心血管疾病的智能算法

### 1 前言

心血管疾病是全球范围内危及人类生命健康的“头号杀手”，占每年全球死亡病人的 29%；中国每 5 例病患死亡者中有 2例死于心血管病，目前心血管病患者大约 2.9 亿，其中冠心病 1100 万，心力衰竭 450 万，风湿性心脏病 250 万，先天性心脏病 200 万，高血压 2.7 亿。巨大的心血管疾病患者基数产生了巨大的市场空间。

目前心血管疾病的预防诊治普遍采用Holter心电记录仪监测动态心电数据，仪器需要与皮肤接触的电极传感器24小时长期佩戴来采集心电信号，使用不便且成本较高，同时对心脏泵血功能、瓣膜疾病、心肌炎等器质性病变的病症甚至无法监测，而使用心音数据，则可以很好地避免上述问题，关键是对心音数据（PCG）、算法进行准确处理，是其中的关键因素！基于良好的数据、算法的心音智能终端，可以达到实时监测反馈用户心脏健康问题，通过少量数据准确评估心肌泵血功能，检测发现心脏瓣膜、心律失常等疾病风险。

心音数据的病理自动分类已经有50多年的历史了，但是分类正确率仍然是非常严峻的挑战。传统的心音数据分类方法可以归类为以下几种：人工神经网络模型分类、支持向量机模型分类、隐马尔可夫模型分类以及聚类模型分类[1]。Physionet在2016年的心音数据分类比赛，旨在鼓励人们开发更准确的心音数据分类算法。很多优秀的选手开发了分类正确率非常高的算法，比赛的前几名都使用到了Springer分割算法，首先对心音数据进行分割，然后进行特征提取或者使用神经网络等模型提取特征。

虽然Springer分割算法能自动识别心音数据周期中四个状态，并可以达到非常高的准确率。为了保留信号的原始信息，我们开发了一种不分割就可以自动识别分类心音数据的算法。

#### 2 比赛数据

本次比赛训练集包含3240个样本，样本一共有两个类别：正常（-1）和不正常（1）。其中类别为正常（-1）的样本占80%，不正常（1）样本占20%，样本采样频率为2000Hz,时间长度从5s到120s不等。

#### 3 算法设计

##### 3.1小波变换与重构

本次比赛的数据存在许多噪声，包括环境噪声、受试者的腹部发出的声音以及其他噪声等[2]。滤除噪声，保证信号质量，这是保证算法能准确自动识别分类的第一步。

小波变换具有较低的复杂度与良好的时域-频域分析特性，在生医信号分析中具有广泛的应用。SWT可以去除原始信号中冗余的特征，并且对于抑制噪声有良好的效果。我们采用一阶SWT重构原始信号，并将重构信号用于后面的分析中。图1显示了a0009.wav部分原始信号与SWT重构的相应的部分信号，可以看出，SWT重构后的信号明显的抑制了信号中的白噪声。

![1552887278933](F:\pythoncode\东南大学list团队\source_code\1552887278933.png)



##### 3.2 移除峰值

虽然SWT重构信号可以抑制原始信号中带有的噪声，但是在信号采集时可能会由于摩擦或受试者的移动而引起信号剧烈的震荡，造成非常高的峰值，为了去掉这些异常峰值，我们采用了Schmidt[3]提出的移除峰值算法。该算法的执行过程如下：

（1） 记录被分成500ms大小的窗口

（2）找到每个窗口的最大绝对振幅(MAA)

（3）如果至少有一个MAA超过了MAA的中值的3倍，则执行下面步骤，否则跳转到步骤4

   （a）选择窗口中最高的MAA

   （b）在被选择的窗口中，识别MAA点的位置并将其作为噪音峰值的顶端。

   （c）噪音峰值的开始定义为MAA点之前的最后零交叉点。

   （d）噪音峰值的结束定义为最大值点之后的第一个零交叉点

   （e）用0替换噪音的峰值

   （f）回到步骤2

（4）程序结束

图2显示了a0010.wav的原始信号与执行Schmidt移除峰值算法之后的信号。

![1552887745243](F:\pythoncode\东南大学list团队\source_code\1552887745243.png)

##### 3.3梅尔倒频谱系数

梅尔频率倒谱（MFC）是基于声音的非线性梅尔刻度的对数能量频谱的线性变换[4]。将梅尔倒频谱系数（MFCC）作为特征广泛应用于自动语音识别中，我们用MFCC作为输入模型的时频特征。梅尔倒频谱系数的计算过程如下：

（1）将一段语音信号分解为多个讯框。

（2）将语音信号预强化，通过一个高通滤波器。

（3）进行傅立叶变换，将信号变换至频域。

（4）将每个讯框获得的频谱通过梅尔滤波器(三角重叠窗口)，得到梅尔刻度。

（5）在每个梅尔刻度上提取对数能量。

（6）对上面获得的结果进行离散傅里叶反变换，变换到倒频谱域。

MFCC就是这个倒频谱图的幅度(amplitudes)。一般使用12个系数，与讯框能量叠加得13维的系数。我们的实验中采用了40个系数。

##### 3.4长短时记忆网络

循环神经网络（RNN）已经在多个领域取得了非凡的成就，而长短时记忆网络（LSTM）是RNN中的翘楚。LSTM可以记忆当前时间之前的信息，非常适合处理时间序列。我们采用LSTM对心音数据进行分类。我们采用Dropout和L2正则化以缓解模型的过拟合，具体见源代码。

##### 3.5Focal Loss

针对训练样本中类别不均衡的问题，我们在训练模型的时候加入了focal loss以缓解该问题。

##### 3.6 K-fold 交叉验证

为何防止模型的过拟合，提高分类的准确性，我们采用了5折交叉验证。将训练集划分成5个不相交的子集，每次模型训练从分好的子集中拿出1个作为测试集，其他的作为训练集，最后将五个模型得出的结果综合。

#### 4.项目实现

本项目采用python和tensorflow框架实现。离散小波变换通过调用Pywt的程序包实现，移除峰值算法基于Python实现，MFCC通过librosa包中的mfcc函数实现。网络框架如下，具体参见源代码。

 ![1552889010040](F:\pythoncode\东南大学list团队\source_code\github\1552889010040.png)

