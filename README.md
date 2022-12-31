# 基于DDPG的自动泊车

本模块基于`DDPG`算法实现智能车在不同场景下的自动泊车。DDPG算法使用`stable-baseline3`实现，提供的场景包括`highwai-env`自带的`parking-v0`和自定义的`modified_parking_env-v0`，使用`gym`进行场景模拟。

## 1.快速使用

建议使用`Anaconda`搭建虚拟环境进行使用。本节说明使用`Anaconda`搭建虚拟环境，并以该环境调用模块。

### 1.1环境搭建

以下Bash命令以使用Anaconda Prompt，且其所处目录下有本模块根目录`GymParking`文件夹为前提。

```bash
conda create -n GymParkingTest python=3.7
conda activate GymParkingTest
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 1.2模块测试

可以直接调用模块的`__init__.py`文件。

```bash
python GymParking\__init__.py
```

也可以在其他文件中调用该模组。下文中`example.py`是一个示例，假设其所处目录下有本模块根目录`GymParking`文件夹。

``` python
# example.py
import GymParking as Gp
Gp.Train() #训练模型
Gp.Eval() #查看模型效果
```

下为两函数的参数含义解释。

```python
def Train(envName:str = 'parking-v0',trainTime:int = int(1e6) ,saveDir:str = r"DDPG\model", logDir:str = r"DDPG\log") -> None:
    '''
    Training model with DDPG.
    Parameter(s):
        - envName:str -> Training environment. 
                         Default: parking-v0 (from highway-env). 
                         modified_parking_env (from authors) is recommended.
        - saveDir:str -> Directory of the model.
                         Default: "DDPG/model".
        - logDir :str -> Directory of the log.
                         Default: "DDPG/log".
                         To activate log: tensorboard.exe --logdir={Absolute path of 										   log directory(NOT A FILE)}
        - trainTime:int -> Total time step of training.
                           Default: int(1e6).
	Return:
        None
	'''
    pass
    
def Eval(envName:str = 'parking-v0', modelDir:str = r"DDPG\model") -> None:
    '''
    Showing the result of training.
    Parameter(s):
        - envName :str -> Training environment. 
                          Default: parking-v0 (from highway-env). 
                          modified_parking_env (from authors) is recommended.
                          Should be same as training.
        - modelDir:str -> Directory of the model.
                          Default: "DDPG/model".
                          A pre-trained model "Modified_Parking_DDPG\model" is 								  recommended.
    Return:
        None
    '''
    pass
```

## 2.项目介绍
自动泊车是现代与近未来交通领域的热门应用，对自动泊车技术的研究在当下具有重要意义。项目实现了智能车在随机初始位置处自主寻找车位，在沿车道线行进、避让障碍物的同时将车倒入停车位的全自动泊车技术。智能车的位姿由此前我组开发的[基于PP-YOLOv2的智能车位姿识别模型](https://aistudio.baidu.com/aistudio/projectdetail/5110116)提供，车道线信息由此前我组开发的[基于PaddleSeg与BiSeNet的自动驾驶道路分割模型](https://aistudio.baidu.com/aistudio/projectdetail/5297082)提供。

本次项目使用强化学习算法**DDPG**(解决了DQN网络难以处理的连续Action情形问题)进行自动泊车技术的开发，在框架上采用了**stable-baseline3**，现实场景重现与三维重建技术使用**Gym与Pybullet**实现。智能车位姿确定与车道线分割技术此前已进行过开发，分别使用PP-YOLOv2与BiSeNet算法。


## 2.1问题分析

### 2.1.1问题重述

&emsp;&emsp;根据发布的作业要求，我们可以将本次项目问题重述如下：

* 标准输入：地图车道线(由车道线划分模型得到)、智能车相关参数(由智能车位姿模型)、泊车位置(人工或程序判断可泊车位置)

* 标准输出：智能车泊车连续动作(Action)、可视化的自动泊车模拟过程、可部署的强化学习模型

&emsp;&emsp;此外，该问题需要使用百度飞桨框架与强化学习算法求解。

### 2.1.2连续状态or离散状态？从Q-learning到DQN


&emsp;&emsp;在使用强化学习算法进行自动泊车的模型训练前，我们首先要思考一个这样的问题：

* **自动泊车的状态(State)是离散的还是连续的？**

&emsp;&emsp;这一问题的解答对于项目开发初期的算法选择十分重要。如果自动泊车的状态是离散的，我们可以直接使用传统且可参考资料丰富的Q-learning算法进行问题求解；反之，若状态是连续的，这就意味着通过查找Q-table返回最大Q值动作的Q-learning无法工作——因为“连续的状态”意味着无穷的State-Action键值对，我们必须改为采用状态连续算法。

&emsp;&emsp;什么样的状态能够被称作是离散的状态呢？让我们来看一个经典的强化学习问题——Grid World：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://ai-studio-static-online.cdn.bcebos.com/5a7d258f094e490fa6b344e372e12a05cd21e41b2a094e6096bda3d15d59a897" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
    图1 Grid World智能体训练<br>(图为本人大二学年的机器学习实践作业)
</center>

&emsp;&emsp;如图1所示，$Agent$就是红色的方块，每次它可以选取上、下、左、右四个方向，并向选择的方向移动一格（如果触碰到边缘则不动），蓝色的方块是不能触碰的陷阱（若触碰则得到负数的reward并重新开始），金色方块则是$Agent$需要前往的目的地。不难看出，GridWorld问题的$State$由且仅由$Agent$目前所处的位置决定，是一个离散的$State$，在一个$n\times n$的方阵中，State至多是$O(n^2)$。外加上$n$的数据范围不大，此时使用Q-learning算法就能高效快速的进行求解。

&emsp;&emsp;自动泊车与GridWorld不同，自动泊车的状态是连续的。自动泊车的状态由智能车的位姿决定，无论是朝向的角度还是中心位置的坐标，都是连续的状态量。当然，我们也可以取一个极小变化量为单位（如0.1cm、0.01°），强行将该问题的状态变为离散的使用Q-learning进行处理，但这样会造成极大的算力与资源浪费（Q-table会非常非常大，每次查找的耗时也极长），是不建议的。
    
&emsp;&emsp;状态连续的例子还有很多，如Gym经典游戏CartPole，杆的倾角是连续而不是离散的、在Atari中，状态也是连续的。那么我们如何处理连续状态的强化学习问题呢？
    
&emsp;&emsp;一个经典的处理连续状态的强化学习算法就是DQN算法，DQN算法使用连续的函数建立起State$\rightarrow$Action的映射关系：

    
   $$F(State)=Action$$
    
    
&emsp;&emsp;这样我们就能够处理连续状态的强化学习问题。而我们很容易想到一个满足以上要求的可更新的、连续的映射——神经网络。事实上也是如此，Q-learning+Deep Network就构成了经典的DQN算法。
    
### 2.1.3连续动作or离散动作？从DQN到DDPG
    
&emsp;&emsp;在考虑完自动泊车$State$的连续/离散情况后，我们还要对$Action$的情况进行考虑。如果自动泊车的$Action$是离散的，我们就可以直接使用DQN算法进行问题求解；反之，若$Action$是连续的，这就意味着只能处理离散型$Action$的DQN算法无法解决这样的问题：DQN更新函数为

   $$Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _a Q\left(S^{\prime}, a^{\prime}\right)-Q(S, A)\right]$$
    
&emsp;&emsp;而$\max _a Q\left(S^{\prime}, a^{\prime}\right)$是离散形式的，如果需要处理连续情形的$Action$，沿用此前的思路，我们需要使用连续的函数替代这一部分。
    
&emsp;&emsp;回到Grid World的例子，我们容易看出，Grid World的$Action$是离散的，因为它一共只有向上、向下、向左、向右四种$Action$可选，而自动泊车则不然。以我组创建的Environment为例，智能车看似只有位置横坐标、位置纵坐标、x方向速度、y方向速度、x方向倾角、y方向倾角六个$Action$，但实际上以速度为例，速度是一个连续的变量，而非像“向上”这种动作只有做与不做两种值，是不能够简单地用DQN进行处理的。
    
&emsp;&emsp;DDPG使用连续的$Actor$网络代为找出最大Q-value的动作，解决了连续情形的$Action$选取问题。
	

### 2.2DDPG(Deep Deterministic Policy Gradient)

&emsp;&emsp;深度确定性策略梯度算法能够求解$State$、$Action$在连续情形下的强化学习问题。在讲述DDPG的工作方式前，我们需要引入DDPG的架构，并在DQN的基础上添加特性，以更好地阐明DDPG的原理。

### 2.2.1Actor-Critic架构

&emsp;&emsp;Actor-Critic网络并非是DDPG算法独有的算法架构，而起源于强化学习领域的价值学习体系与策略学习体系的结合，两种体系结合的产物即是Policy Network(Actor)与Value Network(Critic)。形象地说，在DDPG算法中，Actor网络取代了DQN中的Deep Network的位置，好比一个“演员”，根据目前自身的参数估计出目前应该采取的最佳$Action$，而Critic网络则是一个“批评家”，对Actor网络的$Action$进行“打分”。在这个过程中，Actor网络为了得到更高的分数，将会不断更新自身参数，而Critic则会根据环境返回的Reward不断使自己的“打分”变得更加精确。



<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://ai-studio-static-online.cdn.bcebos.com/b090d17a29514d06b9b4f8b38a16e115ba13c2841a3b4382aa9addcf7e0b59c1" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
    图2 Actor-Critic架构
</center>

我们也可以更加形式化地描述Actor-Critic网路架构：
    
首先我们有状态价值方程：
    
$$V_{\pi}(s)=\sum_a \pi(a|s)\times Q_{\pi}(s,a)$$
    
* Actor(Policy Network)
    

使用网络$\pi(a|s;\theta)$来近似$\pi(a|s)$。
    
$\theta$是可通过训练不断更新的网络参数。

梯度上升更新$\pi(a|s;\theta)$来增加状态价值$V(s;\theta,w)$。
    

* Critic(Value Network)
    
使用网络$Q_{\pi}(s,a;w)$来近似$Q_{\pi}(s,a)$。

w是可通过训练不断更新的网络参数。
    
更新价值网络Q_{\pi}(s,a;w)来更好地估计$Action$的分数。
    
损失函数使用TD error
    
    
### 2.2.2DDPG


&emsp;&emsp;DDPG算法是google DeepMind团队提出的一种用于输出确定性动作的算法。相较上一部分的Actor-Critic架构，它解决Actor-Critic神经网络每次参数更新前后都存在相关性，导致神经网络只能片面的看待问题这一缺点。相较于DQN，它能够解决动作连续情形下的强化学习问题。

&emsp;&emsp;相较DQN算法，DDPG算法加入了以下三大关键技术：

* 经验回放：智能体将得到的经验数据元组$(s,a,r,s^{'},done)$放入Replay Buffer中，更新网络参数时按照批量采样。

* 目标网络：在Actor网络和Critic网络外再使用一套用于估计目标的Target Actor网络和Target Critic网络。在更新目标网络时，为了避免参数更新过快，采用软更新方式。

* 噪声探索：确定性策略输出的动作为确定性动作，缺乏对环境的探索。在训练阶段，给Actor网络输出的动作加入噪声，从而让智能体具备一定的探索能力。

&emsp;&emsp;DDPG的更新流程如下：

1. 重置状态s

1. 选择动作a

1. 与环境互动，获得 s_, r, done 数据

1. 保存数据

1. 如果数据量足够，就对数据进行随机抽样，更新Actor和Critic

1. 把s_赋值给s，开始新的一步


