# 基于DDPG实现自动泊车

------------

本模块基于`DDPG`算法实现智能车在不同场景下的自动泊车。DDPG算法使用`stable-baseline3`实现，提供的场景包括`highwai-env`自带的`parking-v0`和自定义的`modified_parking_env-v0`，使用`gym`进行场景模拟。

## 快速使用

建议使用`Anaconda`搭建虚拟环境进行使用。本节说明使用`Anaconda`搭建虚拟环境，并以该环境调用模块。

#### 环境搭建

以下`Bash`命令以命令行窗口所处目录下有本模块根目录`GymParking`文件夹为前提。

```bash
conda create -n GymParkingTest python=3.7
conda activate GymParkingTest
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

#### 模块测试

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



