## 修改部分

+ 修改了config中GPU和CPU, 实现自动切换
+ 执行`make_data.sh`后, 使用`train.sh`脚本训练训练三个模型

未解决bug: 将NLU和推荐模块一起放入GPU训练会出现反向传播的错误, 分别放入CPU和GPU, 或者全部放入CPU不会报错

## Dependencies

- python3.6
- torch1.3.0
- numpy
- tqdm
- sklearn
- pickle

<del>使用GPU与否，可在`FMConfig.py`, `AgentRuleConfig.py`文件内配置</del>

## 运行方式

python example.py 0 0 --num 3

第一个参数user type，0表示模拟用户，1表示命令行输入

第二个参数agent type，0表示基于规则的系统，1表示基于强化学习的系统

第三个参数（可选），对话次数，默认为1

note：选择命令行输入时会提供目标餐厅信息，根据提供的信息回答系统提出的问题


## 参考论文
Conversational Recommender System

**<https://arxiv.org/abs/1806.03277>**

## 测试结果

average reward | average turn |  success rate  
-|-|-
31.0821 | 3.7336 | 0.9676 

## 生成训练数据（可选）
当前仓库包含已训练好的模型参数

如需重新训练，可使用脚本make_data.sh生成数据

在每个模块的子文件夹中，包含各个模块训练使用的代码

