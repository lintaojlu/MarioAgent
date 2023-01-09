# 超级马里奥兄弟--林韬&王福临
- 本项目基于OpenDILab实验室的[mario_dqn项目](https://github.com/opendilab/DI-adventure/tree/results/mario_dqn)进行马里奥智能体的特征工程探索。
- DI-adventure包含本次实验的代码；DI-engine为支撑引擎，无需关注。
- 本项目的学术海报见mario_poster_06林韬&王福临.pdf

## 环境说明
环境要求见DI-adventure/mario_dqn/requirements.txt

## 训练方法
对智能体进行训练
```bash
python3 -u mario_dqn_main.py -s <SEED> -v <VERSION> -a <ACTION SET> -o <FRAME NUMBER>
# 以下命令的含义是，设置seed=0，游戏版本v0，动作数目为7（即SIMPLE_MOVEMENT），观测通道数目为1（即不进行叠帧）进行训练。
python3 -u mario_dqn_main.py -s 0 -v 0 -a 7 -o 1
```
## 评估方法
选择训练好的模型（check_point）对智能体进行评估，并保存录像
```bash
python3 -u evaluate.py -ckpt <CHECKPOINT_PATH> -v <VERSION> -a <ACTION SET> -o <FRAME NUMBER>
```

## 算法改进
- 可选择DuelingDQN

## 特征空间处理及结论
### 观测空间（observation space）
- 图像降采样，即将游戏版本从`v0`更改为`v1`
  - 去掉背景使得训练的收敛速度得以加速，主要原因是去掉了白云以及小灌木对训练的影响。
- 堆叠四帧作为输入，即输入变为`(4,84,84)`的图像：`-o 4`；
    - 叠帧wrapper可以将连续多帧的图像叠在一起送入网络，补充mario运动的速度等单帧图像无法获取的信息；
- 图像内容简化（尝试游戏版本`v2`、`v3`的效果）：`-v 2/3`；
  - 在3百万次训练下，将版本更改成v2无法收敛，v3可以收敛，主要原因是v2虽然是像素级的内容简化，但是仍然在云朵、小灌木等背景上浪费了大量算力。
  - 在我们去掉位移奖励之后更是发现，学会了躲避小怪的马里奥见到灌木也是小心翼翼地接近然后躲避。而v3将游戏元素都简化成矩形，减少资源消耗使得训练得以收敛。

### 动作空间（action space）
- 动作简化，将 `SIMPLE_ACTION` 替换为 `[['right'], ['right', 'A']]`：`-a 2`；
    - 有时候简化动作种类能有效降低训练前期学习的困难，但可能降低操作上限；
- 增加动作的多样性
  - 12个动作比2个动作在通关时间上表现更好，因为mario学会了刹车，控制速度，这样有利于更顺利通关。
  - 结论：前期效果一般，但是随着训练时间增加表现出上限优势
- 粘性动作 sticky action
  - 粘性动作的含义是，智能体有一定概率直接采用上一帧的动作，可以增加环境的随机性；
  - 容易撞到南墙还不死心， 因为采用上一个动作或者是前进都无法越过，只有在不采用上一个动作的概率下再采用前越才能过障碍，降低了过障碍的概率
  - 同时另一个原因是深度Q网络（DQN）是确定性（Deterministic）策略，导致相同的输入状态肯定得到相同的输出动作，卡住后这里除了右上角的时间外，状态基本上是相同的，导致会一直输出某一个动作，而这一个动作会导致被卡住，从而状态又得不到更新，因此形成了闭环。
  - 尝试解决：
  - 换一个seed1
  - 结论：能收敛，但收敛较慢
- 采用自定义动作集合。7+下蹲
  - 自定义动作与简单动作的表现基本一致，原因是下蹲动作被采用的概率极低，对训练影响十分微小。

### 奖励空间（reward space）
r=v+c+d（位移、时间惩罚、死亡惩罚）
- 金币奖励
  - 只设置金币奖励并不能有效探索到金币，因为探索空间有限。
  - 尝试改进：
  - 去掉达到分数3000就停止训练的设定
  - 把每一个金币奖励值改到100
  - 结果：学会下管道吃金币
- 升级奖励
  - 只设置升级奖励并不能有效吃到蘑菇，因为探索空间有限。
  - 尝试改进：
  - 升级一次的奖励设置为200
  - 去掉时间惩罚，让mario慢慢来；去掉位移奖励，让mario尽情探索
  - 结论：在设置升级奖励为200的基础上去掉位移奖励，智能体在更全面的探索中学会了顶蘑菇，并且前往落地获得蘑菇升级。在此方案中，可以采用手动判断的方式给予升级后的马里奥位移奖励，使其通关。

- 稀疏 reward，只有死亡和过关才给reward
  - 无法判断过程中某个动作的的好坏，e.g.智能体不清楚在沟前跳还是不跳，跳沟在一些情况下死了（否定动作），在一些情况下不死（肯定动作），因此智能体无法判断该动作好坏。 
  - 完全目标导向。有时候在结果出来前无法判断中途的某个动作的好坏。
  - 结果：5e6个step下不收敛

#### 交互`info`信息
每次交互能获得的信息包含在`info`中 
The `info` dictionary returned by the `step` method contains the following
keys:

| Key        | Type   | Description                                           |
|:-----------|:-------|:------------------------------------------------------|
| `coins   ` | `int`  | The number of collected coins                         |
| `flag_get` | `bool` | True if Mario reached a flag or ax                    |
| `life`     | `int`  | The number of lives left, i.e., _{3, 2, 1}_           |
| `score`    | `int`  | The cumulative in-game score                          |
| `stage`    | `int`  | The current stage, i.e., _{1, ..., 4}_                |
| `status`   | `str`  | Mario's status, i.e., _{'small', 'tall', 'fireball'}_ |
| `time`     | `int`  | The time left on the clock                            |
| `world`    | `int`  | The current world, i.e., _{1, ..., 8}_                |
| `x_pos`    | `int`  | Mario's _x_ position in the stage (from the left)     |
| `y_pos`    | `int`  | Mario's _y_ position in the stage (from the bottom)   |

### 特征工程探索表格展示
| seed | version | action | observation | wrapper       | psss 1-1 |
|------|---------|--------|-------------|---------------|----------|
| 0    | 0       | 7      | 1           | \             | n        |
| 0    | 0       | 2      | 4           | \             | n        |
| 0    | 1       | 2      | 4           | \             | y        |
| 1    | 1       | 2      | 4           | \             | y        |
| 2    | 1       | 2      | 4           | \             | y        |
| 0    | 1       | 7      | 4           | \             | y        |
| 0    | 1       | 12     | 4           | \             | y        |
| 0    | 2       | 2      | 4           | \             | n        |
| 0    | 2       | 7      | 4           | \             | n        |
| 0    | 2       | 12     | 4           | \             | n        |
| 0    | 3       | 2      | 4           | \             | y        |
| 0    | 3       | 7      | 4           | \             | y        |
| 0    | 3       | 12     | 4           | \             | y        |
| 0    | 1       | 12     | 4           | sparse reward | n        |
| 0    | 1       | 12     | 4           | sticky action | y        |
| 0    | 1       | 12     | 4           | coin reward   | y        |
| 0    | 1       | 12     | 4           | status reward | n        |
| 0    | 1       | 8      | 4           | status reward | n        |

## 分析方式
### tensorboard 中指标含义如下
tensorboard结果分为 buffer, collector, evaluator, learner 四个部分，以\_iter结尾表明横轴是训练迭代iteration数目，以\_step结尾表明横轴是与环境交互步数step。
一般而言会更加关注与环境交互的步数，即 collector/evaluator/learner\_step。
#### evaluator
评估过程的一些结果，最为重要！展开evaluator_step，主要关注：
- reward_mean：即为任务书中的“episode return”。代表评估分数随着与环境交互交互步数的变化，一般而言，整体上随着交互步数越多（训练了越久），分数越高。
- avg_envstep_per_episode：每局游戏（一个episode）马里奥平均行动了多少step，一般而言认为比较长一点会好；如果很快死亡的话envstep就会很短，但是也不排除卡在某个地方导致超时的情况；如果在某一step突然上升，说明学到了某一个很有用的动作使得过了某一个难关，例如看到坑学会了跳跃。
#### collector
探索过程的一些结果，展开collector_step，其内容和evaluator_step基本一致，但是由于探索过程加了噪声（epsilon-greedy），一般reward_mean会低一些。
#### learner  
学习过程的一些结果，展开learner_step：
- q_value_avg：Q-Network的输出变化，在稳定后一般是稳固上升；
- target_q_value_avg：Target Q-Network的输出变化，和Q-Network基本上一致；
- total_loss_avg：损失曲线，一般不爆炸就不用管，这一点和监督学习有很大差异，思考一下是什么造成了这种差异？
- cur_lr_avg：学习率变化，由于默认不使用学习率衰减，因此会是一条直线；
v1提升性能，因为 将不需要关注的云朵去掉了