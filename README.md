# 超级马里奥兄弟--林韬&王福临
## 算法改进
- 主要为了提升马里奥通关速度
- TODO
## 特征空间处理
### 观测空间（observation space）
- 图像降采样，即将游戏版本从`v0`更改为`v1`，游戏版本的内容请参照[mario游戏仓库](https://github.com/Kautenja/gym-super-mario-bros)：`-v 1`；
- 堆叠四帧作为输入，即输入变为`(4,84,84)`的图像：`-o 4`；
    - 叠帧wrapper可以将连续多帧的图像叠在一起送入网络，补充mario运动的速度等单帧图像无法获取的信息；
- 图像内容简化（尝试游戏版本`v2`、`v3`的效果）：`-v 2/3`；
### 动作空间（action space）
- 动作简化，将 `SIMPLE_ACTION` 替换为 `[['right'], ['right', 'A']]`：`-a 2`；
    - mario提供了不同的[按键组合](https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py)，有时候简化动作种类能有效降低训练前期学习的困难，但可能降低操作上限；
- 增加动作的多样性，将 `SIMPLE_ACTION` 替换为 `COMPLEX_MOVEMENT`：`-a 12`；
    - 也许能提高上限；
- 粘性动作 sticky action（给环境添加 `StickyActionWrapper`，方式和其它自带的 wrapper 相同，即`lambda env: StickyActionWrapper(env)`）
    - 粘性动作的含义是，智能体有一定概率直接采用上一帧的动作，可以增加环境的随机性；
### 奖励空间（reward space）
- 金币奖励
- 稀疏 reward，只有死亡和过关才给reward（给环境添加 `SparseRewardWrapper`，方式和其它自带的 wrapper 相同）
    - 完全目标导向。稀疏奖励是强化学习想要落地必须克服的问题，有时候在结果出来前无法判断中途的某个动作的好坏；

| s   | v   | a   | o   |
|-----|-----|-----|-----|
| 0   | 0   | 7   | 1   |
| 0   | 0   | 2   | 4   |
| 0   | 1   | 2   | 4   |
| 1   | 1   | 2   | 4   |
| 2   | 1   | 2   | 4   |
| 0   | 1   | 7   | 4   |
| 0   | 1   | 12  | 4   |
| 0   | 2   | 2   | 4   |
| 0   | 2   | 7   | 4   |
| 0   | 2   | 12  | 4   |
| 0   | 3   | 2   | 4   |
| 0   | 3   | 7   | 4   |
| 0   | 3   | 12  | 4   |
|     |     |     |     |
|     |     |     |     |
|     |     |     |     |


conda activate mario && cd DI-adventure/mario_dqn && CUDA_VISIBLE_DEVICES=2 python3 -u mario_dqn_main.py -s 0 -v 3 -a 12 -o 4
```bash
python3 -u mario_dqn_main.py -s 0 -v 0 -a 7 -o 1
python3 -u mario_dqn_main.py -s 0 -v 0 -a 2 -o 4

```

## 分析
