# 第一阶段 AI 训练：桃子 vs 20 级桃子

这一阶段先验证从“战斗模拟 → 观察 → 动作 → 奖励 → PPO → 胜率评估”的完整闭环，
暂不同时训练全部角色和地图。

## 固定规则

- AI 操作 P1 桃子，P2 是现有原版逻辑的 20 级桃子。
- 地图固定为第二张 `Mogadishu`。
- 生命模式，每方 3 条命；每局最多 180 秒虚拟游戏时间。
- 道具生成频率只在训练环境中改为 0。
- 战斗物理仍是 25 ms/帧（40 Hz），AI 每 2 帧决策一次（20 Hz）。
- 每局交替左右出生点，避免模型只会从一边开局。
- 不启动窗口、不读取真实键盘，也不播放声音。

训练代码没有被 `play.py`、菜单或普通战斗导入。训练环境持有自己的运行时和清单副本；
因此关闭道具、固定角色、3 条命和地图等设置不会影响正常启动游戏。

## 动作

动作空间是 `MultiDiscrete([3, 3, 4])`，一个动作同时包含三部分：

| 分量 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 水平 | 不动 | 左 | 右 | — |
| 垂直 | 不动 | 上 | 下 | — |
| 战斗 | 无 | 手刀/普通攻击 | 特殊攻击 | 护盾 |

组合动作会在同一个 25 ms 原版逻辑帧触发：

- `上 + 普通攻击`：上手刀；条件不成立时按原逻辑尝试普通手刀。
- `上 + 特殊攻击`：桃子的上特殊攻击/火箭筒；若当前不能用上特殊攻击，按原逻辑尝试枪。
- `上`：跳跃或二段跳。
- `特殊攻击`：地面或空中开枪。
- `下 + 攻击`、移动攻击、空中攻击和护盾均保留原本的状态条件。

模型不直接选择“子弹”或“火箭”；它学习原游戏输入组合以及何时满足对应动作条件。

## 奖励

目标是把人打出界，不是停在场内刷伤害。每次决策的奖励为：

```text
R = 1.0 × (对手掉命 - 自己掉命)
  + 0.5 × 最终胜负
  + 0.15 × (对手向死亡边界的进度 - 自己向死亡边界的进度)
  + 0.0005 × (造成伤害 - 承受伤害)
```

掉命发生时不计算该步的边界进度和伤害塑形，避免复活位置与伤害归零抵消出界奖励。
所以造成 100 点伤害仅值 `0.05`，击出一条命值 `1.0`；最终赢下整局另加 `0.5`。
伤害的用途主要是提高后续击退，学习目标仍然是出界和获胜。

## 安装

建议使用独立虚拟环境，避免 PyTorch、Gymnasium 和 TensorBoard 增大普通游戏环境：

```bash
cd /Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake
# 用已能正常运行游戏的 Python 3.10–3.13 创建；当前不要用 Python 3.14
.venv/bin/python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

在 M1 上第一阶段先用 CPU。当前是小型 MLP 和单个无画面环境，GPU/MPS 的数据搬运
开销通常不会带来优势；以后并行很多环境或改用图像输入/大网络时再转 NVIDIA GPU。
本机 Python 3.13/M1 实测训练环境占磁盘约 800 MB（主要是 PyTorch），单进程 PPO
峰值内存约 1.0 GB。2048 步完整冒烟训练约 6 秒；按此线性估算，100 万步约
50 分钟，建议预留 1–2 小时，具体取决于机器温度和后台负载。

## 1. 先跑基准

```bash
python -m training.benchmark --episodes 20
```

默认基准是一个很小的规则策略，用来确认完整对局、出界、胜负、超时和速度都正常，
不是要替代学习模型。也可以测试随机输入：

```bash
python -m training.benchmark --policy random --episodes 20
```

## 2. 短训练检查

先跑一小段，确认本机依赖和保存流程：

```bash
python -m training.train_ppo --steps 10000 --name peach_smoke
```

这点步数一般不会产生强 AI，只是冒烟测试。正式第一轮建议从 100 万决策开始：

```bash
python -m training.train_ppo --steps 1000000 --name peach_mogadishu_v1
```

终端会显示 PPO 的 `fps`、奖励和对局长度。模型与中间检查点保存在
`training/checkpoints/<name>/`，TensorBoard 日志在 `training/logs/<name>/`；二者都被
Git 忽略，不会把几百 MB 或数 GB 的训练产物推到仓库。

查看训练曲线：

```bash
tensorboard --logdir training/logs
```

如果中断，可从检查点继续：

```bash
python -m training.train_ppo \
  --steps 1000000 \
  --name peach_mogadishu_v1 \
  --resume training/checkpoints/peach_mogadishu_v1/ppo_peach_100000_steps.zip
```

## 3. 独立评估

训练种子和评估种子不同，并交替出生方向：

```bash
python -m training.evaluate \
  --checkpoint training/checkpoints/peach_mogadishu_v1/final_model.zip \
  --episodes 100 \
  --record 3
```

输出包含胜/负/平、正常结束/超时、平均奖励、双方平均击杀、虚拟对局时长和运行速度。
`--record 3` 会把前三局的固定种子输入录像放进 `training/replays/`，可用于复查策略。

阶段一的达标标准不是训练 loss 下降，而是独立 100 局对 20 级 AI 的胜率稳定超过基准，
且换边后没有明显单侧退化。达到后再进入对手池、自我对战、课程学习和真人录像模仿。

## 4. 亲自对战“21级”

专用启动器会保留原版 1–20 级 AI，只在本次启动中把训练模型开放为 21 级：

```bash
source .venv-train/bin/activate
python -m training.play_level21
```

为了最忠实地复现训练位置，菜单中请设置：

1. 进入 `Multiplayer -> VS Mode`。
2. P1 设为 `CP 21` 并选 Peach；P2 保持真人并选 Peach。
3. P3/P4 关闭，`STOCK 3`，地图选 `Mogadishu`。
4. 你操作 P2：方向键移动/跳跃，小键盘 `0` 手刀、`1` 特殊攻击、`2` 护盾。

模型是以 P1 槽位训练的。把 21 级放到 P2、换角色或换地图仍然可以运行，但属于
泛化测试，不能再当成训练胜率的同等复现。实测中它对固定 P2 20 级 AI 能快速
获得 3 个 KO，但对完全不动的对手反而可能僵住，这正是对固定对手过拟合的证据。

## 5. 第二阶段：同时进化 21/22 级

`training.train_league` 不再把 20 级当主要训练价值来源。它建立两个独立 PPO 模型：

1. 每轮先冻结 22，训练 21 对战当前 22。
2. 马上用更新后的 21 做对手，训练 22。
3. 每轮都保存两者并做换槽位的相互评估。

这是“同一阶段训练两个模型”，但不会在一段 PPO rollout 中同时改动双方权重。后者会让
采样时的对手分布不停变化，破坏 PPO 的稳定假设。两个模型都从目前已验证过的旧 21 级
权重初始化：原 142 维观察完整复制，新增加的 38 维输入权重从 0 开始学习。旧 21 模型
保持冻结，只占约 15% 对局用于防止遗忘，不会被覆盖。

其余约 75% 对局是 21/22 互打，约 10% 是不动或持续后撤的探针对手。这两个探针不教
攻击，只检查 AI 是否会主动追人，解决“离远后自己跳和手刀”的固定对手过拟合。每局
随机让学习者处于 P1 或 P2，并独立交换物理出生方向。约 12% 对局从台外回场状态开始，
约 30% 对局按原频率生成地雷/手雷；模型现在能观察最近两个道具及自己手持的道具。

### 真人式输入约束

- 手刀、特殊攻击、跳跃必须先松开才能再次形成按下边沿，不再每 50 ms 自动连点。
- 左右方向至少承诺 100 ms，减少无意义的左右抽搐。
- 护盾有最短保持和 200 ms 再开启间隔，不能每个决策瞬间闪盾。
- 模型看到的是约 100 ms 前的状态，不再拥有零延迟反应。
- `方向 + 手刀`、`上 + 手刀`、`上 + 特殊攻击`仍可在同一原版逻辑帧触发。

技能多样性不会按“按了多少次”奖励。只有命中后才根据出招时机给小额奖励：敌人在上方
时上手刀、空中时空中手刀、远距离对齐时开枪、回场或敌人在上方时用火箭、真正触发背后
抱摔。连续用不同攻击命中也有很小的多样性奖励；普通手刀落空则扣分。因此不能靠随机
乱按组合键刷分。核心目标仍是出界：击掉一条命为 `2.0`，最终胜负为 `±1.0`，造成
100 点伤害只有 `0.02`。

先跑双模型冒烟测试（每个模型各约 1 万步）：

```bash
cd /Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake
.venv-train/bin/python -m training.train_league \
  --rounds 1 \
  --steps-per-round 10000 \
  --name peach_league_test
```

确认终端能依次完成“更新21级”“更新22级”、评估并生成两个 zip 后，开始正式训练：

```bash
.venv-train/bin/python -m training.train_league \
  --rounds 10 \
  --steps-per-round 100000 \
  --name peach_league_v2 \
  --device cpu
```

默认结果分别为：

```text
training/checkpoints/peach_league_v2/level21_model.zip
training/checkpoints/peach_league_v2/level22_model.zip
```

每轮也会保留 `round_XXX_level21.zip` 和 `round_XXX_level22.zip`，所以如果后期策略退化，
可以回到较好轮次。按 `Ctrl-C` 时脚本也会先保存当前两套模型。继续已有联赛：

```bash
.venv-train/bin/python -m training.train_league \
  --rounds 5 \
  --steps-per-round 100000 \
  --resume-dir training/checkpoints/peach_league_v2 \
  --name peach_league_v2
```

M1/M2 上这类单环境小型 MLP 仍建议 CPU；MPS 的频繁小批量搬运通常不划算。默认正式
配置是两个模型各 100 万步，总采样量约为第一阶段的两倍，建议预留约 2–4 小时和约
1.5 GB 内存，温度、后台程序和实际对局长度会造成明显波动。模型本身每个约 2–3 MB，
每轮双份存档会累计约 50–70 MB；全部训练输出已被 Git 忽略。

训练后亲自对战：

```bash
.venv-train/bin/python -m training.play_league
```

菜单会开放 21 和 22。建议固定 Peach vs Peach、`STOCK 3`、`Mogadishu`，分别亲测
追击、回场、抱摔、上手刀、枪、火箭和护盾反应。普通的 `python play.py` 仍只开放原版
1–20 级；联赛的角色、地图、道具概率和奖励配置不会进入普通游戏。

## 自动测试

训练环境测试使用训练虚拟环境：

```bash
python -m unittest discover -s training/tests -v
```

普通游戏测试仍使用原来的环境与命令：

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## 6. 第三阶段：看准再出手的 v3 战术 AI

v2 的主要问题不是单纯“没训够”，而是每 50 ms 直接重选一组原始按键、远程攻击
落空几乎没成本，护盾也不需要真正的威胁。`training.train_tactical` 使用独立的
`peach_tactical_v3` 目录，不会改写已有 `peach_league_v2` 的 21/22 级模型。

### 战术动作和人类式时序

v3 每 100 ms 选一次 `MultiDiscrete([4, 9])` 战术意图，然后转成 4 个真实的
25 ms 游戏逻辑帧：

| 分量 | 选项 |
| --- | --- |
| 移动 | 停止、向对手靠近、远离对手、继续当前意图 |
| 战斗 | 无动作、跳/二段跳、下落穿台、手刀、上手刀、背后抱摔、枪、火箭、护盾 |

- 移动意图至少承诺 200 ms，不能每帧左右抽搐。
- 攻击后必须显式松开，且至少间隔 300 ms 才能再次形成攻击边沿。
- 护盾有最短保持和 300 ms 再开间隔；无子弹或近身攻击威胁时开盾会扣分。
- 枪只在目标处于当前朝向、射线没被平台挡住、距离和纵向预判合理时可选。
- 火箭除了台外回场，只在目标进入原版向上 30 度弹道的预测截击窗口时可选。
- 背后抱摔只在距离、高度和背后几何条件成立时可选，方向+手刀在同一帧下发。

动作掩码只让 PPO 在当前确实能执行的动作中采样，避免用几十万步反复学习
“这个按键在此刻根本不生效”。

### 它现在能看见什么

232 维观察保留 v2 的双方位置、速度、朝向、伤害、生命、攻击状态、平台、
道具和最近投射物，另外增加：

- 当前移动/战斗意图、冷却、护盾保持和再开状态；
- 前两个敌方投射物的相对位置、速度、类型、预计最近时间和错开距离；
- 近身威胁、射线是否清晰、目标纵向预判、是否远距离。

所以枪和火箭不再只能“猜”对手在哪。策略仍有约 100 ms 观察延迟，不是读取
未来帧或瞬时反应。

### 训练课程、对手池和奖励

训练初期约 70% 的局是 20 秒专项课：静止/移动目标枪击、火箭弹道、威胁护盾、
近战/抱摔、台外回场和追击后退目标。到最后一轮专项课逐步降为 20%，其余是完整对局。

完整对局中，21 和 22 仍采用交替冻结训练：更新 21 时 22 不变，然后用新 21 更新
22。对手池在有历史快照后是当前对手 50%、最近 6 个冻结快照合计 35%、v2 21 级
15%，防止双方一起进化成只对当前对手有效的奇怪暗号。

出界、最终胜负仍是主奖励；伤害只是很小的塑形。额外奖励只给实际命中的枪/火箭、
实际挡住的护盾和符合位置的组合技；投射物消失未命中、手刀落空、无威胁开盾都扣分。

### 终端命令

依赖中新增了 `sb3-contrib` 的 `MaskablePPO`，先更新一次训练环境：

```bash
cd /Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake
.venv-train/bin/python -m pip install -r requirements-training.txt
```

先跑冒烟测试（只验证链路，模型不会强）：

```bash
.venv-train/bin/python -m training.train_tactical \
  --rounds 1 \
  --steps-per-round 10000 \
  --eval-episodes 4 \
  --name peach_tactical_test \
  --device cpu
```

通过后跑正式训练：

```bash
.venv-train/bin/python -m training.train_tactical \
  --rounds 12 \
  --steps-per-round 100000 \
  --eval-episodes 20 \
  --name peach_tactical_v3 \
  --device cpu
```

每轮都会输出对当前对手和冻结 v2 21 级的固定种子结果，并同时统计投射物命中率、
每分钟投射物/护盾次数、假护盾率、近战命中率、动作切换率和远距离占比。因此
`best_level21_model.zip` / `best_level22_model.zip` 是胜负与行为质量综合分最高的轮次，
不是盲目使用最后一轮。一旦有模型通过行为门槛，未通过门槛的后续模型不会因为偶然多赢几局就覆盖它。

中断后继续：

```bash
.venv-train/bin/python -m training.train_tactical \
  --rounds 6 \
  --steps-per-round 100000 \
  --resume-dir training/checkpoints/peach_tactical_v3 \
  --name peach_tactical_v3 \
  --device cpu
```

训练完亲自对战（默认优先加载 `best_*`）：

```bash
.venv-train/bin/python -m training.play_tactical
```

本地 M1 仍建议用 CPU。这是 232 维小型 MLP 和单环境物理模拟，MPS 的小批量数据搬运往往比
计算更贵。正式配置是两个模型各 120 万步；建议预留 2–5 小时、1.5–2.5 GB 内存和
约 80–120 MB 快照空间。实际时间会随散热、对局长度和后台负载变化。

## 7. 第四阶段：主动追击、越墙和可晋级的 v4 AI

v4 专门修正 v3 后期出现的“一个只开枪、一个蹲后面开盾”共同退化。
它使用新的 252 维观察和独立模型目录，v2/v3 存档、普通 `play.py`、其他角色和
其他地图配置都不会被训练覆盖。

### 战斗约束

- 无威胁且距离超过 130 时，必须主动向对手靠近；停止或后退不会被动作掩码接受。
- 距离 78 以内优先开放手刀、上手刀和背后抱摔，近距离不允许用手枪代替近战。
- 手枪只在 75–360 距离、面向正确、射线无遮挡且纵向预判误差不超过 32 时可用；
  开枪后约 800 ms 才能再开枪。
- 护盾只在投射物或真实近战威胁下可用，至少保持约 200 ms，最多保持约
  500 ms，释放后约 700 ms 才能重开。同一次开盾只能记一次成功格挡。
- 固定平台上的蹲下被禁止；空中快落和从移动平台下穿仍保留。
- Fixed12/Fixed13 等窄墙使用原版六点探针语义：身体左/右中点命中才挡住，
  只有脚部角点擦过时可滑过。同时保留扫掠检测，高速移动不能穿墙。

### 共享基础、21/22 联赛和冠军门槛

训练先用主动追击脚本、近战脚本、冻结的 v2 21 级和专项课训出一个
`foundation_model.zip`，然后从同一个基础分叉出 21 和 22。更新 21 时 22 是冻结对手，
更新 22 时 21 保持不变，所以 PPO 的一个 rollout 中不会遇到不停变化的对手。

每轮都保存 `candidate_level21_model.zip` / `candidate_level22_model.zip`，但候选必须同时满足
以下条件才会成为 `champion_*`：

- 至少 60% 评估局真正决出胜负，且总胜率至少 35%；所有 `timeout_*` 都按失败评分。
- 远距离发呆不超过 18%，撞墙停滞不超过 15%，护盾占用不超过 8%。
- 假护盾率不超过 15%，开盾不超过每分钟 8 次。
- 有足够的子弹和近战样本，投射物命中率至少 18%，近战机会利用率至少 25%。

一旦已有合格冠军，后续未过门槛或分数更低的候选不会覆盖它，下轮也会回到
冠军继续学习。这样两个模型不能通过一起蹲守来互相“证明”自己变强。

### 先跑小规模验证

这条命令是检查完整链路，这么少的步数不会训出强模型：

```bash
cd /Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake
.venv-train/bin/python -m training.train_v4 \
  --foundation-steps 50000 \
  --rounds 1 \
  --steps-per-round 50000 \
  --eval-episodes 10 \
  --name peach_active_v4_test \
  --device cpu
```

小规模模型很可能不过冠军门槛，用显式候选模式人工试玩：

```bash
.venv-train/bin/python -m training.play_v4 \
  --directory training/checkpoints/peach_active_v4_test \
  --allow-candidate
```

### 正式训练与试玩

```bash
.venv-train/bin/python -m training.train_v4 \
  --foundation-steps 300000 \
  --rounds 8 \
  --steps-per-round 75000 \
  --eval-episodes 40 \
  --name peach_active_v4 \
  --device cpu
```

被 `Ctrl-C` 中断后，脚本会保存当前 candidate。从同名目录继续：

```bash
.venv-train/bin/python -m training.train_v4 \
  --rounds 4 \
  --steps-per-round 75000 \
  --eval-episodes 40 \
  --name peach_active_v4 \
  --device cpu \
  --resume
```

两个级别都有合格冠军后，默认启动器只加载冠军：

```bash
.venv-train/bin/python -m training.play_v4
```

本地 M1 还是建议 CPU。这是单环境物理采样加小型 MLP，性能瓶颈主要在环境而不是
矩阵计算，MPS 不一定更快。默认总量是 30 万共享基础加两个模型各 60 万步，
可先用上面的小规模命令测出本机每万步时间，再线性估算正式训练时长。

## 8. 第五阶段：有目的的 v5 连续战术 AI

v5 不再让 PPO 每 100 ms 直接拼凑方向、跳跃和攻击键。策略只选择 14 种有名称的
战术目的，然后由 40 Hz 执行器完成连续按键：追击、越墙、地面手刀、空中追击、对空手刀、
抱摔、瞄准射击、火箭、击飞脱离、闪避、回场、护盾、有目的落地和继续当前计划。

每个计划有最小承诺时间、目标点、期限、完成条件和失败条件。普通情况不能在计划中途每帧
左右反转；只有回场、击飞脱离或实时威胁可以中断当前计划。

### 墙体与空中追击

- 舞台的每个屋顶和平台是导航节点，根据桃子的单跳、二段跳和水平速度建立可达边。
- 即使双方都在 Fixed1，Fixed12/Fixed13 位于中间时也会生成“墙顶”临时路径点，
  不会因为起点和终点同属 Fixed1 就一直贴墙走。
- 在一跳轨迹已足够越过墙顶时禁止浪费二段跳。空中靠近目标 x 时会主动刹住水平速度以落在窄墙顶。
- 空中追击预测对手未来 100–400 ms 的位置，先按住水平方向起跳，进入命中窗口后保持水平移动并触发 `punchAir`。
- 被击飞者在 `ctrl_loss > 0` 时不能提前操作；v5 在控制即将恢复时缓存脱离方向，在第一个合法帧跳出 `thrown` 并横向避开下方追击。
- 快落只属于有安全落点的落地/闪避计划。跳跃后 400 ms 内的无目的快落被执行器拦截。

### 技能考试后才自对战

共享基础依次学习平台导航、空中追击、击飞脱离、地面击飞接空中手刀，再进入混合课。
正式模式下，任一技能考试未过就不会分叉 21/22，避免两个不会越墙的模型开始互相自证。

冠军门槛额外检查：墙体停滞不超过 5%、有目的跳跃至少 90%、无效“跳+快落”不超过 1%、
空中追击机会利用率至少 30%、空中手刀命中率至少 20%，同时保留真实胜负、枪法、护盾和超时门槛。

### 先跑可玩的小规模版本

这条命令约 10 万总步数。`--allow-unqualified-foundation` 只用于让小规模训练能生成
21/22 candidate 供人工验收，不会绕过冠军门槛：

```bash
cd '/Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake'
.venv-train/bin/python -m training.train_v5 \
  --navigation-steps 10000 \
  --air-steps 10000 \
  --escape-steps 10000 \
  --combo-steps 10000 \
  --mixed-steps 20000 \
  --rounds 1 \
  --steps-per-round 20000 \
  --skill-eval-episodes 8 \
  --eval-episodes 10 \
  --lesson-seconds 8 \
  --name peach_purpose_v5_test \
  --device cpu \
  --allow-unqualified-foundation
```

完成后试玩 candidate：

```bash
.venv-train/bin/python -m training.play_v5 \
  --directory training/checkpoints/peach_purpose_v5_test \
  --allow-candidate
```

### 正式训练

```bash
.venv-train/bin/python -m training.train_v5 --device cpu
```

默认先训 57 万步共享基础，再训 8 轮、每轮两个模型各 7.5 万步。如果技能考试未过，
脚本会保存 `foundation_model.zip` 并在自对战前停止。继续同名基础：

```bash
.venv-train/bin/python -m training.train_v5 --device cpu --resume
```

两个级别都通过冠军门槛后：

```bash
.venv-train/bin/python -m training.play_v5
```

要把真人对战保存为后续模仿学习素材，可在桌面版开启录制：

```bash
.venv-train/bin/python -m training.play_v5 --record-human
```

每局会自动写入 `training/replays/human_v5/`，内容包括固定随机种子、初始状态、
真人和 AI 的逐帧输入及比赛结果。录制不会在对局中改写当前 21/22 冠军；后续只从
这些录像训练独立候选，候选仍须通过技能考试、固定种子联赛和人工试玩后才能晋级。

### 当前一轮：新策略→真人对战→模仿学习

第一步，用已通过的 v5 基础再训一轮 21/22：

```bash
cd '/Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake'
.venv-train/bin/python -m training.train_v5 \
  --resume \
  --rounds 1 \
  --steps-per-round 75000 \
  --freeze-champions \
  --device cpu
```

本轮会先完成尚未完成的混合基础课，再依次冻结 22 更新 21、冻结新 21
更新 22。这次的策略变化是：远距离更主动找人，火箭只轻度收紧，组合技确认和顶部安全
保持中等强度，平台导航、空追、脱离和连招专项课会随机化出生侧、高度、速度与伤害。

第二步，试玩本轮候选并同时录像：

```bash
.venv-train/bin/python -m training.play_v5 --prefer-candidate --record-human
```

菜单中固定 `Peach vs Peach`、`STOCK 3`、`Mogadishu`、关闭道具，对手选 `CP 22`。
每局结束后自动保存，无需退出游戏。建议至少打 3–5 局，有意演示主动追击、地面手刀、
跳跃横移手刀、对空追击、对准后开枪、合理火箭和被击脱离。

第三步，把新版可验证录像模仿到独立的 22 级候选：

```bash
.venv-train/bin/python -m training.learn_from_human \
  --base training/checkpoints/peach_purpose_v5/candidate_level22_model.zip
```

它只接受 `glorton-input-recording-v2` 录像：开局与结束状态摘要都必须逐帧一致，旧 v1
录像和不完整录像会自动跳过。静止帧和无目的蹲下不会成为示范。学习率仅 `5e-5`，并使用
冻结基础策略的 KL 锚定，防止几局真人样本把原来的回场和导航能力洗掉。输出是
`human_candidate_level22_model.zip`，不会覆盖当前冠军。

先直接验收真人模仿候选：

```bash
.venv-train/bin/python -m training.play_v5 --prefer-candidate
```

如果它的出手风格正常，再用 21 级对手巩固一轮，防止只会模仿录像中出现过的局面：

```bash
.venv-train/bin/python -m training.train_v5 \
  --resume \
  --rounds 1 \
  --steps-per-round 75000 \
  --level22-seed training/checkpoints/peach_purpose_v5/human_candidate_level22_model.zip \
  --freeze-champions \
  --device cpu
```

`--prefer-candidate` 只改变当次试玩加载顺序；`--level22-seed` 只改变当次训练起点；
`--freeze-champions` 明确禁止本次训练晋级。因此上面两次训练都不会改写
`champion_level21_model.zip` 或 `champion_level22_model.zip`。

v5 使用独立的 `training/checkpoints/peach_purpose_v5` 目录和 `GLORTON_AI_V5`
启动开关，不会覆盖 v2/v3/v4 模型，普通 `play.py` 也不会加载它。
