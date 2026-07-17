# 全角色 AI 阶段 B：单图并发训练

## 本轮范围

本轮只训练 Mogadishu 上的五个非桃子角色：Default、Trash、Coffee、SBL 和 Aubergine。
桃子 21/22 级冠军保持冻结，手机版和普通 `play.py` 不读取本训练目录。

每个新模型使用 480 维 v6 输入和原来的 14 个目的动作。第 7/8 个动作编号保持兼容，但
含义按角色解释为“普通特殊”和“上特殊”，不再统一当作桃子的枪和火箭。

## 避免重复长训的设计

1. 正式启动前强制验证每个角色的普通特殊、上特殊、480 维观察、合法动作掩码和短局
   步进；任何一项失败都会停止。
2. 从已认可的桃子 22 级模型复制全部网络权重。原 294 维输入权重逐值保持相同，新增加
   的 186 列从零开始，因此初始行为不会被随机网络破坏。
3. 先训练一个混合基础，再由同一基础分出 21/22；每轮先更新 21，再让 22 对新 21 学习。
4. 对手池同时包含当前同角色模型、角色基础、主动近战探针、追击探针和冻结桃子目的
   策略，防止两边一起学会消极拖延。
5. 特殊技必须满足角色自己的距离、朝向、弹道和上边界安全窗口。无命中目的的重复释放
   有小额代价；命中、追击、击飞出界和完成导航计划获得奖励。
6. 每 10000 步覆盖保存一次。中断后会从当前阶段剩余步数继续，不会把整轮重跑。

## 第一步：只跑预检

在项目根目录执行：

```bash
.venv-train/bin/python -m training.run_roster_parallel \
  --preflight-only \
  --run-id roster_b1
```

看到“`五角色预检通过`”后再开始正式训练。预检通常只需几秒。

## 第二步：五个角色同时训练

```bash
caffeinate -dimsu .venv-train/bin/python -m training.run_roster_parallel \
  --run-id roster_b1 \
  --workers 5
```

默认每个角色训练 160000 步基础，再进行 5 轮 21/22 互训，每级每轮 60000 步。五个进程
被限制为各使用一个数值计算线程，避免底层库在每个进程里再次占满全部 CPU。M1 默认用
CPU；不要让五个独立进程同时使用 MPS。

训练输出分别保存在：

```text
training/checkpoints/roster_v6/<角色>/mogadishu/roster_b1/
training/logs/roster_v6/<角色>/mogadishu/roster_b1/launcher.log
```

查看全部实时日志：

```bash
tail -f training/logs/roster_v6/*/mogadishu/roster_b1/launcher.log
```

## 中断和续训

在并发启动终端按一次 `Ctrl+C`，启动器会先向所有子进程发送中断，让它们保存当前模型，
最多等待 20 秒后才结束。继续时使用完全相同的 `run-id`：

```bash
caffeinate -dimsu .venv-train/bin/python -m training.run_roster_parallel \
  --run-id roster_b1 \
  --workers 5 \
  --resume \
  --skip-preflight
```

如果机器温度或响应压力过大，可以中断后把 `--workers 5` 改成 `--workers 3`。任务种子、
模型目录和已完成阶段不变，只是同时运行三个，完成一个后自动启动下一个。

## 训练结束后的含义

每个目录会生成 `candidate_level21_model.zip`、`candidate_level22_model.zip` 和逐轮评估。
评估检查决胜率、远距离空转、卡墙、护盾滥用、近战机会、跳跃目的和特殊技频率。
候选不会自动进入普通游戏；仍需统一试玩和人工验收，合格后才接入正式 1–22 级选择。

## 像普通游戏一样试玩

这个入口和桃子的 `training.play_v5` 一样，会直接弹出完整游戏窗口并停在原主菜单，
不会后台自动开战。进入 `MULTIPLAYER` 后可自行安排真人和 AI：

```bash
.venv-train/bin/python -m training.play_roster_battle \
  --prefer-candidate \
  --record-human
```

选人时将需要交给 AI 的玩家切换成 `CP`，选择任意角色，并把 CP 等级设为 21 或 22。
真人玩家保持 `P1`/`P2` 即可亲自对打。要观看 AI 互打，就把两个玩家都切成 `CP`。
地图选择训练过的 Mogadishu（楼房）。

桃子使用原来的 v5 21/22 级模型；Default、Trash、Coffee、SBL 和 Aubergine 会分别读取
自己训练完成的 roster_v6 模型，不会共用一个模型。`--record-human` 会保存真人输入，
但不会在对局中即时改写模型。
