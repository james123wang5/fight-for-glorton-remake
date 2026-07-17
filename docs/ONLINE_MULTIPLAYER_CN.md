# Glorton 电脑版联机说明

这套联机是独立的电脑版入口，不修改手机 PWA，也不会影响原有单机、普通多人或训练命令。

## 已实现的七层

1. **房间码**：房主创建 6 位房间码，对方输入房间码加入，不要求账号。
2. **同步选人**：两边始终看到同一份房间状态，各自修改自己的角色、颜色和准备状态。房主可添加 P3/P4 电脑。
3. **40Hz 权威服务器**：服务器每 25ms 调用同一套 `BattleSimulation`，死亡、伤害、飞出界、子弹、火箭、道具和 AI 都以服务器为准。
4. **预测、插值和校正**：本机移动先做短距离视觉预测，确认帧之间用 `prev_pos` 插值，每 20 帧比对摘要。不一致时恢复服务器快照。
5. **抖动与断线处理**：输入预留 2 帧缓冲，一次性按键不会因丢包重复触发；显示 RTT 和抖动；断线后服务器暂停最多 25 秒。
6. **轻量 AI**：1–20 级保留原版 AI；21/22 级用当前候选模型导出的 NumPy 权重，不需要 PyTorch。桃子为 294 维，其余五角色为各自 480 维策略。
7. **可部署服务**：`online/Dockerfile` 和 `render.yaml` 可运行同一服务。服务器不依赖玩家的 Mac，两台电脑连入公网 `wss://.../ws`。

## 本机双开测试

首次安装依赖：

```bash
cd '/Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake'
.venv-train/bin/python -m pip install -r requirements-online.txt
```

终端 1：

```bash
cd '/Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake'
.venv-train/bin/python -m online.server --host 127.0.0.1 --port 8765
```

终端 2 和终端 3 分别运行：

```bash
.venv-train/bin/python -m online.client --name '玩家1'
.venv-train/bin/python -m online.client --name '玩家2'
```

第一个窗口点击 `CREATE ROOM`（或按 `C`）创建房间，第二个窗口点击 `JOIN ROOM`（或按 `J`）并输入房间码。

## 免费临时公网联机（房主电脑保持开机）

这是无需云服务器、无需 Cloudflare 账号或银行卡的兜底方案。它通过 [Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/) 把本机联机服务器临时映射到公网；Cloudflare Tunnel [原生支持 WebSocket](https://developers.cloudflare.com/cloudflare-one/faq/cloudflare-tunnels-faq/)。

房主在项目目录运行：

```bash
cd '/Users/wenxinzheng/Desktop/一些无关紧要的项目/pythonpygame/glorton_remake'
.venv-train/bin/python -m online.tunnel
```

终端会显示一个 `wss://随机名称.trycloudflare.com/ws` 地址和完整客户端命令。房主与朋友分别运行该命令，然后正常创建/加入 6 位房间码。

- 房主终端必须保持运行，Mac 也必须联网；脚本会阻止空闲自动睡眠，但 Mac 合盖通常仍会断开。
- 按 `Ctrl+C` 会关闭隧道以及由脚本启动的本机服务器。
- 临时地址每次运行都会变化，没有可用性保证；它适合两三位朋友临时游玩，不是正式长期服务。
- 只通过加密的出站连接开放 HTTP/WebSocket，不需要路由器端口转发。

## 联机大厅操作

- 鼠标：点击自己角色卡两侧箭头换角色，点击四色方块换颜色，点击底部 `READY` 准备。两名真人的选择会实时同步显示。
- 房主可直接点击 P3/P4 卡片中的 `ADD`、角色箭头和 `- / +`，配置 1–22 级 AI；顶部按钮控制地图、生命数和道具。
- `←/→`：更换当前选中槽位的角色；`↑/↓`：更换颜色。
- `Enter`：自己准备/取消准备。两名真人都准备后自动开战。
- 房主 `[` / `]`：选地图；`-` / `+`：改生命数；`I`：开关道具。
- 房主 `F3/F4`：添加或删除 P3/P4 AI。
- 房主 `1/3/4`：选中自己、P3 AI 或 P4 AI；方向键换人/颜色，`,` / `.` 调整 AI 1–22 级。

21/22 级全角色候选目前只在 `Mogadishu`（楼房第二张图）训练过。其他地图可用 1–20 级 AI。

## 新电脑安装

仓库不存放 1.4GB 原始导出目录。首次运行联机客户端时，如果缺少 manifest，会从最新 GitHub Release 下载、校验并安装约 43MB 的 1× 运行素材包。也可手动执行：

```bash
python tools/install_runtime_assets.py
```

macOS/Linux：

```bash
python3 -m venv .venv-online
source .venv-online/bin/activate
python -m pip install -r requirements-online.txt
python -m online.client --server 'wss://服务器地址/ws' --name '玩家名'
```

Windows PowerShell：

```powershell
py -3.13 -m venv .venv-online
.\.venv-online\Scripts\Activate.ps1
python -m pip install -r requirements-online.txt
python -m online.client --server "wss://服务器地址/ws" --name "玩家名"
```

## 服务端边界

- 每客户端最多 120 条消息/秒，单条客户端消息最大 16KB。
- 房间无连接 3 分钟后清理，结束局 90 秒后清理。
- 重连 token 不会在公开房间状态中广播。
- 当前没有帐号、排位、观战、跨区匹配或完整回滚网码。当前是输入缓冲、确定性同步加快照校正的房间码版本。
