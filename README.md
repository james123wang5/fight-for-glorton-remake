# 疯狂小人战斗复刻版（Glorton Remake）

这是使用 Python 和 Pygame 对 Flash 网页游戏 *The Fight for Glorton*
进行的非官方复刻项目。项目以本地 SWF、FFDec 导出的 ActionScript、时间轴、
变换矩阵和矢量素材为依据，目标是还原原版菜单、角色、地图与战斗手感，而不是
根据截图猜测参数。

> 当前仍是开发版本，并非 100% 完成。联网对战尚未接入，部分动作衔接、打击反馈、
> 摄像机细节和画面表现仍需要继续与原版逐项校准。

## 当前进度

- 主流程：启动、主菜单、单人/多人设置、选人、选地图、倒计时、暂停、结算和返回主菜单。
- 角色：SBL、Peach、Trash、Coffee、Default、Aubergine，以及原版四种配色。
- 地图：Rooftop、Mogadishu、B52、Space，包括固定/移动平台、出生点和主要机关。
- 战斗：移动、二段跳、下蹲/快速下落、护盾、闪避、抱摔、攻击、投射物、伤害、击退、死亡和复活。
- 模式：1 至 4 个槽位、生命、计时和耐力规则，以及本地双人控制和 CPU AI。
- 表现：原版音效事件、Flash 时间轴动画、Retina 原生像素倍率检测、设备像素对齐和独立的 HIGH 超采样输出。
- 校验：固定种子、输入录像/回放、99 项视觉覆盖矩阵、逐帧差异热图，以及覆盖战斗、角色、地图、菜单、音频、AI、HUD、结算和完整流程的自动化测试。
- 战斗核心：`BattleSimulation.step(inputs)` 以 25 ms 固定步长无画面运行，可生成 JSON 状态快照、摘要、录像并严格回放。

完整的操作组合、角色重量/速度、普通与特殊攻击、投射物、连击伤害、击退公式、
手雷/地雷、爆炸半径、复活、计分和 AI 条件见
[`docs/COMBAT_SYSTEM_CN.md`](docs/COMBAT_SYSTEM_CN.md)。

SWF 中注册了 `RedGuy`，但原版根时间轴和可选角色列表都没有使用它，因此当前没有
将其加入可玩角色。

## 运行环境

- Python 3.10 或更高版本
- Pygame 2.6 或更高版本
- Pillow 10 或更高版本
- macOS、Windows 或 Linux（当前主要在 macOS 上开发和验证）

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 使用：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

准备好本地生成的 `assets/` 后启动游戏：

```bash
python play.py
```

打开 FFDec 清单检查器：

```bash
python main.py
```

## 默认按键

| 动作 | 玩家 1 | 玩家 2 |
| --- | --- | --- |
| 左右移动 | `A` / `D` | `←` / `→` |
| 跳跃 | `W` | `↑` |
| 下蹲/快速下落 | `S` | `↓` |
| 普通攻击 | `J` | 小键盘 `0` |
| 特殊攻击 | `K` | 小键盘 `1` |
| 护盾 | `Shift` | 小键盘 `2` |

`P` 暂停，`F1` 显示碰撞与边界调试层，`Esc` 退出。四个本地槽位的按键
也可以在游戏的 Controls 页面重新设置。

## 原版素材说明

公开仓库**不包含**从原版 SWF 导出的图片、字体、声音、ActionScript 导出物和生成清单。
这些文件约 1.4 GB，版权归原作者及相关权利人所有。仓库只保存复刻代码、测试和资源
构建工具；使用者需要自行合法取得原版 SWF，并在本地生成资源。

现有构建脚本默认采用以下相邻目录结构：

```text
pythonpygame/
├── fight-for-glorton.swf
├── FFDec.app/
├── glorton_peach/
│   ├── raw_ffdec_export/
│   ├── raw_ffdec_export_scripts/
│   └── raw_ffdec_xml/
└── glorton_remake/
```

在 FFDec 导出物准备完成后，于项目根目录依次运行：

```bash
python tools/export_highres_peach.py
python tools/export_menu_assets.py
python tools/export_sound_assets.py
python tools/build_manifest.py
python tools/build_stages.py
```

这些脚本会创建运行所需的 `assets/`。具体说明见
[`assets/README.md`](assets/README.md)。

## 测试

运行完整测试：

```bash
python -m unittest discover -s tests -v
```

运行 ActionScript 覆盖审计（需要相邻的 `glorton_peach` FFDec 导出目录）：

```bash
python tools/audit_source_coverage.py
```

## 逐帧视觉基准

固定测试定义在 `tools/visual_scenarios.json`，默认种子为 `20260714`、画布为
600×400、战斗步长为 25 ms。覆盖矩阵会自动从清单展开为 99 项：菜单、倒计时、
6 个角色×4 种颜色、6 个角色×8 种攻击、6 个受击序列、2 种道具和 4 张地图。

先准备原版 SWF 的本地 Ruffle 运行器：

```bash
python tools/visual_regression.py prepare-original \
  --swf ../fight-for-glorton.swf \
  --ruffle artifacts/visual/runtime/ruffle
python tools/visual_regression.py prepare-ko-fixture \
  --swf ../fight-for-glorton.swf \
  --ffdec ../FFDec.app/Contents/Resources/ffdec \
  --scripts ../glorton_peach/raw_ffdec_export_scripts/scripts
python -m http.server 8766 --bind 127.0.0.1 \
  --directory artifacts/visual/original_runner
```

在 Chrome 打开 `http://127.0.0.1:8766/?seed=20260714`，按
`tools/visual_scenarios.json` 的坐标与按键采集原版帧到
`artifacts/visual/original/<case>/frame_XXXXXX.png`。这里的原版基准由 Ruffle
模拟执行，并不等同于仍能运行 Adobe Flash Player 的机器；查询参数会记录种子，
但 Ruffle 中的 AVM1 `random()` 目前不能从外部注入种子，因此需要逐像素比较的原版
用例会避开随机道具生成，随机组合另做状态/容差校验。

`koAttack` 是一个特殊的源极端状态：原版 `Fighter.DoCommon` 会在
`Fighter.Attack` 满足 KO 反击条件之前先退出 KO，因此正常输入无法到达这个标签。
`prepare-ko-fixture` 只在 SWF 副本的第 51 帧增加 `U` 键监听，调用原类自身的
`AnimateAttack("koAttack")`，不替换动画或角色素材。FFDec 批量回编译会令这个夹具
的部分无关 HUD 值显示为 `undefined`，报告会把这 6 项单独标记为源时间轴夹具，不能
当成普通运行帧。

生成完整的重制版 HIGH 基准、差异热图和覆盖报告：

```bash
python tools/visual_regression.py record-remake --suite full --quality HIGH
python tools/visual_regression.py normalize-original --pixel-ratio 2
python tools/visual_regression.py compare \
  --reference artifacts/visual/original \
  --actual artifacts/visual/remake \
  --output artifacts/visual/diff
python tools/visual_regression.py coverage
```

报告分别位于 `artifacts/visual/diff/index.html`、
`artifacts/visual/diff/report.json` 和 `artifacts/visual/coverage.json`。每个攻击与
受击目录还包含 `recording.json`，用于证明相同输入可以严格回放到相同最终状态。

## 无画面战斗核心

服务器、回归测试或后续训练环境可以直接推进固定步长模拟，不需要打开游戏窗口：

```python
from src.simulation import BattleSimulation

simulation = BattleSimulation.headless(seed=20260714)
simulation.start_recording({"match": "smoke"})
snapshot = simulation.step([{"right": True}, {}])
snapshot = simulation.step([{"punch_pressed": True}, {}])
recording = simulation.stop_recording()
simulation.replay(recording)  # strict=True 时同时校验初始与最终 SHA-256
```

输入字段固定为左右、上方向轨迹、下蹲、跳跃/普通攻击/特殊攻击/护盾的按下沿和
护盾释放沿。快照包含角色、地图时间、摄像机、道具、投射物、爆炸、随机数状态和
上一帧输入；录像还保存实体为空的初始快照，因此可以从自定义站位严格恢复并回放。
当前 `restore_snapshot()` 明确拒绝在恢复边界已经存在道具、投射物或爆炸的快照，
而不会静默构造不完整对象；这些动态实体可以在恢复后的输入步中正常生成。它与渲染
循环完全分开，方便后续接权威服务器、状态校正和录像回放。

## AI 训练

已经提供独立的 Gymnasium/PPO 训练入口：固定 P1 桃子对原版逻辑 20 级 P2 桃子、
Mogadishu、3 条命、180 秒上限并关闭道具。奖励以击出界和最终胜负为主，伤害仅作为
较弱的塑形信号；动作包含移动、跳跃、手刀、枪、上特殊火箭和护盾的同时输入组合。

训练依赖、模型和日志均与普通游戏隔离，运行 `python play.py` 不会进入训练规则。
安装、基准、短训练、正式训练、续训、评估和录像命令见
[`training/README.md`](training/README.md)。
训练完成后可用 `python -m training.play_level21` 在菜单中临时开放 21 级模型，
原版 1–20 级保持不变。

第二阶段提供 21/22 双模型轮流自对战：使用真人式按键边沿与反应延迟，随机双方槽位，
加入追击/回场课程、部分道具局，以及只在真实命中时计算的抱摔、上手刀、枪与火箭时机
奖励。运行 `python -m training.train_league` 会同时更新并分别保存两套模型；训练后用
`python -m training.play_league` 临时开放 21/22 级。普通 `python play.py` 仍只使用
原版 1–20 级逻辑。

`HIGH` 不再是 `MEDIUM` 的别名：战斗画面先按至少 2×（Retina 上采用系统 backing
scale）渲染，再缩回最终窗口；`MEDIUM` 使用单次平滑缩放，`LOW` 使用最近邻缩放。
可用 `GLORTON_PIXEL_RATIO=2` 覆盖自动检测，便于无 Retina 的 CI 做相同验证。

## iPhone / PWA

无需打开 Mac 的私有云端版已经部署：

<https://glorton-mobile-fight.zeran-wang5.chatgpt.site>

在 iPhone 的 Safari 中打开，用部署站点的同一个 ChatGPT/OpenAI 账号登录，保持页面
在前台完成约 81 MiB 的首次下载；进入主菜单后即可“共享 → 添加到主屏幕”。以后从
主屏幕图标启动，Mac 可以合盖或关机。完整安装、离线缓存和更新说明见
[`mobile/README.md`](mobile/README.md)。

手机版不是另一套战斗：Pygbag/WebAssembly 入口直接复用 `src/`，只在
`GLORTON_MOBILE=1` 时加入左摇杆、`ATK`/`SP`/`DEF` 三键、暂停键、iPhone 安全区、
横屏提示和 PWA 离线缓存。云端使用紧凑 1× 素材，本地可构建 2× Retina 素材；原有 `python play.py`、
1–20 级 AI 和 21/22 级训练入口不会读取这个开关。

## 目录结构

```text
glorton_remake/
├── play.py                 # 游戏入口
├── main.py                 # FFDec 清单检查器入口
├── src/
│   ├── runtime.py          # 固定步长战斗、角色、地图、摄像机和渲染
│   ├── simulation.py       # 无画面固定步长、种子、快照、录像和回放
│   ├── display.py          # Retina 检测、最终缩放和设备像素对齐
│   ├── mobile_controls.py  # 仅手机入口启用的多点触控层
│   ├── menu.py             # 原版菜单与流程
│   ├── audio.py            # 声音注册和播放
│   ├── assets.py           # 延迟加载和内存受限缓存
│   └── visual_check.py     # 资源/时间轴检查工具
├── tests/                  # 对照与回归测试
├── training/               # 独立 Gymnasium/PPO 训练、基准与评估
├── mobile/                 # 隔离的 Pygbag/PWA 入口与文档
├── tools/                  # FFDec 资源导出和清单构建工具
└── assets/                 # 本地生成，不提交到 Git
```

## 后续计划

1. 继续用逐帧报告校准极端组合动作、镜头和音效触发帧。
2. 加入房间模式、双端输入同步、预测与回滚联网对战。
3. 改进并评估 CPU AI。

## 版权与用途

本项目为学习、研究和游戏复刻实践用途，与原游戏作者及 Armor Games 无官方关联。
项目名称、角色、美术、音频和原游戏内容的权利归各自权利人所有。请勿将原版素材用于
商业用途，也不要在未获得授权的情况下重新分发原版资源。
