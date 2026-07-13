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
- 表现：原版音效事件、Flash 时间轴动画、高分辨率矢量栅格化和按内存上限加载的图片缓存。
- 校验：覆盖战斗、角色、地图、菜单、音频、AI、HUD、结算和完整流程的自动化测试。

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

## 目录结构

```text
glorton_remake/
├── play.py                 # 游戏入口
├── main.py                 # FFDec 清单检查器入口
├── src/
│   ├── runtime.py          # 固定步长战斗、角色、地图、摄像机和渲染
│   ├── menu.py             # 原版菜单与流程
│   ├── audio.py            # 声音注册和播放
│   ├── assets.py           # 延迟加载和内存受限缓存
│   └── visual_check.py     # 资源/时间轴检查工具
├── tests/                  # 对照与回归测试
├── tools/                  # FFDec 资源导出和清单构建工具
└── assets/                 # 本地生成，不提交到 Git
```

## 后续计划

1. 继续校准跑跳、组合攻击、受击翻滚、击退和抱摔条件。
2. 完成摄像机、出生动画、HUD、菜单动画和原版画质对照。
3. 抽离可序列化的固定步长战斗模拟。
4. 加入房间模式、双端输入同步、预测与回滚联网对战。
5. 改进并评估 CPU AI。

## 版权与用途

本项目为学习、研究和游戏复刻实践用途，与原游戏作者及 Armor Games 无官方关联。
项目名称、角色、美术、音频和原游戏内容的权利归各自权利人所有。请勿将原版素材用于
商业用途，也不要在未获得授权的情况下重新分发原版资源。

