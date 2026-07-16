# 安装、迁移与备份

## 新电脑快速安装

推荐使用 Python 3.10–3.13。普通游戏不需要 PyTorch。

```bash
git clone https://github.com/james123wang5/fight-for-glorton-remake.git
cd fight-for-glorton-remake
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/install_runtime_assets.py
python play.py
```

Windows PowerShell：

```powershell
git clone https://github.com/james123wang5/fight-for-glorton-remake.git
cd fight-for-glorton-remake
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/install_runtime_assets.py
python play.py
```

安装器会从最新 GitHub Release 下载 `glorton-runtime-assets-1x.zip`，校验一同发布的
SHA-256，再安全解压。Git 仓库本身保存代码、配置和正式 AI；精简动画资源单独发布，
避免每次拉取代码都下载约 1.4 GB 的开发导出物。

## 验证安装

```bash
python -m unittest discover -s tests -v
```

无显示器或 CI 可使用 SDL dummy 驱动进行启动冒烟检查：

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -c \
  "from src.runtime import RuntimeApp; app = RuntimeApp(); print(app.stage.name)"
```

## 安装训练与正式 21/22 级

训练依赖单独安装，不污染普通游戏环境：

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python -m training.play_v5
```

仓库只跟踪人工认可的 `champion_level21_model.zip`、
`champion_level22_model.zip` 和审批元数据。训练日志、候选、回放与自动存档默认忽略，
防止一次训练把仓库扩大数 GB。

## 两档素材

- Release 1×：约 35 MB 压缩包，可完整游玩，适合迁移、普通电脑和快速安装。
- 本地完整导出：约 1.4 GB，保留高分辨率和中间资源，适合逐帧校准与 Retina/HIGH 开发。

精简包由已有 Pygbag 构建中的实际依赖生成：

```bash
python tools/package_runtime_assets.py
```

如果不使用 GitHub，也可以把 ZIP 拷到另一台电脑后离线安装：

```bash
python tools/install_runtime_assets.py \
  --archive /path/to/glorton-runtime-assets-1x.zip \
  --sha256 <发布页记录的SHA-256>
```

## 不应提交或复制的内容

- `.venv*`、`node_modules`：与操作系统和 Python/Node 版本绑定，应该在新电脑重新安装。
- `training/logs`、重复的每轮模型、临时回放：只保留有人工验收价值的结果。
- `artifacts/visual`：视觉差异证据可单独归档，不要混入普通代码提交。
- `build/mobile`、站点 `dist`：属于可重建输出，不是源码。

原游戏素材权利仍归原作者及相关权利人所有。公开发布精简运行包前，应确认拥有相应的
再分发授权。
