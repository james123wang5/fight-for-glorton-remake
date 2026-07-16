# iPhone / PWA 版

这个入口通过 Pygbag 将现有 Pygame 运行时编译为 WebAssembly。战斗规则、角色、
地图、菜单和素材清单仍共用 `src/`；只有 `mobile/main.py` 会打开触屏层和
2× 网页素材。普通 `python play.py` 不读取这些开关。

## 直接在 iPhone 玩（Mac 不用开）

已部署的私有 HTTPS 地址：

<https://glorton-mobile-fight.zeran-wang5.chatgpt.site>

这个版本从云端运行，Mac 合盖、关机或不在身边都不影响。站点使用独立的游戏账号
和密码，不要求朋友拥有 ChatGPT/OpenAI 账号。登录页勾选“记住我”后会在该设备保留
30天登录状态。

第一次在 iPhone 上使用：

1. 用 **Safari** 打开上面的地址。
2. 输入游戏账号和密码，并保持“记住我”勾选。
3. 首次会下载约 81 MiB，保持 Safari 在前台并等待游戏主菜单出现；不要在加载途中
   锁屏或切换应用。
4. 将手机横过来游玩。左侧是移动摇杆，右侧是三个动作键。
5. 确认完整进入过游戏后，在 Safari 点“共享”→“添加到主屏幕”。以后可以从主屏幕
   图标直接启动，不需要打开 Mac。

Android 使用 Chrome 打开同一个地址，输入相同账号密码，横屏进入游戏；浏览器菜单中
选择“安装应用”或“添加到主屏幕”即可作为PWA启动。触控、刘海/挖孔安全区、横屏和
2×设备像素比均共用同一套自适应逻辑。

第一次完整加载会把分块游戏包和运行文件写入浏览器缓存；之后即使短时断网也可尝试
从主屏幕启动。iOS 可能在空间不足或长期未使用时清理网站缓存，因此“随处联网运行”
是稳定能力，离线运行则以 Safari 是否仍保留缓存为准。更新版本后，联网打开一次并
等到主菜单，即可让缓存切换到新版本。

云端为紧凑的 1× 素材包，手机专用图片使用自适应调色板压缩以减少首次下载，但保留六个角色、四张地图、全部动画、1至20级原版规则AI，
以及21/22级轻量目的驱动AI；
本地仍可选择构建 2× Retina 素材包。云端部署与下面的局域网预览是两种不同入口。

## 控制

- 左侧虚拟摇杆：左右移动，下推下蹲/下穿，上推跳跃；回中后再上推触发二段跳。
- `ATK`：手刀/普通攻击。
- `SP`：手枪或角色特殊攻击。
- `DEF`：按住护盾，松手关闭。
- 上推与 `ATK`/`SP` 同时按下会走原有上手刀/上特殊技组合；方向与手刀的
  抱摔判定也没有另写规则。
- 右上角按钮：暂停/继续。

## 构建

```bash
python3 -m venv .venv-web
.venv-web/bin/python -m pip install -r requirements-web.txt
.venv-web/bin/python tools/build_mobile_bundle.py
```

构建器会在被 Git 忽略的 `build/mobile/` 中创建非破坏性副本，不会覆盖
`assets/`。它会把 4× 原素材缩为适合 Retina 输出的 2×，按内容哈希去掉重复帧，
然后生成 PWA manifest、Service Worker 和 iPhone 安全区页面。成功封包后会自动删除与
`app.apk` 重复的临时素材；需要调试中间目录时可加 `--keep-staging`。

仅重做素材副本：

```bash
.venv-web/bin/python tools/build_mobile_bundle.py --prepare-only
```

仅给已有 Pygbag 输出补 PWA 文件：

```bash
.venv-web/bin/python tools/build_mobile_bundle.py --postprocess-only
```

## 本地预览（仅开发调试）

```bash
.venv-web/bin/python -m http.server 8000 \
  --bind 0.0.0.0 \
  --directory build/mobile/app/build/web
```

Mac 上打开 `http://127.0.0.1:8000`。iPhone 与 Mac 在同一 Wi-Fi 时，可用
`ipconfig getifaddr en0` 查到 Mac 地址后访问 `http://<Mac-IP>:8000`做临时触屏测试。

这个局域网地址只用于开发，Mac 合盖后当然会失效；正常手机游玩请使用上面的 HTTPS
云端地址。

“添加到主屏幕”、Service Worker 和离线启动需要 HTTPS（`localhost` 例外不适用于
iPhone 访问 Mac 的局域网 IP）。正式测试时将 `build/mobile/app/build/web/` 整个
目录放到支持 HTTPS 和大文件的静态托管，再用 Safari 的“共享 → 添加到主屏幕”。

完整 6 角色/4 地图的 2× 包目前约 225 MiB，首次打开需要下载约 211 MiB 的
`app.apk`；成功运行一次后 Service Worker 才会把大包放入离线缓存。建议手机
预留至少 700 MiB 可用空间，首次下载时保持 Safari 在前台。

正式 HTTPS 手机站使用 `--asset-scale 1` 的紧凑包，保留全部角色、地图和动画，
以原版逻辑分辨率输出；2× Retina 包仍可在本地按上述默认命令构建。
