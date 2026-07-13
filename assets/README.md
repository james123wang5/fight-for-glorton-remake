# 本地生成资源目录

此目录中的运行资源由项目工具从使用者本地合法取得的
`fight-for-glorton.swf` 和 FFDec 导出物生成，不随公开 Git 仓库分发。

在项目根目录依次运行：

```bash
python tools/export_highres_peach.py
python tools/export_menu_assets.py
python tools/export_sound_assets.py
python tools/build_manifest.py
python tools/build_stages.py
```

生成内容包括角色帧、地图、菜单、特效、音频、字体和运行清单。除本说明文件外，
`assets/` 下的内容均被 `.gitignore` 排除。
