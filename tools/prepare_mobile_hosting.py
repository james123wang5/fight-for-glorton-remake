from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_BUILD = ROOT / "build" / "mobile" / "app" / "build" / "web"
PUBLIC_GAME = ROOT / "mobile" / "site-host" / "public" / "game"
CHUNK_SIZE = 8 * 1024 * 1024


def _bootstrap_html(download_mib: int) -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,height=device-height,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
  <meta name="theme-color" content="#000000">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Glorton">
  <link rel="manifest" href="manifest.webmanifest">
  <link rel="apple-touch-icon" href="apple-touch-icon.png">
  <title>The Fight for Glorton</title>
  <style>
    html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#050608;color:#fff;font-family:-apple-system,BlinkMacSystemFont,sans-serif}
    main{height:100%;display:grid;place-content:center;text-align:center;padding:24px;box-sizing:border-box}
    h1{margin:0 0 12px;font-size:clamp(24px,6vw,48px)}
    p{margin:4px 0;color:#cbd5e1;line-height:1.5}.bar{width:min(72vw,360px);height:8px;margin:24px auto 0;border-radius:8px;background:#222a35;overflow:hidden}
    .bar::after{display:block;width:42%;height:100%;border-radius:inherit;background:#ef4444;animation:load 1.15s ease-in-out infinite alternate;content:""}@keyframes load{to{transform:translateX(140%)}}
  </style>
</head>
<body>
  <main><div><h1>GLORTON</h1><p id="status">正在准备手机版…</p><p>首次需下载约 __DOWNLOAD_MIB__ MiB，请保持 Safari 在前台。</p><div class="bar"></div></div></main>
  <script>
    (async () => {
      const status = document.getElementById("status");
      try {
        if (!("serviceWorker" in navigator)) {
          location.replace("./game.html");
          return;
        }
        await navigator.serviceWorker.register("./sw.js", {scope: "./"});
        await navigator.serviceWorker.ready;
        if (!navigator.serviceWorker.controller) {
          status.textContent = "正在安装离线组件…";
          location.reload();
          return;
        }
        location.replace("./game.html");
      } catch (error) {
        status.textContent = "启动失败，请检查网络后刷新。";
        console.error(error);
      }
    })();
  </script>
</body>
</html>
""".replace("__DOWNLOAD_MIB__", str(download_mib))


def _service_worker(build_id: str, chunks: list[dict[str, int]], total_size: int) -> str:
    chunk_names = json.dumps([item["name"] for item in chunks], separators=(",", ":"))
    return f"""const CACHE_NAME = "glorton-hosted-{build_id}";
const SHELL = ["./", "./index.html", "./game.html", "./manifest.webmanifest", "./apple-touch-icon.png", "./favicon.png"];
const CHUNKS = {chunk_names};
const APK_SIZE = {total_size};
const PYGAME_CDN = "pygame-web.github.io";

self.addEventListener("install", (event) => {{
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
}});

self.addEventListener("activate", (event) => {{
  event.waitUntil(caches.keys().then((keys) => Promise.all(
    keys.filter((key) => key.startsWith("glorton-hosted-") && key !== CACHE_NAME).map((key) => caches.delete(key))
  )));
  self.clients.claim();
}});

async function chunkResponse(name) {{
  const cache = await caches.open(CACHE_NAME);
  const request = new Request(new URL(name, self.registration.scope), {{credentials: "same-origin"}});
  let response = await cache.match(request);
  if (!response) {{
    response = await fetch(request);
    if (!response.ok) throw new Error(`Chunk ${{name}} returned ${{response.status}}`);
    await cache.put(request, response.clone());
  }}
  return response;
}}

function streamedApk() {{
  const body = new ReadableStream({{
    async start(controller) {{
      try {{
        for (const name of CHUNKS) {{
          const response = await chunkResponse(name);
          if (!response.body) throw new Error(`Chunk ${{name}} has no body`);
          const reader = response.body.getReader();
          while (true) {{
            const {{done, value}} = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }}
        }}
        controller.close();
      }} catch (error) {{
        controller.error(error);
      }}
    }}
  }});
  return new Response(body, {{
    status: 200,
    headers: {{
      "Content-Type": "application/octet-stream",
      "Content-Length": String(APK_SIZE),
      "Cache-Control": "no-store"
    }}
  }});
}}

self.addEventListener("fetch", (event) => {{
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin === self.location.origin && url.pathname.endsWith("/game/app.apk")) {{
    event.respondWith(streamedApk());
    return;
  }}
  if (request.headers.has("range")) return;
  const cacheable = url.origin === self.location.origin || url.hostname === PYGAME_CDN;
  if (!cacheable) return;
  event.respondWith(caches.match(request).then((cached) => cached || fetch(request).then((response) => {{
    if (response.ok || response.type === "opaque") {{
      const copy = response.clone();
      caches.open(CACHE_NAME).then((cache) => cache.put(request, copy)).catch(() => {{}});
    }}
    return response;
  }}).catch(() => caches.match("./index.html"))));
}});
"""


def prepare() -> dict[str, object]:
    apk = WEB_BUILD / "app.apk"
    if not apk.is_file():
        raise SystemExit("Mobile bundle missing. Run tools/build_mobile_bundle.py first.")
    if PUBLIC_GAME.exists():
        shutil.rmtree(PUBLIC_GAME)
    chunks_dir = PUBLIC_GAME / "chunks"
    chunks_dir.mkdir(parents=True)

    for name in ("manifest.webmanifest", "apple-touch-icon.png", "favicon.png"):
        shutil.copy2(WEB_BUILD / name, PUBLIC_GAME / name)
    shutil.copy2(WEB_BUILD / "index.html", PUBLIC_GAME / "game.html")

    digest = hashlib.sha256()
    chunks: list[dict[str, int]] = []
    total_size = 0
    with apk.open("rb") as source:
        index = 0
        while data := source.read(CHUNK_SIZE):
            digest.update(data)
            total_size += len(data)
            name = f"chunks/app-{index:03d}.bin"
            (PUBLIC_GAME / name).write_bytes(data)
            chunks.append({"name": name, "size": len(data)})
            index += 1
    build_id = digest.hexdigest()[:12]
    download_mib = round(total_size / 1024 / 1024)
    (PUBLIC_GAME / "index.html").write_text(
        _bootstrap_html(download_mib), encoding="utf-8"
    )
    (PUBLIC_GAME / "sw.js").write_text(
        _service_worker(build_id, chunks, total_size), encoding="utf-8"
    )
    report: dict[str, object] = {
        "build_id": build_id,
        "sha256": digest.hexdigest(),
        "apk_size": total_size,
        "chunk_size": CHUNK_SIZE,
        "chunks": chunks,
    }
    (PUBLIC_GAME / "chunks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = prepare()
    print(
        f"Prepared hosted PWA {result['build_id']}: "
        f"{len(result['chunks'])} chunks, {result['apk_size']} bytes"
    )
