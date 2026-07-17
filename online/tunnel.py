from __future__ import annotations

import argparse
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
QUICK_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)


def extract_quick_tunnel_url(line: str) -> str | None:
    match = QUICK_TUNNEL_RE.search(line)
    return match.group(0).lower() if match else None


def websocket_url(public_url: str) -> str:
    return public_url.rstrip("/").replace("https://", "wss://", 1) + "/ws"


def client_command(python: str, public_url: str, name: str = "玩家名") -> str:
    return " ".join(
        [
            shlex.quote(python),
            "-m online.client --server",
            shlex.quote(websocket_url(public_url)),
            "--name",
            shlex.quote(name),
        ]
    )


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def _healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_health_url(port), timeout=timeout) as response:
            return response.status == 200 and b'"ok":true' in response.read(512)
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_health(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"联机服务器提前退出，退出码 {process.returncode}")
        if _healthy(port):
            return
        time.sleep(0.25)
    raise RuntimeError(f"联机服务器在 {timeout:.0f} 秒内没有通过健康检查")


def _reader(stream: TextIO, lines: queue.Queue[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            print(f"[tunnel] {line}", end="", flush=True)
            lines.put(line)
    finally:
        stream.close()


def _wait_for_tunnel_url(
    process: subprocess.Popen[str],
    lines: queue.Queue[str],
    timeout: float,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Cloudflare 隧道提前退出，退出码 {process.returncode}")
        try:
            line = lines.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        url = extract_quick_tunnel_url(line)
        if url:
            return url
    raise RuntimeError("30 秒内没有取得 trycloudflare.com 临时地址")


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _print_handoff(public_url: str, python: str) -> None:
    ws_url = websocket_url(public_url)
    host_command = client_command(python, public_url)
    portable_command = client_command("python", public_url)
    bar = "=" * 76
    print(f"\n{bar}")
    print("Glorton 临时公网联机已启动（完全免费，不需要 Cloudflare 账号）")
    print(f"健康检查: {public_url}/health")
    print(f"游戏服务器: {ws_url}")
    print("\n房主另开一个终端，复制下面两行；名字可以修改：")
    print(f"cd {shlex.quote(str(ROOT))}")
    print(host_command)
    print("\n发给朋友的命令（朋友先进入自己的项目目录并启用虚拟环境）：")
    print(portable_command)
    print("\n随后一人点击 CREATE ROOM，把 6 位房间码发给另一人。")
    print("这个终端必须保持运行；按 Ctrl+C 会同时关闭公网入口和本机服务器。")
    print("临时网址在每次重新启动后都会变化。Mac 合盖通常仍会断开。")
    print(f"{bar}\n", flush=True)


def run(port: int, tunnel_timeout: float = 30.0) -> int:
    cloudflared = shutil.which("cloudflared")
    if cloudflared is None:
        print("没有找到 cloudflared。macOS 可先运行: brew install cloudflared", file=sys.stderr)
        return 2

    server: subprocess.Popen[str] | None = None
    tunnel: subprocess.Popen[str] | None = None
    caffeinate: subprocess.Popen[str] | None = None
    reused_server = _healthy(port)
    try:
        if reused_server:
            print(f"复用已经运行的本机 Glorton 服务器: {_health_url(port)}")
        else:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "online.server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                text=True,
            )
            _wait_for_health(port, server, timeout=60.0)
            print(f"本机 Glorton 服务器已就绪: {_health_url(port)}")

        if sys.platform == "darwin" and shutil.which("caffeinate"):
            caffeinate = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )

        tunnel = subprocess.Popen(
            [
                cloudflared,
                "tunnel",
                "--url",
                f"http://127.0.0.1:{port}",
                "--no-autoupdate",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert tunnel.stdout is not None
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=_reader, args=(tunnel.stdout, lines), daemon=True).start()
        public_url = _wait_for_tunnel_url(tunnel, lines, tunnel_timeout)
        _print_handoff(public_url, sys.executable)

        while True:
            if tunnel.poll() is not None:
                raise RuntimeError(f"Cloudflare 隧道已断开，退出码 {tunnel.returncode}")
            if server is not None and server.poll() is not None:
                raise RuntimeError(f"Glorton 服务器已停止，退出码 {server.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在关闭临时公网联机……")
        return 0
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1
    finally:
        _terminate(tunnel)
        if not reused_server:
            _terminate(server)
        _terminate(caffeinate)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动免费的 Glorton Cloudflare 临时公网隧道")
    parser.add_argument("--port", type=int, default=8765, help="本机联机服务器端口")
    args = parser.parse_args()
    raise SystemExit(run(args.port))


if __name__ == "__main__":
    main()
