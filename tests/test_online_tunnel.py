from __future__ import annotations

import unittest

from online.tunnel import client_command, extract_quick_tunnel_url, websocket_url


class OnlineTunnelTests(unittest.TestCase):
    def test_extracts_cloudflare_quick_tunnel_url(self) -> None:
        line = "INF |  https://Kind-Tree-123.trycloudflare.com  |"
        self.assertEqual(
            extract_quick_tunnel_url(line),
            "https://kind-tree-123.trycloudflare.com",
        )

    def test_ignores_unrelated_cloudflare_log_lines(self) -> None:
        self.assertIsNone(extract_quick_tunnel_url("Registered tunnel connection"))

    def test_converts_public_health_origin_to_websocket_endpoint(self) -> None:
        self.assertEqual(
            websocket_url("https://kind-tree.trycloudflare.com/"),
            "wss://kind-tree.trycloudflare.com/ws",
        )

    def test_client_command_quotes_interpreter_paths_with_spaces(self) -> None:
        command = client_command(
            "/tmp/a project/.venv/bin/python",
            "https://kind-tree.trycloudflare.com",
        )
        self.assertIn("'/tmp/a project/.venv/bin/python'", command)
        self.assertIn("wss://kind-tree.trycloudflare.com/ws", command)


if __name__ == "__main__":
    unittest.main()
