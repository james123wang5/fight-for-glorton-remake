from __future__ import annotations

import unittest

from online.protocol import (
    ProtocolError,
    controls_without_edges,
    decode_message,
    normalize_controls,
    room_code,
    validate_lobby_patch,
)
from online.room import RoomManager


class OnlineProtocolTests(unittest.TestCase):
    def test_protocol_rejects_non_objects_and_bad_room_codes(self) -> None:
        with self.assertRaises(ProtocolError):
            decode_message("[]")
        with self.assertRaises(ProtocolError):
            room_code("O0BAD!")
        self.assertEqual(room_code("abc234"), "ABC234")

    def test_controls_clear_one_shot_edges_when_a_packet_is_missing(self) -> None:
        controls = normalize_controls(
            {"left": True, "punch_pressed": True, "shield_pressed": True}
        )
        held = controls_without_edges(controls)
        self.assertTrue(held["left"])
        self.assertFalse(held["punch_pressed"])
        self.assertFalse(held["shield_pressed"])

    def test_lobby_patch_clamps_level_and_rejects_unknown_fighter(self) -> None:
        self.assertEqual(validate_lobby_patch({"level": 99})["level"], 22)
        with self.assertRaises(ProtocolError):
            validate_lobby_patch({"fighter": "UnknownPlayer"})


class OnlineRoomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.manager = RoomManager()
        self.sent: dict[int, list[dict]] = {0: [], 1: []}

        async def send0(message: dict) -> None:
            self.sent[0].append(message)

        async def send1(message: dict) -> None:
            self.sent[1].append(message)

        self.room, self.host_token, _slot = self.manager.create("Host")
        self.room.add_connection(self.host_token, 0, send0)
        self.guest_token, _ = self.room.join("Guest")
        self.room.add_connection(self.guest_token, 1, send1)

    async def test_both_humans_see_each_others_selection(self) -> None:
        await self.room.update_lobby(
            1,
            {"slot": 1, "patch": {"fighter": "CoffeePlayer", "color": 3}},
        )
        self.assertEqual(self.room.slots[1].fighter, "CoffeePlayer")
        self.assertEqual(self.sent[0][-1]["slots"][1]["fighter"], "CoffeePlayer")
        self.assertEqual(self.sent[1][-1]["slots"][1]["color"], 3)

    async def test_guest_cannot_edit_host_or_ai(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot_edit_other_human"):
            await self.room.update_lobby(1, {"slot": 0, "patch": {"color": 2}})
        with self.assertRaisesRegex(ValueError, "host_only"):
            await self.room.update_lobby(1, {"slot": 2, "patch": {"enabled": True}})

    async def test_ready_requires_two_connected_humans(self) -> None:
        await self.room.update_lobby(0, {"slot": 0, "patch": {"ready": True}})
        self.assertFalse(self.room.can_start())
        await self.room.update_lobby(1, {"slot": 1, "patch": {"ready": True}})
        self.assertTrue(self.room.can_start())

    async def test_input_delay_and_edge_debounce(self) -> None:
        self.room.phase = "playing"
        self.room.server_tick = 10
        self.room.queue_input(
            0,
            {
                "seq": 1,
                "tick": 12,
                "controls": {"right": True, "punch_pressed": True},
            },
        )
        first = self.room._controls_for_tick(0, 12)
        missing = self.room._controls_for_tick(0, 13)
        self.assertTrue(first["right"] and first["punch_pressed"])
        self.assertTrue(missing["right"])
        self.assertFalse(missing["punch_pressed"])

    async def test_disconnect_token_resumes_the_same_slot(self) -> None:
        resumed: list[dict] = []

        async def send(message: dict) -> None:
            resumed.append(message)

        self.room.disconnect(self.guest_token)
        self.assertFalse(self.room.slots[1].connected)
        slot = self.room.resume(self.guest_token, send)
        self.assertEqual(slot, 1)
        self.assertTrue(self.room.slots[1].connected)

    async def test_enabling_p4_also_keeps_p3_active(self) -> None:
        await self.room.update_lobby(0, {"slot": 3, "patch": {"enabled": True}})
        self.assertTrue(self.room.slots[2].enabled)
        self.assertTrue(self.room.slots[3].enabled)

    def test_public_room_state_never_exposes_reconnect_tokens(self) -> None:
        state = self.room.public_state()
        self.assertNotIn("token", state["slots"][0])
        self.assertNotIn("token", state["slots"][1])


if __name__ == "__main__":
    unittest.main()
