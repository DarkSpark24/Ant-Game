from __future__ import annotations

import json
import os
import struct
import sys
from typing import List

from logic.constant import row, col
from logic.gamedata import CellType, Farmer, MainGenerals, SubGenerals
from logic.gamestate import GameState
from ai import ai_func


def _log(msg: str) -> None:
    sys.stderr.write(f"[AI] {msg}\n")
    sys.stderr.flush()

def _recv_packet() -> bytes:
    hdr = sys.stdin.buffer.read(8)
    if len(hdr) < 8:
        return b""
    length, _target = struct.unpack(">Ii", hdr)
    if length <= 0:
        return b""
    return sys.stdin.buffer.read(length)


def _send_packet(payload: bytes, target: int = 0) -> None:
    header = struct.pack(">Ii", len(payload), target)
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


class AIProcess:
    def __init__(self, seat: int = 0):
        self.seat = int(seat)
        self.state = GameState()
        self.initialized = False
        self.ai = ai_func

    def _apply_init(self, rep: dict) -> None:
        ct = rep.get("Cell_type", "")
        if ct:
            for i in range(row * col):
                t = int(ct[i])
                self.state.board[i // col][i % col].type = CellType(t)
        for cell in rep.get("Cells", []):
            (x, y), owner, army = cell
            c = self.state.board[x][y]
            c.player = owner
            c.army = army
        self._rebuild_generals(rep.get("Generals", []))
        if "Coins" in rep:
            self.state.coin = list(rep["Coins"])  # type: ignore
        if "Tech_level" in rep:
            self.state.tech_level = [list(rep["Tech_level"][0]), list(rep["Tech_level"][1])]  # type: ignore
        if "Weapon_cds" in rep:
            self.state.super_weapon_cd = list(rep["Weapon_cds"])  # type: ignore
        self.state.round = int(rep.get("Round", 1))
        self.initialized = True

    def _apply_update(self, rep: dict) -> None:
        for cell in rep.get("Cells", []):
            (x, y), owner, army = cell
            c = self.state.board[x][y]
            c.player = owner
            c.army = army
        self._rebuild_generals(rep.get("Generals", []))
        if "Coins" in rep:
            self.state.coin = list(rep["Coins"])  # type: ignore
        if "Tech_level" in rep:
            self.state.tech_level = [list(rep["Tech_level"][0]), list(rep["Tech_level"][1])]  # type: ignore
        if "Weapon_cds" in rep:
            self.state.super_weapon_cd = list(rep["Weapon_cds"])  # type: ignore
        if "Round" in rep:
            self.state.round = int(rep["Round"])  # keep in sync

    def _rebuild_generals(self, gens: List[dict]) -> None:
        for i in range(row):
            for j in range(col):
                self.state.board[i][j].generals = None
        self.state.generals.clear()
        for g in gens:
            gid = int(g.get("Id", 0))
            owner = int(g.get("Player", -1))
            gtype = int(g.get("Type", 3))
            x, y = list(g.get("Position", [0, 0]))
            lvl = list(g.get("Level", [1, 1, 1]))
            scd = list(g.get("Skill_cd", [0, 0, 0, 0, 0]))
            srest = list(g.get("Skill_rest", [0, 0, 0]))
            alive = int(g.get("Alive", 1))
            if alive == 0:
                continue
            if gtype == 1:
                obj = MainGenerals(id=gid, player=owner)
                obj.produce_level = max(1, 2 * (int(lvl[0]) - 1))
                obj.defense_level = int(lvl[1])
                obj.mobility_level = max(1, int(lvl[2]))
            elif gtype == 2:
                obj = SubGenerals(id=gid, player=owner)
                obj.produce_level = max(1, 2 * (int(lvl[0]) - 1))
                obj.defense_level = int(lvl[1])
                obj.mobility_level = max(1, int(lvl[2]))
            else:
                obj = Farmer(id=gid, player=owner)
                obj.produce_level = int(lvl[0])
                if int(lvl[1]) == 2:
                    obj.defense_level = 1.5  # type: ignore[assignment]
                elif int(lvl[1]) >= 3:
                    obj.defense_level = int(lvl[1]) - 1
                else:
                    obj.defense_level = 1
                obj.mobility_level = 0
            obj.position = [x, y]
            obj.skills_cd = scd
            obj.skill_duration = srest
            self.state.generals.append(obj)
            self.state.board[x][y].generals = obj
            if owner != -1:
                self.state.board[x][y].player = owner

    @staticmethod
    def _ops_to_str(ops: List[List[int]]) -> str:
        lines: List[str] = []
        for op in ops:
            if not op:
                continue
            lines.append(" ".join(str(x) for x in op))
            if op[0] == 8:
                break
        if not lines or (lines and lines[-1] != "8"):
            # ensure end-turn present
            if not lines or lines[-1].split()[0] != "8":
                lines.append("8")
        return "\n".join(lines) + "\n"

    def loop(self) -> None:
        while True:
            data = _recv_packet()
            if not data:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            state_id = msg.get("state", 1)
            if state_id == 0:
                # round config, ignore
                continue

            content = msg.get("content", [])
            listen = msg.get("listen", [])

            # Initialize or update local state from content
            try:
                if isinstance(content, list) and len(content) >= 1:
                    if len(content) == 2:
                        reps = [json.loads(content[0]), json.loads(content[1])]
                        chosen = None
                        for rep in reps:
                            if int(rep.get("Player", -1)) == self.seat:
                                chosen = rep
                                break
                        if chosen is None:
                            chosen = reps[0]
                        self._apply_init(chosen)
                    else:
                        rep = json.loads(content[0])
                        if not self.initialized:
                            self._apply_init(rep)
                        else:
                            self._apply_update(rep)
            except Exception:
                # 不因状态解析失败而中断
                pass

            # Act only if we are in listen
            if isinstance(listen, list) and (self.seat in listen or int(self.seat) in listen):
                try:
                    ops = self.ai(self.state)
                except Exception:
                    ops = [[8]]
                out = {"player": self.seat, "content": self._ops_to_str(ops)}
                _send_packet(json.dumps(out).encode("utf-8"), target=0)


def main():
    seat = int(os.environ.get("AI_SEAT", "0"))
    AIProcess(seat=seat).loop()


if __name__ == "__main__":
    main()

