"""
AI that uses a trained reduced-action policy saved under AI/selfplay/model.pt.

Interface:
  - Exposes function `policy(round_idx, my_seat, state)` returning a command list
    compatible with logic.ai2logic (list of lists), ending with [8] to finish turn.

Weights:
  - Default path: AI/selfplay/model.pt
  - To update: run `python SDK/train_selfplay.py --save AI/selfplay/model.pt`
"""

from typing import List
import os

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

import numpy as np

from SDK.policies import obs_to_tensor as sdk_obs_to_tensor
from logic.constant import row, col
from logic.map import PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP
from logic.gamedata import Direction
from logic.gamestate import GameState


MODEL_PATH = os.environ.get("SELFPLAY_MODEL", os.path.join("AI", "selfplay", "model.pt"))
MAX_ARMY_OBS = 10000


if nn is None:
    class SmallPolicy:  # type: ignore[override]
        def __init__(self):
            raise RuntimeError("PyTorch not installed. Please install torch to load the self-play policy model.")
else:
    class SmallPolicy(nn.Module):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            inp = row * col * 4 + 2 + 2 + 1
            hid = 256
            self.net = nn.Sequential(
                nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 9 + row + col + 4 + 5 + 10)
            )

        def forward(self, x):
            return self.net(x)


def _build_obs(state: GameState, player: int):
    owner = np.empty((row, col), dtype=np.int8)
    army = np.empty((row, col), dtype=np.int32)
    terrain = np.empty((row, col), dtype=np.int8)
    generals_owner = np.full((row, col), -1, dtype=np.int8)
    for i in range(row):
        for j in range(col):
            cell = state.board[i][j]
            owner[i, j] = cell.player
            army[i, j] = min(cell.army, MAX_ARMY_OBS)
            terrain[i, j] = int(cell.type)
            if cell.generals is not None:
                generals_owner[i, j] = cell.generals.player
    return {
        "board_owner": owner,
        "board_army": army,
        "board_terrain": terrain,
        "generals_owner": generals_owner,
        "coins": np.array(state.coin, dtype=np.int32),
        "rest_moves": np.array(state.rest_move_step, dtype=np.int32),
        "current_player": int(player),
    }


def _obs_to_tensor(obs) -> np.ndarray:
    return sdk_obs_to_tensor(obs)


def _pick_action(policy: SmallPolicy, obs) -> np.ndarray:
    x = torch.from_numpy(_obs_to_tensor(obs)).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy(x)[0]
        # Use similar logic as logits_to_action
        type_logits = logits[:9]
        x_logits = logits[9:9+row]
        y_logits = logits[9+row:9+row+col]
        dir_logits = logits[9+row+col:9+row+col+4]
        num_logits = logits[9+row+col+4:9+row+col+4+5]
        gid_logits = logits[9+row+col+4+5:]

        type_probs = torch.softmax(type_logits, dim=-1)
        atype = torch.multinomial(type_probs, 1).item()

        act = np.zeros(8, dtype=np.int64)
        if atype == 0:
            act[0] = 0
            return act

        act[0] = atype

        x_probs = torch.softmax(x_logits, dim=-1)
        x = torch.multinomial(x_probs, 1).item()
        y_probs = torch.softmax(y_logits, dim=-1)
        y = torch.multinomial(y_probs, 1).item()
        dir_probs = torch.softmax(dir_logits, dim=-1)
        d = torch.multinomial(dir_probs, 1).item()
        num_probs = torch.softmax(num_logits, dim=-1)
        n = torch.multinomial(num_probs, 1).item() + 1
        gid_probs = torch.softmax(gid_logits, dim=-1)
        gid = torch.multinomial(gid_probs, 1).item()

        if atype == 1:  # ArmyMove
            act[1] = x
            act[2] = y
            act[3] = d + 1
            act[4] = n
        elif atype == 2:  # GeneralMove
            act[1] = gid
            act[2] = x
            act[3] = y
        elif atype == 3:  # LevelUp
            act[1] = gid
            act[2] = (d % 3) + 1
        elif atype == 4:  # GeneralSkill
            act[1] = gid
            act[2] = (d % 2) + 1
            act[3] = x
            act[4] = y
        elif atype == 5:  # TechUpdate
            act[1] = d
        elif atype == 6:  # SuperWeapon
            act[1] = (d % 4) + 1
            act[2] = x
            act[3] = y
        elif atype == 7:  # CallGenerals
            act[1] = x
            act[2] = y
        elif atype == 8:  # GiveUp
            pass
        return act


def _main_gen(state: GameState, player: int):
    for g in state.generals:
        if type(g).__name__ == "MainGenerals" and g.player == player:
            return g
    return None


def policy(round_idx: int, my_seat: int, state: GameState) -> List[List[int]]:
    # intercept nearby ants before consulting the neural policy
    base = PLAYER_0_BASE_CAMP if my_seat == 0 else PLAYER_1_BASE_CAMP
    for ant in getattr(state, "ants", []):
        if ant.player != my_seat:
            dist = state.map.distance(ant.pos, base)
            if dist <= 2:
                g = _main_gen(state, my_seat)
                if g and state.board[g.position[0]][g.position[1]].army > 1 and state.rest_move_step[my_seat] > 0:
                    x, y = g.position
                    if ant.pos[0] < x:
                        d = Direction.UP
                    elif ant.pos[0] > x:
                        d = Direction.DOWN
                    elif ant.pos[1] < y:
                        d = Direction.LEFT
                    else:
                        d = Direction.RIGHT
                    num = min(3, state.board[x][y].army - 1)
                    return [[1, x, y, int(d) + 1, num], [8]]
    # Fallback if torch or model unavailable
    if torch is None or not os.path.exists(MODEL_PATH):
        return [[8]]
    # Load once and cache on module
    global _POLICY
    try:
        _ = _POLICY  # type: ignore[name-defined]
    except Exception:
        _POLICY = SmallPolicy()  # type: ignore[assignment]
        _POLICY.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _POLICY.eval()
    obs = _build_obs(state, my_seat)
    action_vector = _pick_action(_POLICY, obs)  # type: ignore[arg-type]
    # Map to commands
    if action_vector[0] == 0:
        return [[8]]
    else:
        return [action_vector.tolist(), [8]]

def ai_func(state: GameState) -> List[List[int]]:
    """Adapter for controller-style API (single-state -> ops)."""
    round_idx = getattr(state, "round", 0) + 1
    return policy(round_idx, 0, state)
