import json
import random
import struct
import sys
import time

from logic.ai2logic import execute_single_command
from logic.constant import *
from logic.gamedata import Farmer, MainGenerals, SubGenerals, SuperWeapon, WeaponType
from logic.gamestate import GameState, init_generals, update_round
from logic.generate_round_replay import get_single_round_replay
from logic.game_rules import is_game_over  # use canonical rule implementation

error_map = ["RE", "TLE", "OLE"]


# 发送数据包给 Judger
def send_to_judger(data, target=-1):
    try:
        length = len(data)
        # 逻辑发给 Judger 的头是 8 字节: length(4) + target(4)
        header = struct.pack(">Ii", length, target)
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(data)
        sys.stdout.flush() # 【必须】立即刷新缓冲区
    except Exception:
        sys.stderr.write("IO Error in send_to_judger:\n")
        sys.stderr.write(traceback.format_exc())


def receive_from_judger():
    try:
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            return b""
        length = struct.unpack(">I", header)[0]
        data = sys.stdin.buffer.read(length)
        return data
    except Exception:
        sys.stderr.write("IO Error in receive_from_judger:\n")
        sys.stderr.write(traceback.format_exc())
        return b""


# judger 向逻辑发送初始化信息
def receive_init_info():
    # 接收来自 Judger 的字节流数据
    init_info_bytes = receive_from_judger()
    # 将字节流解码成字符串
    init_info_str = init_info_bytes.decode("utf-8")
    # 将JSON格式的字符串解析为 python 数据类型
    init_info = json.loads(init_info_str)
    return init_info


# 逻辑向judger发送回合配置信息
def send_round_config(time: int, length: int):
    round_config = {"state": 0, "time": time, "length": length}
    round_config_str = json.dumps(round_config)
    round_config_bytes = round_config_str.encode("utf-8")
    send_to_judger(round_config_bytes)


# 逻辑向judger发生地图信息
def send_map_info(state: int, content: list[str]):
    round_info = {"state": state, "listen": [], "player": [0, 1], "content": content}
    round_info_bytes = json.dumps(round_info).encode("utf-8")
    send_to_judger(round_info_bytes)


# 逻辑向 judger 发送正常回合消息
def send_round_info(
    state: int, listen: list[int], player: list[int], content: list[str]
):
    round_info = {
        "state": state,
        "listen": listen,
        "player": player,
        "content": content,
    }

    round_info_bytes = json.dumps(round_info).encode("utf-8")
    send_to_judger(round_info_bytes)


# 逻辑向 judger 发送观战消息
def send_watch_info(watch: str):
    watch_info = {"watch": watch}
    watch_info_bytes = json.dumps(watch_info).encode("utf-8")
    send_to_judger(watch_info_bytes)


# judger 向逻辑发送 AI 正常或异常消息
def receive_ai_info():
    ai_info_bytes = receive_from_judger()
    ai_info_str = ai_info_bytes.decode("utf-8")
    ai_info = json.loads(ai_info_str)
    return ai_info


# 逻辑表示对局已结束，向 judger 请求 AI 结束状态
def request_ai_end_state():
    request_end = {"action": "request_end_state"}
    request_end_bytes = json.dumps(request_end).encode("utf-8")
    send_to_judger(request_end_bytes)


# judger 向逻辑回复 AI 结束状态
def receive_ai_end_state():
    ai_end_state_bytes = receive_from_judger()
    ai_end_state_str = ai_end_state_bytes.decode("utf-8")
    ai_end_state = json.loads(ai_end_state_str)
    return ai_end_state


# 逻辑向judger发送游戏结束信息
def send_game_end_info(end_info: str, end_state: str):
    game_end_info = {"state": -1, "end_info": end_info, "end_state": end_state}
    game_end_info_bytes = json.dumps(game_end_info).encode("utf-8")
    send_to_judger(game_end_info_bytes)


# 判断游戏是否结束：改为使用 logic.game_rules.is_game_over（保持与核心规则一致）


def convert_command_list_str(command_list_str: str):
    res: list[list[int]] = []
    with open("command_list_str.txt", "w") as f:
        f.write(command_list_str)
    for command_str in command_list_str.split("\n"):
        command = command_str.split()
        command = [int(num) for num in command]
        res.append(command)
    return res


def quit_running(er):
    with open(gamestate.replay_file, "a") as f:
        f.write(er + "\n")
    end_state = json.dumps(["IA", "IA"])
    end_info = {"0": 0, "1": 0}
    send_game_end_info(json.dumps(end_info), end_state)


def write_debug_into_replay(gamestate, message):
    with open(gamestate.replay_file, "a") as f:
        f.write(message + "\n")


def write_end_info(gamestate):
    with open(gamestate.replay_file, "a") as f:
        f.write(
            str(
                {
                    "Round": gamestate.round,
                    "Player": gamestate.winner,
                    "Action": [9],
                }
            ).replace("'", '"')
            + "\n"
        )


def read_human_information_and_apply(gamestate: GameState, player, enemy_human):
    command_list_str = ""
    while 1:
        # 持续收消息
        ai_info = receive_ai_info()
        # 出现错误，退出游戏
        if ai_info["player"] == -1:
            gamestate.winner = 1 - player
            write_end_info(gamestate)
            time.sleep(sleep_time)
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                player,
            )
            if enemy_human:
                # 如果对方是播放器，同时向对方转发结果
                # time.sleep(0.5)
                send_to_judger(
                    (
                        open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                    ).encode("utf-8"),
                    1 - player,
                )
            end_list = ["OK", "OK"]
            end_list[json.loads(ai_info["content"])["player"]] = error_map[
                json.loads(ai_info["content"])["error"]
            ]
            end_info = {
                "0": json.loads(ai_info["content"])["player"],
                "1": 1 - json.loads(ai_info["content"])["player"],
            }
            send_game_end_info(json.dumps(end_info), json.dumps(end_list))
            return False, ""

        if ai_info["content"] == "8\n":
            # 操作结束，停止接收消息
            command_list_str += "8\n"
            break
        elif ai_info["content"] == "9\n":
            gamestate.winner = 1 - player
            write_end_info(gamestate)
            time.sleep(sleep_time)
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                player,
            )
            if enemy_human:
                # 如果对方是播放器，同时向对方转发结果
                # time.sleep(sleep_time)
                send_to_judger(
                    (
                        open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                    ).encode("utf-8"),
                    1 - player,
                )
            end_state = json.dumps(["OK", "OK"])
            end_info = {"0": 1 - gamestate.winner, "1": gamestate.winner}
            send_game_end_info(json.dumps(end_info), end_state)
            return False, ""
        elif ai_info["content"] == "10\n":
            gamestate.coin[player] += 999
        else:
            # 执行操作
            command = ai_info["content"][:-1].split()
            command = [int(i) for i in command]
            success = execute_single_command(player, gamestate, command[0], command[1:])
            if success:
                # 操作成功，返回操作结果
                command_list_str += ai_info["content"]
                # time.sleep(0.5)
                send_to_judger(
                    (
                        open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                    ).encode("utf-8"),
                    player,
                )
                if enemy_human:
                    # 如果对方是播放器，同时向对方转发结果
                    # time.sleep(0.5)
                    send_to_judger(
                        (
                            open(gamestate.replay_file, "r").readlines()[-1].strip()
                            + "\n"
                        ).encode("utf-8"),
                        1 - player,
                    )
                # time.sleep(0.5)
                # 检查游戏是否结束，如果结束则停止操作
                gamestate.winner = is_game_over(gamestate)
                if gamestate.winner != -1:
                    break
            else:
                # 操作不成功，通知播放器
                # time.sleep(0.5)
                send_to_judger(
                    str(get_single_round_replay(gamestate, [], player, [-1])).encode(
                        "utf-8"
                    ),
                    player,
                )
                # time.sleep(0.5)

    # 玩家操作结束后，判断游戏是否结束
    if gamestate.winner != -1:
        write_end_info(gamestate)
        time.sleep(sleep_time)
        send_to_judger(
            (open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n").encode(
                "utf-8"
            ),
            player,
        )
        if enemy_human:
            # time.sleep(0.5)
            # 如果对方是播放器，同时向对方转发结果
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                1 - player,
            )
        # time.sleep(0.5)
        end_state = json.dumps(["OK", "OK"])
        end_info = {"0": 1 - gamestate.winner, "1": gamestate.winner}
        send_game_end_info(json.dumps(end_info), end_state)
        return False, ""

    if player == 0:
        # time.sleep(0.5)
        send_to_judger(
            (
                str(get_single_round_replay(gamestate, [], player, [8])).replace(
                    "'", '"'
                )
                + "\n"
            ).encode("utf-8"),
            player,
        )
        if enemy_human:
            # 如果对方是播放器，同时向对方转发结果
            # time.sleep(0.5)
            send_to_judger(
                (
                    str(get_single_round_replay(gamestate, [], player, [8])).replace(
                        "'", '"'
                    )
                    + "\n"
                ).encode("utf-8"),
                1 - player,
            )
        # time.sleep(0.5)
    else:
        # 如果是后手，更新回合
        update_round(gamestate)
        # 向播放器发送回合更新信息
        update_info = json.loads(
            open(gamestate.replay_file, "r").readlines()[-1].strip()
        )
        update_info["Player"] = player
        # time.sleep(0.5)
        send_to_judger(
            (str(update_info).replace("'", '"') + "\n").encode("utf-8"),
            player,
        )
        if enemy_human:
            # 如果对方是播放器，同时向对方转发结果
            # time.sleep(0.5)
            send_to_judger(
                (str(update_info).replace("'", '"') + "\n").encode("utf-8"),
                1 - player,
            )
        # time.sleep(0.5)
        # 判断游戏是否结束
        gamestate.winner = is_game_over(gamestate)
        if gamestate.winner != -1:
            write_end_info(gamestate)
            if enemy_human:
                time.sleep(sleep_time)
                send_to_judger(
                    (
                        open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                    ).encode("utf-8"),
                    player,
                )
            end_state = json.dumps(["OK", "OK"])
            end_info = {"0": 1 - gamestate.winner, "1": gamestate.winner}
            send_game_end_info(json.dumps(end_info), end_state)
            return False, ""
    return True, command_list_str


def read_ai_information_and_apply(gamestate: GameState, player, enemy_human):
    ai_info = receive_ai_info()
    if ai_info["player"] == -1:
        # 如果信息异常，胜负已分，游戏结束
        gamestate.winner = 1 - player
        write_end_info(gamestate)
        if enemy_human:
            time.sleep(sleep_time)
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                player,
            )
        end_list = ["OK", "OK"]
        end_list[json.loads(ai_info["content"])["player"]] = error_map[
            json.loads(ai_info["content"])["error"]
        ]
        end_info = {
            "0": json.loads(ai_info["content"])["player"],
            "1": 1 - json.loads(ai_info["content"])["player"],
        }
        send_game_end_info(json.dumps(end_info), json.dumps(end_list))
        return False, ""

    # 进行操作
    command_list_str = ai_info["content"]

    # 检查操作字符串合法性
    if command_list_str[-2:] != "8\n":
        command_list_str = ai_info["content"]
        gamestate.winner = 1 - player
        write_end_info(gamestate)
        if enemy_human:
            time.sleep(sleep_time)
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                player,
            )
        end_list = ["OK", "OK"]
        end_list[player] = "IA"
        end_info = {"0": player, "1": 1 - player}
        send_game_end_info(json.dumps(end_info), json.dumps(end_list))
        return False, ""

    try:
        command_list = convert_command_list_str(command_list_str)
    except Exception:
        command_list_str = ai_info["content"]
        gamestate.winner = 1 - player
        write_end_info(gamestate)
        if enemy_human:
            time.sleep(sleep_time)
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                player,
            )
        end_list = ["OK", "OK"]
        end_list[player] = "IA"
        end_info = {"0": player, "1": 1 - player}
        send_game_end_info(json.dumps(end_info), json.dumps(end_list))
        return False, ""

    success = True
    for command in command_list:
        if command[0] == 8:
            break
        if command[0] != 8:
            success = execute_single_command(player, gamestate, command[0], command[1:])
            # 如果操作失败，胜负已分
            if not success:
                gamestate.winner = 1 - player
                break
            # 如果对方是播放器，向对方发送操作结果
            if enemy_human:
                time.sleep(sleep_time)
                send_to_judger(
                    (
                        open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                    ).encode("utf-8"),
                    1 - player,
                )
            # 如果满足条件，胜负已分
            gamestate.winner = is_game_over(gamestate)  # isgameover返回-1代表没结束
            if gamestate.winner != -1:
                break

    # 玩家0的操作回合结束后，判断游戏是否结束
    if gamestate.winner != -1:
        write_end_info(gamestate)
        if enemy_human:
            time.sleep(sleep_time)
            send_to_judger(
                (
                    open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                ).encode("utf-8"),
                1 - player,
            )
        end_list = ["OK", "OK"]
        if not success:
            end_list[player] = "IA"
        end_info = {"0": 1 - gamestate.winner, "1": gamestate.winner}
        send_game_end_info(json.dumps(end_info), json.dumps(end_list))
        return False, ""

    # 发送正常回合消息
    # quit_running("before send round info")
    if player == 0:
        if enemy_human:
            # time.sleep(sleep_time)
            # 如果对方是播放器，同时向对方转发结果
            time.sleep(sleep_time)
            send_to_judger(
                (
                    str(get_single_round_replay(gamestate, [], player, [8])).replace(
                        "'", '"'
                    )
                    + "\n"
                ).encode("utf-8"),
                1 - player,
            )
    elif player == 1:
        update_round(gamestate)
        update_info = json.loads(
            open(gamestate.replay_file, "r").readlines()[-1].strip()
        )
        update_info["Player"] = player
        if enemy_human:
            # time.sleep(0.5)
            # 如果对方是播放器，同时向对方转发结果
            time.sleep(sleep_time)
            send_to_judger(
                (str(update_info).replace("'", '"') + "\n").encode("utf-8"),
                1 - player,
            )

        # 判断游戏是否结束
        gamestate.winner = is_game_over(gamestate)
        if gamestate.winner != -1:
            write_end_info(gamestate)
            if enemy_human:
                time.sleep(sleep_time)
                send_to_judger(
                    (
                        open(gamestate.replay_file, "r").readlines()[-1].strip() + "\n"
                    ).encode("utf-8"),
                    1 - player,
                )
            end_state = json.dumps(["OK", "OK"])
            end_info = {"0": 1 - gamestate.winner, "1": gamestate.winner}
            send_game_end_info(json.dumps(end_info), end_state)
            return False, ""

    return True, command_list_str


if __name__ == "__main__":
    try:
        from logic.ai2logic import execute_single_command
        from logic.constant import *
        from logic.gamedata import Farmer, MainGenerals, SubGenerals, SuperWeapon, WeaponType, CellType
        from logic.gamestate import GameState, init_generals, update_round
        from logic.generate_round_replay import get_single_round_replay
        from logic.game_rules import is_game_over
        from logic.gamedata import CellType

        # constant.mountain_persent = random.uniform(0.05, 0.1)
        # constant.bog_percent = random.uniform(0.05, 0.25)
        gamestate = GameState()  # 每局游戏唯一的游戏状态类，所有的修改应该在此对象中进行
        init_generals(gamestate)
        gamestate.coin = [40, 40]

        # init_json = """{"Round": 0, "Player": -1, "Action": [8], "Cells": [[[0, 0], -1, 0], [[0, 1], -1, 0], [[0, 2], -1, 0], [[0, 3], -1, 0], [[0, 4], -1, 0], [[0, 5], -1, 0], [[0, 6], -1, 0], [[0, 7], -1, 0], [[0, 8], -1, 0], [[0, 9], -1, 0], [[0, 10], -1, 0], [[0, 11], -1, 0], [[0, 12], -1, 0], [[0, 13], -1, 0], [[0, 14], -1, 0], [[1, 0], -1, 0], [[1, 1], -1, 0], [[1, 2], -1, 5], [[1, 3], -1, 0], [[1, 4], -1, 0], [[1, 5], -1, 0], [[1, 6], -1, 0], [[1, 7], -1, 0], [[1, 8], -1, 0], [[1, 9], -1, 0], [[1, 10], -1, 0], [[1, 11], -1, 0], [[1, 12], -1, 0], [[1, 13], -1, 0], [[1, 14], -1, 0], [[2, 0], -1, 0], [[2, 1], -1, 0], [[2, 2], -1, 0], [[2, 3], -1, 0], [[2, 4], -1, 0], [[2, 5], -1, 0], [[2, 6], -1, 0], [[2, 7], -1, 0], [[2, 8], -1, 0], [[2, 9], -1, 0], [[2, 10], -1, 0], [[2, 11], -1, 0], [[2, 12], -1, 0], [[2, 13], -1, 0], [[2, 14], -1, 0], [[3, 0], -1, 0], [[3, 1], -1, 0], [[3, 2], -1, 0], [[3, 3], -1, 0], [[3, 4], -1, 0], [[3, 5], -1, 0], [[3, 6], -1, 0], [[3, 7], -1, 0], [[3, 8], -1, 14], [[3, 9], -1, 0], [[3, 10], -1, 0], [[3, 11], -1, 0], [[3, 12], -1, 0], [[3, 13], -1, 0], [[3, 14], -1, 0], [[4, 0], -1, 0], [[4, 1], -1, 0], [[4, 2], -1, 0], [[4, 3], -1, 0], [[4, 4], -1, 0], [[4, 5], -1, 0], [[4, 6], -1, 0], [[4, 7], -1, 0], [[4, 8], -1, 0], [[4, 9], -1, 0], [[4, 10], -1, 0], [[4, 11], -1, 0], [[4, 12], -1, 0], [[4, 13], -1, 0], [[4, 14], -1, 0], [[5, 0], -1, 0], [[5, 1], -1, 0], [[5, 2], -1, 0], [[5, 3], -1, 0], [[5, 4], -1, 0], [[5, 5], -1, 5], [[5, 6], -1, 0], [[5, 7], -1, 0], [[5, 8], -1, 0], [[5, 9], -1, 0], [[5, 10], -1, 0], [[5, 11], -1, 3], [[5, 12], -1, 0], [[5, 13], -1, 0], [[5, 14], -1, 0], [[6, 0], -1, 0], [[6, 1], -1, 0], [[6, 2], -1, 0], [[6, 3], -1, 0], [[6, 4], -1, 3], [[6, 5], -1, 0], [[6, 6], -1, 0], [[6, 7], -1, 0], [[6, 8], -1, 0], [[6, 9], -1, 0], [[6, 10], -1, 5], [[6, 11], -1, 19], [[6, 12], -1, 0], [[6, 13], -1, 0], [[6, 14], -1, 0], [[7, 0], -1, 0], [[7, 1], -1, 0], [[7, 2], -1, 0], [[7, 3], -1, 0], [[7, 4], -1, 0], [[7, 5], -1, 0], [[7, 6], -1, 0], [[7, 7], -1, 0], [[7, 8], -1, 0], [[7, 9], -1, 0], [[7, 10], -1, 0], [[7, 11], 0, 0], [[7, 12], -1, 0], [[7, 13], -1, 0], [[7, 14], -1, 0], [[8, 0], -1, 0], [[8, 1], -1, 0], [[8, 2], -1, 0], [[8, 3], -1, 0], [[8, 4], -1, 0], [[8, 5], -1, 3], [[8, 6], -1, 0], [[8, 7], -1, 5], [[8, 8], -1, 0], [[8, 9], -1, 0], [[8, 10], -1, 0], [[8, 11], -1, 0], [[8, 12], -1, 0], [[8, 13], -1, 0], [[8, 14], -1, 0], [[9, 0], -1, 0], [[9, 1], -1, 0], [[9, 2], -1, 0], [[9, 3], -1, 0], [[9, 4], -1, 0], [[9, 5], -1, 0], [[9, 6], 1, 0], [[9, 7], -1, 0], [[9, 8], -1, 0], [[9, 9], -1, 0], [[9, 10], -1, 0], [[9, 11], -1, 0], [[9, 12], -1, 0], [[9, 13], -1, 0], [[9, 14], -1, 0], [[10, 0], -1, 0], [[10, 1], -1, 0], [[10, 2], -1, 0], [[10, 3], -1, 0], [[10, 4], -1, 0], [[10, 5], -1, 0], [[10, 6], -1, 0], [[10, 7], -1, 0], [[10, 8], -1, 13], [[10, 9], -1, 0], [[10, 10], -1, 0], [[10, 11], -1, 0], [[10, 12], -1, 0], [[10, 13], -1, 0], [[10, 14], -1, 0], [[11, 0], -1, 16], [[11, 1], -1, 0], [[11, 2], -1, 0], [[11, 3], -1, 0], [[11, 4], -1, 0], [[11, 5], -1, 0], [[11, 6], -1, 0], [[11, 7], -1, 0], [[11, 8], -1, 0], [[11, 9], -1, 0], [[11, 10], -1, 0], [[11, 11], -1, 0], [[11, 12], -1, 0], [[11, 13], -1, 0], [[11, 14], -1, 0], [[12, 0], -1, 0], [[12, 1], -1, 0], [[12, 2], -1, 0], [[12, 3], -1, 0], [[12, 4], -1, 0], [[12, 5], -1, 0], [[12, 6], -1, 0], [[12, 7], -1, 0], [[12, 8], -1, 0], [[12, 9], -1, 0], [[12, 10], -1, 0], [[12, 11], -1, 0], [[12, 12], -1, 0], [[12, 13], -1, 0], [[12, 14], -1, 0], [[13, 0], -1, 0], [[13, 1], -1, 0], [[13, 2], -1, 0], [[13, 3], -1, 0], [[13, 4], -1, 0], [[13, 5], -1, 0], [[13, 6], -1, 0], [[13, 7], -1, 0], [[13, 8], -1, 0], [[13, 9], -1, 0], [[13, 10], -1, 0], [[13, 11], -1, 0], [[13, 12], -1, 0], [[13, 13], -1, 0], [[13, 14], -1, 0], [[14, 0], -1, 0], [[14, 1], -1, 0], [[14, 2], -1, 0], [[14, 3], -1, 0], [[14, 4], -1, 3], [[14, 5], -1, 0], [[14, 6], -1, 0], [[14, 7], -1, 0], [[14, 8], -1, 0], [[14, 9], -1, 0], [[14, 10], -1, 0], [[14, 11], -1, 0], [[14, 12], -1, 0], [[14, 13], -1, 0], [[14, 14], -1, 0]], "Generals": [{"Id": 0, "Player": 0, "Type": 1, "Position": [7, 11], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 1, "Player": 1, "Type": 1, "Position": [9, 6], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 2, "Player": -1, "Type": 2, "Position": [6, 11], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 3, "Player": -1, "Type": 2, "Position": [10, 8], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 4, "Player": -1, "Type": 2, "Position": [3, 8], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 5, "Player": -1, "Type": 2, "Position": [11, 0], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 6, "Player": -1, "Type": 3, "Position": [14, 4], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 7, "Player": -1, "Type": 3, "Position": [6, 10], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 8, "Player": -1, "Type": 3, "Position": [8, 7], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 9, "Player": -1, "Type": 3, "Position": [5, 11], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 10, "Player": -1, "Type": 3, "Position": [1, 2], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 11, "Player": -1, "Type": 3, "Position": [8, 5], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 12, "Player": -1, "Type": 3, "Position": [5, 5], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}, {"Id": 13, "Player": -1, "Type": 3, "Position": [6, 4], "Level": [1, 1, 1], "Skill_cd": [0, 0, 0, 0, 0], "Skill_rest": [0, 0, 0], "Alive": 1}], "Weapons": [], "Weapon_cds": [-1, -1], "Tech_level": [[1, 0, 0, 0], [1, 0, 0, 0]], "Coins": [40, 40], "Cell_type": "000202000002100000000020200001000001000000000002000100101022000100000000120100000010020001000000000000000001000000000010000000000000100012000000002000022000000001000000000000000000000000101000000000000100100000000000200000000"}"""
        # dict = json.loads(init_json)
        # map = dict["Cells"]
        # types = dict["Cell_type"]
        # generals = dict["Generals"]
        # for i in range(len(map)):
        #     gamestate.board[int(i / row)][i % col].type = CellType(int(types[i]))
        #     gamestate.board[int(i / row)][i % col].player = map[i][1]
        #     gamestate.board[int(i / row)][i % col].army = map[i][2]
        # for i in range(len(generals)):
        #     id, player = generals[i]["Id"], generals[i]["Player"]
        #     position = generals[i]["Position"]
        #     if generals[i]["Type"] == 1:
        #         general = MainGenerals(id, player, position)
        #     elif generals[i]["Type"] == 2:
        #         general = SubGenerals(id, player, position)
        #     else:
        #         general = Farmer(id, player, position)
        #     gamestate.generals.append(general)
        #     gamestate.board[position[0]][position[1]].generals = general
        # gamestate.coin = dict["Coins"]
        # gamestate.next_generals_id = len(generals)

        # 接收judger的初始化信息
        init_info = receive_init_info()
        gamestate.replay_file = init_info["replay"]
        player_type = init_info["player_list"]
        state = 0

        # 写入初始化json
        init_json = gamestate.trans_state_to_init_json(-1)
        init_json["Round"] = 0
        with open(gamestate.replay_file, "w") as f:
            f.write(str(init_json).replace("'", '"') + "\n")
        f.close()

        state += 1

        if player_type[0] == 1:
            send_round_config(1, 1024)
        else:
            send_round_config(180, 1024)
        # 向双方AI发送初始化信息
        json0 = gamestate.trans_state_to_init_json(0)
        json0["Round"] = 0
        json1 = gamestate.trans_state_to_init_json(1)
        json1["Round"] = 0
        send_round_info(
            state,
            [0],
            [0, 1],
            [
                str(json0).replace("'", '"') + "\n",
                str(json1).replace("'", '"') + "\n",
            ],
        )
        # state += 1

        player = 0
        game_continue = True
        while game_continue:
            # send_round_config(state, 1, 1024)
            if player_type[player] == 1:
                game_continue, operation_string = read_ai_information_and_apply(
                    gamestate, player, player_type[1 - player] == 2
                )
            elif player_type[player] == 2:
                game_continue, operation_string = read_human_information_and_apply(
                    gamestate, player, player_type[1 - player] == 2
                )

            if not game_continue:
                break
            player = 1 - player
            state += 1
            if player_type[player] == 1:
                send_round_config(1, 1024)
                send_round_info(
                    state,
                    [player],
                    [player],
                    [operation_string],
                )
            elif player_type[player] == 2:
                send_round_config(180, 1024)
                send_round_info(
                    state,
                    [player],
                    [],
                    [],
                )
    except Exception as e:
        # 【修复后的异常处理】
        
        # 1. 将详细报错打印到 标准错误输出 (stderr)
        # 评测机通常会把 stderr 的内容显示在另外的日志里，这是最保险的
        sys.stderr.write("!!! Logic Process Crashed !!!\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()

        # 2. 尝试写入回放文件（如果路径已获取）
        try:
            if gamestate and hasattr(gamestate, 'replay_file') and gamestate.replay_file:
                with open(gamestate.replay_file, "a") as f:
                    f.write("\n--- Critical Error ---\n")
                    f.write(traceback.format_exc())
        except:
            # 如果连写文件都失败了，就只依赖 stderr
            pass
        
        # 3. 尝试按照协议通知 Judger 发生了错误 (防止 Judger 等待超时)
        # 构造一个合法的结束包，防止评测机报空
        try:
            # 这里的格式必须严格符合 judger 对 logic 的结束要求
            # 如果前面没能初始化，发送这个可能也没用，但值得一试
            end_info = {"0": 0, "1": 0} 
            end_state = json.dumps(["RE", "RE"]) # RE = Runtime Error
            
            # 手动构建包发送，避免依赖可能损坏的函数
            msg = json.dumps({
                "state": -1, 
                "end_info": json.dumps(end_info), 
                "end_state": end_state
            }).encode("utf-8")
            
            header = struct.pack(">Ii", len(msg), -1)
            sys.stdout.buffer.write(header)
            sys.stdout.buffer.write(msg)
            sys.stdout.flush()
        except:
            pass
