from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from SDK.constants import (
    BASE_HP,
    BASE_UPGRADE_COST,
    BASIC_INCOME,
    CENTERLINE_WEIGHTS,
    HIGHLAND_CELLS,
    INITIAL_COINS,
    LAMBDA_DENOM,
    LAMBDA_NUM,
    MAP_SIZE,
    MAX_ROUND,
    OFFSET,
    OperationType,
    PHEROMONE_FAIL_BONUS_INT,
    PHEROMONE_FLOOR,
    PHEROMONE_INIT_INT,
    PHEROMONE_SCALE,
    PHEROMONE_SUCCESS_BONUS_INT,
    PHEROMONE_TOO_OLD_BONUS_INT,
    TAU_BASE_ADD_INT,
    PLAYER_BASES,
    PLAYER_COUNT,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TOWER_BUILD_BASE_COST,
    TOWER_BUILD_RATIO,
    TOWER_DOWNGRADE_REFUND_RATIO,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    TowerType,
    WeaponStats,
    AntStatus,
    LEVEL2_TOWER_UPGRADE_COST,
    LEVEL3_TOWER_UPGRADE_COST,
)
from SDK.geometry import hex_distance, is_highland, is_path, is_valid_pos, neighbors
from SDK.model import Ant, Base, Operation, Tower, WeaponEffect


@dataclass(slots=True)
class PublicRoundState:
    round_index: int
    towers: list[tuple[int, int, int, int, int, int]]
    ants: list[tuple[int, int, int, int, int, int, int, int]]
    coins: tuple[int, int]
    camps_hp: tuple[int, int]


@dataclass(slots=True)
class TurnResolution:
    operations: tuple[list[Operation], list[Operation]]
    illegal: tuple[list[Operation], list[Operation]]
    terminal: bool
    winner: int | None


@dataclass(slots=True)
class GameState:
    seed: int = 0
    round_index: int = 0
    towers: list[Tower] = field(default_factory=list)
    ants: list[Ant] = field(default_factory=list)
    bases: list[Base] = field(default_factory=list)
    coins: list[int] = field(default_factory=lambda: [INITIAL_COINS, INITIAL_COINS])
    pheromone: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.int32))
    weapon_cooldowns: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, 5), dtype=np.int16))
    active_effects: list[WeaponEffect] = field(default_factory=list)
    old_count: list[int] = field(default_factory=lambda: [0, 0])
    die_count: list[int] = field(default_factory=lambda: [0, 0])
    super_weapon_usage: list[int] = field(default_factory=lambda: [0, 0])
    ai_time: list[int] = field(default_factory=lambda: [0, 0])
    next_ant_id: int = 0
    next_tower_id: int = 0
    terminal: bool = False
    winner: int | None = None

    @classmethod
    def initial(cls, seed: int = 0) -> GameState:
        state = cls(seed=seed)
        state.bases = [Base(0, *PLAYER_BASES[0], hp=BASE_HP), Base(1, *PLAYER_BASES[1], hp=BASE_HP)]
        state._init_pheromone(seed)
        return state

    def clone(self) -> GameState:
        return GameState(
            seed=self.seed,
            round_index=self.round_index,
            towers=[tower.clone() for tower in self.towers],
            ants=[ant.clone() for ant in self.ants],
            bases=[base.clone() for base in self.bases],
            coins=list(self.coins),
            pheromone=self.pheromone.copy(),
            weapon_cooldowns=self.weapon_cooldowns.copy(),
            active_effects=[effect.clone() for effect in self.active_effects],
            old_count=list(self.old_count),
            die_count=list(self.die_count),
            super_weapon_usage=list(self.super_weapon_usage),
            ai_time=list(self.ai_time),
            next_ant_id=self.next_ant_id,
            next_tower_id=self.next_tower_id,
            terminal=self.terminal,
            winner=self.winner,
        )

    def _init_pheromone(self, seed: int) -> None:
        value = seed & ((1 << 48) - 1)
        for player in range(PLAYER_COUNT):
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    value = (25214903917 * value) & ((1 << 48) - 1)
                    self.pheromone[player, x, y] = PHEROMONE_INIT_INT + (value * 10000 >> 46)

    def tower_count(self, player: int) -> int:
        return sum(1 for tower in self.towers if tower.player == player)

    def towers_of(self, player: int) -> list[Tower]:
        return [tower for tower in self.towers if tower.player == player]

    def ants_of(self, player: int) -> list[Ant]:
        return [ant for ant in self.ants if ant.player == player and ant.is_alive()]

    def tower_at(self, x: int, y: int) -> Tower | None:
        for tower in self.towers:
            if tower.x == x and tower.y == y:
                return tower
        return None

    def tower_by_id(self, tower_id: int) -> Tower | None:
        for tower in self.towers:
            if tower.tower_id == tower_id:
                return tower
        return None

    def strategic_slots(self, player: int) -> tuple[tuple[int, int], ...]:
        return STRATEGIC_BUILD_ORDER[player]

    def build_tower_cost(self, tower_count: int | None = None) -> int:
        if tower_count is None:
            tower_count = self.tower_count(0)
        return int(TOWER_BUILD_BASE_COST * (TOWER_BUILD_RATIO ** tower_count))

    def upgrade_tower_cost(self, target_type: TowerType) -> int:
        if target_type.value < 10:
            return LEVEL2_TOWER_UPGRADE_COST
        return LEVEL3_TOWER_UPGRADE_COST

    def destroy_tower_income(self, tower_count: int) -> int:
        return int(self.build_tower_cost(tower_count - 1) * TOWER_DOWNGRADE_REFUND_RATIO)

    def downgrade_tower_income(self, tower_type: TowerType) -> int:
        return int(self.upgrade_tower_cost(tower_type) * TOWER_DOWNGRADE_REFUND_RATIO)

    def upgrade_base_cost(self, level: int) -> int:
        return BASE_UPGRADE_COST[level]

    def weapon_cost(self, weapon_type: SuperWeaponType) -> int:
        return SUPER_WEAPON_STATS[weapon_type].cost

    def nearest_ant_distance(self, player: int) -> int:
        base_x, base_y = PLAYER_BASES[player]
        enemies = [hex_distance(ant.x, ant.y, base_x, base_y) for ant in self.ants if ant.player != player and ant.is_alive()]
        return min(enemies) if enemies else 32

    def frontline_distance(self, player: int) -> int:
        base_x, base_y = PLAYER_BASES[1 - player]
        ants = [hex_distance(ant.x, ant.y, base_x, base_y) for ant in self.ants if ant.player == player and ant.is_alive()]
        return min(ants) if ants else 32

    def safe_coin_threshold(self, player: int) -> int:
        enemy = 1 - player
        emp_cd = int(self.weapon_cooldowns[enemy, SuperWeaponType.EMP_BLASTER])
        enemy_coin = int(self.coins[enemy])
        if emp_cd >= 90:
            return 0
        if emp_cd > 0:
            return max(int(min(enemy_coin, 149) - emp_cd * 1.66), 0)
        return min(enemy_coin, 149)

    def current_and_neighbors_empty(self, x: int, y: int) -> bool:
        if (x, y) in PLAYER_BASES:
            return False
        if self.tower_at(x, y) is not None:
            return False
        for _, nx, ny in neighbors(x, y):
            if is_valid_pos(nx, ny) and ((nx, ny) in PLAYER_BASES or self.tower_at(nx, ny) is not None):
                return False
        return True

    def is_shielded_by_emp(self, player: int, x: int, y: int) -> bool:
        for effect in self.active_effects:
            if effect.weapon_type == SuperWeaponType.EMP_BLASTER and effect.player != player and effect.in_range(x, y):
                return True
        return False

    def is_shielded_by_deflector(self, ant: Ant) -> bool:
        for effect in self.active_effects:
            if effect.weapon_type == SuperWeaponType.DEFLECTOR and effect.player == ant.player and effect.in_range(ant.x, ant.y):
                return True
        return False

    def weapon_effect(self, weapon_type: SuperWeaponType, player: int) -> WeaponEffect | None:
        for effect in self.active_effects:
            if effect.weapon_type == weapon_type and effect.player == player:
                return effect
        return None

    def _operation_income(self, player: int, operation: Operation, tower_count_hint: int | None = None) -> int:
        if operation.op_type == OperationType.BUILD_TOWER:
            return -self.build_tower_cost(self.tower_count(player) if tower_count_hint is None else tower_count_hint)
        if operation.op_type == OperationType.UPGRADE_TOWER:
            return -self.upgrade_tower_cost(TowerType(operation.arg1))
        if operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None:
                return 0
            if tower.tower_type == TowerType.BASIC:
                count = self.tower_count(player) if tower_count_hint is None else tower_count_hint
                return self.destroy_tower_income(count)
            return self.downgrade_tower_income(tower.tower_type)
        if operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return -self.weapon_cost(SuperWeaponType(operation.op_type % 10))
        if operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            return -self.upgrade_base_cost(self.bases[player].generation_level)
        if operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            return -self.upgrade_base_cost(self.bases[player].ant_level)
        return 0

    def can_apply_operation(self, player: int, operation: Operation, pending: Iterable[Operation] = ()) -> bool:
        pending_list = list(pending)
        if operation.op_type == OperationType.BUILD_TOWER:
            if not is_highland(player, operation.arg0, operation.arg1):
                return False
            if (operation.arg0, operation.arg1) in PLAYER_BASES or self.tower_at(operation.arg0, operation.arg1) is not None:
                return False
            if self.is_shielded_by_emp(player, operation.arg0, operation.arg1):
                return False
            if any(op.op_type == OperationType.BUILD_TOWER and op.arg0 == operation.arg0 and op.arg1 == operation.arg1 for op in pending_list):
                return False
        elif operation.op_type == OperationType.UPGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None or tower.player != player:
                return False
            if self.is_shielded_by_emp(player, tower.x, tower.y):
                return False
            if not tower.is_upgrade_type_valid(TowerType(operation.arg1)):
                return False
            if any(op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER) and op.arg0 == operation.arg0 for op in pending_list):
                return False
        elif operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None or tower.player != player:
                return False
            if self.is_shielded_by_emp(player, tower.x, tower.y):
                return False
            if any(op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER) and op.arg0 == operation.arg0 for op in pending_list):
                return False
        elif operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            if not is_valid_pos(operation.arg0, operation.arg1):
                return False
            weapon_type = SuperWeaponType(operation.op_type % 10)
            if self.weapon_cooldowns[player, weapon_type] > 0:
                return False
            if any(op.op_type == operation.op_type for op in pending_list):
                return False
        elif operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            if self.bases[player].generation_level >= 2:
                return False
            if any(op.op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT) for op in pending_list):
                return False
        elif operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            if self.bases[player].ant_level >= 2:
                return False
            if any(op.op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT) for op in pending_list):
                return False
        else:
            return False

        income = 0
        simulated_tower_count = self.tower_count(player)
        for op in (*pending_list, operation):
            if op.op_type == OperationType.BUILD_TOWER:
                income -= self.build_tower_cost(simulated_tower_count)
                simulated_tower_count += 1
            elif op.op_type == OperationType.DOWNGRADE_TOWER:
                tower = self.tower_by_id(op.arg0)
                if tower is None:
                    continue
                if tower.tower_type == TowerType.BASIC:
                    income += self.destroy_tower_income(simulated_tower_count)
                    simulated_tower_count -= 1
                else:
                    income += self.downgrade_tower_income(tower.tower_type)
            else:
                income += self._operation_income(player, op)
        return self.coins[player] + income >= 0

    def apply_operation(self, player: int, operation: Operation) -> None:
        self.coins[player] += self._operation_income(player, operation)
        if operation.op_type == OperationType.BUILD_TOWER:
            self.towers.append(
                Tower(
                    self.next_tower_id,
                    player,
                    operation.arg0,
                    operation.arg1,
                    TowerType.BASIC,
                    TOWER_STATS[TowerType.BASIC].speed,
                )
            )
            self.next_tower_id += 1
            return
        if operation.op_type == OperationType.UPGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            assert tower is not None
            tower.upgrade(TowerType(operation.arg1))
            return
        if operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            assert tower is not None
            destroy = tower.downgrade_or_destroy()
            if destroy:
                self.towers = [item for item in self.towers if item.tower_id != tower.tower_id]
            return
        if operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            weapon_type = SuperWeaponType(operation.op_type % 10)
            stats = SUPER_WEAPON_STATS[weapon_type]
            self.weapon_cooldowns[player, weapon_type] = stats.cooldown
            self.super_weapon_usage[player] += 1
            if weapon_type == SuperWeaponType.EMERGENCY_EVASION:
                for ant in self.ants:
                    if ant.player == player and hex_distance(operation.arg0, operation.arg1, ant.x, ant.y) <= stats.attack_range:
                        ant.shield = 2
            self.active_effects.append(WeaponEffect(weapon_type, player, operation.arg0, operation.arg1, stats.duration))
            return
        if operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            self.bases[player].generation_level += 1
            return
        if operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            self.bases[player].ant_level += 1
            return

    def apply_operation_list(self, player: int, operations: Iterable[Operation]) -> list[Operation]:
        illegal: list[Operation] = []
        accepted: list[Operation] = []
        for operation in operations:
            if self.can_apply_operation(player, operation, accepted):
                self.apply_operation(player, operation)
                accepted.append(operation)
            else:
                illegal.append(operation)
        return illegal

    def _apply_lightning_storm(self) -> None:
        for effect in self.active_effects:
            if effect.weapon_type != SuperWeaponType.LIGHTNING_STORM:
                continue
            for ant in self.ants:
                if ant.player != effect.player and ant.is_alive() and effect.in_range(ant.x, ant.y):
                    ant.hp -= 100
                    ant.refresh_status()

    def _attack_ants(self) -> None:
        self._apply_lightning_storm()
        for ant in self.ants:
            ant.deflector = False
            ant.frozen = False
        for ant in self.ants:
            ant.deflector = self.is_shielded_by_deflector(ant)
            ant.refresh_status()
        for tower in self.towers:
            if self.is_shielded_by_emp(tower.player, tower.x, tower.y):
                continue
            tower.tick()
            if not tower.ready_to_fire():
                continue
            attacked = self._tower_attack(tower)
            if attacked:
                tower.reset_cooldown()

    def _tower_attack(self, tower: Tower) -> bool:
        targets = self._find_targets(tower)
        if not targets:
            return False
        attacked_any = False
        repetitions = int(round(1 / tower.speed)) if tower.speed < 1 else 1
        for _ in range(repetitions):
            local_targets = self._find_targets(tower)
            if not local_targets:
                break
            for ant in self._expand_attack_targets(tower, local_targets):
                ant.take_damage(tower.damage, apply_freeze=tower.tower_type == TowerType.ICE)
                attacked_any = True
        return attacked_any

    def _find_targets(self, tower: Tower) -> list[Ant]:
        candidates = [
            ant for ant in self.ants
            if ant.player != tower.player and ant.is_alive() and hex_distance(ant.x, ant.y, tower.x, tower.y) <= tower.attack_range
        ]
        candidates.sort(key=lambda ant: (hex_distance(ant.x, ant.y, tower.x, tower.y), ant.ant_id))
        if tower.tower_type == TowerType.DOUBLE:
            return candidates[:2]
        return candidates[:1]

    def _expand_attack_targets(self, tower: Tower, targets: list[Ant]) -> list[Ant]:
        expanded: list[Ant] = []
        for target in targets:
            if tower.tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
                expanded.extend(self._ants_in_range(tower.player, target.x, target.y, 1))
            elif tower.tower_type == TowerType.PULSE:
                expanded.extend(self._ants_in_range(tower.player, tower.x, tower.y, tower.attack_range))
            elif tower.tower_type == TowerType.MISSILE:
                expanded.extend(self._ants_in_range(tower.player, target.x, target.y, 2))
            else:
                expanded.append(target)
        unique: dict[int, Ant] = {}
        for ant in expanded:
            unique[ant.ant_id] = ant
        return list(unique.values())

    def _ants_in_range(self, player: int, x: int, y: int, attack_range: int) -> list[Ant]:
        return [ant for ant in self.ants if ant.player != player and ant.is_alive() and hex_distance(ant.x, ant.y, x, y) <= attack_range]

    def _choose_ant_move(self, ant: Ant) -> int:
        target_x, target_y = PLAYER_BASES[1 - ant.player]
        current_distance = hex_distance(ant.x, ant.y, target_x, target_y)
        best_direction = -1
        best_weighted = -1
        best_raw = -1
        for direction, nx, ny in neighbors(ant.x, ant.y):
            if ant.path and ant.path[-1] == (direction + 3) % 6:
                continue
            if not is_path(nx, ny):
                continue
            pheromone_raw = int(self.pheromone[ant.player, nx, ny])
            next_distance = hex_distance(nx, ny, target_x, target_y)
            if next_distance < current_distance:
                eta_scaled = 12500  # 1.25
            elif next_distance == current_distance:
                eta_scaled = 10000  # 1.0
            else:
                eta_scaled = 7500   # 0.75
            weighted = pheromone_raw * eta_scaled // PHEROMONE_SCALE
            if weighted > best_weighted or (weighted == best_weighted and pheromone_raw > best_raw):
                best_direction = direction
                best_weighted = weighted
                best_raw = pheromone_raw
        return best_direction

    def _move_ants(self) -> None:
        for ant in self.ants:
            ant.refresh_status()
            direction = -1
            if ant.status == AntStatus.ALIVE:
                direction = self._choose_ant_move(ant)
            ant.path.append(direction)
            if direction != -1:
                dx, dy = OFFSET[ant.y % 2][direction]
                ant.x += dx
                ant.y += dy
            ant.refresh_status()

    def _update_pheromone(self) -> None:
        # Global attenuation: p_new = 0.97*p + 0.03*10 (integer arithmetic)
        self.pheromone = np.maximum(
            0,
            (LAMBDA_NUM * self.pheromone + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM,
        )
        for ant in self.ants:
            if ant.status in (AntStatus.ALIVE, AntStatus.FROZEN):
                continue
            if ant.status == AntStatus.SUCCESS:
                delta = PHEROMONE_SUCCESS_BONUS_INT
            elif ant.status == AntStatus.FAIL:
                delta = PHEROMONE_FAIL_BONUS_INT
            elif ant.status == AntStatus.TOO_OLD:
                delta = PHEROMONE_TOO_OLD_BONUS_INT
            else:
                continue
            visited: set[tuple[int, int]] = set()
            x, y = PLAYER_BASES[ant.player]
            if (x, y) not in visited:
                self.pheromone[ant.player, x, y] = max(0, self.pheromone[ant.player, x, y] + delta)
                visited.add((x, y))
            for direction in ant.path:
                if direction == -1:
                    continue
                dx, dy = OFFSET[y % 2][direction]
                x += dx
                y += dy
                if (x, y) in visited:
                    continue
                self.pheromone[ant.player, x, y] = max(0, self.pheromone[ant.player, x, y] + delta)
                visited.add((x, y))

    def _resolve_ant_lifecycle(self) -> None:
        remaining: list[Ant] = []
        for ant in self.ants:
            ant.refresh_status()
            if ant.status == AntStatus.SUCCESS:
                self.bases[1 - ant.player].hp -= 1
                self.coins[ant.player] += 5
                if self.bases[1 - ant.player].hp <= 0:
                    self.terminal = True
                    self.winner = ant.player
            elif ant.status == AntStatus.FAIL:
                self.coins[1 - ant.player] += ant.kill_reward
                self.die_count[ant.player] += 1
            elif ant.status == AntStatus.TOO_OLD:
                self.old_count[ant.player] += 1
            else:
                remaining.append(ant)
        self.ants = remaining

    def _spawn_ants(self) -> None:
        for base in self.bases:
            if base.should_spawn(self.round_index):
                self.ants.append(base.spawn_ant(self.next_ant_id))
                self.next_ant_id += 1

    def _increase_ant_age(self) -> None:
        for ant in self.ants:
            ant.age += 1
            ant.refresh_status()

    def _tick_effects(self) -> None:
        for player in range(PLAYER_COUNT):
            for weapon_index in range(1, 5):
                if self.weapon_cooldowns[player, weapon_index] > 0:
                    self.weapon_cooldowns[player, weapon_index] -= 1
        next_effects: list[WeaponEffect] = []
        for effect in self.active_effects:
            effect.remaining_turns -= 1
            if effect.remaining_turns > 0 and effect.weapon_type != SuperWeaponType.EMERGENCY_EVASION:
                next_effects.append(effect)
        self.active_effects = next_effects

    def _judge_timeout_winner(self) -> None:
        if self.bases[0].hp != self.bases[1].hp:
            self.winner = 0 if self.bases[0].hp > self.bases[1].hp else 1
            return
        if self.die_count[0] != self.die_count[1]:
            self.winner = 0 if self.die_count[0] > self.die_count[1] else 1
            return
        if self.super_weapon_usage[0] != self.super_weapon_usage[1]:
            self.winner = 0 if self.super_weapon_usage[0] < self.super_weapon_usage[1] else 1
            return
        if self.ai_time[0] != self.ai_time[1]:
            self.winner = 0 if self.ai_time[0] < self.ai_time[1] else 1
            return
        self.winner = 0

    def advance_round(self) -> None:
        if self.terminal:
            return
        self._attack_ants()
        self._move_ants()
        self._update_pheromone()
        self._resolve_ant_lifecycle()
        self._spawn_ants()
        self._increase_ant_age()
        for player in range(PLAYER_COUNT):
            self.coins[player] += BASIC_INCOME
        self._tick_effects()
        self.round_index += 1
        if self.round_index >= MAX_ROUND and not self.terminal:
            self.terminal = True
            self._judge_timeout_winner()
        if not self.terminal:
            for ant in self.ants:
                ant.refresh_status()
                if ant.status == AntStatus.TOO_OLD:
                    self.old_count[ant.player] += 1
            self.ants = [ant for ant in self.ants if ant.status != AntStatus.TOO_OLD]

    def resolve_turn(self, operations0: Iterable[Operation], operations1: Iterable[Operation]) -> TurnResolution:
        illegal0 = self.apply_operation_list(0, operations0)
        illegal1 = self.apply_operation_list(1, operations1)
        self.advance_round()
        return TurnResolution((list(operations0), list(operations1)), (illegal0, illegal1), self.terminal, self.winner)

    def to_public_round_state(self) -> PublicRoundState:
        towers = [
            (tower.tower_id, tower.player, tower.x, tower.y, int(tower.tower_type), tower.display_cooldown())
            for tower in sorted(self.towers, key=lambda item: item.tower_id)
        ]
        ants = [
            (ant.ant_id, ant.player, ant.x, ant.y, ant.hp, ant.level, ant.age, int(ant.status))
            for ant in sorted(self.ants, key=lambda item: item.ant_id)
        ]
        return PublicRoundState(
            round_index=self.round_index,
            towers=towers,
            ants=ants,
            coins=(self.coins[0], self.coins[1]),
            camps_hp=(self.bases[0].hp, self.bases[1].hp),
        )

    def sync_public_round_state(self, public_state: PublicRoundState) -> None:
        self.round_index = public_state.round_index
        self.coins[0], self.coins[1] = public_state.coins
        self.bases[0].hp, self.bases[1].hp = public_state.camps_hp
        tower_map = {tower.tower_id: tower for tower in self.towers}
        synced_towers: list[Tower] = []
        for tower_id, player, x, y, tower_type, cooldown in public_state.towers:
            tower = tower_map.get(tower_id, Tower(tower_id, player, x, y, TowerType(tower_type), float(cooldown)))
            tower.player = player
            tower.x = x
            tower.y = y
            tower.tower_type = TowerType(tower_type)
            tower.cooldown_clock = float(cooldown)
            synced_towers.append(tower)
        self.towers = synced_towers
        ant_map = {ant.ant_id: ant for ant in self.ants}
        synced_ants: list[Ant] = []
        for ant_id, player, x, y, hp, level, age, status in public_state.ants:
            ant = ant_map.get(ant_id, Ant(ant_id, player, x, y, hp, level, age=age, status=AntStatus(status)))
            ant.player = player
            ant.x = x
            ant.y = y
            ant.hp = hp
            ant.level = level
            ant.age = age
            ant.status = AntStatus(status)
            synced_ants.append(ant)
        self.ants = synced_ants
        if self.towers:
            self.next_tower_id = max(tower.tower_id for tower in self.towers) + 1
        if self.ants:
            self.next_ant_id = max(ant.ant_id for ant in self.ants) + 1

    def tower_spread_score(self, player: int) -> float:
        towers = self.towers_of(player)
        if len(towers) < 2:
            return 0.0
        penalty = 0.0
        for index, tower in enumerate(towers[:-1]):
            for other in towers[index + 1 :]:
                distance = hex_distance(tower.x, tower.y, other.x, other.y)
                if distance <= 3:
                    penalty += 5.0
                elif distance <= 6:
                    penalty += 2.0
        return -penalty

    def slot_priority(self, player: int, x: int, y: int) -> float:
        try:
            order = self.strategic_slots(player).index((x, y))
        except ValueError:
            order = len(self.strategic_slots(player))
        priority = max(0.0, 24.0 - order * 0.6)
        priority *= CENTERLINE_WEIGHTS.get((x, y), 1.0)
        base_x, base_y = PLAYER_BASES[player]
        priority += hex_distance(x, y, base_x, base_y) * 0.4
        return priority
