#include "../include/coin.h"
#include <algorithm>
#include <cmath>
#include <tuple>

constexpr int INITIAL_COIN = 50, INITIAL_TOWER_BUILD_PRICE = 15,
              INITIAL_BARRACK_BUILD_PRICE = 0, INITIAL_BASIC_INCOME = 3,
              INITIAL_PENALTY = 0;
constexpr float TOWER_PRICE_INCREASING_RATIO = 2,
                BARRACK_PRICE_INCREASING_RATIO = 2;

Coin::Coin()
    : coin(INITIAL_COIN), basic_income(INITIAL_BASIC_INCOME),
      barrack_building_price(INITIAL_BARRACK_BUILD_PRICE),
      tower_building_price(INITIAL_TOWER_BUILD_PRICE),
      penalty(INITIAL_PENALTY) {}

int Coin::get_coin() const { return coin; }

void Coin::set_coin(int change) { coin += change; }

std::tuple<bool, int> Coin::basic_income_and_penalty() const {
    int income = basic_income - penalty;
    return std::tuple<bool, int>(
        income >= 0 || coin > 0,
        income); // if basic_income < 0 && coin <= 0, it will return false
}

void Coin::income_ant_kill(const Ant &killed_ant) {
    int level = killed_ant.get_level();
    const int coin_list[3] = {6, 10, 14};
    coin += coin_list[level];
}
void Coin::income_ant_arrive() {
    coin += 10;
}

bool Coin::isEnough_tower_build() const { return coin >= tower_building_price; }
constexpr int LEVEL1_PRICE = 60, LEVEL2_PRICE = 200;
void Coin::income_tower_destroy(const DefenseTower &tower) {
    switch (tower.get_level()) {
    case 0: // level 0 -> -1
        tower_building_price /= TOWER_PRICE_INCREASING_RATIO;
        coin += static_cast<int>(
            (9LL * tower_building_price * std::max(tower.get_hp(), 0)) /
            (10LL * std::max(tower.get_hp_limit(), 1)));
        break;
    case 1: // level 1 -> 0
        coin += static_cast<int>(
            (9LL * LEVEL1_PRICE * std::max(tower.get_hp(), 0)) /
            (10LL * std::max(tower.get_hp_limit(), 1)));
        break;
    case 2: // level 2 -> 1
        coin += static_cast<int>(
            (9LL * LEVEL2_PRICE * std::max(tower.get_hp(), 0)) /
            (10LL * std::max(tower.get_hp_limit(), 1)));
        break;
    }
}

bool Coin::isEnough_tower_upgrade(const DefenseTower &tower) const {
    switch (tower.get_level()) {
    case 0: // level 0 -> 1
        return coin >= LEVEL1_PRICE;
    case 1: // level 1 -> 2
        return coin >= LEVEL2_PRICE;
    default:
        return false;
    }
}

void Coin::cost_tower_build() {
    coin -= tower_building_price;
    tower_building_price = tower_building_price * TOWER_PRICE_INCREASING_RATIO;
}

void Coin::cost_tower_upgrade(const DefenseTower &tower) {
    switch (tower.get_level()) {
    case 0:
        coin -= LEVEL1_PRICE;
        return;
    case 1:
        coin -= LEVEL2_PRICE;
        return;

    default:
        return;
    }
}

bool Coin::isEnough_base_camp_upgrade(const int& level) const {
    const int cost[2] = {200, 250};
    return coin >= cost[level];
}

void Coin::cost_base_camp_upgrade(const int& level) {
    const int cost[2] = {200, 250};
    coin -= cost[level];
}

bool Coin::isEnough_item_applied(ItemType item) const {
    const int cost[4] = {150, 150, 100, 100};
    return coin >= cost[item];
}

void Coin::cost_item(ItemType item) {
    const int cost[4] = {150, 150, 100, 100};
    coin -= cost[item];
}
