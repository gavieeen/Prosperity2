from datamodel import OrderDepth, TradingState, Order
from typing import List, Tuple
import numpy as np
import pandas as pd
import collections
import copy
from collections import defaultdict
import math
import string

VWAP_DEPTH: int = 3
OFI_DEPTH: int = 3
TICK_SIZE: int = 1
MOMENTUM_WINDOW: int = 100


class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the data.

        Parameters:
        - X: Independent variable (features)
        - y: Dependent variable (target)
        """
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = 0
        denominator = 0

        for i in range(len(X)):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * x_mean)

    def predict(self, X):
        """
        Predict the target variable for new data points.

        Parameters:
        - X: New data points (features)

        Returns:
        - Predicted target variable
        """
        return self.slope * X + self.intercept


def min_qty(book: OrderDepth, level: int) -> int:
    ask_qty = 0
    ask_depth = len(book.sell_orders)
    level__ = min(level, ask_depth)
    sell_levels_list = list(book.sell_orders.items())

    for idx in range(level__):
        if idx >= ask_depth:
            break

        arr = sell_levels_list[idx]
        if float(-arr[1]) <= 0.0:
            level__ += 1
            continue
        ask_qty += float(-arr[1])

    bid_qty = 0
    bid_depth = len(book.buy_orders)
    level__ = min(level, bid_depth)
    buy_levels_list = list(book.buy_orders.items())

    for idx in range(level__):
        if idx >= bid_depth:
            break
        arr = buy_levels_list[idx]
        if float(arr[1]) <= 0.0:
            level__ += 1
            continue
        bid_qty += float(arr[1])

    return min(bid_qty, ask_qty)


def ask_vwap_qty(book: OrderDepth, level: int, qty: int) -> float:
    price__ = 0
    rem_qty = qty
    ask_depth = len(book.sell_orders)
    level__ = min(level, ask_depth)
    sell_levels_list = list(book.sell_orders.items())

    for idx in range(level__):
        if idx >= ask_depth:
            break
        arr = sell_levels_list[idx]
        if float(-arr[1]) <= 0.0:
            level__ += 1
            continue
        rem_qty -= float(-arr[1])
        if rem_qty <= 0:
            rem_qty += float(-arr[1])
            price__ += float(arr[0]) * rem_qty
            rem_qty = 0
            break
        else:
            price__ += float(arr[0]) * float(-arr[1])

    return price__ / qty


def bid_vwap_qty(book: OrderDepth, level: int, qty: int) -> float:
    price__ = 0
    rem_qty = qty
    bid_depth = len(book.buy_orders)
    level__ = min(level, bid_depth)
    buy_levels_list = list(book.buy_orders.items())

    for idx in range(level__):
        if idx >= bid_depth:
            break
        arr = buy_levels_list[idx]
        if float(arr[1]) <= 0.0:
            level__ += 1
            continue
        rem_qty -= float(arr[1])
        if rem_qty <= 0:
            rem_qty += float(arr[1])
            price__ += float(arr[0]) * rem_qty
            rem_qty = 0
            break
        else:
            price__ += float(arr[0]) * float(arr[1])

    return price__ / qty


def update_vwap(
    book: OrderDepth,
) -> float:  # passing the book so that requests are minimised
    vol_wt_depth = VWAP_DEPTH
    qty = min_qty(book, vol_wt_depth)
    if qty <= 0:
        return 0.0

    assert vol_wt_depth > 0
    bid_to_use = bid_vwap_qty(book, vol_wt_depth, qty)
    ask_to_use = ask_vwap_qty(book, vol_wt_depth, qty)
    vwap__ = (bid_to_use + ask_to_use) / 2.0

    return vwap__


def get_inventory_adjusted_min_dist(min_dist, inv) -> int:
    return min_dist + max(0, inv) * min_dist


def get_bid_price(px, position, spread, tick) -> int:
    adjusted_px = px - (spread + max(0, position) / 32)
    return int((adjusted_px // tick) * tick)


def get_ask_price(px, position, spread, tick) -> int:
    adjusted_px = px + (spread + max(0, -position) / 32)
    truncated_px = (adjusted_px // tick) * tick

    if truncated_px < adjusted_px:
        return int(truncated_px + tick)
    else:
        return int(truncated_px)


def get_order_flow_imbalance(book: OrderDepth, depth: int) -> float:
    """
    Calculate the order flow imbalance (OFI) using the specified depth of the order book.
    The OFI is a measure of the excess of buy-side or sell-side pressure in the order book.

    Args:
        book (OrderDepth): The current order book state.
        depth (int): The number of levels to consider in the order book.

    Returns:
        float: The order flow imbalance value, positive for buy-side pressure, negative for sell-side pressure.
    """
    total_buy_volume = sum(
        float(vol) for price, vol in list(book.buy_orders.items())[:depth]
    )
    total_sell_volume = sum(
        float(-vol) for price, vol in list(book.sell_orders.items())[:depth]
    )

    total_volume = total_buy_volume + total_sell_volume
    if total_volume == 0:
        return 0.0

    ofi = (total_buy_volume - total_sell_volume) / total_volume
    return ofi


def get_price_prediction(
    symbol: str, ob_list: List[OrderDepth], position: int
) -> Tuple[float, float, float]:
    spread = 1
    vwap = update_vwap(ob_list[-1])
    price_list = [update_vwap(ob) for ob in ob_list[-MOMENTUM_WINDOW:]]
    momentum = (
        (price_list[-1] - (sum(price_list) / len(price_list))) / price_list[-1] * 100
    )
    vwap += (-20) * momentum

    return (
        vwap,
        get_bid_price(vwap, position, spread, TICK_SIZE),
        get_ask_price(vwap, position, spread, TICK_SIZE),
    )


class Trader:
    POSITION_LIMIT = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250, "STRAWBERRIES": 350,
                    "ROSES": 60, "GIFT_BASKET": 60}
    steps = 0
    last_sunlight = -1
    last_humidity = -1
    buy_gear = False
    sell_gear = False
    last_dg_price = 0

    def __init__(self):
        self.symbol_ob_collection = {}
        self.models = {}  # Dictionary to store linear regression models for each symbol

    def train_model(self, symbol: str, ob_list: List[OrderDepth], position: int):
        if symbol not in self.models:
            self.models[symbol] = LinearRegression()

        # Extract features from order book data
        features = []
        for ob in ob_list[-MOMENTUM_WINDOW:]:
            features.append(update_vwap(ob))

        # Convert features to numpy array and reshape for fitting the model
        X = np.array(features)

        # Calculate target variable (price) based on position and VWAP
        y = np.array(
            [update_vwap(ob) - (position * 20) for ob in ob_list[-MOMENTUM_WINDOW:]]
        )

        # Fit the model
        self.models[symbol].fit(X, y)

    def predict_price(self, symbol: str, position: int) -> float:
        if symbol not in self.models:
            return 0.0  # Return default price if model not trained

        # Predict price based on the latest VWAP and position
        latest_vwap = update_vwap(self.symbol_ob_collection[symbol][-1])
        return self.models[symbol].predict(latest_vwap) + (position * 20)

    def pairs_trading_orchids(self, position, order_depth, observations):
        orders = {"ORCHIDS": []}
        prods = ["ORCHIDS"]
        (
            osell,
            obuy,
            best_sell,
            best_buy,
            worst_sell,
            worst_buy,
            mid_price,
            vol_buy,
            vol_sell,
        ) = ({}, {}, {}, {}, {}, {}, {}, {}, {})
        for p in prods:
            osell[p] = collections.OrderedDict(
                sorted(order_depth[p].sell_orders.items())
            )
            obuy[p] = collections.OrderedDict(
                sorted(order_depth[p].buy_orders.items(), reverse=True)
            )

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol
            for price, vol in osell[p].items():
                vol_sell[p] += -vol

        if self.last_sunlight != -1 and (
            (observations["ORCHIDS"].sunlight - self.last_sunlight > 1)
            or (observations["ORCHIDS"].humidity - self.last_humidity > 1)
        ):
            self.buy_gear = True
        else:
            self.buy_gear = False

        if self.last_sunlight != -1 and (
            observations["ORCHIDS"].sunlight - self.last_sunlight < -1
            or observations["ORCHIDS"].humidity - self.last_humidity < -1
        ):
            self.sell_gear = True
        else:
            self.sell_gear = False

        if self.buy_gear:
            vol = self.POSITION_LIMIT["ORCHIDS"] - position["ORCHIDS"]
            orders["ORCHIDS"].append(Order("ORCHIDS", worst_sell["ORCHIDS"], vol))
        if self.sell_gear:
            vol = self.position["ORCHIDS"] + self.POSITION_LIMIT["ORCHIDS"]
            orders["ORCHIDS"].append(Order("ORCHIDS", worst_buy["ORCHIDS"], -vol))
        self.last_sunlight = observations["ORCHIDS"].sunlight
        self.last_humidity = observations["ORCHIDS"].humidity

        self.last_dg_price = mid_price["ORCHIDS"]

        return orders

    def run(self, state: TradingState):
        self.steps = (self.steps + 1)
        self.last_humidity = state.observations.conversionObservations["ORCHIDS"].humidity
        self.last_sunlight = state.observations.conversionObservations["ORCHIDS"].sunlight
            
        result = {}
        
        for product, order_depth in state.order_depths.items():
            orders = []
            if product not in self.symbol_ob_collection:
                self.symbol_ob_collection[product] = []
            self.symbol_ob_collection[product].append(order_depth)
            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue

            acceptable_price = 0
            std = 0

            if product in ["STARFRUIT"]:
                self.train_model(
                    product,
                    self.symbol_ob_collection[product],
                    state.position.get(product, 0),
                )
                acceptable_price = self.predict_price(
                    product, state.position.get(product, 0)
                )
            elif product == "AMETHYSTS":
                if order_depth.sell_orders:
                    orders.append(Order(product, 9995, 3))
                    orders.append(Order(product, 9996, 5))
                    orders.append(Order(product, 9998, 12))
                
                if order_depth.buy_orders:
                    orders.append(Order(product, 10005, -3))
                    orders.append(Order(product, 10004, -5))
                    orders.append(Order(product, 10002, -12))
                    
                result[product] = orders
            else:
                acceptable_price, bid_price, ask_price = get_price_prediction(product, self.symbol_ob_collection[product], state.position.get(product,0))

            # elif product == "ORCHIDS":
            #     orders = self.pairs_trading_orchids(
            #         state.position,
            #         state.order_depths,
            #         state.observations.conversionObservations,
            #     )
            #     if "ORCHIDS" not in result:
            #         result["ORCHIDS"] = []
            #     result["ORCHIDS"] += orders["ORCHIDS"]
            
            
            if product not in ["AMETHYST", "ORCHIDS"]:
                if order_depth.sell_orders:
                    best_ask_price, best_ask_amount = next(
                        iter(order_depth.sell_orders.items())
                    )
                    if acceptable_price - std > best_ask_price:
                        print("BUY", -best_ask_amount, best_ask_price)
                        orders.append(Order(product, best_ask_price, -best_ask_amount))

                if order_depth.buy_orders:
                    best_bid_price, best_bid_amount = next(
                        iter(order_depth.buy_orders.items())
                    )
                    if acceptable_price + std < best_bid_price:
                        print("SELL", best_bid_amount, best_bid_price)
                        orders.append(Order(product, best_bid_price, -best_bid_amount))

                result[product] = orders

        trader_data = "SAMPLE"
        conversions = 1
        return result, conversions, trader_data
