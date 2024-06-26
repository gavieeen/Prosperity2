from datamodel import OrderDepth, TradingState, Order
from typing import List
import numpy as np
import collections
import copy
from collections import defaultdict
import math

empty_dict = {"AMETHYST": 0, "STARFRUIT": 0}


def def_value():
    return copy.deepcopy(empty_dict)


INF = int(1e9)


class Trader:
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        orders: List[Order] = []

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            # Calculate acceptable price and standard deviation for the current product
            if product == "AMETHYST":
                acceptable_price = 9999.897360703812
                std = 3.1830079410159877
            elif product == "STARFRUIT":
                acceptable_price = 5049.300368756286
                std = 13.641363961465302
                # Implement Fourier sine transform for STARFRUIT
                orders += self.fourier_trading_strategy(order_depth)
            else:
                print("Unknown product:", product)
                continue

            print("Product:", product)
            print("Acceptable price:", acceptable_price)
            print("Buy Order depth:", len(order_depth.buy_orders))
            print("Sell Order depth:", len(order_depth.sell_orders))

            # Process sell orders
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price - std:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            # Process buy orders
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price + std:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on the next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData

    def fourier_trading_strategy(self, order_depth: OrderDepth) -> List[Order]:
        # Extract price data
        prices = list(order_depth.sell_orders.keys()) + list(order_depth.buy_orders.keys())
        # Calculate the Fourier sine transform
        fourier_transform = np.fft.fft(np.sin(prices))
        # Find the dominant frequency
        dominant_freq = np.argmax(np.abs(fourier_transform))
        
        # Based on the dominant frequency, decide whether to buy or sell
        if dominant_freq < 0:
            # If dominant frequency is positive, place a buy order
            print("Dominant frequency is positive, place BUY order")
            # Assuming a quantity of 1 for simplicity
            return [Order("STARFRUIT", max(prices), 1)]
        else:
            # If dominant frequency is negative, place a sell order
            print("Dominant frequency is negative, place SELL order")
            # Assuming a quantity of 1 for simplicity
            return [Order("STARFRUIT", min(prices), -1)]