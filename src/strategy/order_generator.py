"""
Order generation module for creating buy/sell orders based on portfolio and signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from src.strategy.portfolio_manager import Portfolio, PortfolioManager

logger = logging.getLogger(__name__)


class OrderGenerator:
    """Generates trading orders based on portfolio state and LLM signals."""

    def __init__(self, portfolio_manager: PortfolioManager, config: Dict[str, Any]):
        """
        Initialize order generator.

        Args:
            portfolio_manager: Portfolio manager instance
            config: Configuration dictionary with portfolio settings
        """
        self.portfolio_manager = portfolio_manager
        self.config = config

        # Extract portfolio configuration
        self.max_position_size = config.get('max_position_size', 0.15)
        self.max_total_positions = config.get('max_total_positions', 10)
        self.min_trade_size = config.get('min_trade_size', 1000)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)

        logger.info(f"Order generator initialized with max position size: {self.max_position_size:.1%}")

    def calculate_position_size(self, symbol: str, signal: Dict, current_price: float,
                              market_data: Optional[pd.DataFrame] = None) -> Tuple[int, float]:
        """
        Calculate optimal position size based on risk and signal strength.

        Args:
            symbol: Stock symbol
            signal: Trading signal dictionary
            current_price: Current stock price
            market_data: Optional market data for volatility calculation

        Returns:
            Tuple of (shares, risk_score)
        """
        try:
            # Get signal parameters
            signal_type = signal.get('signal', 'HOLD')
            confidence = signal.get('confidence', 0.5)
            strength = signal.get('strength', 0.0)

            if signal_type == 'HOLD':
                return 0, 0.0

            # Get current portfolio state
            portfolio = self.portfolio_manager.portfolio
            current_prices = {symbol: current_price}
            total_value = portfolio.get_total_value(current_prices)
            available_cash = portfolio.get_cash()

            # Calculate base position size based on risk per trade
            risk_amount = total_value * self.risk_per_trade

            # Adjust for signal confidence and strength
            confidence_multiplier = confidence
            strength_multiplier = min(strength * 2, 1.0)  # Cap at 1.0

            # Calculate volatility adjustment if market data available
            volatility_multiplier = 1.0
            if market_data is not None and 'returns' in market_data.columns:
                volatility = market_data['returns'].std() * np.sqrt(252)  # Annualized
                # Higher volatility = smaller position
                volatility_multiplier = max(0.5, 1.0 - volatility)

            # Calculate adjusted risk amount
            adjusted_risk = risk_amount * confidence_multiplier * strength_multiplier * volatility_multiplier

            # Calculate shares based on risk
            shares_by_risk = int(adjusted_risk / current_price) if current_price > 0 else 0

            # Calculate shares based on available cash and max position size
            max_position_value = total_value * self.max_position_size
            shares_by_position = int(max_position_value / current_price) if current_price > 0 else 0

            # Calculate shares based on available cash
            shares_by_cash = int(available_cash / current_price) if current_price > 0 else 0

            # Take minimum of all constraints
            shares = min(shares_by_risk, shares_by_position, shares_by_cash)

            # Apply minimum trade size constraint
            trade_value = shares * current_price
            if trade_value < self.min_trade_size:
                logger.debug(f"Trade value {trade_value} below minimum {self.min_trade_size} for {symbol}")
                shares = 0

            logger.debug(f"Position sizing for {symbol}: risk={shares_by_risk}, position={shares_by_position}, cash={shares_by_cash}, final={shares}")

            # Calculate risk score (0-1, lower is better)
            position_value = shares * current_price
            risk_score = position_value / total_value if total_value > 0 else 0

            logger.info(f"Position sizing for {symbol}: {shares} shares, risk score: {risk_score:.3f}")

            return shares, risk_score

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0, 1.0

    def generate_orders(self, signals: Dict[str, Dict], market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Generate orders based on portfolio and signals.

        Args:
            signals: Dictionary of trading signals by symbol
            market_data: Dictionary of market data by symbol

        Returns:
            List of order dictionaries
        """
        orders = []
        portfolio = self.portfolio_manager.portfolio

        # Get current prices from market data
        current_prices = {}
        for symbol, data in market_data.items():
            if data is not None and not data.empty and 'close' in data.columns:
                current_prices[symbol] = data['close'].iloc[-1]

        # Sort signals by confidence (highest first)
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1].get('confidence', 0),
            reverse=True
        )

        for symbol, signal in sorted_signals:
            if symbol not in current_prices:
                logger.warning(f"No current price available for {symbol}")
                continue

            current_price = current_prices[symbol]
            signal_type = signal.get('signal', 'HOLD')

            logger.debug(f"Processing {symbol}: {signal_type} signal")

            if signal_type == 'HOLD':
                logger.debug(f"Skipping {symbol} - HOLD signal")
                continue

            # Calculate position size
            market_df = market_data.get(symbol)
            shares, risk_score = self.calculate_position_size(
                symbol, signal, current_price, market_df
            )

            if shares <= 0:
                continue

            # Check if we already have a position
            current_position = portfolio.get_position(symbol)

            # Generate order based on signal and current position
            if signal_type == 'BUY':
                if current_position and current_position['shares'] > 0:
                    # Already have position, skip or consider adding
                    logger.info(f"Already have position in {symbol}, skipping BUY signal")
                    continue
                else:
                    # Generate BUY order
                    order = self._create_buy_order(symbol, shares, current_price, signal, risk_score)
                    orders.append(order)

            elif signal_type == 'SELL':
                if current_position and current_position['shares'] > 0:
                    # Generate SELL order for existing position
                    sell_shares = min(shares, current_position['shares'])
                    order = self._create_sell_order(symbol, sell_shares, current_price, signal, risk_score)
                    orders.append(order)
                else:
                    # No position to sell, skip
                    logger.info(f"No position in {symbol} to sell")
                    continue

        # Sort orders by risk score (lowest risk first)
        orders.sort(key=lambda x: x.get('risk_score', 1.0))

        logger.info(f"Generated {len(orders)} orders")
        return orders

    def _create_buy_order(self, symbol: str, shares: int, price: float,
                         signal: Dict, risk_score: float) -> Dict:
        """Create a BUY order."""
        return {
            "symbol": symbol,
            "action": "BUY",
            "order_type": "MARKET",
            "quantity": shares,
            "estimated_price": price,
            "estimated_cost": shares * price,
            "reason": f"LLM signal: {signal.get('signal', 'BUY')}, "
                     f"confidence: {signal.get('confidence', 0):.1%}, "
                     f"strength: {signal.get('strength', 0):.2f}",
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat(),
            "signal_data": {
                "confidence": signal.get('confidence', 0),
                "strength": signal.get('strength', 0),
                "prediction": signal.get('prediction', 0)
            }
        }

    def _create_sell_order(self, symbol: str, shares: int, price: float,
                          signal: Dict, risk_score: float) -> Dict:
        """Create a SELL order."""
        return {
            "symbol": symbol,
            "action": "SELL",
            "order_type": "MARKET",
            "quantity": shares,
            "estimated_price": price,
            "estimated_proceeds": shares * price,
            "reason": f"LLM signal: {signal.get('signal', 'SELL')}, "
                     f"confidence: {signal.get('confidence', 0):.1%}, "
                     f"strength: {signal.get('strength', 0):.2f}",
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat(),
            "signal_data": {
                "confidence": signal.get('confidence', 0),
                "strength": signal.get('strength', 0),
                "prediction": signal.get('prediction', 0)
            }
        }

    def validate_orders(self, orders: List[Dict]) -> List[Dict]:
        """
        Validate orders against portfolio constraints.

        Args:
            orders: List of orders to validate

        Returns:
            List of valid orders
        """
        valid_orders = []
        portfolio = self.portfolio_manager.portfolio

        for order in orders:
            symbol = order['symbol']
            action = order['action']
            quantity = order['quantity']
            price = order['estimated_price']

            # Check if we can afford the trade
            if action == 'BUY':
                cost = quantity * price
                if not self.portfolio_manager.can_afford_trade(symbol, quantity, price):
                    logger.warning(f"Cannot afford BUY order for {symbol}: ${cost:,.2f}")
                    continue

                # Check position size limit
                current_prices = {symbol: price}
                total_value = portfolio.get_total_value(current_prices)
                position_value = quantity * price

                if position_value > total_value * self.max_position_size:
                    # Reduce quantity to fit within limits
                    max_shares = int((total_value * self.max_position_size) / price)
                    if max_shares > 0:
                        order['quantity'] = max_shares
                        order['estimated_cost'] = max_shares * price
                        logger.info(f"Reduced BUY order for {symbol} to {max_shares} shares")
                    else:
                        continue

            elif action == 'SELL':
                # Check if we have enough shares
                current_position = portfolio.get_position(symbol)
                if not current_position or current_position['shares'] < quantity:
                    logger.warning(f"Insufficient shares to sell {quantity} of {symbol}")
                    continue

            valid_orders.append(order)

        logger.info(f"Validated {len(valid_orders)} out of {len(orders)} orders")
        return valid_orders

    def get_order_summary(self, orders: List[Dict]) -> Dict[str, Any]:
        """
        Get summary of generated orders.

        Args:
            orders: List of orders

        Returns:
            Order summary dictionary
        """
        if not orders:
            return {
                "total_orders": 0,
                "buy_orders": 0,
                "sell_orders": 0,
                "total_cost": 0.0,
                "total_proceeds": 0.0,
                "average_risk_score": 0.0
            }

        buy_orders = [o for o in orders if o['action'] == 'BUY']
        sell_orders = [o for o in orders if o['action'] == 'SELL']

        total_cost = sum(o.get('estimated_cost', 0) for o in buy_orders)
        total_proceeds = sum(o.get('estimated_proceeds', 0) for o in sell_orders)
        avg_risk_score = np.mean([o.get('risk_score', 0) for o in orders])

        return {
            "total_orders": len(orders),
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
            "total_cost": total_cost,
            "total_proceeds": total_proceeds,
            "net_cash_flow": total_proceeds - total_cost,
            "average_risk_score": avg_risk_score
        }


def create_order_generator(portfolio_manager: PortfolioManager, config: Dict[str, Any]) -> OrderGenerator:
    """
    Factory function to create order generator.

    Args:
        portfolio_manager: Portfolio manager instance
        config: Configuration dictionary

    Returns:
        Order generator instance
    """
    return OrderGenerator(portfolio_manager, config)
