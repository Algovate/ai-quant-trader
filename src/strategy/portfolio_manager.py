"""
Portfolio management module for tracking positions and generating orders.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio class to track positions, cash, and transactions."""

    def __init__(self, initial_cash: float = 100000, portfolio_file: str = "data/portfolio.json"):
        """
        Initialize portfolio.

        Args:
            initial_cash: Starting cash amount
            portfolio_file: Path to portfolio JSON file
        """
        self.portfolio_file = portfolio_file
        self.data = {
            "cash": initial_cash,
            "positions": {},
            "transactions": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "initial_cash": initial_cash
            }
        }

        # Load existing portfolio if file exists
        if os.path.exists(portfolio_file):
            self.load_from_file()

        logger.info(f"Portfolio initialized with ${initial_cash:,.2f} cash")

    def load_from_file(self) -> bool:
        """Load portfolio from JSON file."""
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Portfolio loaded from {self.portfolio_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load portfolio from {self.portfolio_file}: {e}")
            return False

    def save_to_file(self) -> bool:
        """Save portfolio to JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)

            # Update metadata
            self.data["metadata"]["updated_at"] = datetime.now().isoformat()

            # Save with backup
            backup_file = self.portfolio_file + ".backup"
            if os.path.exists(self.portfolio_file):
                os.rename(self.portfolio_file, backup_file)

            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)

            # Remove backup if save successful
            if os.path.exists(backup_file):
                os.remove(backup_file)

            logger.info(f"Portfolio saved to {self.portfolio_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save portfolio to {self.portfolio_file}: {e}")
            # Restore backup if save failed
            backup_file = self.portfolio_file + ".backup"
            if os.path.exists(backup_file):
                os.rename(backup_file, self.portfolio_file)
            return False

    def get_cash(self) -> float:
        """Get available cash."""
        return self.data["cash"]

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        return self.data["positions"].copy()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol."""
        return self.data["positions"].get(symbol)

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current value of a position."""
        position = self.get_position(symbol)
        if not position:
            return 0.0
        return position["shares"] * current_price

    def get_total_position_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total value of all positions."""
        total = 0.0
        for symbol, position in self.data["positions"].items():
            if symbol in current_prices:
                total += self.get_position_value(symbol, current_prices[symbol])
        return total

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        return self.get_cash() + self.get_total_position_value(current_prices)

    def add_position(self, symbol: str, shares: int, price: float,
                    transaction_type: str = "BUY") -> bool:
        """
        Add or update position.

        Args:
            symbol: Stock symbol
            shares: Number of shares (positive for buy, negative for sell)
            price: Price per share
            transaction_type: "BUY" or "SELL"
        """
        try:
            current_position = self.get_position(symbol)
            cost = shares * price

            if transaction_type == "BUY":
                if current_position:
                    # Update existing position
                    total_shares = current_position["shares"] + shares
                    total_cost = (current_position["shares"] * current_position["avg_price"] +
                                shares * price)
                    new_avg_price = total_cost / total_shares if total_shares > 0 else 0

                    current_position["shares"] = total_shares
                    current_position["avg_price"] = new_avg_price
                    current_position["current_price"] = price
                else:
                    # Create new position
                    self.data["positions"][symbol] = {
                        "shares": shares,
                        "avg_price": price,
                        "current_price": price
                    }

                # Deduct cost from cash
                self.data["cash"] -= cost

            elif transaction_type == "SELL":
                if not current_position or current_position["shares"] < shares:
                    logger.warning(f"Insufficient shares to sell {shares} of {symbol}")
                    return False

                # Update position
                current_position["shares"] -= shares
                current_position["current_price"] = price

                # Add proceeds to cash
                self.data["cash"] += cost

                # Remove position if no shares left
                if current_position["shares"] <= 0:
                    del self.data["positions"][symbol]

            # Record transaction
            transaction = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": transaction_type,
                "shares": abs(shares),
                "price": price,
                "total_cost": cost if transaction_type == "BUY" else -cost,
                "cash_after": self.data["cash"]
            }
            self.data["transactions"].append(transaction)

            logger.info(f"Position updated: {transaction_type} {shares} {symbol} @ ${price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to update position for {symbol}: {e}")
            return False

    def close_position(self, symbol: str, current_price: float) -> bool:
        """Close entire position for a symbol."""
        position = self.get_position(symbol)
        if not position:
            return False

        return self.add_position(symbol, -position["shares"], current_price, "SELL")

    def get_transactions(self, limit: Optional[int] = None) -> List[Dict]:
        """Get transaction history."""
        transactions = self.data["transactions"]
        if limit:
            return transactions[-limit:]
        return transactions.copy()

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get portfolio summary."""
        total_position_value = self.get_total_position_value(current_prices)
        total_value = self.get_cash() + total_position_value

        return {
            "cash": self.data["cash"],
            "total_position_value": total_position_value,
            "total_value": total_value,
            "position_count": len(self.data["positions"]),
            "positions": {
                symbol: {
                    "shares": pos["shares"],
                    "avg_price": pos["avg_price"],
                    "current_price": current_prices.get(symbol, pos.get("current_price", 0)),
                    "current_value": self.get_position_value(symbol, current_prices.get(symbol, 0)),
                    "unrealized_pnl": self.get_position_value(symbol, current_prices.get(symbol, 0)) -
                                    (pos["shares"] * pos["avg_price"])
                }
                for symbol, pos in self.data["positions"].items()
            },
            "metadata": self.data["metadata"]
        }

    def reset_portfolio(self, initial_cash: float = 100000) -> bool:
        """Reset portfolio to initial state."""
        try:
            self.data = {
                "cash": initial_cash,
                "positions": {},
                "transactions": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "initial_cash": initial_cash
                }
            }
            logger.info(f"Portfolio reset with ${initial_cash:,.2f} cash")
            return True
        except Exception as e:
            logger.error(f"Failed to reset portfolio: {e}")
            return False


class PortfolioManager:
    """Portfolio manager with additional utility methods."""

    def __init__(self, portfolio: Portfolio):
        """Initialize with a Portfolio instance."""
        self.portfolio = portfolio

    def can_afford_trade(self, symbol: str, shares: int, price: float) -> bool:
        """Check if portfolio can afford a trade."""
        cost = shares * price
        return self.portfolio.get_cash() >= cost

    def get_max_shares(self, symbol: str, price: float) -> int:
        """Calculate maximum shares that can be bought."""
        available_cash = self.portfolio.get_cash()
        return int(available_cash / price) if price > 0 else 0

    def get_position_weight(self, symbol: str, current_prices: Dict[str, float]) -> float:
        """Calculate position weight as percentage of total portfolio."""
        if symbol not in current_prices:
            return 0.0

        position_value = self.portfolio.get_position_value(symbol, current_prices[symbol])
        total_value = self.portfolio.get_total_value(current_prices)

        return position_value / total_value if total_value > 0 else 0.0

    def get_available_cash_percentage(self) -> float:
        """Get available cash as percentage of total portfolio."""
        # This is a simplified calculation - in practice you'd need current prices
        # For now, return cash as percentage of initial cash
        initial_cash = self.portfolio.data["metadata"]["initial_cash"]
        return self.portfolio.get_cash() / initial_cash if initial_cash > 0 else 0.0
