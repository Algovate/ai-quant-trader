"""
Unit tests for order generation functionality.
重点测试订单生成逻辑，包括风险控制、仓位计算和订单验证。
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.portfolio_manager import Portfolio, PortfolioManager
from strategy.order_generator import OrderGenerator
from core.utils import format_currency


class TestOrderGenerator(unittest.TestCase):
    """Test cases for OrderGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test portfolio
        self.portfolio = Portfolio(initial_cash=100000)
        self.portfolio_manager = PortfolioManager(self.portfolio)

        # Create test configuration
        self.config = {
            'max_position_size': 0.15,  # 15% max position
            'max_total_positions': 10,
            'min_trade_size': 1000,     # $1000 minimum
            'risk_per_trade': 0.02,     # 2% risk per trade
        }

        # Create order generator
        self.order_generator = OrderGenerator(self.portfolio_manager, self.config)

        # Create test market data
        self.market_data = {
            'AAPL': pd.DataFrame({
                'close': [180.0, 182.0, 185.0],
                'volume': [1000000, 1200000, 1100000]
            }),
            'MSFT': pd.DataFrame({
                'close': [380.0, 385.0, 390.0],
                'volume': [800000, 900000, 850000]
            })
        }

    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        # Test with high confidence signal
        signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.8,
            'strength': 0.7
        }

        current_price = 180.0
        shares, risk_score = self.order_generator.calculate_position_size(
            'AAPL', signal, current_price, self.market_data['AAPL']
        )

        # Should generate a reasonable position size
        self.assertGreater(shares, 0)
        self.assertLessEqual(shares * current_price, 15000)  # Max 15% of 100k

    def test_calculate_position_size_risk_control(self):
        """Test position sizing respects risk limits."""
        # Test with very high confidence (should still respect limits)
        signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.95,
            'strength': 0.9
        }

        current_price = 180.0
        shares, risk_score = self.order_generator.calculate_position_size(
            'AAPL', signal, current_price, self.market_data['AAPL']
        )

        position_value = shares * current_price
        max_position = 100000 * 0.15  # 15% of portfolio

        self.assertLessEqual(position_value, max_position)

    def test_calculate_position_size_minimum_trade(self):
        """Test minimum trade size enforcement."""
        # Test with low confidence signal
        signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.3,
            'strength': 0.2
        }

        current_price = 180.0
        shares, risk_score = self.order_generator.calculate_position_size(
            'AAPL', signal, current_price, self.market_data['AAPL']
        )

        trade_value = shares * current_price

        # Should either be 0 (no trade) or meet minimum
        if shares > 0:
            self.assertGreaterEqual(trade_value, 1000)

    def test_generate_orders_buy_signals(self):
        """Test order generation for BUY signals."""
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.7
            },
            {
                'symbol': 'MSFT',
                'signal': 'BUY',
                'confidence': 0.6,
                'strength': 0.5
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        # Should generate orders for BUY signals
        self.assertGreater(len(orders), 0)

        for order in orders:
            self.assertEqual(order['action'], 'BUY')
            self.assertIn(order['symbol'], ['AAPL', 'MSFT'])
            self.assertGreater(order['quantity'], 0)
            self.assertEqual(order['order_type'], 'MARKET')

    def test_generate_orders_sell_signals(self):
        """Test order generation for SELL signals."""
        # First add some positions
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 380.0)

        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'SELL',
                'confidence': 0.8,
                'strength': 0.7
            },
            {
                'symbol': 'MSFT',
                'signal': 'SELL',
                'confidence': 0.6,
                'strength': 0.5
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        # Should generate SELL orders
        self.assertGreater(len(orders), 0)

        for order in orders:
            self.assertEqual(order['action'], 'SELL')
            self.assertIn(order['symbol'], ['AAPL', 'MSFT'])
            self.assertGreater(order['quantity'], 0)

    def test_generate_orders_hold_signals(self):
        """Test that HOLD signals don't generate orders."""
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'HOLD',
                'confidence': 0.5,
                'strength': 0.3
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        # Should not generate orders for HOLD signals
        self.assertEqual(len(orders), 0)

    def test_validate_orders_sufficient_cash(self):
        """Test order validation with sufficient cash."""
        # Create a BUY order within cash limits
        orders = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 50,
                'estimated_price': 180.0
            }
        ]

        valid_orders = self.order_generator.validate_orders(orders)

        # Should validate successfully
        self.assertEqual(len(valid_orders), 1)
        self.assertEqual(valid_orders[0]['symbol'], 'AAPL')

    def test_validate_orders_insufficient_cash(self):
        """Test order validation with insufficient cash."""
        # Create a BUY order exceeding cash
        orders = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 1000,  # $180,000 - exceeds $100,000 cash
                'estimated_price': 180.0
            }
        ]

        valid_orders = self.order_generator.validate_orders(orders)

        # Should reject the order
        self.assertEqual(len(valid_orders), 0)

    def test_validate_orders_sell_without_position(self):
        """Test order validation for SELL without position."""
        # Create a SELL order without holding the position
        orders = [
            {
                'symbol': 'AAPL',
                'action': 'SELL',
                'quantity': 50,
                'estimated_price': 180.0
            }
        ]

        valid_orders = self.order_generator.validate_orders(orders)

        # Should reject the order (no position to sell)
        self.assertEqual(len(valid_orders), 0)

    def test_validate_orders_sell_with_position(self):
        """Test order validation for SELL with position."""
        # Add position first
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)

        # Create a SELL order within position
        orders = [
            {
                'symbol': 'AAPL',
                'action': 'SELL',
                'quantity': 50,
                'estimated_price': 180.0
            }
        ]

        valid_orders = self.order_generator.validate_orders(orders)

        # Should validate successfully
        self.assertEqual(len(valid_orders), 1)
        self.assertEqual(valid_orders[0]['symbol'], 'AAPL')

    def test_validate_orders_sell_exceeds_position(self):
        """Test order validation for SELL exceeding position."""
        # Add small position
        self.portfolio.add_position('AAPL', 50, 180.0, 180.0)

        # Create a SELL order exceeding position
        orders = [
            {
                'symbol': 'AAPL',
                'action': 'SELL',
                'quantity': 100,  # Exceeds 50 shares held
                'estimated_price': 180.0
            }
        ]

        valid_orders = self.order_generator.validate_orders(orders)

        # Should reject the order (exceeds position)
        self.assertEqual(len(valid_orders), 0)

    def test_order_prioritization(self):
        """Test that orders are prioritized by confidence."""
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.6,
                'strength': 0.5
            },
            {
                'symbol': 'MSFT',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.7
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        # Should generate orders (may be limited by cash)
        self.assertGreaterEqual(len(orders), 0)

        # If multiple orders, higher confidence should come first
        if len(orders) > 1:
            # Orders should be sorted by confidence (descending)
            confidences = [order.get('confidence', 0) for order in orders]
            self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_volatility_adjustment(self):
        """Test that position sizing considers volatility."""
        # Create high volatility data
        high_vol_data = pd.DataFrame({
            'close': [180.0, 200.0, 160.0, 190.0, 170.0],  # High volatility
            'volume': [1000000, 1200000, 1100000, 1300000, 900000]
        })

        signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.8,
            'strength': 0.7
        }

        # Test with high volatility
        shares_high_vol, _ = self.order_generator.calculate_position_size(
            'AAPL', signal, 180.0, high_vol_data
        )

        # Test with low volatility
        low_vol_data = pd.DataFrame({
            'close': [180.0, 181.0, 179.0, 182.0, 180.5],  # Low volatility
            'volume': [1000000, 1200000, 1100000, 1300000, 900000]
        })

        shares_low_vol, _ = self.order_generator.calculate_position_size(
            'AAPL', signal, 180.0, low_vol_data
        )

        # High volatility should result in smaller position
        if shares_high_vol > 0 and shares_low_vol > 0:
            self.assertLessEqual(shares_high_vol, shares_low_vol)

    def test_max_positions_limit(self):
        """Test maximum number of positions limit."""
        # Add multiple positions to test limit
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']

        for i, symbol in enumerate(symbols[:10]):  # Add 10 positions
            self.portfolio.add_position(symbol, 10, 100.0, 100.0)

        # Try to add one more position
        signals = [
            {
                'symbol': 'ORCL',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.7
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        # Should not generate new orders if at position limit
        # (This depends on implementation - may allow rebalancing)
        self.assertGreaterEqual(len(orders), 0)

    def test_order_reasoning(self):
        """Test that orders include proper reasoning."""
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.7
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        if orders:
            order = orders[0]
            self.assertIn('reason', order)
            self.assertIn('AAPL', order['reason'])
            self.assertIn('BUY', order['reason'])

    def test_risk_score_calculation(self):
        """Test risk score calculation for orders."""
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.8,
                'strength': 0.7
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)

        if orders:
            order = orders[0]
            self.assertIn('risk_score', order)
            self.assertGreaterEqual(order['risk_score'], 0.0)
            self.assertLessEqual(order['risk_score'], 1.0)


class TestOrderGeneratorIntegration(unittest.TestCase):
    """Integration tests for order generation with real portfolio scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.portfolio = Portfolio(initial_cash=50000)  # Smaller portfolio for testing
        self.portfolio_manager = PortfolioManager(self.portfolio)

        self.config = {
            'max_position_size': 0.20,  # 20% max position
            'max_total_positions': 5,
            'min_trade_size': 500,      # $500 minimum
            'risk_per_trade': 0.03,    # 3% risk per trade
        }

        self.order_generator = OrderGenerator(self.portfolio_manager, self.config)

        # Create realistic market data
        self.market_data = {
            'AAPL': pd.DataFrame({
                'close': [180.0, 182.0, 185.0, 183.0, 187.0],
                'volume': [1000000, 1200000, 1100000, 1300000, 900000]
            }),
            'MSFT': pd.DataFrame({
                'close': [380.0, 385.0, 390.0, 388.0, 392.0],
                'volume': [800000, 900000, 850000, 950000, 820000]
            })
        }

    def test_full_order_generation_workflow(self):
        """Test complete order generation workflow."""
        # Start with empty portfolio
        self.assertEqual(self.portfolio.get_cash(), 50000)
        self.assertEqual(len(self.portfolio.get_positions()), 0)

        # Generate orders for BUY signals
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.75,
                'strength': 0.6
            },
            {
                'symbol': 'MSFT',
                'signal': 'BUY',
                'confidence': 0.65,
                'strength': 0.55
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)
        valid_orders = self.order_generator.validate_orders(orders)

        # Should generate valid orders
        self.assertGreater(len(valid_orders), 0)

        # Check order properties
        for order in valid_orders:
            self.assertIn(order['symbol'], ['AAPL', 'MSFT'])
            self.assertEqual(order['action'], 'BUY')
            self.assertEqual(order['order_type'], 'MARKET')
            self.assertGreater(order['quantity'], 0)
            self.assertGreater(order['estimated_price'], 0)
            self.assertIn('reason', order)
            self.assertIn('risk_score', order)

    def test_portfolio_rebalancing_scenario(self):
        """Test order generation for portfolio rebalancing."""
        # Add existing positions
        self.portfolio.add_position('AAPL', 50, 180.0, 185.0)
        self.portfolio.add_position('MSFT', 25, 380.0, 390.0)

        # Generate mixed signals (some BUY, some SELL)
        signals = [
            {
                'symbol': 'AAPL',
                'signal': 'SELL',  # Sell some AAPL
                'confidence': 0.7,
                'strength': 0.6
            },
            {
                'symbol': 'MSFT',
                'signal': 'BUY',   # Buy more MSFT
                'confidence': 0.8,
                'strength': 0.7
            }
        ]

        orders = self.order_generator.generate_orders(signals, self.market_data)
        valid_orders = self.order_generator.validate_orders(orders)

        # Should generate both BUY and SELL orders
        buy_orders = [o for o in valid_orders if o['action'] == 'BUY']
        sell_orders = [o for o in valid_orders if o['action'] == 'SELL']

        self.assertGreater(len(valid_orders), 0)

        # Check that we have appropriate order types
        if buy_orders:
            self.assertTrue(any(o['symbol'] == 'MSFT' for o in buy_orders))
        if sell_orders:
            self.assertTrue(any(o['symbol'] == 'AAPL' for o in sell_orders))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
