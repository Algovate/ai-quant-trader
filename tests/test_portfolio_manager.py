"""
Unit tests for portfolio management functionality.
重点测试投资组合管理，包括持仓跟踪、现金管理和交易记录。
"""

import unittest
import json
import os
import tempfile
import sys
from datetime import datetime
from unittest.mock import patch, mock_open

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.portfolio_manager import Portfolio, PortfolioManager


class TestPortfolio(unittest.TestCase):
    """Test cases for Portfolio class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.portfolio_file = self.temp_file.name

        self.portfolio = Portfolio(
            initial_cash=100000,
            portfolio_file=self.portfolio_file
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.portfolio_file):
            os.unlink(self.portfolio_file)

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.get_cash(), 100000)
        self.assertEqual(len(self.portfolio.get_positions()), 0)
        self.assertEqual(len(self.portfolio.get_transactions()), 0)
        self.assertIsNotNone(self.portfolio.data.get('metadata'))

    def test_add_position(self):
        """Test adding a position."""
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)

        positions = self.portfolio.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertIn('AAPL', positions)

        aapl_pos = positions['AAPL']
        self.assertEqual(aapl_pos['shares'], 100)
        self.assertEqual(aapl_pos['avg_price'], 180.0)
        self.assertEqual(aapl_pos['current_price'], 180.0)

    def test_add_multiple_positions(self):
        """Test adding multiple positions."""
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 380.0)

        positions = self.portfolio.get_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn('AAPL', positions)
        self.assertIn('MSFT', positions)

    def test_update_position_price(self):
        """Test updating position price."""
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)
        self.portfolio.update_position_price('AAPL', 185.0)

        positions = self.portfolio.get_positions()
        aapl_pos = positions['AAPL']
        self.assertEqual(aapl_pos['current_price'], 185.0)
        self.assertEqual(aapl_pos['avg_price'], 180.0)  # Should not change

    def test_add_to_existing_position(self):
        """Test adding shares to existing position."""
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)
        self.portfolio.add_position('AAPL', 50, 185.0, 185.0)

        positions = self.portfolio.get_positions()
        aapl_pos = positions['AAPL']

        # Should have 150 total shares
        self.assertEqual(aapl_pos['shares'], 150)

        # Average price should be weighted average
        expected_avg = (100 * 180.0 + 50 * 185.0) / 150
        self.assertAlmostEqual(aapl_pos['avg_price'], expected_avg, places=2)

    def test_close_position(self):
        """Test closing a position."""
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)
        self.portfolio.close_position('AAPL', 185.0)

        positions = self.portfolio.get_positions()
        self.assertNotIn('AAPL', positions)

    def test_calculate_position_value(self):
        """Test position value calculation."""
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)

        position_value = self.portfolio.calculate_position_value('AAPL')
        expected_value = 100 * 185.0
        self.assertEqual(position_value, expected_value)

    def test_calculate_total_value(self):
        """Test total portfolio value calculation."""
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)

        total_value = self.portfolio.calculate_total_value()
        expected_value = 100000 + (100 * 185.0)  # Cash + position value
        self.assertEqual(total_value, expected_value)

    def test_transaction_logging(self):
        """Test transaction logging."""
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)

        transactions = self.portfolio.get_transactions()
        self.assertEqual(len(transactions), 1)

        transaction = transactions[0]
        self.assertEqual(transaction['symbol'], 'AAPL')
        self.assertEqual(transaction['action'], 'BUY')
        self.assertEqual(transaction['shares'], 100)
        self.assertEqual(transaction['price'], 180.0)
        self.assertIn('timestamp', transaction)

    def test_save_and_load_portfolio(self):
        """Test portfolio save and load functionality."""
        # Add some data
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 390.0)

        # Save portfolio
        success = self.portfolio.save_to_file()
        self.assertTrue(success)

        # Create new portfolio and load
        new_portfolio = Portfolio(initial_cash=0, portfolio_file=self.portfolio_file)
        success = new_portfolio.load_from_file()
        self.assertTrue(success)

        # Check loaded data
        self.assertEqual(new_portfolio.get_cash(), 100000)
        positions = new_portfolio.get_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn('AAPL', positions)
        self.assertIn('MSFT', positions)

    def test_portfolio_metadata(self):
        """Test portfolio metadata tracking."""
        metadata = self.portfolio.data.get('metadata', {})
        self.assertIn('created_at', metadata)
        self.assertIn('updated_at', metadata)
        self.assertEqual(metadata.get('initial_cash'), 100000)

    def test_cash_management(self):
        """Test cash management operations."""
        # Test initial cash
        self.assertEqual(self.portfolio.get_cash(), 100000)

        # Test cash deduction (simulating buy order)
        self.portfolio.deduct_cash(18000)  # Buy 100 shares at $180
        self.assertEqual(self.portfolio.get_cash(), 82000)

        # Test cash addition (simulating sell order)
        self.portfolio.add_cash(18500)  # Sell 100 shares at $185
        self.assertEqual(self.portfolio.get_cash(), 100500)


class TestPortfolioManager(unittest.TestCase):
    """Test cases for PortfolioManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.portfolio_file = self.temp_file.name

        self.portfolio = Portfolio(
            initial_cash=100000,
            portfolio_file=self.portfolio_file
        )
        self.portfolio_manager = PortfolioManager(self.portfolio)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.portfolio_file):
            os.unlink(self.portfolio_file)

    def test_can_afford_trade(self):
        """Test trade affordability check."""
        # Should be able to afford small trade
        self.assertTrue(self.portfolio_manager.can_afford_trade('AAPL', 50, 180.0))

        # Should not be able to afford large trade
        self.assertFalse(self.portfolio_manager.can_afford_trade('AAPL', 1000, 180.0))

    def test_get_position_info(self):
        """Test getting position information."""
        # Add position
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)

        # Get position info
        position_info = self.portfolio_manager.get_position_info('AAPL')
        self.assertIsNotNone(position_info)
        self.assertEqual(position_info['shares'], 100)
        self.assertEqual(position_info['avg_price'], 180.0)
        self.assertEqual(position_info['current_price'], 185.0)

    def test_get_position_info_nonexistent(self):
        """Test getting position info for non-existent position."""
        position_info = self.portfolio_manager.get_position_info('NONEXISTENT')
        self.assertIsNone(position_info)

    def test_calculate_position_pnl(self):
        """Test position P&L calculation."""
        # Add position
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)

        # Calculate P&L
        pnl = self.portfolio_manager.calculate_position_pnl('AAPL')
        expected_pnl = 100 * (185.0 - 180.0)  # 100 shares * $5 gain
        self.assertEqual(pnl, expected_pnl)

    def test_calculate_total_pnl(self):
        """Test total portfolio P&L calculation."""
        # Add positions with gains and losses
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)  # +$500
        self.portfolio.add_position('MSFT', 50, 380.0, 375.0)   # -$250

        total_pnl = self.portfolio_manager.calculate_total_pnl()
        expected_pnl = 100 * (185.0 - 180.0) + 50 * (375.0 - 380.0)
        self.assertEqual(total_pnl, expected_pnl)

    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        # Add some positions
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 390.0)

        summary = self.portfolio_manager.get_portfolio_summary()

        # Check summary structure
        self.assertIn('cash', summary)
        self.assertIn('total_value', summary)
        self.assertIn('positions_count', summary)
        self.assertIn('total_pnl', summary)

        # Check values
        self.assertEqual(summary['cash'], 100000)
        self.assertEqual(summary['positions_count'], 2)
        self.assertGreater(summary['total_value'], 100000)

    def test_portfolio_rebalancing_scenario(self):
        """Test portfolio rebalancing scenario."""
        # Start with some positions
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 390.0)

        # Simulate rebalancing: sell some AAPL, buy more MSFT
        self.portfolio.close_position('AAPL', 185.0)  # Sell all AAPL
        self.portfolio.add_position('MSFT', 25, 390.0, 390.0)  # Buy more MSFT

        # Check final state
        positions = self.portfolio.get_positions()
        self.assertNotIn('AAPL', positions)
        self.assertIn('MSFT', positions)
        self.assertEqual(positions['MSFT']['shares'], 75)  # 50 + 25

    def test_portfolio_performance_metrics(self):
        """Test portfolio performance metrics calculation."""
        # Add positions with different performance
        self.portfolio.add_position('AAPL', 100, 180.0, 200.0)  # +11.1%
        self.portfolio.add_position('MSFT', 50, 380.0, 360.0)   # -5.3%

        # Calculate performance metrics
        total_value = self.portfolio.calculate_total_value()
        total_pnl = self.portfolio_manager.calculate_total_pnl()

        # Check calculations
        expected_total = 100000 + (100 * 200.0) + (50 * 360.0)
        self.assertEqual(total_value, expected_total)

        expected_pnl = 100 * (200.0 - 180.0) + 50 * (360.0 - 380.0)
        self.assertEqual(total_pnl, expected_pnl)

    def test_portfolio_risk_metrics(self):
        """Test portfolio risk metrics."""
        # Add multiple positions
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 390.0)
        self.portfolio.add_position('GOOGL', 25, 2800.0, 2900.0)

        # Calculate position weights
        total_value = self.portfolio.calculate_total_value()
        aapl_weight = (100 * 185.0) / total_value
        msft_weight = (50 * 390.0) / total_value
        googl_weight = (25 * 2900.0) / total_value

        # Weights should sum to position value / total value
        position_value = (100 * 185.0) + (50 * 390.0) + (25 * 2900.0)
        expected_total_weight = position_value / total_value

        calculated_weight = aapl_weight + msft_weight + googl_weight
        self.assertAlmostEqual(calculated_weight, expected_total_weight, places=4)


class TestPortfolioIntegration(unittest.TestCase):
    """Integration tests for portfolio management."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.portfolio_file = self.temp_file.name

        self.portfolio = Portfolio(
            initial_cash=200000,  # Larger portfolio for testing
            portfolio_file=self.portfolio_file
        )
        self.portfolio_manager = PortfolioManager(self.portfolio)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.portfolio_file):
            os.unlink(self.portfolio_file)

    def test_complete_trading_workflow(self):
        """Test complete trading workflow."""
        # Initial state
        self.assertEqual(self.portfolio.get_cash(), 200000)
        self.assertEqual(len(self.portfolio.get_positions()), 0)

        # Buy some stocks
        self.portfolio.add_position('AAPL', 100, 180.0, 180.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 380.0)

        # Check state after buying
        positions = self.portfolio.get_positions()
        self.assertEqual(len(positions), 2)

        # Update prices (market movement)
        self.portfolio.update_position_price('AAPL', 185.0)
        self.portfolio.update_position_price('MSFT', 390.0)

        # Check P&L
        aapl_pnl = self.portfolio_manager.calculate_position_pnl('AAPL')
        msft_pnl = self.portfolio_manager.calculate_position_pnl('MSFT')

        self.assertEqual(aapl_pnl, 500)  # 100 * (185 - 180)
        self.assertEqual(msft_pnl, 500)  # 50 * (390 - 380)

        # Sell some positions
        self.portfolio.close_position('AAPL', 185.0)

        # Check final state
        positions = self.portfolio.get_positions()
        self.assertNotIn('AAPL', positions)
        self.assertIn('MSFT', positions)

        # Check transactions
        transactions = self.portfolio.get_transactions()
        self.assertGreater(len(transactions), 0)

    def test_portfolio_persistence(self):
        """Test portfolio data persistence across sessions."""
        # Create initial portfolio with positions
        self.portfolio.add_position('AAPL', 100, 180.0, 185.0)
        self.portfolio.add_position('MSFT', 50, 380.0, 390.0)

        # Save portfolio
        self.portfolio.save_to_file()

        # Create new portfolio instance and load
        new_portfolio = Portfolio(initial_cash=0, portfolio_file=self.portfolio_file)
        new_portfolio.load_from_file()

        # Verify loaded data
        self.assertEqual(new_portfolio.get_cash(), 200000)
        positions = new_portfolio.get_positions()
        self.assertEqual(len(positions), 2)
        self.assertIn('AAPL', positions)
        self.assertIn('MSFT', positions)

        # Verify position details
        aapl_pos = positions['AAPL']
        self.assertEqual(aapl_pos['shares'], 100)
        self.assertEqual(aapl_pos['avg_price'], 180.0)
        self.assertEqual(aapl_pos['current_price'], 185.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
