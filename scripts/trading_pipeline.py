#!/usr/bin/env python3
"""
LLM-Powered AI Trading System Demo
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import sys
import os
import json
import pickle
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.llm_predictor import LLMPredictor, create_llm_predictor
from src.strategy.llm_signal_generator import LLMSignalGenerator, create_llm_signal_generator
from src.strategy.portfolio_manager import Portfolio, PortfolioManager
from src.strategy.order_generator import OrderGenerator, create_order_generator
from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.data.feature_engineer import FeatureEngineer
from src.core.utils import (
    load_config, setup_logging, get_default_model, print_section_header,
    print_step_header, print_success, print_error, print_warning,
    ensure_api_key, format_percentage, format_currency, load_portfolio
)

logger = logging.getLogger(__name__)


def save_intermediate_result(data, filename, step_name, output_dir="data/results"):
    """Save intermediate results to files for later verification."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Always save as JSON format
    if not filename.endswith('.json'):
        filename = filename.replace('.pkl', '.json').replace('.csv', '.json')

    filepath = os.path.join(output_dir, f"{step_name}_{timestamp}_{filename}")

    # Convert data to JSON-serializable format
    json_data = convert_to_json_serializable(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Saved {step_name} results to: {filepath}")
    return filepath


def convert_to_json_serializable(data):
    """Convert data to JSON-serializable format."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, date

    if isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to dict with proper formatting
        df_dict = {
            'data': data.to_dict('records'),
            'index': data.index.tolist(),
            'columns': data.columns.tolist(),
            'shape': data.shape,
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        return df_dict
    elif isinstance(data, pd.Series):
        return {
            'data': data.to_dict(),
            'index': data.index.tolist(),
            'name': data.name,
            'dtype': str(data.dtype)
        }
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data


def create_llm_config(api_key: str, model: str) -> dict:
    """Create LLM configuration."""
    llm_config = {
        'api_key': api_key,
        'model': model,
        'base_url': 'https://openrouter.ai/api/v1',
        'min_confidence': 0.6,
        'signal_threshold': 0.02
    }

    logger.info(f"LLM configuration created for model: {model}")
    return llm_config


def run_llm_demo():
    """Run the main LLM demo."""
    print_section_header("LLM-POWERED AI QUANTITATIVE TRADING SYSTEM DEMO")

    config = load_config()
    setup_logging(config)

    api_key = ensure_api_key()
    if not api_key:
        return

    model = get_default_model()
    print(f"\nUsing default model: {model}")

    llm_config = create_llm_config(api_key, model)
    llm_predictor = create_llm_predictor(api_key, model)
    logger.info(f"LLM system initialized with model: {model}")

    data_fetcher = DataFetcher()
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    signal_generator = create_llm_signal_generator(llm_predictor)

    symbols = config['data']['symbols']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']

    # Display configuration
    print(f"\n📊 Configuration:")
    print(f"   Model: {model}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Date Range: {start_date} to {end_date}")
    print(f"   Total Symbols: {len(symbols)}")
    print()

    # Step 1: Fetch Data
    print_step_header("📈 Step 1", "Fetching Historical Data")

    logger.info("Fetching historical data...")
    raw_data = data_fetcher.fetch_data(symbols, start_date, end_date)

    if not raw_data:
        print_error("No data fetched. Please check your symbols and date range.")
        return

    for symbol, data in raw_data.items():
        logger.info(f"Fetched data for {symbol}: {len(data)} rows")

    # Save raw data
    save_intermediate_result(raw_data, "raw_data.json", "step1_fetch")

    print_success(f"Successfully fetched data for {len(raw_data)} symbols")

    # Step 2: Process Data
    print_step_header("🔧 Step 2", "Processing Data")

    logger.info("Processing data...")
    processed_data = data_processor.process_data(raw_data)

    if not processed_data:
        print_error("No data processed. Please check your data.")
        return

    # Save processed data
    save_intermediate_result(processed_data, "processed_data.json", "step2_process")

    print_success(f"Successfully processed data for {len(processed_data)} symbols")

    # Step 3: Engineer Features
    print_step_header("⚙️ Step 3", "Engineering Features")

    logger.info("Engineering features...")
    features_data = feature_engineer.engineer_features(processed_data)

    for symbol, features in features_data.items():
        logger.info(f"Engineered features for {symbol}: {len(features.columns)} features")

    # Save engineered features
    save_intermediate_result(features_data, "engineered_features.json", "step3_features")

    print_success(f"Successfully engineered features for {len(features_data)} symbols")

    # Step 4: Get LLM Predictions
    print_step_header("🧠 Step 4", "Getting LLM Predictions")

    logger.info(f"Getting LLM predictions for {len(features_data)} symbols")
    llm_predictions = llm_predictor.predict(features_data)

    print_section_header("🧠 LLM PREDICTIONS", 60)
    for symbol, pred_data in llm_predictions.items():
        prediction = pred_data['prediction']
        confidence = pred_data['confidence']

        # Add visual indicators for prediction direction
        if prediction > 0.01:
            direction = "📈 Bullish"
        elif prediction < -0.01:
            direction = "📉 Bearish"
        else:
            direction = "➡️ Neutral"

        print(f"\n{symbol}:")
        print(f"  {direction}")
        print(f"  Prediction: {format_percentage(prediction)}")
        print(f"  Confidence: {format_percentage(confidence)}")
        print(f"  Time: {pred_data['timestamp'].strftime('%H:%M:%S')}")

    # Save LLM predictions with detailed information
    llm_predictions_detailed = {
        'predictions': llm_predictions,
        'model_info': {
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(llm_predictions)
        },
        'prompt_details': getattr(llm_predictor, 'last_prompts', {}),
        'response_details': getattr(llm_predictor, 'last_responses', {})
    }
    save_intermediate_result(llm_predictions_detailed, "llm_predictions.json", "step4_predictions")

    print_success(f"Successfully got LLM predictions for {len(llm_predictions)} symbols")

    # Step 5: Generate Trading Signals
    print_step_header("🎯 Step 5", "Generating Trading Signals")

    logger.info(f"Generating LLM signals for {len(llm_predictions)} symbols")
    trading_signals = signal_generator.generate_signals_from_predictions(llm_predictions)

    print_section_header("🎯 TRADING SIGNALS", 60)
    buy_count = 0
    sell_count = 0
    hold_count = 0
    total_confidence = 0.0

    for symbol, signal_data in trading_signals.items():
        signal = signal_data['signal']
        strength = signal_data['strength']
        confidence = signal_data['confidence']

        # Add visual indicators for signal type
        if signal == "BUY":
            signal_icon = "🟢 BUY"
            buy_count += 1
        elif signal == "SELL":
            signal_icon = "🔴 SELL"
            sell_count += 1
        else:
            signal_icon = "🟡 HOLD"
            hold_count += 1

        print(f"\n{symbol}:")
        print(f"  Signal: {signal_icon}")
        print(f"  Strength: {strength:.2f}")
        print(f"  Confidence: {format_percentage(confidence)}")

        total_confidence += confidence

    print(f"\n📊 Signal Summary:")
    print(f"  Total: {len(trading_signals)} symbols")
    print(f"  🟢 Buy: {buy_count}")
    print(f"  🔴 Sell: {sell_count}")
    print(f"  🟡 Hold: {hold_count}")
    if len(trading_signals) > 0:
        print(f"  Avg Confidence: {format_percentage(total_confidence / len(trading_signals))}")

    # Save trading signals
    save_intermediate_result(trading_signals, "trading_signals.json", "step5_signals")

    print_success(f"Successfully generated signals for {len(trading_signals)} symbols")

    # Step 6: Generate Portfolio Orders
    print_step_header("🎯 Step 6", "Generating Portfolio Orders")

    logger.info("Loading portfolio and generating orders...")

    # Load portfolio
    portfolio_config = config.get('portfolio', {})
    portfolio_file = portfolio_config.get('portfolio_file', 'data/portfolio.json')
    portfolio = load_portfolio(portfolio_file)
    portfolio_manager = PortfolioManager(portfolio)

    # Get current prices from market data
    current_prices = {}
    for symbol, data in processed_data.items():
        if not data.empty and 'close' in data.columns:
            current_prices[symbol] = data['close'].iloc[-1]

    # Display current portfolio status
    portfolio_summary = portfolio.get_portfolio_summary(current_prices)
    print(f"\n📊 Current Portfolio:")
    print(f"   Cash: {format_currency(portfolio_summary['cash'])}")
    print(f"   Positions: {portfolio_summary['position_count']}")
    print(f"   Total Value: {format_currency(portfolio_summary['total_value'])}")

    if portfolio_summary['positions']:
        print(f"\n📈 Current Positions:")
        for symbol, pos in portfolio_summary['positions'].items():
            pnl = pos['unrealized_pnl']
            pnl_str = f"+{format_currency(pnl)}" if pnl >= 0 else format_currency(pnl)
            print(f"   {symbol}: {pos['shares']} shares @ {format_currency(pos['avg_price'])} "
                  f"(Current: {format_currency(pos['current_price'])}, P&L: {pnl_str})")

    # Generate orders
    order_generator = create_order_generator(portfolio_manager, portfolio_config)
    orders = order_generator.generate_orders(trading_signals, processed_data)

    # Validate orders
    valid_orders = order_generator.validate_orders(orders)

    print_section_header("🎯 GENERATED ORDERS", 60)

    if valid_orders:
        total_cost = 0
        total_proceeds = 0

        for order in valid_orders:
            symbol = order['symbol']
            action = order['action']
            quantity = order['quantity']
            price = order['estimated_price']
            reason = order['reason']
            risk_score = order['risk_score']

            if action == 'BUY':
                cost = order['estimated_cost']
                total_cost += cost
                action_icon = "🟢 BUY"
                print(f"\n{action_icon} {symbol}: {quantity} shares @ {format_currency(price)} ({format_currency(cost)})")
            else:
                proceeds = order['estimated_proceeds']
                total_proceeds += proceeds
                action_icon = "🔴 SELL"
                print(f"\n{action_icon} {symbol}: {quantity} shares @ {format_currency(price)} ({format_currency(proceeds)})")

            print(f"   Reason: {reason}")
            print(f"   Risk Score: {risk_score:.3f}")

        print(f"\n📊 Order Summary:")
        print(f"   Total Orders: {len(valid_orders)}")
        print(f"   Total Cost: {format_currency(total_cost)}")
        print(f"   Total Proceeds: {format_currency(total_proceeds)}")
        print(f"   Net Cash Flow: {format_currency(total_proceeds - total_cost)}")

        # Save orders
        save_intermediate_result(valid_orders, "portfolio_orders.json", "step6_orders")

        # Save updated portfolio
        portfolio_manager.portfolio.save_to_file()

        print_success(f"Successfully generated {len(valid_orders)} orders")
    else:
        print_warning("No valid orders generated")
        print("   This could be due to:")
        print("   - Insufficient cash for buy orders")
        print("   - No positions to sell for sell orders")
        print("   - Orders not meeting minimum trade size requirements")

        # Save empty orders
        save_intermediate_result([], "portfolio_orders.json", "step6_orders")

        # Save portfolio (even with no orders)
        portfolio_manager.portfolio.save_to_file()

    print_section_header("✅ DEMO COMPLETED SUCCESSFULLY!")

    print(f"\n📁 Generated Files:")
    print(f"   - data/cache/: Cached historical data")
    print(f"   - data/results/: Intermediate results for verification")
    print(f"     • step1_fetch_*_raw_data.json: Raw fetched data")
    print(f"     • step2_process_*_processed_data.json: Processed data with technical indicators")
    print(f"     • step3_features_*_engineered_features.json: Engineered features")
    print(f"     • step4_predictions_*_llm_predictions.json: LLM predictions with prompts & responses")
    print(f"     • step5_signals_*_trading_signals.json: Trading signals")
    print(f"     • step6_orders_*_portfolio_orders.json: Generated portfolio orders")
    print(f"   - data/portfolio.json: Portfolio state (positions, cash, transactions)")
    print(f"   - Logs: System processing logs")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Review predictions and signals above")
    print(f"   2. Modify config/config.yaml for different models")
    print(f"   3. Try different symbols or date ranges")
    print(f"   4. Add market context and news sentiment")
    print(f"   5. Integrate with real-time data feeds")
    print(f"\n💡 Tip: Set a valid OPENROUTER_API_KEY for real LLM predictions!")


def run_quick_test():
    """Run a quick test of the LLM system."""
    print_section_header("🧪 QUICK LLM TEST", 60)

    api_key = ensure_api_key()
    if not api_key:
        return

    llm_predictor = create_llm_predictor(api_key, "anthropic/claude-3.5-sonnet")

    sample_data = {
        'AAPL': pd.DataFrame({
            'close': [150, 151, 152, 153, 154],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'rsi': [50, 55, 60, 65, 70],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5],
            'ma_20': [149, 150, 151, 152, 153],
            'ma_50': [148, 149, 150, 151, 152],
            'returns': [0.01, 0.02, 0.01, 0.02, 0.01]
        }, index=pd.date_range('2023-01-01', periods=5))
    }

    predictions = llm_predictor.predict(sample_data, ['AAPL'])

    if 'AAPL' in predictions:
        pred = predictions['AAPL']
        prediction = pred['prediction']
        confidence = pred['confidence']

        # Add visual indicator
        if prediction > 0.01:
            direction = "📈 Bullish"
        elif prediction < -0.01:
            direction = "📉 Bearish"
        else:
            direction = "➡️ Neutral"

        print_success("✅ LLM prediction successful!")
        print(f"  {direction}")
        print(f"  Prediction: {format_percentage(prediction)}")
        print(f"  Confidence: {format_percentage(confidence)}")
    else:
        print_error("❌ LLM prediction failed")


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_quick_test()
    else:
        run_llm_demo()


if __name__ == "__main__":
    main()