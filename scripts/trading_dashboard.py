#!/usr/bin/env python3
"""
AI Quant Trader
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Please run: uv add gradio")
    exit(1)

class TradingDashboard:
    """Professional trading dashboard"""

    def __init__(self, results_dir="data/results"):
        self.results_dir = Path(results_dir)
        self.portfolio_file = Path("data/portfolio.json")

    def load_latest_results(self):
        """Load the latest analysis results"""
        try:
            step_files = {
                'step1': self._find_latest_file('step1_fetch_*_raw_data.json'),
                'step2': self._find_latest_file('step2_process_*_processed_data.json'),
                'step3': self._find_latest_file('step3_features_*_engineered_features.json'),
                'step4': self._find_latest_file('step4_predictions_*_llm_predictions.json'),
                'step5': self._find_latest_file('step5_signals_*_trading_signals.json'),
                'step6': self._find_latest_file('step6_orders_*_portfolio_orders.json')
            }

            results = {}
            for step, file_path in step_files.items():
                if file_path and file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results[step] = json.load(f)
                else:
                    results[step] = None

            return results
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}

    def _find_latest_file(self, pattern):
        """Find the latest matching file"""
        files = list(self.results_dir.glob(pattern))
        if files:
            return max(files, key=lambda x: x.stat().st_mtime)
        return None

    def load_portfolio_data(self):
        """Load portfolio data"""
        try:
            if self.portfolio_file.exists():
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return None

    def create_market_overview(self, results):
        """Create market overview"""
        if not results.get('step1'):
            return None, "No market data"

        # Get latest price data
        market_data = results['step1']
        symbols = list(market_data.keys())

        overview_data = []
        for symbol in symbols:
            if symbol in market_data and 'data' in market_data[symbol]:
                data = market_data[symbol]['data']
                if data:
                    latest = data[-1]
                    overview_data.append({
                        'Symbol': symbol,
                        'Price': f"${latest.get('close', 0):.2f}",
                        'Change': f"{latest.get('change', 0):.2f}",
                        'Volume': f"{latest.get('volume', 0):,}"
                    })

        df = pd.DataFrame(overview_data)
        return df, f"📊 Market Overview - {len(symbols)} stocks"

    def create_price_chart(self, symbol, results):
        """Create price chart"""
        if not results.get('step3') or symbol not in results['step3']:
            return None

        data = results['step3'][symbol]['data']
        df = pd.DataFrame(data)

        if df.empty:
            return None

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price Trend', 'Volume', 'Technical Indicators'),
            row_heights=[0.5, 0.2, 0.3]
        )

        # Price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )

        # Moving averages
        if 'ma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ma_20'], name='MA20',
                          line=dict(color='orange', width=2)),
                row=1, col=1
            )
        if 'ma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ma_50'], name='MA50',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )

        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume',
                   marker_color='lightblue', opacity=0.7),
            row=2, col=1
        )

        # RSI indicator
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )

        return fig

    def create_ai_analysis(self, results):
        """Create AI analysis panel"""
        if not results.get('step4'):
            return None, None, "No AI prediction data"

        predictions = results['step4']['predictions']

        # Prediction summary
        analysis_data = []
        for symbol, pred in predictions.items():
            prediction = pred['prediction']
            confidence = pred['confidence']

            # Prediction direction
            if prediction > 0.01:
                direction = "📈 Bullish"
                color = "green"
            elif prediction < -0.01:
                direction = "📉 Bearish"
                color = "red"
            else:
                direction = "➡️ Neutral"
                color = "orange"

            analysis_data.append({
                'Stock': symbol,
                'Prediction': f"{prediction:.2%}",
                'Confidence': f"{confidence:.1%}",
                'Direction': direction,
                'Color': color
            })

        df_analysis = pd.DataFrame(analysis_data)

        # Prediction distribution chart
        symbols = list(predictions.keys())
        pred_values = [predictions[s]['prediction']*100 for s in symbols]
        confidences = [predictions[s]['confidence']*100 for s in symbols]

        fig = go.Figure()

        # Prediction value bar chart
        colors = ['green' if p > 0 else 'red' for p in pred_values]
        fig.add_trace(go.Bar(
            x=symbols,
            y=pred_values,
            name='Prediction (%)',
            marker_color=colors,
            text=[f"{p:.1f}%" for p in pred_values],
            textposition='auto'
        ))

        fig.update_layout(
            title="AI Prediction Analysis",
            xaxis_title="Stock Symbol",
            yaxis_title="Predicted Change (%)",
            height=400,
            template='plotly_dark'
        )

        return df_analysis, fig, f"🤖 AI Analysis - {len(predictions)} stocks"

    def create_trading_signals(self, results):
        """Create trading signals panel"""
        if not results.get('step5'):
            return None, None, "No trading signals data"

        signals = results['step5']

        # Signal summary
        signal_data = []
        for symbol, signal in signals.items():
            signal_type = signal['signal']
            strength = signal['strength']
            confidence = signal['confidence']

            # Signal icon
            if signal_type == 'BUY':
                icon = "🟢 Buy"
                color = "green"
            elif signal_type == 'SELL':
                icon = "🔴 Sell"
                color = "red"
            else:
                icon = "🟡 Hold"
                color = "orange"

            signal_data.append({
                'Stock': symbol,
                'Signal': icon,
                'Strength': f"{strength:.2f}",
                'Confidence': f"{confidence:.1%}",
                'Color': color
            })

        df_signals = pd.DataFrame(signal_data)

        # Signal distribution chart
        symbols = list(signals.keys())
        signal_types = [signals[s]['signal'] for s in symbols]
        strengths = [signals[s]['strength']*100 for s in symbols]

        fig = go.Figure()

        # Signal strength bar chart
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        bar_colors = [colors.get(s, 'gray') for s in signal_types]

        fig.add_trace(go.Bar(
            x=symbols,
            y=strengths,
            name='Signal Strength',
            marker_color=bar_colors,
            text=signal_types,
            textposition='auto'
        ))

        fig.update_layout(
            title="Trading Signals Analysis",
            xaxis_title="Stock Symbol",
            yaxis_title="Signal Strength (%)",
            height=400,
            template='plotly_dark'
        )

        return df_signals, fig, f"📈 Trading Signals - {len(signals)} stocks"

    def create_portfolio_dashboard(self, portfolio_data):
        """Create portfolio dashboard"""
        if not portfolio_data:
            return None, None, None, "No portfolio data"

        cash = portfolio_data.get('cash', 0)
        positions = portfolio_data.get('positions', {})

        # Portfolio overview
        total_value = cash
        position_values = []

        for symbol, pos in positions.items():
            shares = pos.get('shares', 0)
            current_price = pos.get('current_price', 0)
            value = shares * current_price
            total_value += value
            position_values.append({
                'Stock': symbol,
                'Shares': shares,
                'Current Price': f"${current_price:.2f}",
                'Market Value': f"${value:,.2f}",
                'P&L': f"${pos.get('unrealized_pnl', 0):,.2f}"
            })

        df_positions = pd.DataFrame(position_values)

        # Asset allocation pie chart
        if total_value > 0:
            cash_pct = (cash / total_value) * 100
            positions_pct = ((total_value - cash) / total_value) * 100

            fig_pie = go.Figure(data=[go.Pie(
                labels=['Cash', 'Stocks'],
                values=[cash, total_value - cash],
                hole=0.3,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
            )])

            fig_pie.update_layout(
                title="Asset Allocation",
                height=400,
                template='plotly_dark'
            )
        else:
            fig_pie = None

        # Portfolio overview
        overview_data = {
            'Metric': ['Total Assets', 'Cash', 'Stock Value', 'Position Count'],
            'Value': [
                f"${total_value:,.2f}",
                f"${cash:,.2f}",
                f"${total_value - cash:,.2f}",
                len(positions)
            ]
        }
        df_overview = pd.DataFrame(overview_data)

        return df_overview, df_positions, fig_pie, f"💼 Portfolio - Total Value ${total_value:,.2f}"

    def create_orders_panel(self, results):
        """Create orders panel"""
        if not results.get('step6'):
            return None, None, "No orders data"

        orders = results['step6']
        if not orders:
            return None, None, "No orders generated"

        # Order summary
        order_data = []
        total_cost = 0
        total_proceeds = 0

        for order in orders:
            symbol = order.get('symbol', '')
            action = order.get('action', '')
            quantity = order.get('quantity', 0)
            price = order.get('estimated_price', 0)
            reason = order.get('reason', '')

            if action == 'BUY':
                cost = order.get('estimated_cost', 0)
                total_cost += cost
                value_str = f"${cost:,.2f}"
                icon = "🟢 Buy"
            else:
                proceeds = order.get('estimated_proceeds', 0)
                total_proceeds += proceeds
                value_str = f"${proceeds:,.2f}"
                icon = "🔴 Sell"

            order_data.append({
                'Stock': symbol,
                'Action': icon,
                'Quantity': quantity,
                'Price': f"${price:.2f}",
                'Amount': value_str,
                'Reason': reason[:30] + '...' if len(reason) > 30 else reason
            })

        df_orders = pd.DataFrame(order_data)

        # Order statistics
        net_flow = total_proceeds - total_cost
        stats_text = f"""
        📊 Order Statistics:
        • Total Orders: {len(orders)}
        • Buy Cost: ${total_cost:,.2f}
        • Sell Proceeds: ${total_proceeds:,.2f}
        • Net Cash Flow: ${net_flow:,.2f}
        """

        return df_orders, stats_text, f"🎯 Trading Orders - {len(orders)} orders"

    def create_dashboard(self):
        """Create main dashboard"""

        # Custom CSS styles
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """

        with gr.Blocks(
            title="AI Trading System - Professional Dashboard",
            css=custom_css,
            theme=gr.themes.Soft()
        ) as demo:

            # Page title
            gr.HTML("""
            <div class="dashboard-header">
                <h1>🚀 AI Trading System - Professional Dashboard</h1>
                <p>Intelligent Quantitative Trading Analysis Platform | AI-driven Market Analysis and Investment Decisions</p>
            </div>
            """)

            # Status control panel
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh Data", variant="primary", size="lg")
                    run_analysis_btn = gr.Button("⚡ Run Analysis", variant="secondary", size="lg")
                with gr.Column(scale=3):
                    status_text = gr.Textbox(
                        label="📈 System Status",
                        value="Ready - Click 'Refresh Data' to start analysis",
                        interactive=False,
                        lines=2
                    )

            # Data state
            results_state = gr.State()

            # Main analysis tabs
            with gr.Tabs():

                # Market overview tab
                with gr.Tab("📊 Market Overview", id="market_overview"):
                    gr.Markdown("### 📈 Real-time Market Data")
                    with gr.Row():
                        market_table = gr.Dataframe(
                            label="Market Overview",
                            interactive=False,
                            wrap=True
                        )
                        market_summary = gr.Textbox(
                            label="Market Summary",
                            interactive=False,
                            lines=3
                        )

                # Technical analysis tab
                with gr.Tab("📈 Technical Analysis", id="technical_analysis"):
                    gr.Markdown("### 🔍 Technical Indicators Analysis")

                    # Stock selection for technical analysis
                    with gr.Row():
                        symbol_radio = gr.Radio(
                            choices=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                            value="AAPL",
                            interactive=True,
                            type="value",
                            show_label=False
                        )

                    with gr.Row():
                        price_chart = gr.Plot(label="Price Trend Chart", show_label=False)

                # AI analysis tab
                with gr.Tab("🤖 AI Analysis", id="ai_analysis"):
                    gr.Markdown("### 🧠 AI Market Prediction")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ai_table = gr.Dataframe(
                                label="AI Prediction Results",
                                interactive=False
                            )
                        with gr.Column(scale=2):
                            ai_chart = gr.Plot(label="Prediction Analysis Chart", show_label=False)
                    ai_summary = gr.Textbox(
                        label="AI Analysis Summary",
                        interactive=False,
                        lines=2
                    )

                # Trading signals tab
                with gr.Tab("📈 Trading Signals", id="trading_signals"):
                    gr.Markdown("### 🎯 Intelligent Trading Signals")
                    with gr.Row():
                        with gr.Column(scale=1):
                            signals_table = gr.Dataframe(
                                label="Trading Signals",
                                interactive=False
                            )
                        with gr.Column(scale=2):
                            signals_chart = gr.Plot(label="Signal Analysis Chart", show_label=False)
                    signals_summary = gr.Textbox(
                        label="Signal Summary",
                        interactive=False,
                        lines=2
                    )

                # Portfolio tab
                with gr.Tab("💼 Portfolio", id="portfolio"):
                    gr.Markdown("### 💰 Portfolio Management")

                    # Portfolio overview
                    with gr.Row():
                        with gr.Column(scale=1):
                            portfolio_overview = gr.Dataframe(
                                label="Portfolio Overview",
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            portfolio_pie = gr.Plot(label="Asset Allocation", show_label=False)

                    # Position details
                    gr.Markdown("#### 📊 Position Details")
                    positions_table = gr.Dataframe(
                        label="Current Positions",
                        interactive=False
                    )
                    portfolio_summary = gr.Textbox(
                        label="Portfolio Summary",
                        interactive=False,
                        lines=2
                    )

                # Trading orders tab
                with gr.Tab("🎯 Trading Orders", id="orders"):
                    gr.Markdown("### 📋 Intelligent Order Generation")
                    orders_table = gr.Dataframe(
                        label="Generated Trading Orders",
                        interactive=False
                    )
                    orders_stats = gr.Textbox(
                        label="Order Statistics",
                        interactive=False,
                        lines=4
                    )
                    orders_summary = gr.Textbox(
                        label="Order Summary",
                        interactive=False,
                        lines=2
                    )

            # Event handling functions
            def load_and_refresh():
                """Load and refresh all data"""
                results = self.load_latest_results()
                portfolio_data = self.load_portfolio_data()

                if not results:
                    return (
                        "❌ No analysis data found, please run the trading pipeline first",
                        results,
                        None, None, None,  # market
                        None, None, None,   # ai
                        None, None, None,   # signals
                        None, None, None, None,  # portfolio
                        None, None, None,   # orders
                        None  # price_chart
                    )

                # Market overview
                market_df, market_text = self.create_market_overview(results)

                # AI analysis
                ai_df, ai_fig, ai_text = self.create_ai_analysis(results)

                # Trading signals
                signals_df, signals_fig, signals_text = self.create_trading_signals(results)

                # Portfolio
                portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio_data)

                # Orders
                orders_df, orders_stats_text, orders_text = self.create_orders_panel(results)

                # Create initial price chart for default symbol (AAPL)
                initial_chart = self.create_price_chart("AAPL", results)

                return (
                    f"✅ Data loaded successfully - {len(results)} analysis steps",
                    results,
                    market_df, market_text,  # market
                    ai_df, ai_fig, ai_text,  # ai
                    signals_df, signals_fig, signals_text,  # signals
                    portfolio_overview_df, positions_df, portfolio_fig, portfolio_text,  # portfolio
                    orders_df, orders_stats_text, orders_text,  # orders
                    initial_chart  # price_chart
                )

            def update_chart(symbol, results):
                """Update price chart"""
                if not symbol or not results:
                    return None
                return self.create_price_chart(symbol, results)

            # Bind events
            refresh_btn.click(
                load_and_refresh,
                outputs=[
                    status_text, results_state,
                    market_table, market_summary,
                    ai_table, ai_chart, ai_summary,
                    signals_table, signals_chart, signals_summary,
                    portfolio_overview, positions_table, portfolio_pie, portfolio_summary,
                    orders_table, orders_stats, orders_summary,
                    price_chart  # Add price chart to refresh outputs
                ]
            )

            run_analysis_btn.click(
                load_and_refresh,
                outputs=[
                    status_text, results_state,
                    market_table, market_summary,
                    ai_table, ai_chart, ai_summary,
                    signals_table, signals_chart, signals_summary,
                    portfolio_overview, positions_table, portfolio_pie, portfolio_summary,
                    orders_table, orders_stats, orders_summary,
                    price_chart  # Add price chart to refresh outputs
                ]
            )

            symbol_radio.change(
                update_chart,
                inputs=[symbol_radio, results_state],
                outputs=[price_chart]
            )

        return demo

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI Trading System Professional Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trading_dashboard.py                    # Start dashboard locally
  python trading_dashboard.py --share           # Start dashboard with public sharing
  python trading_dashboard.py --port 8080       # Start on custom port
  python trading_dashboard.py --share --port 8080  # Share on custom port
        """
    )

    parser.add_argument(
        '--share',
        action='store_true',
        help='Enable public sharing (creates public URL via Gradio)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port number for the dashboard (default: 7860)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind to (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )

    args = parser.parse_args()

    dashboard = TradingDashboard()
    demo = dashboard.create_dashboard()

    print("🚀 Starting AI Trading System Professional Dashboard...")
    print("📊 Modern interface design, professional trading experience")

    if args.share:
        print("🌐 Public sharing enabled - Dashboard will be accessible via public URL")
    else:
        print("🔒 Local access only - Dashboard accessible at localhost")

    print(f"🔗 Server: {args.host}:{args.port}")

    if not args.no_browser:
        print("🌐 Dashboard will open in browser")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=not args.no_browser
    )

if __name__ == "__main__":
    main()