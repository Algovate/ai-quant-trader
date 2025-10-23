#!/usr/bin/env python3
"""
AI Trading System - Professional Trading Dashboard
现代化、专业的交易分析仪表板
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Please run: uv add gradio")
    exit(1)

class TradingDashboard:
    """专业交易仪表板"""
    
    def __init__(self, results_dir="/Users/rodin/Workspace/algovate/lab/stock/data/results"):
        self.results_dir = Path(results_dir)
        self.portfolio_file = Path("data/portfolio.json")
        
    def load_latest_results(self):
        """加载最新的分析结果"""
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
        """查找最新的匹配文件"""
        files = list(self.results_dir.glob(pattern))
        if files:
            return max(files, key=lambda x: x.stat().st_mtime)
        return None

    def load_portfolio_data(self):
        """加载投资组合数据"""
        try:
            if self.portfolio_file.exists():
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return None

    def create_market_overview(self, results):
        """创建市场概览"""
        if not results.get('step1'):
            return None, "无市场数据"
        
        # 获取最新价格数据
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
        return df, f"📊 市场概览 - {len(symbols)} 只股票"

    def create_price_chart(self, symbol, results):
        """创建价格图表"""
        if not results.get('step3') or symbol not in results['step3']:
            return None
        
        data = results['step3'][symbol]['data']
        df = pd.DataFrame(data)
        
        if df.empty:
            return None
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} 价格走势', '成交量', '技术指标'),
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # 价格K线图
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='价格',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # 移动平均线
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
        
        # 成交量
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='成交量', 
                   marker_color='lightblue', opacity=0.7),
            row=2, col=1
        )
        
        # RSI指标
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
            # RSI超买超卖线
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} 技术分析',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

    def create_ai_analysis(self, results):
        """创建AI分析面板"""
        if not results.get('step4'):
            return None, None, "无AI预测数据"
        
        predictions = results['step4']['predictions']
        
        # 预测摘要
        analysis_data = []
        for symbol, pred in predictions.items():
            prediction = pred['prediction']
            confidence = pred['confidence']
            
            # 预测方向
            if prediction > 0.01:
                direction = "📈 看涨"
                color = "green"
            elif prediction < -0.01:
                direction = "📉 看跌"
                color = "red"
            else:
                direction = "➡️ 中性"
                color = "orange"
            
            analysis_data.append({
                '股票': symbol,
                '预测': f"{prediction:.2%}",
                '置信度': f"{confidence:.1%}",
                '方向': direction,
                '颜色': color
            })
        
        df_analysis = pd.DataFrame(analysis_data)
        
        # 预测分布图
        symbols = list(predictions.keys())
        pred_values = [predictions[s]['prediction']*100 for s in symbols]
        confidences = [predictions[s]['confidence']*100 for s in symbols]
        
        fig = go.Figure()
        
        # 预测值条形图
        colors = ['green' if p > 0 else 'red' for p in pred_values]
        fig.add_trace(go.Bar(
            x=symbols, 
            y=pred_values,
            name='预测值 (%)',
            marker_color=colors,
            text=[f"{p:.1f}%" for p in pred_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="AI预测分析",
            xaxis_title="股票代码",
            yaxis_title="预测涨跌幅 (%)",
            height=400,
            template='plotly_dark'
        )
        
        return df_analysis, fig, f"🤖 AI分析 - {len(predictions)} 只股票"

    def create_trading_signals(self, results):
        """创建交易信号面板"""
        if not results.get('step5'):
            return None, None, "无交易信号数据"
        
        signals = results['step5']
        
        # 信号摘要
        signal_data = []
        for symbol, signal in signals.items():
            signal_type = signal['signal']
            strength = signal['strength']
            confidence = signal['confidence']
            
            # 信号图标
            if signal_type == 'BUY':
                icon = "🟢 买入"
                color = "green"
            elif signal_type == 'SELL':
                icon = "🔴 卖出"
                color = "red"
            else:
                icon = "🟡 持有"
                color = "orange"
            
            signal_data.append({
                '股票': symbol,
                '信号': icon,
                '强度': f"{strength:.2f}",
                '置信度': f"{confidence:.1%}",
                '颜色': color
            })
        
        df_signals = pd.DataFrame(signal_data)
        
        # 信号分布图
        symbols = list(signals.keys())
        signal_types = [signals[s]['signal'] for s in symbols]
        strengths = [signals[s]['strength']*100 for s in symbols]
        
        fig = go.Figure()
        
        # 信号强度条形图
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        bar_colors = [colors.get(s, 'gray') for s in signal_types]
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=strengths,
            name='信号强度',
            marker_color=bar_colors,
            text=signal_types,
            textposition='auto'
        ))
        
        fig.update_layout(
            title="交易信号分析",
            xaxis_title="股票代码",
            yaxis_title="信号强度 (%)",
            height=400,
            template='plotly_dark'
        )
        
        return df_signals, fig, f"📈 交易信号 - {len(signals)} 只股票"

    def create_portfolio_dashboard(self, portfolio_data):
        """创建投资组合仪表板"""
        if not portfolio_data:
            return None, None, None, "无投资组合数据"
        
        cash = portfolio_data.get('cash', 0)
        positions = portfolio_data.get('positions', {})
        
        # 投资组合概览
        total_value = cash
        position_values = []
        
        for symbol, pos in positions.items():
            shares = pos.get('shares', 0)
            current_price = pos.get('current_price', 0)
            value = shares * current_price
            total_value += value
            position_values.append({
                '股票': symbol,
                '股数': shares,
                '当前价格': f"${current_price:.2f}",
                '市值': f"${value:,.2f}",
                '盈亏': f"${pos.get('unrealized_pnl', 0):,.2f}"
            })
        
        df_positions = pd.DataFrame(position_values)
        
        # 资产配置饼图
        if total_value > 0:
            cash_pct = (cash / total_value) * 100
            positions_pct = ((total_value - cash) / total_value) * 100
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['现金', '股票'],
                values=[cash, total_value - cash],
                hole=0.3,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
            )])
            
            fig_pie.update_layout(
                title="资产配置",
                height=400,
                template='plotly_dark'
            )
        else:
            fig_pie = None
        
        # 投资组合概览
        overview_data = {
            '指标': ['总资产', '现金', '股票市值', '持仓数量'],
            '数值': [
                f"${total_value:,.2f}",
                f"${cash:,.2f}",
                f"${total_value - cash:,.2f}",
                len(positions)
            ]
        }
        df_overview = pd.DataFrame(overview_data)
        
        return df_overview, df_positions, fig_pie, f"💼 投资组合 - 总价值 ${total_value:,.2f}"

    def create_orders_panel(self, results):
        """创建订单面板"""
        if not results.get('step6'):
            return None, None, "无订单数据"
        
        orders = results['step6']
        if not orders:
            return None, None, "暂无生成订单"
        
        # 订单摘要
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
                icon = "🟢 买入"
            else:
                proceeds = order.get('estimated_proceeds', 0)
                total_proceeds += proceeds
                value_str = f"${proceeds:,.2f}"
                icon = "🔴 卖出"
            
            order_data.append({
                '股票': symbol,
                '操作': icon,
                '数量': quantity,
                '价格': f"${price:.2f}",
                '金额': value_str,
                '原因': reason[:30] + '...' if len(reason) > 30 else reason
            })
        
        df_orders = pd.DataFrame(order_data)
        
        # 订单统计
        net_flow = total_proceeds - total_cost
        stats_text = f"""
        📊 订单统计:
        • 总订单数: {len(orders)}
        • 买入成本: ${total_cost:,.2f}
        • 卖出收入: ${total_proceeds:,.2f}
        • 净现金流: ${net_flow:,.2f}
        """
        
        return df_orders, stats_text, f"🎯 交易订单 - {len(orders)} 笔"

    def create_dashboard(self):
        """创建主仪表板"""
        
        # 自定义CSS样式
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
            
            # 页面标题
            gr.HTML("""
            <div class="dashboard-header">
                <h1>🚀 AI Trading System - Professional Dashboard</h1>
                <p>智能量化交易分析平台 | AI驱动的市场分析与投资决策</p>
            </div>
            """)
            
            # 状态控制面板
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 刷新数据", variant="primary", size="lg")
                    run_analysis_btn = gr.Button("⚡ 运行分析", variant="secondary", size="lg")
                with gr.Column(scale=2):
                    symbol_dropdown = gr.Dropdown(
                        label="📊 选择股票",
                        choices=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                        value="AAPL",
                        interactive=True
                    )
                with gr.Column(scale=2):
                    status_text = gr.Textbox(
                        label="📈 系统状态",
                        value="准备就绪 - 点击'刷新数据'开始分析",
                        interactive=False,
                        lines=2
                    )
            
            # 数据状态
            results_state = gr.State()
            
            # 主要分析标签页
            with gr.Tabs():
                
                # 市场概览标签页
                with gr.Tab("📊 市场概览", id="market_overview"):
                    gr.Markdown("### 📈 实时市场数据")
                    with gr.Row():
                        market_table = gr.Dataframe(
                            label="市场概览",
                            interactive=False,
                            wrap=True
                        )
                        market_summary = gr.Textbox(
                            label="市场摘要",
                            interactive=False,
                            lines=3
                        )
                
                # 技术分析标签页
                with gr.Tab("📈 技术分析", id="technical_analysis"):
                    gr.Markdown("### 🔍 技术指标分析")
                    with gr.Row():
                        price_chart = gr.Plot(label="价格走势图", show_label=False)
                
                # AI分析标签页
                with gr.Tab("🤖 AI分析", id="ai_analysis"):
                    gr.Markdown("### 🧠 人工智能市场预测")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ai_table = gr.Dataframe(
                                label="AI预测结果",
                                interactive=False
                            )
                        with gr.Column(scale=2):
                            ai_chart = gr.Plot(label="预测分析图", show_label=False)
                    ai_summary = gr.Textbox(
                        label="AI分析摘要",
                        interactive=False,
                        lines=2
                    )
                
                # 交易信号标签页
                with gr.Tab("📈 交易信号", id="trading_signals"):
                    gr.Markdown("### 🎯 智能交易信号")
                    with gr.Row():
                        with gr.Column(scale=1):
                            signals_table = gr.Dataframe(
                                label="交易信号",
                                interactive=False
                            )
                        with gr.Column(scale=2):
                            signals_chart = gr.Plot(label="信号分析图", show_label=False)
                    signals_summary = gr.Textbox(
                        label="信号摘要",
                        interactive=False,
                        lines=2
                    )
                
                # 投资组合标签页
                with gr.Tab("💼 投资组合", id="portfolio"):
                    gr.Markdown("### 💰 投资组合管理")
                    
                    # 投资组合概览
                    with gr.Row():
                        with gr.Column(scale=1):
                            portfolio_overview = gr.Dataframe(
                                label="投资组合概览",
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            portfolio_pie = gr.Plot(label="资产配置", show_label=False)
                    
                    # 持仓详情
                    gr.Markdown("#### 📊 持仓详情")
                    positions_table = gr.Dataframe(
                        label="当前持仓",
                        interactive=False
                    )
                    portfolio_summary = gr.Textbox(
                        label="投资组合摘要",
                        interactive=False,
                        lines=2
                    )
                
                # 交易订单标签页
                with gr.Tab("🎯 交易订单", id="orders"):
                    gr.Markdown("### 📋 智能订单生成")
                    orders_table = gr.Dataframe(
                        label="生成的交易订单",
                        interactive=False
                    )
                    orders_stats = gr.Textbox(
                        label="订单统计",
                        interactive=False,
                        lines=4
                    )
                    orders_summary = gr.Textbox(
                        label="订单摘要",
                        interactive=False,
                        lines=2
                    )
            
            # 事件处理函数
            def load_and_refresh():
                """加载和刷新所有数据"""
                results = self.load_latest_results()
                portfolio_data = self.load_portfolio_data()
                
                if not results:
                    return (
                        "❌ 未找到分析数据，请先运行交易流水线",
                        results,
                        None, None, None,  # market
                        None, None, None,   # ai
                        None, None, None,   # signals
                        None, None, None, None,  # portfolio
                        None, None, None    # orders
                    )
                
                # 市场概览
                market_df, market_text = self.create_market_overview(results)
                
                # AI分析
                ai_df, ai_fig, ai_text = self.create_ai_analysis(results)
                
                # 交易信号
                signals_df, signals_fig, signals_text = self.create_trading_signals(results)
                
                # 投资组合
                portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio_data)
                
                # 订单
                orders_df, orders_stats_text, orders_text = self.create_orders_panel(results)
                
                return (
                    f"✅ 数据加载完成 - {len(results)} 个分析步骤",
                    results,
                    market_df, market_text,  # market
                    ai_df, ai_fig, ai_text,  # ai
                    signals_df, signals_fig, signals_text,  # signals
                    portfolio_overview_df, positions_df, portfolio_fig, portfolio_text,  # portfolio
                    orders_df, orders_stats_text, orders_text  # orders
                )
            
            def update_chart(symbol, results):
                """更新价格图表"""
                if not symbol or not results:
                    return None
                return self.create_price_chart(symbol, results)
            
            # 绑定事件
            refresh_btn.click(
                load_and_refresh,
                outputs=[
                    status_text, results_state,
                    market_table, market_summary,
                    ai_table, ai_chart, ai_summary,
                    signals_table, signals_chart, signals_summary,
                    portfolio_overview, positions_table, portfolio_pie, portfolio_summary,
                    orders_table, orders_stats, orders_summary
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
                    orders_table, orders_stats, orders_summary
                ]
            )
            
            symbol_dropdown.change(
                update_chart,
                inputs=[symbol_dropdown, results_state],
                outputs=[price_chart]
            )
        
        return demo

def main():
    """主函数"""
    dashboard = TradingDashboard()
    demo = dashboard.create_dashboard()
    
    print("🚀 启动AI交易系统专业仪表板...")
    print("📊 现代化界面设计，专业交易体验")
    print("🔗 仪表板将在浏览器中打开")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()