#!/usr/bin/env python3
"""
Simple Final Working Gradio UI for Stock Analysis Pipeline Results Visualization
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Please run: uv add gradio")
    exit(1)

class StockAnalysisUI:
    def __init__(self, results_dir="/Users/rodin/Workspace/algovate/lab/stock/data/results"):
        self.results_dir = Path(results_dir)
    
    def load_latest_results(self):
        """Load the latest results from each step"""
        try:
            step_files = {
                'step1': self._find_latest_file('step1_fetch_*_raw_data.json'),
                'step2': self._find_latest_file('step2_process_*_processed_data.json'),
                'step3': self._find_latest_file('step3_features_*_engineered_features.json'),
                'step4': self._find_latest_file('step4_predictions_*_llm_predictions.json'),
                'step5': self._find_latest_file('step5_signals_*_trading_signals.json')
            }
            
            results = {}
            for step, file_path in step_files.items():
                if file_path and file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results[step] = json.load(f)
                else:
                    results[step] = None
                    
            return results
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            print(f"Error loading results: {e}")
            return {}
    
    def _find_latest_file(self, pattern):
        """Find the latest file matching the pattern"""
        files = list(self.results_dir.glob(pattern))
        if files:
            return max(files, key=lambda x: x.stat().st_mtime)
        return None
    
    def create_price_chart(self, symbol, step_data):
        """Create price chart with technical indicators"""
        if not step_data or symbol not in step_data:
            return None
            
        data = step_data[symbol]['data']
        df = pd.DataFrame(data)
        
        # Create date range if no date column
        if 'date' not in df.columns:
            df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'ma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['ma_20'], name='MA 20', line=dict(color='orange')),
                row=1, col=1
            )
        if 'ma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['ma_50'], name='MA 50', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        return fig
    
    def create_prediction_summary(self, predictions_data):
        """Create prediction summary table and charts"""
        if not predictions_data or 'predictions' not in predictions_data:
            return None, None
            
        predictions = predictions_data['predictions']
        
        # Create summary table
        summary_data = []
        for symbol, pred in predictions.items():
            summary_data.append({
                'Symbol': symbol,
                'Prediction (%)': f"{pred['prediction']*100:.2f}%",
                'Confidence': f"{pred['confidence']*100:.1f}%",
                'Timestamp': pred['timestamp']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Create prediction chart
        symbols = list(predictions.keys())
        pred_values = [predictions[s]['prediction']*100 for s in symbols]
        confidences = [predictions[s]['confidence']*100 for s in symbols]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Prediction Values', 'Confidence Levels'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['green' if p > 0 else 'red' for p in pred_values]
        fig.add_trace(
            go.Bar(x=symbols, y=pred_values, name='Prediction %', marker_color=colors),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=symbols, y=confidences, name='Confidence %', marker_color='blue'),
            row=1, col=2
        )
        
        fig.update_layout(title="LLM Predictions Summary", height=400)
        
        return df_summary, fig
    
    def create_signals_summary(self, signals_data):
        """Create trading signals summary"""
        if not signals_data:
            return None, None
            
        # Create signals table
        signals_summary = []
        for symbol, signal in signals_data.items():
            signals_summary.append({
                'Symbol': symbol,
                'Signal': signal['signal'],
                'Strength': f"{signal['strength']*100:.2f}%",
                'Confidence': f"{signal['confidence']*100:.1f}%",
                'Prediction': f"{signal['prediction']*100:.2f}%"
            })
        
        df_signals = pd.DataFrame(signals_summary)
        
        # Create signals chart
        symbols = list(signals_data.keys())
        signals = [signals_data[s]['signal'] for s in symbols]
        strengths = [signals_data[s]['strength']*100 for s in symbols]
        
        # Color mapping for signals
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        colors = [signal_colors.get(s, 'gray') for s in signals]
        
        fig = go.Figure(data=[
            go.Bar(x=symbols, y=strengths, marker_color=colors, text=signals, textposition='auto')
        ])
        
        fig.update_layout(
            title="Trading Signals Summary",
            xaxis_title="Symbol",
            yaxis_title="Signal Strength (%)",
            height=400
        )
        
        return df_signals, fig
    
    def create_ui(self):
        """Create the main Gradio interface"""
        
        # Create the interface
        with gr.Blocks(title="Stock Analysis Results") as demo:
            gr.Markdown("# 📈 Stock Analysis Pipeline Results Viewer")
            gr.Markdown("Visualize intermediate results from your stock analysis pipeline")
            
            with gr.Row():
                with gr.Column(scale=1):
                    load_btn = gr.Button("🔄 Load Latest Results", variant="primary")
                    symbol_dropdown = gr.Dropdown(
                        label="Select Symbol",
                        choices=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                        value="AAPL",
                        interactive=True
                    )
                
                with gr.Column(scale=3):
                    status_text = gr.Textbox(
                        label="Status",
                        value="Click 'Load Latest Results' to start",
                        interactive=False
                    )
            
            # State for storing results
            results_state = gr.State()
            
            # Tabs for different views
            with gr.Tabs():
                with gr.Tab("📊 Price Analysis"):
                    gr.Markdown("### Price Charts with Technical Indicators")
                    price_plot = gr.Plot(label="Price Chart")
                    
                    gr.Markdown("### Latest Data Point")
                    latest_data_json = gr.JSON(label="Latest Data")
                
                with gr.Tab("🤖 LLM Predictions"):
                    gr.Markdown("### Prediction Summary")
                    pred_table = gr.Dataframe(label="Predictions Table")
                    pred_plot = gr.Plot(label="Predictions Chart")
                
                with gr.Tab("📈 Trading Signals"):
                    gr.Markdown("### Trading Signals Summary")
                    signals_table = gr.Dataframe(label="Signals Table")
                    signals_plot = gr.Plot(label="Signals Chart")
                
                with gr.Tab("📋 Raw Data"):
                    gr.Markdown("### Raw Data Files")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Step 1: Raw Data")
                            raw_data_json = gr.JSON(label="Raw Data")
                        
                        with gr.Column():
                            gr.Markdown("#### Step 2: Processed Data")
                            processed_data_json = gr.JSON(label="Processed Data")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Step 3: Engineered Features")
                            features_data_json = gr.JSON(label="Features Data")
                        
                        with gr.Column():
                            gr.Markdown("#### Step 4: LLM Predictions")
                            predictions_data_json = gr.JSON(label="Predictions Data")
                    
                    gr.Markdown("#### Step 5: Trading Signals")
                    signals_data_json = gr.JSON(label="Signals Data")
            
            # Event handlers
            def load_and_update():
                results = self.load_latest_results()
                if not results:
                    return "No data found. Please run the analysis pipeline first.", results, None, None, None, None, None, None, None, None, None, None, None
                
                symbols = []
                if results.get('step1'):
                    symbols = list(results['step1'].keys())
                
                # Update predictions and signals
                pred_summary, pred_chart = self.create_prediction_summary(results.get('step4'))
                signals_summary, signals_chart = self.create_signals_summary(results.get('step5'))
                
                return (
                    f"Loaded data for {len(symbols)} symbols: {', '.join(symbols)}", 
                    results,
                    None, None,  # price chart and latest data
                    pred_summary, pred_chart,  # predictions
                    signals_summary, signals_chart,  # signals
                    results.get('step1'), results.get('step2'), results.get('step3'),  # raw data
                    results.get('step4'), results.get('step5')
                )
            
            def update_analysis(symbol, results):
                if not symbol or not results:
                    return None, None
                
                price_chart = self.create_price_chart(symbol, results.get('step3'))
                
                latest_data = None
                if results.get('step3') and symbol in results['step3']:
                    data = results['step3'][symbol]['data']
                    if data:
                        latest_data = data[-1]
                
                return price_chart, latest_data
            
            # Connect events
            load_btn.click(
                load_and_update,
                outputs=[
                    status_text, results_state,
                    price_plot, latest_data_json,
                    pred_table, pred_plot,
                    signals_table, signals_plot,
                    raw_data_json, processed_data_json, features_data_json, predictions_data_json, signals_data_json
                ]
            )
            
            symbol_dropdown.change(
                update_analysis,
                inputs=[symbol_dropdown, results_state],
                outputs=[price_plot, latest_data_json]
            )
        
        return demo

def main():
    """Main function to run the Gradio UI"""
    ui = StockAnalysisUI()
    demo = ui.create_ui()
    
    print("🚀 Starting Stock Analysis Results Viewer...")
    print("📊 The interface will open in your browser")
    print("🔗 If it doesn't open automatically, check the terminal for the local URL")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
