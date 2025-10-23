# 📊 量化交易方法综述

## 📋 概述

本文档全面介绍了除LLM之外的各种量化交易方法，包括传统技术分析、机器学习、统计套利、因子投资、高频交易、风险管理等多个维度。这些方法可以与现有的LLM系统形成互补，构建更全面的量化交易解决方案。

## 🎯 目录

1. [传统技术分析方法](#1-传统技术分析方法)
2. [机器学习方法](#2-机器学习方法)
3. [统计套利策略](#3-统计套利策略)
4. [因子投资策略](#4-因子投资策略)
5. [高频交易策略](#5-高频交易策略)
6. [风险管理策略](#6-风险管理策略)
7. [多资产配置策略](#7-多资产配置策略)
8. [事件驱动策略](#8-事件驱动策略)
9. [量化策略组合](#9-量化策略组合)
10. [方法对比与选择](#10-方法对比与选择)
11. [实际应用建议](#11-实际应用建议)

---

## 1. 传统技术分析方法

### 1.1 趋势跟踪策略

#### 移动平均线策略

```python
def ma_crossover_strategy(data, short_window=5, long_window=20):
    """移动平均线交叉策略"""
    data['ma_short'] = data['close'].rolling(window=short_window).mean()
    data['ma_long'] = data['close'].rolling(window=long_window).mean()

    # 生成信号
    data['signal'] = 0
    data['signal'][short_window:] = np.where(
        data['ma_short'][short_window:] > data['ma_long'][short_window:], 1, 0
    )
    data['positions'] = data['signal'].diff()

    return data
```

#### MACD策略
```python
def macd_strategy(data):
    """MACD策略"""
    # 计算MACD
    ema_12 = data['close'].ewm(span=12).mean()
    ema_26 = data['close'].ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line

    # 生成信号
    data['macd_signal'] = np.where(macd_line > signal_line, 1, -1)
    return data
```

#### 布林带策略
```python
def bollinger_bands_strategy(data, window=20, std_dev=2):
    """布林带策略"""
    data['bb_middle'] = data['close'].rolling(window=window).mean()
    data['bb_std'] = data['close'].rolling(window=window).std()
    data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
    data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)

    # 生成信号
    data['bb_signal'] = np.where(
        data['close'] < data['bb_lower'], 1,  # 买入信号
        np.where(data['close'] > data['bb_upper'], -1, 0)  # 卖出信号
    )
    return data
```

### 1.2 均值回归策略

#### RSI策略
```python
def rsi_strategy(data, period=14, oversold=30, overbought=70):
    """RSI均值回归策略"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # 生成信号
    data['rsi_signal'] = np.where(
        data['rsi'] < oversold, 1,  # 买入信号
        np.where(data['rsi'] > overbought, -1, 0)  # 卖出信号
    )
    return data
```

#### 随机振荡器策略
```python
def stochastic_strategy(data, k_period=14, d_period=3):
    """随机振荡器策略"""
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    data['stoch_k'] = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()

    # 生成信号
    data['stoch_signal'] = np.where(
        (data['stoch_k'] < 20) & (data['stoch_d'] < 20), 1,  # 买入信号
        np.where((data['stoch_k'] > 80) & (data['stoch_d'] > 80), -1, 0)  # 卖出信号
    )
    return data
```

### 1.3 成交量分析策略

#### 成交量价格趋势(VPT)
```python
def vpt_strategy(data):
    """成交量价格趋势策略"""
    data['vpt'] = (data['volume'] * (data['close'] - data['close'].shift(1)) / data['close'].shift(1)).cumsum()
    data['vpt_ma'] = data['vpt'].rolling(window=20).mean()

    # 生成信号
    data['vpt_signal'] = np.where(data['vpt'] > data['vpt_ma'], 1, -1)
    return data
```

---

## 2. 机器学习方法

### 2.1 监督学习

#### 随机森林分类器
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def random_forest_strategy(data, features, target_col='future_return'):
    """随机森林策略"""
    # 准备特征和标签
    X = data[features].dropna()
    y = (data[target_col] > 0).astype(int)

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 预测
    predictions = rf_model.predict(X_test)
    probabilities = rf_model.predict_proba(X_test)[:, 1]

    return rf_model, predictions, probabilities
```

#### 支持向量机
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def svm_strategy(data, features, target_col='future_return'):
    """支持向量机策略"""
    # 数据预处理
    scaler = StandardScaler()
    X = data[features].dropna()
    y = (data[target_col] > 0).astype(int)

    X_scaled = scaler.fit_transform(X)

    # 训练SVM
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_scaled, y)

    # 预测
    predictions = svm_model.predict(X_scaled)
    probabilities = svm_model.predict_proba(X_scaled)[:, 1]

    return svm_model, predictions, probabilities
```

### 2.2 深度学习

#### LSTM时间序列预测
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def lstm_strategy(data, sequence_length=60, features=['close', 'volume', 'rsi']):
    """LSTM时间序列预测策略"""
    # 准备数据
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    # 创建序列
    X, y = create_sequences(scaled_data, sequence_length)

    # 构建LSTM模型
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

    return model, scaler
```

#### 卷积神经网络(CNN)
```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

def cnn_strategy(data, sequence_length=60):
    """CNN时间序列策略"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model
```

### 2.3 无监督学习

#### K-means聚类
```python
from sklearn.cluster import KMeans

def kmeans_strategy(data, features, n_clusters=5):
    """K-means聚类策略"""
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # 计算每个聚类的平均收益
    data['cluster'] = clusters
    cluster_returns = data.groupby('cluster')['returns'].mean()

    # 生成交易信号
    data['cluster_signal'] = data['cluster'].map(cluster_returns).apply(
        lambda x: 1 if x > 0 else -1
    )

    return kmeans, clusters
```

---

## 3. 统计套利策略

### 3.1 配对交易

#### 协整检验
```python
from statsmodels.tsa.stattools import coint
from scipy.stats import pearsonr

def find_cointegrated_pairs(data, threshold=0.8):
    """寻找协整股票对"""
    n = len(data.columns)
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            stock1, stock2 = data.columns[i], data.columns[j]

            # 计算相关系数
            corr, _ = pearsonr(data[stock1], data[stock2])

            if corr > threshold:
                # 协整检验
                score, pvalue, _ = coint(data[stock1], data[stock2])
                if pvalue < 0.05:  # 协整关系显著
                    pairs.append((stock1, stock2, score, pvalue))

    return pairs
```

#### 配对交易策略
```python
def pairs_trading_strategy(stock1_prices, stock2_prices, lookback=20):
    """配对交易策略"""
    # 计算价差
    spread = stock1_prices - stock2_prices

    # 计算价差的均值和标准差
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()

    # 计算Z-score
    z_score = (spread - spread_mean) / spread_std

    # 生成交易信号
    signals = pd.Series(0, index=spread.index)
    signals[z_score > 2] = -1  # 卖出价差
    signals[z_score < -2] = 1  # 买入价差
    signals[abs(z_score) < 0.5] = 0  # 平仓

    return signals, z_score
```

### 3.2 均值回归策略

#### 价格偏离策略
```python
def price_deviation_strategy(prices, window=20, threshold=2):
    """价格偏离均值回归策略"""
    # 计算移动平均和标准差
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()

    # 计算Z-score
    z_score = (prices - ma) / std

    # 生成信号
    signals = pd.Series(0, index=prices.index)
    signals[z_score > threshold] = -1  # 卖出信号
    signals[z_score < -threshold] = 1  # 买入信号

    return signals, z_score
```

---

## 4. 因子投资策略

### 4.1 多因子模型

#### 价值因子
```python
def value_factors(data):
    """价值因子计算"""
    # 市盈率倒数
    data['pe_ratio'] = data['market_cap'] / data['net_income']
    data['earnings_yield'] = 1 / data['pe_ratio']

    # 市净率倒数
    data['pb_ratio'] = data['market_cap'] / data['book_value']
    data['book_to_market'] = 1 / data['pb_ratio']

    # 市销率倒数
    data['ps_ratio'] = data['market_cap'] / data['revenue']
    data['sales_yield'] = 1 / data['ps_ratio']

    return data
```

#### 动量因子
```python
def momentum_factors(data, lookback_periods=[1, 3, 6, 12]):
    """动量因子计算"""
    for period in lookback_periods:
        data[f'momentum_{period}m'] = data['close'].pct_change(period * 21)  # 月度数据

    # 相对强弱指数
    data['rsi_momentum'] = data['close'].rolling(window=14).apply(
        lambda x: 100 - (100 / (1 + x.pct_change().where(x.pct_change() > 0, 0).mean() /
                               (-x.pct_change().where(x.pct_change() < 0, 0)).mean()))
    )

    return data
```

#### 质量因子
```python
def quality_factors(data):
    """质量因子计算"""
    # ROE
    data['roe'] = data['net_income'] / data['shareholders_equity']

    # ROA
    data['roa'] = data['net_income'] / data['total_assets']

    # 毛利率
    data['gross_margin'] = (data['revenue'] - data['cost_of_goods_sold']) / data['revenue']

    # 净利率
    data['net_margin'] = data['net_income'] / data['revenue']

    # 债务权益比
    data['debt_to_equity'] = data['total_debt'] / data['shareholders_equity']

    return data
```

### 4.2 因子组合

#### 多因子评分模型
```python
def multi_factor_score(data, factor_weights=None):
    """多因子评分模型"""
    if factor_weights is None:
        factor_weights = {
            'value': 0.3,
            'momentum': 0.3,
            'quality': 0.4
        }

    # 标准化因子
    normalized_factors = {}
    for factor_name in factor_weights.keys():
        factor_data = data[f'{factor_name}_score']
        normalized_factors[factor_name] = (factor_data - factor_data.mean()) / factor_data.std()

    # 计算综合评分
    composite_score = sum(
        factor_weights[name] * factor for name, factor in normalized_factors.items()
    )

    return composite_score
```

---

## 5. 高频交易策略

### 5.1 微观结构策略

#### 订单流不平衡
```python
def order_flow_imbalance(bid_volume, ask_volume):
    """订单流不平衡策略"""
    ofi = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    return ofi

def order_flow_strategy(data):
    """基于订单流的策略"""
    data['ofi'] = order_flow_imbalance(data['bid_volume'], data['ask_volume'])

    # 生成信号
    data['ofi_signal'] = np.where(data['ofi'] > 0.1, 1,  # 买入信号
                                 np.where(data['ofi'] < -0.1, -1, 0))  # 卖出信号

    return data
```

#### 买卖价差策略
```python
def spread_arbitrage_strategy(bid_prices, ask_prices, mid_prices):
    """价差套利策略"""
    spreads = ask_prices - bid_prices
    spread_ratios = spreads / mid_prices

    # 价差过大时的套利机会
    arbitrage_opportunities = spread_ratios > 0.001  # 0.1%的价差阈值

    return arbitrage_opportunities, spreads
```

### 5.2 动量策略

#### 短期动量
```python
def short_term_momentum_strategy(data, window=5):
    """短期动量策略"""
    # 计算短期收益率
    data['short_return'] = data['close'].pct_change(window)

    # 计算成交量加权平均价格
    data['vwap'] = (data['close'] * data['volume']).rolling(window=window).sum() / \
                   data['volume'].rolling(window=window).sum()

    # 价格相对于VWAP的位置
    data['price_vwap_ratio'] = data['close'] / data['vwap']

    # 生成信号
    data['momentum_signal'] = np.where(
        (data['short_return'] > 0.01) & (data['price_vwap_ratio'] > 1.001), 1,  # 买入
        np.where((data['short_return'] < -0.01) & (data['price_vwap_ratio'] < 0.999), -1, 0)  # 卖出
    )

    return data
```

---

## 6. 风险管理策略

### 6.1 动态对冲

#### Delta对冲
```python
def delta_hedge(portfolio_value, market_exposure, hedge_ratio=0.5):
    """Delta对冲策略"""
    hedge_amount = portfolio_value * market_exposure * hedge_ratio
    return hedge_amount

def dynamic_hedge_strategy(portfolio, market_data, hedge_threshold=0.1):
    """动态对冲策略"""
    # 计算投资组合的Beta
    portfolio_beta = calculate_portfolio_beta(portfolio, market_data)

    # 当Beta超过阈值时进行对冲
    if abs(portfolio_beta) > hedge_threshold:
        hedge_amount = delta_hedge(portfolio.value, portfolio_beta)
        return hedge_amount

    return 0
```

### 6.2 VaR计算

#### 历史模拟法
```python
def historical_var(returns, confidence_level=0.05):
    """历史模拟法计算VaR"""
    return np.percentile(returns, confidence_level * 100)

def monte_carlo_var(returns, confidence_level=0.05, n_simulations=10000):
    """蒙特卡洛模拟VaR"""
    # 估计参数
    mu = returns.mean()
    sigma = returns.std()

    # 蒙特卡洛模拟
    simulated_returns = np.random.normal(mu, sigma, n_simulations)

    return np.percentile(simulated_returns, confidence_level * 100)
```

### 6.3 压力测试

#### 情景分析
```python
def stress_test(portfolio, scenarios):
    """压力测试"""
    results = {}

    for scenario_name, scenario_data in scenarios.items():
        # 计算情景下的投资组合价值
        portfolio_value = calculate_portfolio_value(portfolio, scenario_data)
        results[scenario_name] = portfolio_value

    return results

def scenario_analysis():
    """情景分析"""
    scenarios = {
        'market_crash': {'market_return': -0.2},
        'interest_rate_shock': {'interest_rate_change': 0.02},
        'volatility_spike': {'volatility_multiplier': 2.0}
    }

    return scenarios
```

---

## 7. 多资产配置策略

### 7.1 现代投资组合理论

#### Markowitz优化
```python
from scipy.optimize import minimize

def markowitz_optimization(returns, risk_free_rate=0.02):
    """Markowitz投资组合优化"""
    n_assets = len(returns.columns)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(returns.cov(), weights))

    def portfolio_return(weights):
        return np.sum(returns.mean() * weights)

    def sharpe_ratio(weights):
        return (portfolio_return(weights) - risk_free_rate) / np.sqrt(portfolio_variance(weights))

    # 约束条件
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    # 优化夏普比率
    result = minimize(lambda x: -sharpe_ratio(x),
                     x0=np.array([1/n_assets]*n_assets),
                     method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x
```

#### 风险平价
```python
def risk_parity_weights(returns):
    """风险平价权重"""
    cov_matrix = returns.cov()
    inv_cov = np.linalg.inv(cov_matrix)

    # 计算风险贡献
    risk_contrib = np.diag(inv_cov)
    weights = risk_contrib / np.sum(risk_contrib)

    return weights
```

### 7.2 动态资产配置

#### 战术资产配置
```python
def tactical_asset_allocation(returns, market_regime):
    """战术资产配置"""
    if market_regime == 'bull':
        # 牛市：增加股票权重
        equity_weight = 0.7
        bond_weight = 0.3
    elif market_regime == 'bear':
        # 熊市：增加债券权重
        equity_weight = 0.3
        bond_weight = 0.7
    else:  # 震荡市
        equity_weight = 0.5
        bond_weight = 0.5

    return {'equity': equity_weight, 'bond': bond_weight}
```

---

## 8. 事件驱动策略

### 8.1 财报发布策略

#### 财报前交易
```python
def earnings_announcement_strategy(earnings_date, current_date, earnings_beat):
    """财报发布策略"""
    days_to_earnings = (earnings_date - current_date).days

    if days_to_earnings <= 5:  # 财报前5天
        if earnings_beat > 0.1:  # 预期大幅超预期
            return 'buy'
        elif earnings_beat < -0.1:  # 预期大幅不及预期
            return 'sell'

    return 'hold'
```

### 8.2 并购套利

#### 并购交易策略
```python
def merger_arbitrage_strategy(target_price, offer_price, probability, time_to_close):
    """并购套利策略"""
    # 计算预期收益
    expected_return = (offer_price - target_price) / target_price

    # 年化收益率
    annualized_return = expected_return * (365 / time_to_close)

    # 风险调整收益
    risk_adjusted_return = annualized_return * probability

    if risk_adjusted_return > 0.1:  # 10%的阈值
        return 'buy'
    else:
        return 'hold'
```

---

## 9. 量化策略组合

### 9.1 策略组合优化

#### 多策略组合
```python
class StrategyPortfolio:
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.weights = weights or [1/len(strategies)] * len(strategies)

    def get_combined_signal(self, data):
        """获取组合信号"""
        signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            signals.append(signal)

        # 加权平均信号
        combined_signal = sum(w * s for w, s in zip(self.weights, signals))
        return combined_signal

    def optimize_weights(self, historical_data):
        """优化权重"""
        from scipy.optimize import minimize

        def objective(weights):
            # 计算组合收益
            portfolio_returns = self.calculate_portfolio_returns(historical_data, weights)
            # 最大化夏普比率
            return -portfolio_returns.mean() / portfolio_returns.std()

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.strategies)))

        result = minimize(objective, x0=self.weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        self.weights = result.x
        return result.x
```

### 9.2 策略轮动

#### 基于表现的策略轮动
```python
def strategy_rotation(strategies, lookback_period=30):
    """基于表现的策略轮动"""
    strategy_performance = {}

    for strategy in strategies:
        # 计算策略表现
        returns = strategy.get_returns(lookback_period)
        sharpe_ratio = returns.mean() / returns.std()
        strategy_performance[strategy.name] = sharpe_ratio

    # 选择表现最好的策略
    best_strategy = max(strategy_performance, key=strategy_performance.get)
    return best_strategy
```

---

## 10. 方法对比与选择

### 10.1 方法特点对比

| 方法类别 | 优势 | 劣势 | 适用场景 |
|---------|------|------|----------|
| 技术分析 | 简单直观，易于理解 | 滞后性，容易产生假信号 | 趋势明显的市场 |
| 机器学习 | 能捕捉复杂模式 | 需要大量数据，容易过拟合 | 数据丰富的市场 |
| 统计套利 | 市场中性，风险较低 | 需要高相关性，机会有限 | 相关性高的资产 |
| 因子投资 | 理论基础扎实 | 因子可能失效 | 长期投资 |
| 高频交易 | 收益稳定 | 技术要求高，成本高 | 流动性好的市场 |
| 风险管理 | 控制下行风险 | 可能限制收益 | 所有策略都需要 |

### 10.2 选择标准

#### 数据要求
- **技术分析**：价格和成交量数据
- **机器学习**：历史价格、基本面、宏观经济数据
- **统计套利**：多个相关资产的历史数据
- **因子投资**：基本面数据、财务报表
- **高频交易**：tick级别数据

#### 技术能力要求
- **技术分析**：基础编程能力
- **机器学习**：Python、scikit-learn、TensorFlow
- **统计套利**：统计学知识、协整分析
- **因子投资**：财务分析能力
- **高频交易**：系统编程、低延迟技术

#### 资金要求
- **技术分析**：小资金即可
- **机器学习**：中等资金
- **统计套利**：大资金（需要分散化）
- **因子投资**：大资金（长期投资）
- **高频交易**：大资金（技术投入高）

---

## 11. 实际应用建议

### 11.1 策略开发流程

1. **数据准备**
   - 收集历史数据
   - 数据清洗和预处理
   - 特征工程

2. **策略设计**
   - 选择合适的方法
   - 定义交易规则
   - 设置参数

3. **回测验证**
   - 历史数据回测
   - 参数优化
   - 风险评估

4. **实盘测试**
   - 小资金实盘
   - 监控表现
   - 调整参数

5. **正式运行**
   - 扩大资金规模
   - 持续监控
   - 定期优化

### 11.2 与LLM系统集成

#### 混合策略框架
```python
class HybridStrategy:
    def __init__(self, llm_predictor, traditional_strategies):
        self.llm_predictor = llm_predictor
        self.traditional_strategies = traditional_strategies

    def generate_signal(self, data):
        # LLM预测
        llm_signal = self.llm_predictor.predict(data)

        # 传统策略信号
        traditional_signals = []
        for strategy in self.traditional_strategies:
            signal = strategy.generate_signal(data)
            traditional_signals.append(signal)

        # 信号融合
        combined_signal = self.fuse_signals(llm_signal, traditional_signals)
        return combined_signal

    def fuse_signals(self, llm_signal, traditional_signals):
        """信号融合算法"""
        # 加权平均
        weights = [0.4] + [0.6/len(traditional_signals)] * len(traditional_signals)
        all_signals = [llm_signal] + traditional_signals

        combined = sum(w * s for w, s in zip(weights, all_signals))
        return combined
```

### 11.3 风险管理集成

#### 多层次风险管理
```python
class RiskManager:
    def __init__(self, max_position_size=0.1, max_drawdown=0.15):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown

    def check_risk_limits(self, portfolio, new_signal):
        """检查风险限制"""
        # 检查单笔交易风险
        if abs(new_signal) > self.max_position_size:
            return False

        # 检查总仓位风险
        total_exposure = sum(abs(pos) for pos in portfolio.positions.values())
        if total_exposure > 1.0:  # 100%仓位限制
            return False

        # 检查回撤风险
        current_drawdown = self.calculate_drawdown(portfolio)
        if current_drawdown > self.max_drawdown:
            return False

        return True
```

### 11.4 性能监控

#### 策略表现监控
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, returns):
        """计算性能指标"""
        metrics = {
            'total_return': returns.sum(),
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean()
        }
        return metrics

    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

---

## 📚 总结

本文档全面介绍了除LLM之外的各种量化交易方法，包括：

1. **传统技术分析**：基于价格和成交量的技术指标
2. **机器学习**：监督学习、深度学习、无监督学习
3. **统计套利**：配对交易、均值回归
4. **因子投资**：多因子模型、因子组合
5. **高频交易**：微观结构策略、短期动量
6. **风险管理**：动态对冲、VaR计算、压力测试
7. **多资产配置**：现代投资组合理论、动态配置
8. **事件驱动**：财报策略、并购套利
9. **策略组合**：多策略组合、策略轮动

这些方法可以与您现有的LLM系统形成互补，构建更全面、更稳健的量化交易解决方案。建议根据您的具体需求、数据可用性和技术能力选择合适的方法进行实施。

---

*文档版本: 1.0*
*最后更新: 2024-10-23*
*作者: AI Quant Trader Team*
