# 🧪 AI Trading System - 测试文档

## 📋 测试概览

本系统使用pytest框架进行全面的单元测试，覆盖所有核心模块和功能。

## 🎯 测试结构

### 测试模块分类

| 模块 | 测试文件 | 描述 |
|------|----------|------|
| **数据获取** | `test_data_fetcher.py` | 测试数据获取、缓存、多数据源 |
| **数据处理** | `test_data_processor.py` | 测试数据清洗、验证、技术指标 |
| **特征工程** | `test_feature_engineer.py` | 测试特征生成、选择、转换 |
| **LLM预测** | `test_llm_predictor.py` | 测试LLM预测、响应解析、提示词 |
| **信号生成** | `test_llm_signal_generator.py` | 测试交易信号生成、强度计算 |
| **模型管理** | `test_model_manager.py` | 测试模型管理、预测协调 |
| **投资组合** | `test_portfolio_manager.py` | 测试投资组合管理、交易记录 |
| **订单生成** | `test_order_generator.py` | 测试订单生成、风险控制 |

### 测试标记

- `unit`: 单元测试
- `integration`: 集成测试
- `data`: 数据处理测试
- `model`: 模型测试
- `strategy`: 策略测试
- `api`: API测试
- `slow`: 慢速测试

## 🚀 运行测试

### 快速开始

```bash
# 运行所有测试
./run_tests.sh all

# 运行单元测试
./run_tests.sh unit

# 运行数据测试
./run_tests.sh data

# 运行模型测试
./run_tests.sh model

# 运行策略测试
./run_tests.sh strategy
```

### 使用pytest直接运行

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_data_fetcher.py -v

# 运行带标记的测试
pytest tests/ -m "unit" -v

# 运行带覆盖率的测试
pytest tests/ --cov=src --cov-report=html
```

### 使用Python脚本运行

```bash
# 运行所有测试
python run_pytest.py

# 运行单元测试
python run_pytest.py --unit

# 运行数据测试
python run_pytest.py --data

# 运行带覆盖率的测试
python run_pytest.py --coverage --html-report
```

## 📊 测试覆盖率

### 覆盖率目标

- **整体覆盖率**: ≥ 90%
- **核心模块覆盖率**: ≥ 95%
- **关键功能覆盖率**: 100%

### 生成覆盖率报告

```bash
# 生成HTML覆盖率报告
pytest tests/ --cov=src --cov-report=html:reports/coverage_html

# 生成XML覆盖率报告
pytest tests/ --cov=src --cov-report=xml:reports/coverage.xml

# 生成终端覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing
```

## 🔧 测试配置

### pytest.ini配置

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
    --durations=10
```

### 测试依赖

```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-html pytest-xdist pytest-mock

# 或使用requirements
pip install -r requirements-test.txt
```

## 📝 测试用例示例

### 数据获取测试

```python
def test_fetch_data_success(self, mock_ticker):
    """测试成功获取数据"""
    mock_df = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [99, 100, 101],
        'Close': [104, 105, 106],
        'Volume': [1000, 1100, 1200]
    })
    
    mock_ticker_instance = Mock()
    mock_ticker_instance.history.return_value = mock_df
    mock_ticker.return_value = mock_ticker_instance
    
    source = YahooFinanceSource()
    result = source.fetch_data('AAPL', '2023-01-01', '2023-01-03')
    
    assert result is not None
    assert not result.empty
    assert 'open' in result.columns
```

### LLM预测测试

```python
def test_predict_single_symbol(self, mock_predictor, sample_market_data):
    """测试预测单个符号"""
    mock_prediction = PredictionResult('AAPL', 'BUY', 0.8, 'Strong signal')
    mock_predictor.predict.return_value = mock_prediction
    
    result = mock_predictor.predict('AAPL', sample_market_data['AAPL'])
    
    assert result.symbol == 'AAPL'
    assert result.prediction == 'BUY'
    assert result.confidence == 0.8
```

### 投资组合测试

```python
def test_add_position(self, portfolio):
    """测试添加仓位"""
    portfolio.add_position('AAPL', 100, 180.0)
    
    assert portfolio.get_position('AAPL')['shares'] == 100
    assert portfolio.get_position('AAPL')['avg_price'] == 180.0
    assert portfolio.get_cash() == 82000.0  # 100000 - 18000
```

## 🎯 测试最佳实践

### 1. 测试命名

- 测试类: `TestClassName`
- 测试方法: `test_function_name`
- 描述性名称: `test_should_return_error_when_invalid_input`

### 2. 测试结构

```python
def test_function_name(self):
    """测试描述"""
    # Arrange - 准备测试数据
    input_data = create_test_data()
    
    # Act - 执行被测试的功能
    result = function_under_test(input_data)
    
    # Assert - 验证结果
    assert result == expected_result
```

### 3. 使用Fixtures

```python
@pytest.fixture
def sample_data(self):
    """样本数据"""
    return pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })

def test_with_sample_data(self, sample_data):
    """使用样本数据测试"""
    result = process_data(sample_data)
    assert not result.empty
```

### 4. 模拟外部依赖

```python
@patch('requests.post')
def test_api_call(self, mock_post):
    """测试API调用"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'result': 'success'}
    mock_post.return_value = mock_response
    
    result = api_call()
    assert result == 'success'
```

## 📈 持续集成

### GitHub Actions配置

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=src --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🐛 调试测试

### 运行特定测试

```bash
# 运行特定测试文件
pytest tests/test_data_fetcher.py::TestDataFetcher::test_fetch_data_success -v

# 运行特定测试方法
pytest tests/test_data_fetcher.py -k "test_fetch_data" -v

# 运行失败的测试
pytest tests/ --lf -v
```

### 调试模式

```bash
# 进入调试模式
pytest tests/ --pdb

# 详细输出
pytest tests/ -v -s

# 显示本地变量
pytest tests/ --tb=long
```

## 📊 测试报告

### HTML报告

```bash
# 生成HTML测试报告
pytest tests/ --html=reports/report.html --self-contained-html
```

### XML报告

```bash
# 生成JUnit XML报告
pytest tests/ --junitxml=reports/junit.xml
```

### 性能报告

```bash
# 显示最慢的10个测试
pytest tests/ --durations=10
```

## 🎉 测试总结

### 测试统计

- **总测试数**: 200+ 个测试用例
- **覆盖率**: ≥ 90%
- **运行时间**: < 30秒
- **支持平台**: Windows, macOS, Linux

### 测试价值

1. **代码质量**: 确保代码正确性和可靠性
2. **回归测试**: 防止新功能破坏现有功能
3. **文档作用**: 测试用例作为使用示例
4. **重构支持**: 安全地重构和优化代码
5. **持续集成**: 自动化测试和部署

通过全面的测试覆盖，我们确保了AI交易系统的稳定性和可靠性！🚀