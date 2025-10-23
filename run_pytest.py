#!/usr/bin/env python3
"""
AI Trading System - pytest测试运行器

运行所有单元测试，支持不同的测试模式和选项。
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_tests(test_path=None, markers=None, verbose=False, coverage=False, 
              parallel=False, html_report=False, xml_report=False):
    """运行测试"""
    
    # 构建pytest命令
    cmd = ['python', '-m', 'pytest']
    
    # 测试路径
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append('tests/')
    
    # 详细输出
    if verbose:
        cmd.extend(['-v', '-s'])
    else:
        cmd.append('-v')
    
    # 标记过滤
    if markers:
        cmd.extend(['-m', markers])
    
    # 并行运行
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(['-n', 'auto'])
        except ImportError:
            print("Warning: pytest-xdist not installed, running tests sequentially")
    
    # 覆盖率
    if coverage:
        try:
            import pytest_cov
            cmd.extend(['--cov=src', '--cov-report=term-missing'])
            if html_report:
                cmd.extend(['--cov-report=html:htmlcov'])
        except ImportError:
            print("Warning: pytest-cov not installed, coverage disabled")
    
    # HTML报告
    if html_report:
        cmd.extend(['--html=reports/report.html', '--self-contained-html'])
    
    # XML报告
    if xml_report:
        cmd.extend(['--junitxml=reports/junit.xml'])
    
    # 其他选项
    cmd.extend([
        '--tb=short',
        '--strict-markers',
        '--disable-warnings',
        '--color=yes',
        '--durations=10'
    ])
    
    # 创建报告目录
    os.makedirs('reports', exist_ok=True)
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    # 运行测试
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run AI Trading System tests')
    
    parser.add_argument('--path', '-p', 
                       help='Test path or file to run')
    parser.add_argument('--markers', '-m',
                       help='pytest markers to filter tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Run with coverage')
    parser.add_argument('--parallel', '-n', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--xml-report', action='store_true',
                       help='Generate XML report')
    parser.add_argument('--unit', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--data', action='store_true',
                       help='Run only data processing tests')
    parser.add_argument('--model', action='store_true',
                       help='Run only model tests')
    parser.add_argument('--strategy', action='store_true',
                       help='Run only strategy tests')
    
    args = parser.parse_args()
    
    # 确定测试路径
    test_path = args.path
    if not test_path:
        if args.unit:
            test_path = 'tests/test_*.py'
        elif args.integration:
            test_path = 'tests/test_integration_*.py'
        elif args.data:
            test_path = 'tests/test_data_*.py'
        elif args.model:
            test_path = 'tests/test_*model*.py'
        elif args.strategy:
            test_path = 'tests/test_*strategy*.py'
    
    # 确定标记
    markers = args.markers
    if not markers:
        if args.unit:
            markers = 'unit'
        elif args.integration:
            markers = 'integration'
        elif args.data:
            markers = 'data'
        elif args.model:
            markers = 'model'
        elif args.strategy:
            markers = 'strategy'
    
    # 运行测试
    return run_tests(
        test_path=test_path,
        markers=markers,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        html_report=args.html_report,
        xml_report=args.xml_report
    )


if __name__ == '__main__':
    sys.exit(main())
