#!/usr/bin/env python3
"""
Test Report Generator
Aggregates and formats test results from multiple sources
"""

import os
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

class TestReportGenerator:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.report_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {},
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'security_tests': {},
            'load_tests': {},
            'coverage': {}
        }
        
    def collect_cargo_test_results(self) -> Dict[str, Any]:
        """Collect Rust cargo test results"""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'ignored': 0,
            'duration_ms': 0,
            'failures': []
        }
        
        # Look for cargo test output files
        for test_file in self.results_dir.glob('*test*.log'):
            with open(test_file, 'r') as f:
                content = f.read()
                # Parse cargo test output (simplified)
                if 'test result:' in content:
                    lines = content.split('\\n')
                    for line in lines:
                        if 'test result:' in line:
                            # Extract test counts
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'passed;':
                                    results['passed'] = int(parts[i-1])
                                elif part == 'failed;':
                                    results['failed'] = int(parts[i-1])
                                elif part == 'ignored;':
                                    results['ignored'] = int(parts[i-1])
                            
                            results['total'] = results['passed'] + results['failed'] + results['ignored']
        
        return results
        
    def collect_performance_results(self) -> Dict[str, Any]:
        """Collect performance benchmark results"""
        results = {
            'benchmarks': [],
            'summary': {
                'total_benchmarks': 0,
                'avg_performance_score': 0.0,
                'regressions': []
            }
        }
        
        perf_files = list(self.results_dir.glob('*perf*.json'))
        
        for perf_file in perf_files:
            try:
                with open(perf_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results['benchmarks'].extend(data)
                    elif isinstance(data, dict):
                        results['benchmarks'].append(data)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        results['summary']['total_benchmarks'] = len(results['benchmarks'])
        
        return results
        
    def collect_security_results(self) -> Dict[str, Any]:
        """Collect security audit results"""
        results = {
            'vulnerabilities': [],
            'clippy_warnings': 0,
            'audit_passed': True
        }
        
        # Security audit results
        audit_file = self.results_dir / 'security-audit.json'
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    audit_data = json.load(f)
                    if 'vulnerabilities' in audit_data:
                        results['vulnerabilities'] = audit_data['vulnerabilities']
                        results['audit_passed'] = len(audit_data['vulnerabilities']) == 0
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return results
        
    def collect_load_test_results(self) -> Dict[str, Any]:
        """Collect k6 load test results"""
        results = {
            'requests': 0,
            'failures': 0,
            'avg_response_time': 0.0,
            'p95_response_time': 0.0,
            'throughput': 0.0
        }
        
        load_file = self.results_dir / 'load-test-results.json'
        if load_file.exists():
            try:
                with open(load_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    
                    # Extract key metrics
                    if 'http_reqs' in metrics:
                        results['requests'] = metrics['http_reqs']['count']
                    
                    if 'http_req_duration' in metrics:
                        duration = metrics['http_req_duration']
                        results['avg_response_time'] = duration.get('avg', 0)
                        results['p95_response_time'] = duration.get('p(95)', 0)
                    
                    if 'http_req_failed' in metrics:
                        results['failures'] = metrics['http_req_failed']['count']
                    
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return results
        
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate overall test summary"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'success_rate': 0.0,
            'overall_status': 'UNKNOWN',
            'test_categories': {
                'unit': 'UNKNOWN',
                'integration': 'UNKNOWN', 
                'performance': 'UNKNOWN',
                'security': 'UNKNOWN',
                'load': 'UNKNOWN'
            }
        }
        
        # Aggregate from all test categories
        for category in ['unit_tests', 'integration_tests']:
            if category in self.report_data:
                data = self.report_data[category]
                summary['total_tests'] += data.get('total', 0)
                summary['passed_tests'] += data.get('passed', 0)
                summary['failed_tests'] += data.get('failed', 0)
        
        # Calculate success rate
        if summary['total_tests'] > 0:
            summary['success_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
        
        # Determine overall status
        if summary['failed_tests'] == 0 and summary['total_tests'] > 0:
            summary['overall_status'] = 'PASS'
        elif summary['failed_tests'] > 0:
            summary['overall_status'] = 'FAIL'
        else:
            summary['overall_status'] = 'NO_TESTS'
        
        # Category statuses
        for category, key in [
            ('unit', 'unit_tests'),
            ('integration', 'integration_tests')
        ]:
            if key in self.report_data:
                data = self.report_data[key]
                if data.get('failed', 0) == 0 and data.get('total', 0) > 0:
                    summary['test_categories'][category] = 'PASS'
                elif data.get('failed', 0) > 0:
                    summary['test_categories'][category] = 'FAIL'
                else:
                    summary['test_categories'][category] = 'NO_TESTS'
        
        # Security status
        sec_data = self.report_data.get('security_tests', {})
        if sec_data.get('audit_passed', False):
            summary['test_categories']['security'] = 'PASS'
        else:
            summary['test_categories']['security'] = 'FAIL'
        
        return summary
        
    def generate_html_report(self) -> str:
        """Generate HTML test report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Doc-RAG Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric { padding: 20px; border-radius: 8px; text-align: center; }
        .metric.pass { background-color: #d4edda; border-left: 5px solid #28a745; }
        .metric.fail { background-color: #f8d7da; border-left: 5px solid #dc3545; }
        .metric.warning { background-color: #fff3cd; border-left: 5px solid #ffc107; }
        .metric h3 { margin: 0 0 10px 0; color: #333; }
        .metric .value { font-size: 24px; font-weight: bold; margin: 5px 0; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .status-pass { color: #28a745; font-weight: bold; }
        .status-fail { color: #dc3545; font-weight: bold; }
        .footer { text-align: center; margin-top: 30px; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Doc-RAG Test Report</h1>
            <p>Generated on {timestamp}</p>
            <p>Overall Status: <span class="status-{overall_status_class}">{overall_status}</span></p>
        </div>
        
        <div class="summary">
            <div class="metric {total_class}">
                <h3>Total Tests</h3>
                <div class="value">{total_tests}</div>
            </div>
            <div class="metric {passed_class}">
                <h3>Passed</h3>
                <div class="value">{passed_tests}</div>
            </div>
            <div class="metric {failed_class}">
                <h3>Failed</h3>
                <div class="value">{failed_tests}</div>
            </div>
            <div class="metric {success_rate_class}">
                <h3>Success Rate</h3>
                <div class="value">{success_rate:.1f}%</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Test Categories</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Status</th>
                        <th>Tests</th>
                        <th>Passed</th>
                        <th>Failed</th>
                    </tr>
                </thead>
                <tbody>
                    {category_rows}
                </tbody>
            </table>
        </div>
        
        {performance_section}
        
        {security_section}
        
        <div class="footer">
            <p>Generated by Doc-RAG Test Runner v1.0</p>
        </div>
    </div>
</body>
</html>
        """
        
        summary = self.report_data['summary']
        
        # Format values
        values = {
            'timestamp': datetime.fromisoformat(self.report_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'overall_status': summary['overall_status'],
            'overall_status_class': 'pass' if summary['overall_status'] == 'PASS' else 'fail',
            'total_tests': summary['total_tests'],
            'passed_tests': summary['passed_tests'],
            'failed_tests': summary['failed_tests'],
            'success_rate': summary['success_rate'],
            'total_class': 'pass' if summary['total_tests'] > 0 else 'warning',
            'passed_class': 'pass' if summary['passed_tests'] > 0 else 'warning',
            'failed_class': 'fail' if summary['failed_tests'] > 0 else 'pass',
            'success_rate_class': 'pass' if summary['success_rate'] >= 90 else 'warning' if summary['success_rate'] >= 70 else 'fail'
        }
        
        # Generate category rows
        category_rows = []
        for category, status in summary['test_categories'].items():
            data = self.report_data.get(f'{category}_tests', {})
            row = f"""
                <tr>
                    <td>{category.title()}</td>
                    <td><span class="status-{status.lower()}">{status}</span></td>
                    <td>{data.get('total', 0)}</td>
                    <td>{data.get('passed', 0)}</td>
                    <td>{data.get('failed', 0)}</td>
                </tr>
            """
            category_rows.append(row)
        
        values['category_rows'] = ''.join(category_rows)
        
        # Performance section
        perf_data = self.report_data.get('performance_tests', {})
        if perf_data.get('summary', {}).get('total_benchmarks', 0) > 0:
            values['performance_section'] = f"""
                <div class="section">
                    <h2>⚡ Performance Results</h2>
                    <p>Total Benchmarks: {perf_data['summary']['total_benchmarks']}</p>
                </div>
            """
        else:
            values['performance_section'] = ""
        
        # Security section
        sec_data = self.report_data.get('security_tests', {})
        vuln_count = len(sec_data.get('vulnerabilities', []))
        values['security_section'] = f"""
            <div class="section">
                <h2>🔒 Security Results</h2>
                <p>Vulnerabilities Found: <span class="{'status-fail' if vuln_count > 0 else 'status-pass'}">{vuln_count}</span></p>
                <p>Audit Status: <span class="{'status-pass' if sec_data.get('audit_passed', False) else 'status-fail'}">
                    {'PASS' if sec_data.get('audit_passed', False) else 'FAIL'}
                </span></p>
            </div>
        """
        
        return html_template.format(**values)
    
    def generate_json_report(self) -> str:
        """Generate JSON test report"""
        return json.dumps(self.report_data, indent=2)
        
    def run(self):
        """Main execution method"""
        print("🔍 Collecting test results...")
        
        # Collect results from different sources
        self.report_data['unit_tests'] = self.collect_cargo_test_results()
        self.report_data['integration_tests'] = self.collect_cargo_test_results()
        self.report_data['performance_tests'] = self.collect_performance_results()
        self.report_data['security_tests'] = self.collect_security_results()
        self.report_data['load_tests'] = self.collect_load_test_results()
        
        # Calculate summary
        self.report_data['summary'] = self.calculate_summary()
        
        print("📊 Generating reports...")
        
        # Generate HTML report
        html_report = self.generate_html_report()
        html_file = self.results_dir / 'test-report.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        # Generate JSON report
        json_report = self.generate_json_report()
        json_file = self.results_dir / 'test-report.json'
        with open(json_file, 'w') as f:
            f.write(json_report)
        
        print(f"✅ Reports generated:")
        print(f"   HTML: {html_file}")
        print(f"   JSON: {json_file}")
        
        # Print summary to stdout
        summary = self.report_data['summary']
        print(f"\\n📋 Test Summary:")
        print(f"   Overall Status: {summary['overall_status']}")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        # Exit with appropriate code
        if summary['overall_status'] == 'PASS':
            sys.exit(0)
        else:
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Generate Doc-RAG test reports')
    parser.add_argument('results_dir', nargs='?', default='/test-results',
                       help='Directory containing test results (default: /test-results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    generator = TestReportGenerator(args.results_dir)
    generator.run()

if __name__ == '__main__':
    main()