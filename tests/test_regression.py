#!/usr/bin/env python3
"""
Simple regression testing for Doc-RAG system
Generates test PDFs, runs queries, compares to baseline
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

API_URL = "http://localhost:8080"

# Test cases with expected answers
TEST_CASES = [
    {
        "pdf_content": """
        PCI DSS Compliance Requirements
        
        Organizations must encrypt cardholder data at rest using AES-256 encryption.
        Network segmentation is required to isolate the cardholder data environment.
        Regular security testing must be performed quarterly.
        """,
        "questions": [
            ("What encryption is required?", "AES-256"),
            ("How often is security testing required?", "quarterly"),
            ("What must be isolated?", "cardholder data environment")
        ]
    },
    {
        "pdf_content": """
        Machine Learning Best Practices
        
        Always split your data into training, validation, and test sets.
        Use cross-validation to avoid overfitting.
        Monitor model drift in production environments.
        """,
        "questions": [
            ("How should data be split?", "training, validation, and test"),
            ("What helps avoid overfitting?", "cross-validation"),
            ("What should be monitored in production?", "model drift")
        ]
    }
]

def create_test_pdf(content: str, filename: str) -> Path:
    """Create a simple text file as PDF placeholder for testing"""
    path = Path(f"tests/generated/{filename}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(content)
    return path

def upload_document(filepath: Path) -> str:
    """Upload document and return doc_id"""
    with open(filepath, 'rb') as f:
        response = requests.post(
            f"{API_URL}/upload",
            files={'file': f},
            data={'name': filepath.stem}
        )
    response.raise_for_status()
    return response.json()['id']

def query_document(doc_id: str, question: str) -> Dict:
    """Query document and return response"""
    response = requests.post(
        f"{API_URL}/query",
        json={'doc_id': doc_id, 'question': question}
    )
    response.raise_for_status()
    return response.json()

def calculate_accuracy(answer: str, expected: str) -> float:
    """Simple accuracy calculation based on keyword presence"""
    expected_words = expected.lower().split()
    answer_lower = answer.lower()
    matches = sum(1 for word in expected_words if word in answer_lower)
    return matches / len(expected_words) if expected_words else 0.0

def run_regression_tests(baseline_file: str = None) -> Dict:
    """Run all regression tests"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_cases': [],
        'summary': {}
    }
    
    total_accuracy = 0
    total_tests = 0
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"\n📄 Test Case {i+1}")
        
        # Create and upload test document
        pdf_path = create_test_pdf(test_case['pdf_content'], f"test_{i+1}.txt")
        print(f"   Uploading {pdf_path.name}...", end=" ")
        doc_id = upload_document(pdf_path)
        print(f"✅ (ID: {doc_id})")
        
        case_results = {
            'doc_id': doc_id,
            'questions': []
        }
        
        # Test each question
        for question, expected in test_case['questions']:
            print(f"   Q: {question}")
            
            start_time = time.time()
            response = query_document(doc_id, question)
            elapsed = time.time() - start_time
            
            answer = response.get('answer', '')
            accuracy = calculate_accuracy(answer, expected)
            
            print(f"   A: {answer[:100]}...")
            print(f"   Accuracy: {accuracy:.1%}, Time: {elapsed:.2f}s")
            
            case_results['questions'].append({
                'question': question,
                'expected': expected,
                'answer': answer,
                'accuracy': accuracy,
                'time_ms': int(elapsed * 1000),
                'citations': len(response.get('citations', []))
            })
            
            total_accuracy += accuracy
            total_tests += 1
        
        results['test_cases'].append(case_results)
    
    # Calculate summary
    results['summary'] = {
        'total_tests': total_tests,
        'average_accuracy': total_accuracy / total_tests if total_tests else 0,
        'passed': total_accuracy / total_tests >= 0.9 if total_tests else False
    }
    
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY")
    print(f"   Total Tests: {total_tests}")
    print(f"   Average Accuracy: {results['summary']['average_accuracy']:.1%}")
    print(f"   Status: {'✅ PASS' if results['summary']['passed'] else '❌ FAIL'}")
    
    # Compare to baseline if provided
    if baseline_file and Path(baseline_file).exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        baseline_acc = baseline.get('summary', {}).get('average_accuracy', 0)
        current_acc = results['summary']['average_accuracy']
        
        print(f"\n📈 Regression Analysis:")
        print(f"   Baseline: {baseline_acc:.1%}")
        print(f"   Current:  {current_acc:.1%}")
        print(f"   Change:   {(current_acc - baseline_acc):.1%}")
        
        if current_acc < baseline_acc - 0.05:
            print("   ⚠️  WARNING: Significant regression detected!")
            results['summary']['regression_detected'] = True
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run regression tests for Doc-RAG')
    parser.add_argument('--baseline', help='Baseline results file for comparison')
    parser.add_argument('--output', default='results.json', help='Output file for results')
    args = parser.parse_args()
    
    print("🧪 Doc-RAG Regression Testing")
    print("="*50)
    
    try:
        # Check API health
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print("✅ API is healthy\n")
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return 1
    
    # Run tests
    results = run_regression_tests(args.baseline)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")
    
    return 0 if results['summary']['passed'] else 1

if __name__ == '__main__':
    exit(main())