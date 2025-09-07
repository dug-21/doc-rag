#!/usr/bin/env python3
"""
Live PDF Q&A Testing Interface
Interactive testing for the Doc-RAG system with real PDF documents
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime

class PDFQATest:
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.document_id = None
        self.session_id = f"test-session-{int(time.time())}"
        
    def check_health(self) -> bool:
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def upload_pdf(self, pdf_path: str) -> Optional[str]:
        """Upload a PDF document"""
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            return None
            
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                metadata = {
                    'source': 'test',
                    'uploaded_at': datetime.now().isoformat(),
                    'session_id': self.session_id
                }
                data = {'metadata': json.dumps(metadata)}
                
                response = requests.post(
                    f"{self.api_url}/api/v1/documents",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    doc_id = result.get('document_id', f'mock-{int(time.time())}')
                    print(f"✅ PDF uploaded successfully: {doc_id}")
                    return doc_id
                else:
                    print(f"⚠️  Upload returned {response.status_code}")
                    # Use mock ID for testing
                    doc_id = f"mock-doc-{int(time.time())}"
                    print(f"📝 Using mock document ID: {doc_id}")
                    return doc_id
                    
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None
    
    def query(self, question: str) -> Dict[str, Any]:
        """Send a query to the API"""
        if not self.document_id:
            return {"error": "No document uploaded"}
            
        try:
            payload = {
                "query_id": f"q-{int(time.time() * 1000)}",
                "query": question,
                "document_ids": [self.document_id],
                "max_results": 5,
                "enable_citations": True
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/api/v1/queries",
                json=payload,
                timeout=10
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['response_time'] = f"{elapsed_time:.2f}s"
                return result
            else:
                return {
                    "error": f"Query failed with status {response.status_code}",
                    "response_time": f"{elapsed_time:.2f}s"
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def interactive_session(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("🚀 Doc-RAG Live PDF Q&A Testing")
        print("="*60)
        
        # Check API health
        print("\n📡 Checking API connection...")
        if not self.check_health():
            print("❌ API is not running. Please start the API server first.")
            print("   Run: cargo run --release --bin api")
            return
        print("✅ API is running")
        
        # Upload PDF
        print("\n📄 Select PDF to test:")
        print("1. uploads/thor_resume.pdf (existing)")
        print("2. Enter custom path")
        print("3. Skip upload (use mock data)")
        
        choice = input("\nChoice [1-3]: ").strip()
        
        if choice == "1":
            pdf_path = "uploads/thor_resume.pdf"
        elif choice == "2":
            pdf_path = input("Enter PDF path: ").strip()
        else:
            self.document_id = f"mock-doc-{int(time.time())}"
            print(f"📝 Using mock document: {self.document_id}")
            pdf_path = None
        
        if pdf_path:
            print(f"\n📤 Uploading {pdf_path}...")
            self.document_id = self.upload_pdf(pdf_path)
            if not self.document_id:
                print("❌ Failed to upload PDF")
                return
        
        # Interactive Q&A
        print("\n" + "="*60)
        print("💬 Interactive Q&A Mode")
        print("="*60)
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - Type 'demo' for demo questions")
        print("  - Type 'quit' to exit")
        print("-"*60)
        
        demo_questions = [
            "What is the main experience of the candidate?",
            "What programming languages are mentioned?",
            "What are the key achievements?",
            "What is the educational background?",
            "List the technical skills",
        ]
        
        while True:
            try:
                question = input("\n❓ Question: ").strip()
                
                if question.lower() == 'quit':
                    break
                elif question.lower() == 'demo':
                    print("\n📋 Demo Questions:")
                    for i, q in enumerate(demo_questions, 1):
                        print(f"   {i}. {q}")
                    continue
                elif question.isdigit() and 1 <= int(question) <= len(demo_questions):
                    question = demo_questions[int(question) - 1]
                    print(f"   Selected: {question}")
                elif not question:
                    continue
                
                print("\n⏳ Processing...")
                result = self.query(question)
                
                print("\n" + "-"*40)
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    answer = result.get('answer', result.get('response', 'No answer generated'))
                    print(f"✅ Answer: {answer}")
                    
                    if 'citations' in result:
                        print(f"\n📚 Citations:")
                        for cite in result['citations']:
                            print(f"   - {cite}")
                    
                    if 'confidence' in result:
                        print(f"\n📊 Confidence: {result['confidence']:.2%}")
                    
                    if 'response_time' in result:
                        print(f"⏱️  Response Time: {result['response_time']}")
                print("-"*40)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Session ended. Thank you for testing!")

def main():
    # Check if custom API URL provided
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    tester = PDFQATest(api_url)
    tester.interactive_session()

if __name__ == "__main__":
    main()