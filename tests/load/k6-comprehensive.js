// K6 Comprehensive Load Testing Suite for Doc-RAG API
// Performance testing with realistic user scenarios

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomIntBetween, randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('doc_rag_errors');
const queryLatency = new Trend('doc_rag_query_latency');
const uploadLatency = new Trend('doc_rag_upload_latency');
const embeddingLatency = new Trend('doc_rag_embedding_latency');
const documentCounter = new Counter('doc_rag_documents_processed');
const queryCounter = new Counter('doc_rag_queries_processed');

// Test configuration
export const options = {
  scenarios: {
    // Baseline load test
    baseline_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },   // Ramp up to 10 users
        { duration: '5m', target: 10 },   // Stay at 10 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
    },
    
    // Spike test for peak loads
    spike_test: {
      executor: 'ramping-vus',
      startTime: '10m',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 50 },   // Quick ramp to 50 users
        { duration: '2m', target: 50 },   // Maintain spike
        { duration: '1m', target: 0 },    // Quick ramp down
      ],
    },
    
    // Stress test to find breaking point
    stress_test: {
      executor: 'ramping-vus',
      startTime: '15m',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },
        { duration: '2m', target: 40 },
        { duration: '2m', target: 60 },
        { duration: '2m', target: 80 },
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },  // Stress period
        { duration: '2m', target: 0 },
      ],
    },
    
    // Soak test for memory leaks
    soak_test: {
      executor: 'constant-vus',
      vus: 20,
      duration: '30m',
      startTime: '25m',
    }
  },
  
  thresholds: {
    // HTTP request duration should be below 1s for 95% of requests
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
    
    // Error rate should be below 1%
    http_req_failed: ['rate<0.01'],
    
    // Custom metrics thresholds
    doc_rag_errors: ['rate<0.01'],
    doc_rag_query_latency: ['p(95)<500', 'p(99)<1000'],
    doc_rag_upload_latency: ['p(95)<2000', 'p(99)<5000'],
    doc_rag_embedding_latency: ['p(95)<3000', 'p(99)<10000'],
  },
  
  // Don't save responses to reduce memory usage
  discardResponseBodies: true,
};

// Test data
const sampleDocuments = [
  {
    title: "Introduction to Machine Learning",
    content: "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It involves training models on datasets to make predictions or decisions without being explicitly programmed for every scenario.",
    type: "educational",
    category: "AI/ML"
  },
  {
    title: "Database Design Principles", 
    content: "Effective database design follows normalization principles to reduce redundancy and improve data integrity. Key considerations include choosing appropriate data types, defining relationships, and optimizing for query performance.",
    type: "technical",
    category: "Database"
  },
  {
    title: "Cloud Architecture Patterns",
    content: "Modern cloud applications employ various architectural patterns such as microservices, event-driven architecture, and serverless computing. These patterns enable scalability, resilience, and maintainability.",
    type: "technical", 
    category: "Cloud"
  },
  {
    title: "Project Management Methodologies",
    content: "Agile and waterfall are two primary project management approaches. Agile emphasizes iterative development and customer collaboration, while waterfall follows a sequential approach with defined phases.",
    type: "business",
    category: "Management"
  },
  {
    title: "Cybersecurity Best Practices",
    content: "Organizations must implement multi-layered security measures including access controls, encryption, regular security audits, and employee training to protect against cyber threats and data breaches.",
    type: "security",
    category: "Security"
  }
];

const sampleQueries = [
  "What is machine learning?",
  "How to design a database?", 
  "Explain cloud architecture patterns",
  "What are the benefits of agile methodology?",
  "How to improve cybersecurity?",
  "Compare different AI approaches",
  "What is data normalization?",
  "Explain microservices architecture",
  "How to manage remote teams?",
  "What are common security vulnerabilities?"
];

// Configuration
const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8080';
const API_VERSION = '/api/v1';

// Helper functions
function getRandomDocument() {
  return randomItem(sampleDocuments);
}

function getRandomQuery() {
  return randomItem(sampleQueries);
}

function checkResponse(response, expectedStatus, operationType) {
  const success = check(response, {
    [`${operationType} status is ${expectedStatus}`]: (r) => r.status === expectedStatus,
    [`${operationType} response time < 30s`]: (r) => r.timings.duration < 30000,
  });
  
  if (!success) {
    errorRate.add(1);
  }
  
  return success;
}

// Main test scenarios
export default function() {
  // Randomly choose a user behavior pattern
  const userType = randomIntBetween(1, 100);
  
  if (userType <= 60) {
    // 60% - Query-heavy users (readers)
    queryUser();
  } else if (userType <= 85) {
    // 25% - Mixed users (readers and contributors) 
    mixedUser();
  } else {
    // 15% - Upload-heavy users (content creators)
    uploadUser();
  }
}

function queryUser() {
  group('Query User Workflow', () => {
    // Health check
    group('Health Check', () => {
      const healthResponse = http.get(`${BASE_URL}/health`);
      checkResponse(healthResponse, 200, 'health check');
    });
    
    sleep(randomIntBetween(1, 3));
    
    // Multiple queries to simulate searching behavior
    for (let i = 0; i < randomIntBetween(3, 8); i++) {
      group('Search Query', () => {
        const query = getRandomQuery();
        const queryStart = Date.now();
        
        const queryResponse = http.post(
          `${BASE_URL}${API_VERSION}/query`,
          JSON.stringify({
            query: query,
            limit: randomIntBetween(5, 20),
            include_metadata: true,
            similarity_threshold: 0.7
          }),
          {
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
          }
        );
        
        const queryDuration = Date.now() - queryStart;
        queryLatency.add(queryDuration);
        queryCounter.add(1);
        
        if (checkResponse(queryResponse, 200, 'query')) {
          // Validate response structure
          check(queryResponse, {
            'query has results': (r) => {
              try {
                const body = JSON.parse(r.body);
                return body.results && Array.isArray(body.results);
              } catch (e) {
                return false;
              }
            },
            'query has metadata': (r) => {
              try {
                const body = JSON.parse(r.body);
                return body.metadata && typeof body.metadata === 'object';
              } catch (e) {
                return false;
              }
            },
          });
        }
      });
      
      sleep(randomIntBetween(2, 5));
    }
  });
}

function uploadUser() {
  group('Upload User Workflow', () => {
    // Health check
    const healthResponse = http.get(`${BASE_URL}/health`);
    checkResponse(healthResponse, 200, 'health check');
    
    sleep(randomIntBetween(1, 2));
    
    // Upload multiple documents
    for (let i = 0; i < randomIntBetween(2, 5); i++) {
      group('Document Upload', () => {
        const doc = getRandomDocument();
        const uploadStart = Date.now();
        
        const uploadResponse = http.post(
          `${BASE_URL}${API_VERSION}/documents`,
          JSON.stringify({
            title: doc.title,
            content: doc.content,
            metadata: {
              type: doc.type,
              category: doc.category,
              source: 'load_test',
              uploaded_at: new Date().toISOString()
            }
          }),
          {
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
          }
        );
        
        const uploadDuration = Date.now() - uploadStart;
        uploadLatency.add(uploadDuration);
        documentCounter.add(1);
        
        if (checkResponse(uploadResponse, 201, 'upload')) {
          // Validate upload response
          check(uploadResponse, {
            'upload has document_id': (r) => {
              try {
                const body = JSON.parse(r.body);
                return body.document_id && typeof body.document_id === 'string';
              } catch (e) {
                return false;
              }
            },
            'upload has processing_status': (r) => {
              try {
                const body = JSON.parse(r.body);
                return body.status && ['processing', 'completed'].includes(body.status);
              } catch (e) {
                return false;
              }
            },
          });
          
          // Wait for document processing
          sleep(randomIntBetween(5, 10));
          
          // Check processing status
          group('Check Processing Status', () => {
            try {
              const body = JSON.parse(uploadResponse.body);
              if (body.document_id) {
                const statusResponse = http.get(`${BASE_URL}${API_VERSION}/documents/${body.document_id}/status`);
                checkResponse(statusResponse, 200, 'status check');
              }
            } catch (e) {
              errorRate.add(1);
            }
          });
        }
      });
      
      sleep(randomIntBetween(3, 8));
    }
  });
}

function mixedUser() {
  group('Mixed User Workflow', () => {
    // Health check
    const healthResponse = http.get(`${BASE_URL}/health`);
    checkResponse(healthResponse, 200, 'health check');
    
    sleep(randomIntBetween(1, 3));
    
    // Upload a document
    group('Document Upload', () => {
      const doc = getRandomDocument();
      const uploadStart = Date.now();
      
      const uploadResponse = http.post(
        `${BASE_URL}${API_VERSION}/documents`,
        JSON.stringify({
          title: doc.title,
          content: doc.content,
          metadata: {
            type: doc.type,
            category: doc.category,
            source: 'mixed_user_test'
          }
        }),
        {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
        }
      );
      
      const uploadDuration = Date.now() - uploadStart;
      uploadLatency.add(uploadDuration);
      documentCounter.add(1);
      
      checkResponse(uploadResponse, 201, 'upload');
    });
    
    sleep(randomIntBetween(5, 10));
    
    // Perform queries
    for (let i = 0; i < randomIntBetween(2, 5); i++) {
      group('Search Query', () => {
        const query = getRandomQuery();
        const queryStart = Date.now();
        
        const queryResponse = http.post(
          `${BASE_URL}${API_VERSION}/query`,
          JSON.stringify({
            query: query,
            limit: randomIntBetween(3, 10),
            include_metadata: false
          }),
          {
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
          }
        );
        
        const queryDuration = Date.now() - queryStart;
        queryLatency.add(queryDuration);
        queryCounter.add(1);
        
        checkResponse(queryResponse, 200, 'query');
      });
      
      sleep(randomIntBetween(2, 6));
    }
    
    // Check system metrics
    group('Metrics Check', () => {
      const metricsResponse = http.get(`${BASE_URL}/metrics`);
      check(metricsResponse, {
        'metrics endpoint accessible': (r) => r.status === 200,
        'metrics in prometheus format': (r) => r.body.includes('# HELP'),
      });
    });
  });
}

// Setup function - runs once at the start
export function setup() {
  console.log('🚀 Starting Doc-RAG comprehensive load test');
  console.log(`📊 Testing against: ${BASE_URL}`);
  
  // Verify API is accessible
  const healthResponse = http.get(`${BASE_URL}/health`);
  
  if (healthResponse.status !== 200) {
    console.error(`❌ API health check failed: ${healthResponse.status}`);
    console.error(`Response: ${healthResponse.body}`);
    throw new Error('API is not accessible - aborting test');
  }
  
  console.log('✅ API health check passed');
  return { baseUrl: BASE_URL };
}

// Teardown function - runs once at the end
export function teardown(data) {
  console.log('🏁 Load test completed');
  console.log(`📈 Total documents processed: ${documentCounter.count}`);
  console.log(`🔍 Total queries processed: ${queryCounter.count}`);
}

// Handle summary data
export function handleSummary(data) {
  return {
    'load-test-summary.json': JSON.stringify(data, null, 2),
    'stdout': `
    
📊 LOAD TEST RESULTS SUMMARY
============================

Duration: ${data.state.testRunDurationMs / 1000}s
VU Max: ${data.metrics.vus_max.values.max}

HTTP Metrics:
- Total Requests: ${data.metrics.http_reqs.values.count}
- Failed Requests: ${data.metrics.http_req_failed.values.fails} (${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%)
- Avg Response Time: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- 95th Percentile: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- 99th Percentile: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

Custom Metrics:
- Documents Processed: ${data.metrics.doc_rag_documents_processed ? data.metrics.doc_rag_documents_processed.values.count : 'N/A'}
- Queries Processed: ${data.metrics.doc_rag_queries_processed ? data.metrics.doc_rag_queries_processed.values.count : 'N/A'}
- Error Rate: ${data.metrics.doc_rag_errors ? (data.metrics.doc_rag_errors.values.rate * 100).toFixed(4) : 'N/A'}%

Thresholds:
${Object.entries(data.thresholds)
  .map(([name, threshold]) => `- ${name}: ${threshold.ok ? '✅ PASS' : '❌ FAIL'}`)
  .join('\n')}

`,
  };
}