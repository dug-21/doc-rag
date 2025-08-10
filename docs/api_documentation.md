# Doc-RAG API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Error Handling](#error-handling)
5. [API Endpoints](#api-endpoints)
6. [Data Models](#data-models)
7. [Examples](#examples)
8. [SDKs and Client Libraries](#sdks-and-client-libraries)

## Overview

The Doc-RAG API provides programmatic access to document processing, vector search, and AI-powered query capabilities. The API follows RESTful principles and returns JSON responses.

**Base URL**: `https://api.docrag.com/v1`

**API Version**: 1.0.0

**Content-Type**: `application/json`

## Authentication

### JWT Bearer Token Authentication

All API requests require authentication using JWT (JSON Web Token) bearer tokens.

```bash
# Obtain access token
curl -X POST https://api.docrag.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user": {
    "id": "user123",
    "email": "user@example.com",
    "roles": ["user"]
  }
}
```

**Using the token:**
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  https://api.docrag.com/v1/documents
```

### API Key Authentication (Alternative)

For server-to-server communication, you can use API keys:

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://api.docrag.com/v1/documents
```

## Rate Limiting

API requests are rate-limited to ensure fair usage and system stability.

**Default Limits:**
- **Standard Users**: 100 requests per minute
- **Premium Users**: 1000 requests per minute
- **Enterprise Users**: Custom limits

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1691234567
```

**Rate Limit Exceeded Response:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 100 requests per minute exceeded",
    "details": {
      "limit": 100,
      "reset_time": "2025-08-10T12:34:56Z"
    }
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information in JSON format.

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation errors
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "query",
      "reason": "Query length must be between 1 and 1000 characters"
    },
    "request_id": "req_123456789",
    "timestamp": "2025-08-10T12:34:56Z"
  }
}
```

## API Endpoints

### Authentication Endpoints

#### POST /auth/login
Authenticate user and obtain access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user": {
    "id": "user123",
    "email": "user@example.com",
    "roles": ["user"]
  }
}
```

#### POST /auth/refresh
Refresh an existing access token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### POST /auth/logout
Revoke access token and logout.

**Request Headers:**
```
Authorization: Bearer YOUR_JWT_TOKEN
```

### Document Management

#### GET /documents
List all documents with pagination and filtering.

**Query Parameters:**
- `page` (integer, default: 1) - Page number
- `limit` (integer, default: 20, max: 100) - Items per page
- `search` (string) - Search in document titles and content
- `tags` (string) - Filter by comma-separated tags
- `created_after` (ISO 8601 date) - Filter by creation date
- `created_before` (ISO 8601 date) - Filter by creation date

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  "https://api.docrag.com/v1/documents?page=1&limit=10&search=machine%20learning"
```

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_123",
      "title": "Introduction to Machine Learning",
      "filename": "ml_intro.pdf",
      "content_type": "application/pdf",
      "size": 2048000,
      "created_at": "2025-08-10T10:30:00Z",
      "updated_at": "2025-08-10T10:35:00Z",
      "tags": ["machine-learning", "ai", "tutorial"],
      "status": "processed",
      "chunk_count": 15,
      "metadata": {
        "author": "Dr. Jane Smith",
        "pages": 25
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total_pages": 5,
    "total_items": 47
  }
}
```

#### POST /documents
Upload and process a new document.

**Request (multipart/form-data):**
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@document.pdf" \
  -F "title=Machine Learning Guide" \
  -F "tags=ml,ai,tutorial" \
  -F "description=Comprehensive guide to ML fundamentals" \
  https://api.docrag.com/v1/documents
```

**Response:**
```json
{
  "id": "doc_124",
  "title": "Machine Learning Guide",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "size": 3145728,
  "status": "processing",
  "created_at": "2025-08-10T11:00:00Z",
  "upload_url": "https://api.docrag.com/v1/documents/doc_124/upload",
  "processing_time_estimate": 120
}
```

#### GET /documents/{id}
Get detailed information about a specific document.

**Response:**
```json
{
  "id": "doc_123",
  "title": "Introduction to Machine Learning",
  "filename": "ml_intro.pdf",
  "content_type": "application/pdf",
  "size": 2048000,
  "created_at": "2025-08-10T10:30:00Z",
  "updated_at": "2025-08-10T10:35:00Z",
  "tags": ["machine-learning", "ai", "tutorial"],
  "status": "processed",
  "chunk_count": 15,
  "processing_log": [
    {
      "stage": "uploaded",
      "timestamp": "2025-08-10T10:30:00Z",
      "status": "completed"
    },
    {
      "stage": "chunked",
      "timestamp": "2025-08-10T10:32:00Z", 
      "status": "completed",
      "chunks_created": 15
    },
    {
      "stage": "embedded",
      "timestamp": "2025-08-10T10:35:00Z",
      "status": "completed"
    }
  ],
  "metadata": {
    "author": "Dr. Jane Smith",
    "pages": 25,
    "language": "en",
    "word_count": 5420
  }
}
```

#### PUT /documents/{id}
Update document metadata.

**Request Body:**
```json
{
  "title": "Updated Document Title",
  "tags": ["updated", "machine-learning"],
  "description": "Updated description"
}
```

#### DELETE /documents/{id}
Delete a document and all associated data.

**Response:**
```json
{
  "message": "Document deleted successfully",
  "deleted_at": "2025-08-10T12:00:00Z"
}
```

### Query and Search

#### POST /query
Perform semantic search and question answering.

**Request Body:**
```json
{
  "query": "What are the main types of machine learning algorithms?",
  "limit": 5,
  "filters": {
    "tags": ["machine-learning"],
    "document_ids": ["doc_123", "doc_124"]
  },
  "options": {
    "include_citations": true,
    "confidence_threshold": 0.7,
    "max_context_length": 2000
  }
}
```

**Response:**
```json
{
  "query": "What are the main types of machine learning algorithms?",
  "response": "The main types of machine learning algorithms are supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models for prediction tasks...",
  "confidence": 0.92,
  "response_time_ms": 245,
  "sources": [
    {
      "document_id": "doc_123",
      "title": "Introduction to Machine Learning",
      "chunk_id": "chunk_456",
      "content": "Machine learning algorithms can be categorized into three main types...",
      "relevance_score": 0.95,
      "page_number": 5
    }
  ],
  "citations": [
    "[1] Introduction to Machine Learning, Page 5",
    "[2] ML Fundamentals Guide, Chapter 2"
  ]
}
```

#### POST /search/vector
Perform vector similarity search.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3, ..., 0.384],
  "limit": 10,
  "filters": {
    "document_ids": ["doc_123"],
    "tags": ["technical"]
  },
  "similarity_threshold": 0.8
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_123",
      "chunk_id": "chunk_789",
      "content": "Neural networks are computational models...",
      "similarity_score": 0.94,
      "metadata": {
        "page_number": 12,
        "section": "Deep Learning Basics"
      }
    }
  ],
  "query_time_ms": 45
}
```

#### POST /search/hybrid
Perform hybrid search combining text and vector search.

**Request Body:**
```json
{
  "query": "neural network architecture",
  "vector": [0.1, 0.2, 0.3, ..., 0.384],
  "weights": {
    "text": 0.3,
    "vector": 0.7
  },
  "limit": 10
}
```

### Analytics and Metrics

#### GET /analytics/queries
Get query analytics and performance metrics.

**Query Parameters:**
- `start_date` (ISO 8601 date) - Start date for analytics
- `end_date` (ISO 8601 date) - End date for analytics
- `granularity` (string) - hour, day, week, month

**Response:**
```json
{
  "period": {
    "start_date": "2025-08-01T00:00:00Z",
    "end_date": "2025-08-10T23:59:59Z"
  },
  "metrics": {
    "total_queries": 1523,
    "avg_response_time_ms": 189,
    "avg_confidence_score": 0.84,
    "success_rate": 0.97
  },
  "top_queries": [
    {
      "query": "machine learning algorithms",
      "count": 45,
      "avg_confidence": 0.89
    }
  ],
  "time_series": [
    {
      "timestamp": "2025-08-10T00:00:00Z",
      "queries": 156,
      "avg_response_time_ms": 201
    }
  ]
}
```

#### GET /analytics/documents
Get document analytics.

**Response:**
```json
{
  "metrics": {
    "total_documents": 247,
    "total_chunks": 3891,
    "avg_chunks_per_document": 15.7,
    "total_size_bytes": 524288000,
    "processing_success_rate": 0.99
  },
  "document_types": {
    "application/pdf": 156,
    "text/plain": 67,
    "application/docx": 24
  },
  "popular_tags": [
    {
      "tag": "machine-learning",
      "count": 89
    },
    {
      "tag": "ai",
      "count": 67
    }
  ]
}
```

### System Health

#### GET /health
System health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-10T12:34:56Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "services": {
    "database": "healthy",
    "vector_db": "healthy",
    "cache": "healthy",
    "search": "healthy"
  }
}
```

#### GET /metrics
Prometheus-compatible metrics endpoint.

**Response:** (Prometheus format)
```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 1523
http_requests_total{method="POST",status="200"} 891

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 100
http_request_duration_seconds_bucket{le="0.5"} 450
```

## Data Models

### Document
```json
{
  "id": "string",
  "title": "string",
  "filename": "string",
  "content_type": "string",
  "size": "integer",
  "created_at": "string (ISO 8601)",
  "updated_at": "string (ISO 8601)",
  "tags": ["string"],
  "status": "enum (uploading, processing, processed, failed)",
  "chunk_count": "integer",
  "metadata": "object",
  "processing_log": [
    {
      "stage": "string",
      "timestamp": "string (ISO 8601)",
      "status": "string",
      "details": "object"
    }
  ]
}
```

### Query Response
```json
{
  "query": "string",
  "response": "string",
  "confidence": "number (0-1)",
  "response_time_ms": "integer",
  "sources": [
    {
      "document_id": "string",
      "title": "string",
      "chunk_id": "string",
      "content": "string",
      "relevance_score": "number (0-1)",
      "page_number": "integer"
    }
  ],
  "citations": ["string"]
}
```

### Error Response
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object",
    "request_id": "string",
    "timestamp": "string (ISO 8601)"
  }
}
```

## Examples

### Complete Document Upload and Query Flow

```bash
#!/bin/bash

API_BASE="https://api.docrag.com/v1"
EMAIL="user@example.com"
PASSWORD="your_password"

# 1. Login and get token
TOKEN_RESPONSE=$(curl -s -X POST "$API_BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")

TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')
echo "Obtained token: ${TOKEN:0:20}..."

# 2. Upload document
echo "Uploading document..."
UPLOAD_RESPONSE=$(curl -s -X POST "$API_BASE/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@ml_guide.pdf" \
  -F "title=Machine Learning Guide" \
  -F "tags=ml,ai,tutorial")

DOCUMENT_ID=$(echo $UPLOAD_RESPONSE | jq -r '.id')
echo "Document uploaded: $DOCUMENT_ID"

# 3. Wait for processing
echo "Waiting for document processing..."
while true; do
  STATUS_RESPONSE=$(curl -s "$API_BASE/documents/$DOCUMENT_ID" \
    -H "Authorization: Bearer $TOKEN")
  STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
  
  if [ "$STATUS" = "processed" ]; then
    echo "Document processing completed"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Document processing failed"
    exit 1
  else
    echo "Status: $STATUS"
    sleep 10
  fi
done

# 4. Query the document
echo "Querying document..."
QUERY_RESPONSE=$(curl -s -X POST "$API_BASE/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are supervised learning algorithms?",
    "limit": 3,
    "filters": {
      "document_ids": ["'$DOCUMENT_ID'"]
    },
    "options": {
      "include_citations": true
    }
  }')

echo "Query response:"
echo $QUERY_RESPONSE | jq '.response'

echo "Sources:"
echo $QUERY_RESPONSE | jq '.sources[0].content'
```

### Batch Document Processing

```python
import requests
import time
import os
from pathlib import Path

class DocRagClient:
    def __init__(self, api_base, email, password):
        self.api_base = api_base
        self.session = requests.Session()
        self.login(email, password)
    
    def login(self, email, password):
        response = self.session.post(f"{self.api_base}/auth/login", json={
            "email": email,
            "password": password
        })
        response.raise_for_status()
        token = response.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def upload_document(self, file_path, title=None, tags=None):
        files = {"file": open(file_path, "rb")}
        data = {}
        if title:
            data["title"] = title
        if tags:
            data["tags"] = ",".join(tags)
        
        response = self.session.post(f"{self.api_base}/documents", 
                                   files=files, data=data)
        response.raise_for_status()
        return response.json()["id"]
    
    def wait_for_processing(self, document_id, timeout=300):
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.api_base}/documents/{document_id}")
            response.raise_for_status()
            status = response.json()["status"]
            
            if status == "processed":
                return True
            elif status == "failed":
                return False
            
            time.sleep(10)
        
        return False
    
    def query(self, query, document_ids=None, limit=5):
        data = {"query": query, "limit": limit}
        if document_ids:
            data["filters"] = {"document_ids": document_ids}
        
        response = self.session.post(f"{self.api_base}/query", json=data)
        response.raise_for_status()
        return response.json()

# Usage example
client = DocRagClient("https://api.docrag.com/v1", 
                     "user@example.com", "your_password")

# Upload multiple documents
document_ids = []
pdf_directory = Path("./documents")

for pdf_file in pdf_directory.glob("*.pdf"):
    print(f"Uploading {pdf_file.name}...")
    doc_id = client.upload_document(pdf_file, 
                                  title=pdf_file.stem,
                                  tags=["batch-upload"])
    document_ids.append(doc_id)

# Wait for all documents to process
print("Waiting for documents to process...")
for doc_id in document_ids:
    if client.wait_for_processing(doc_id):
        print(f"Document {doc_id} processed successfully")
    else:
        print(f"Document {doc_id} processing failed")

# Query across all documents
response = client.query("What are the key concepts in machine learning?", 
                       document_ids=document_ids)
print(f"Response: {response['response']}")
```

## SDKs and Client Libraries

### Official SDKs

- **Python**: `pip install docrag-python`
- **Node.js**: `npm install docrag-js`
- **Go**: `go get github.com/docrag/docrag-go`
- **Java**: Maven/Gradle dependency available

### Community SDKs

- **Ruby**: `gem install docrag-ruby`
- **PHP**: `composer require docrag/docrag-php`
- **C#**: NuGet package available

### OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
`https://api.docrag.com/v1/openapi.json`

Generate client libraries using tools like:
- OpenAPI Generator
- Swagger Codegen
- Postman

---

**API Version**: 1.0.0  
**Last Updated**: 2025-08-10  
**Support**: api-support@docrag.com