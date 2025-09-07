// API Mock Service - Fast responses for testing
const express = require('express');
const cors = require('cors');
const multer = require('multer');

const app = express();
const upload = multer({ dest: '/tmp/uploads/' });
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Mock data storage
const documents = new Map();
let requestCount = 0;

// Request logging middleware
app.use((req, res, next) => {
  requestCount++;
  console.log(`[API Mock] ${req.method} ${req.path} - Request #${requestCount}`);
  next();
});

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'api-mock',
    version: '1.0.0-test',
    uptime: process.uptime(),
    requests: requestCount
  });
});

// Upload document - Always succeeds quickly
app.post('/upload', upload.single('file'), (req, res) => {
  const docId = `mock_doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  const document = {
    id: docId,
    name: req.body.name || req.file?.originalname || 'mock-document.pdf',
    size: req.file?.size || Math.floor(Math.random() * 1000000) + 10000,
    uploadTime: new Date().toISOString(),
    chunks: Math.floor(Math.random() * 50) + 10,
    embedded: true
  };
  
  documents.set(docId, document);
  
  // Simulate processing delay (0-100ms)
  setTimeout(() => {
    res.json({
      id: docId,
      status: 'processed',
      message: 'Mock document uploaded successfully',
      metadata: {
        chunks_created: document.chunks,
        processing_time_ms: Math.floor(Math.random() * 100),
        embedding_model: 'mock-embedder-v1'
      }
    });
  }, Math.random() * 100);
});

// Query document - Fast mock responses
app.post('/query', (req, res) => {
  const { doc_id, question } = req.body;
  
  if (!doc_id || !question) {
    return res.status(400).json({
      error: 'Missing required fields: doc_id and question'
    });
  }
  
  const document = documents.get(doc_id) || {
    id: doc_id,
    name: 'mock-document.pdf',
    size: 50000
  };
  
  // Generate realistic mock response
  const confidence = 0.75 + Math.random() * 0.2;
  const processingTime = Math.floor(Math.random() * 200) + 50;
  
  // Simulate processing delay
  setTimeout(() => {
    res.json({
      answer: `Mock response for "${question}"\n\nBased on analysis of ${document.name}, here are the key findings:\n- Neural chunking achieved 95.4% accuracy\n- FACT cache hit rate: 89%\n- Processing completed in ${processingTime}ms\n- Confidence score: ${confidence.toFixed(3)}`,
      citations: [
        {
          source: document.name,
          page: Math.floor(Math.random() * 10) + 1,
          relevance: confidence,
          text: `Relevant excerpt from ${document.name} supporting the answer...`
        },
        {
          source: document.name,
          page: Math.floor(Math.random() * 10) + 1,
          relevance: confidence - 0.1,
          text: `Additional context from ${document.name}...`
        }
      ],
      confidence: confidence,
      doc_id: doc_id,
      question: question,
      processing_time_ms: processingTime,
      metadata: {
        chunks_analyzed: Math.floor(Math.random() * 20) + 5,
        cache_hits: Math.floor(Math.random() * 10),
        model_version: 'mock-v1.0'
      }
    });
  }, Math.random() * 50);
});

// List documents
app.get('/documents', (req, res) => {
  const docs = Array.from(documents.values()).map(doc => ({
    id: doc.id,
    name: doc.name,
    size: doc.size,
    uploadTime: doc.uploadTime,
    chunks: doc.chunks,
    status: 'processed'
  }));
  
  res.json({
    documents: docs,
    count: docs.length,
    total_size: docs.reduce((sum, doc) => sum + doc.size, 0)
  });
});

// Metrics endpoint
app.get('/metrics', (req, res) => {
  res.json({
    requests_total: requestCount,
    documents_stored: documents.size,
    uptime_seconds: Math.floor(process.uptime()),
    memory_usage: process.memoryUsage(),
    response_time_avg_ms: 75 + Math.random() * 50
  });
});

// Error simulation endpoint
app.post('/simulate-error', (req, res) => {
  const { error_type, delay } = req.body;
  
  setTimeout(() => {
    switch (error_type) {
      case 'timeout':
        // Don't respond (simulate timeout)
        break;
      case 'server_error':
        res.status(500).json({ error: 'Mock server error for testing' });
        break;
      case 'bad_request':
        res.status(400).json({ error: 'Mock bad request error' });
        break;
      default:
        res.json({ message: 'Error simulation complete' });
    }
  }, delay || 0);
});

app.listen(PORT, () => {
  console.log(`🔧 API Mock Service running on port ${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/health`);
  console.log(`   Metrics: http://localhost:${PORT}/metrics`);
});