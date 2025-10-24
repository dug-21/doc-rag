# SPARC Architecture Document - TypeScript RAG System

**Version:** 1.0
**Date:** October 24, 2025
**Phase:** 03-ARCHITECTURE (SPARC Methodology)
**Status:** DESIGN COMPLETE

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer (REST API)                  │
│                    Express.js / Fastify                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              API Gateway & Middleware                        │
│    - JWT Authentication                                      │
│    - Rate Limiting (express-rate-limit)                     │
│    - Request Validation (zod)                               │
│    - Logging (pino)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│         Application Layer (TypeScript)                       │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   Ingestion     │  │     Query       │                  │
│  │   Service       │  │   Processor     │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                             │
│  ┌────────▼────────────────────▼────────┐                  │
│  │  Orchestration Layer (agentic-flow)  │                  │
│  │  - Multi-agent coordination          │                  │
│  │  - ReasoningBank learning            │                  │
│  │  - QUIC protocol (<100ms latency)    │                  │
│  └────────┬─────────────────────────────┘                  │
└───────────┼──────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────┐
│              Data & Neural Layer                              │
│  ┌──────────────────┐  ┌───────────────────────────────┐    │
│  │   AgentDB        │  │  ruv-FANN WASM (Neural)       │    │
│  │  - HNSW Index    │  │  - Intent Classification      │    │
│  │  - Quantization  │  │  - Relevance Scoring          │    │
│  │  - Memory Store  │  │  - Topic Modeling             │    │
│  │  - RL Plugins    │  │  - SIMD Optimized             │    │
│  └──────────────────┘  └───────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────┐
│              Infrastructure Layer                             │
│  - Redis Cache (session/query cache)                         │
│  - Prometheus Metrics                                         │
│  - Grafana Dashboards                                         │
│  - Docker Containers                                          │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Technology | Purpose | Performance Target |
|-----------|-----------|---------|-------------------|
| **API Layer** | Express.js | REST endpoints | <50ms overhead |
| **Ingestion** | pdf-parse + AgentDB | Document processing | <5s per PDF |
| **Query** | agentic-flow | Multi-agent orchestration | <600ms P95 |
| **Storage** | AgentDB | Vector + metadata | 150x faster HNSW |
| **Neural** | ruv-FANN WASM | Classification + scoring | <20ms inference |
| **Learning** | ReasoningBank | Adaptive optimization | Online training |
| **Monitoring** | Prometheus + Grafana | Observability | Real-time metrics |

### 1.3 Technology Stack

```typescript
// Core Dependencies
{
  "runtime": "Node.js 20.x",
  "language": "TypeScript 5.x",
  "framework": "Express.js 4.x",
  "orchestration": "agentic-flow@1.6.0",
  "vectorDB": "agentdb@latest",
  "neural": "ruv-swarm-wasm@latest",
  "cache": "redis@4.x",
  "logging": "pino@8.x",
  "validation": "zod@3.x",
  "testing": "vitest@1.x"
}
```

---

## 2. Component Architecture

### 2.1 API Layer (Express.js)

**Responsibilities:**
- RESTful API endpoints
- Request validation
- Authentication/authorization
- Rate limiting
- Error handling

**Interface Design:**

```typescript
// src/api/routes/query.routes.ts
import { Router } from 'express';
import { z } from 'zod';

const QueryRequestSchema = z.object({
  query: z.string().min(3).max(500),
  filters: z.record(z.string()).optional(),
  maxResults: z.number().min(1).max(50).default(10),
  includeMetadata: z.boolean().default(true)
});

const router = Router();

// POST /api/v1/query
router.post('/query',
  authenticate,
  rateLimit({ max: 100, windowMs: 60000 }),
  validate(QueryRequestSchema),
  async (req, res) => {
    const result = await queryProcessor.process(req.body);
    res.json(result);
  }
);

// GET /api/v1/documents/:id
router.get('/documents/:id',
  authenticate,
  async (req, res) => {
    const doc = await docRepository.findById(req.params.id);
    res.json(doc);
  }
);
```

**Endpoints:**
| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | `/api/v1/query` | Execute RAG query | JWT |
| POST | `/api/v1/ingest` | Upload document | JWT + Admin |
| GET | `/api/v1/documents` | List documents | JWT |
| GET | `/api/v1/documents/:id` | Get document | JWT |
| GET | `/api/v1/health` | Health check | None |
| GET | `/api/v1/metrics` | Prometheus metrics | None |

### 2.2 Ingestion Service

**Responsibilities:**
- PDF parsing and extraction
- Intelligent chunking
- Embedding generation
- AgentDB storage

**Architecture:**

```typescript
// src/ingestion/ingestion.service.ts
import { AgenticFlow } from 'agentic-flow';
import { AgentDB } from 'agentdb';
import { NeuralClassifier } from '../neural/classifier';
import pdfParse from 'pdf-parse';

export class IngestionService {
  private swarm: AgenticFlow;
  private db: AgentDB;
  private classifier: NeuralClassifier;

  async ingestPDF(buffer: Buffer): Promise<string> {
    // 1. Parse PDF
    const pdfData = await pdfParse(buffer);
    const text = pdfData.text;

    // 2. Classify document type
    const docType = await this.classifier.classifyDocument(text);

    // 3. Initialize ingestion swarm
    const swarmId = await this.swarm.init({
      topology: 'mesh',
      maxAgents: 4,
      strategy: 'parallel'
    });

    // 4. Spawn specialized agents (parallel)
    const [sections, chunks, embeddings] = await Promise.all([
      this.extractSections(text, docType),
      this.chunkDocument(text, { size: 512, overlap: 50 }),
      this.generateEmbeddings(text)
    ]);

    // 5. Store in AgentDB with HNSW indexing
    const docId = await this.db.insertBatch({
      collection: 'technical_standards',
      vectors: embeddings,
      metadata: chunks.map((chunk, i) => ({
        doc_id: docId,
        chunk_index: i,
        doc_type: docType,
        section: sections[i]?.title,
        page: chunk.page,
        text: chunk.text,
        timestamp: new Date()
      }))
    });

    // 6. Initialize learning session
    await this.db.createSession({
      sessionId: docId,
      sessionType: 'document_context',
      metadata: { doc_type: docType }
    });

    return docId;
  }

  private async chunkDocument(
    text: string,
    options: { size: number, overlap: number }
  ): Promise<Chunk[]> {
    // Intelligent chunking with semantic boundaries
    const chunks: Chunk[] = [];
    let start = 0;

    while (start < text.length) {
      const end = start + options.size;
      const chunkText = text.slice(start, end);

      // Find sentence boundary for clean split
      const boundaryIndex = this.findSentenceBoundary(chunkText);

      chunks.push({
        text: chunkText.slice(0, boundaryIndex),
        start,
        end: start + boundaryIndex,
        page: this.calculatePage(start, text)
      });

      start += boundaryIndex - options.overlap;
    }

    return chunks;
  }
}
```

**Pipeline Flow:**
```
PDF Upload → Parse → Classify → Chunk → Embed → Store → Index
    ↓          ↓        ↓         ↓       ↓       ↓       ↓
  100KB    1-2s     200ms     500ms    2s     1s     500ms

Total Latency: ~5 seconds per document
```

### 2.3 Query Processor (agentic-flow)

**Responsibilities:**
- Query intent classification
- Multi-agent coordination
- Retrieval strategy selection
- Response synthesis
- Verification

**Architecture:**

```typescript
// src/query/query.processor.ts
import { AgenticFlow } from 'agentic-flow';
import { AgentDB } from 'agentdb';
import { NeuralNet } from 'ruv-swarm-wasm';

export class QueryProcessor {
  private classifier: NeuralNet;
  private db: AgentDB;
  private swarm: AgenticFlow;

  async process(request: QueryRequest): Promise<QueryResponse> {
    // 1. Classify query intent (WASM neural network)
    const intent = await this.classifier.predict(
      this.extractQueryFeatures(request.query)
    );

    // 2. Select swarm topology based on complexity
    const topology = this.selectTopology(intent.complexity);

    // 3. Initialize query swarm
    await this.swarm.init({
      topology,
      maxAgents: 6,
      strategy: 'adaptive'
    });

    // 4. Spawn retrieval agents (parallel strategies)
    const retrievalTasks = [
      this.spawnRetrievalAgent('hnsw', request),
      this.spawnRetrievalAgent('hybrid', request)
    ];

    if (intent.complexity === 'complex') {
      retrievalTasks.push(
        this.spawnRetrievalAgent('rerank', request),
        this.spawnRetrievalAgent('graph_walk', request)
      );
    }

    const retrievalResults = await Promise.all(retrievalTasks);

    // 5. Merge and deduplicate results
    const mergedDocs = this.mergeResults(retrievalResults);

    // 6. Score relevance (WASM neural network)
    const scoredDocs = await this.scoreRelevance(
      request.query,
      mergedDocs
    );

    // 7. Spawn reasoning agent
    const reasoningResult = await this.swarm.spawn('reasoning', {
      query: request.query,
      documents: scoredDocs.slice(0, 10),
      patterns: await this.db.queryPatterns(request.query),
      use_memory: true
    });

    // 8. Spawn synthesis agent
    const synthesisResult = await this.swarm.spawn('synthesis', {
      evidence: scoredDocs,
      reasoning: reasoningResult,
      format: 'structured'
    });

    // 9. Spawn verification agent
    const verificationResult = await this.swarm.spawn('verification', {
      response: synthesisResult,
      query: request.query,
      threshold: 0.97
    });

    // 10. Record trajectory for ReasoningBank learning
    await this.recordTrajectory({
      query: request.query,
      strategy: topology,
      accuracy: verificationResult.accuracy,
      response: verificationResult.response
    });

    return verificationResult;
  }

  private selectTopology(complexity: string): SwarmTopology {
    const topologyMap = {
      simple: 'ring',      // Sequential processing
      moderate: 'mesh',    // Peer-to-peer
      complex: 'hierarchical' // Coordinator pattern
    };
    return topologyMap[complexity] || 'mesh';
  }

  private async spawnRetrievalAgent(
    strategy: RetrievalStrategy,
    request: QueryRequest
  ): Promise<Document[]> {
    return this.swarm.spawn('retrieval', {
      strategy,
      query: request.query,
      filters: request.filters,
      limit: request.maxResults,
      use_cache: true
    });
  }
}
```

**Query Flow:**
```
Query → Intent → Topology → Retrieval → Scoring → Reasoning → Synthesis → Verify
  ↓       ↓         ↓          ↓          ↓          ↓           ↓          ↓
Input   20ms     10ms      100ms       50ms      200ms       100ms      50ms

Total Latency (P95): <600ms
```

### 2.4 Storage Layer (AgentDB)

**Responsibilities:**
- Vector storage (HNSW indexing)
- Metadata filtering
- Session memory management
- Reinforcement learning plugins
- Pattern learning

**Schema Design:**

```typescript
// src/storage/agentdb.schema.ts
export interface AgentDBConfig {
  collection: 'technical_standards',
  vectorConfig: {
    size: 1536,  // ada-002 embedding dimension
    distance: 'cosine',
    hnsw: {
      m: 16,              // Connections per layer
      efConstruction: 200, // Build-time accuracy
      efSearch: 100       // Query-time accuracy
    },
    quantization: {
      enabled: true,
      method: 'scalar',
      compressionRatio: 4  // 4x memory reduction
    }
  },
  metadataSchema: {
    doc_id: 'uuid',
    chunk_index: 'integer',
    doc_type: 'keyword',
    section: 'text',
    page: 'integer',
    text: 'text',
    timestamp: 'datetime',
    verified: 'boolean'
  },
  sessionConfig: {
    enableMemory: true,
    memoryType: 'long_term',
    consolidationStrategy: 'importance_based',
    maxContextLength: 10000
  },
  learningConfig: {
    enabled: true,
    algorithms: [
      'decision_transformer',
      'actor_critic',
      'q_learning'
    ],
    trainingMode: 'online',
    checkpointInterval: 1000
  }
}
```

**Operations:**

```typescript
// src/storage/agentdb.repository.ts
export class AgentDBRepository {
  private client: AgentDB;

  async searchHNSW(
    query: string,
    options: SearchOptions
  ): Promise<Document[]> {
    const queryVector = await this.embedQuery(query);

    return this.client.search({
      collection: 'technical_standards',
      vector: queryVector,
      limit: options.k || 20,
      useHNSW: true,
      efSearch: 100,
      filter: options.filter
    });
  }

  async searchHybrid(
    query: string,
    options: SearchOptions
  ): Promise<Document[]> {
    const queryVector = await this.embedQuery(query);

    return this.client.search({
      collection: 'technical_standards',
      vector: queryVector,
      limit: options.k || 20,
      filter: {
        must: [
          { match: { chunk_type: 'requirement' } },
          { range: { confidence: { gte: 0.8 } } }
        ]
      }
    });
  }

  async recordTrajectory(trajectory: Trajectory): Promise<void> {
    await this.client.recordTrajectory({
      state: {
        query_type: trajectory.queryType,
        complexity: trajectory.complexity
      },
      action: {
        retrieval_strategy: trajectory.strategy,
        num_agents: trajectory.agentsUsed
      },
      reward: trajectory.accuracy,
      nextState: {
        success: trajectory.success
      }
    });

    // Trigger learning if batch size reached
    if (await this.shouldTrain('query_routing')) {
      await this.client.trainPlugin({
        pluginId: 'query_routing',
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001
      });
    }
  }
}
```

### 2.5 Neural Layer (ruv-FANN WASM)

**Responsibilities:**
- Intent classification
- Relevance scoring
- Topic modeling
- Semantic analysis

**Implementation:**

```typescript
// src/neural/classifier.ts
import { NeuralNet } from 'ruv-swarm-wasm';

export class NeuralClassifier {
  private intentClassifier: NeuralNet;
  private relevanceScorer: NeuralNet;

  async initialize(): Promise<void> {
    // Intent classification network
    this.intentClassifier = await NeuralNet.new({
      inputSize: 128,
      hiddenLayers: [64, 32],
      outputSize: 10,
      activation: 'sigmoid'
    });

    // Relevance scoring network
    this.relevanceScorer = await NeuralNet.new({
      inputSize: 256,
      hiddenLayers: [128, 64],
      outputSize: 1,
      activation: 'sigmoid'
    });
  }

  async classifyIntent(query: string): Promise<IntentResult> {
    const features = this.extractFeatures(query);
    const output = await this.intentClassifier.predict(features);

    return {
      intent: this.getIntentLabel(output.classId),
      confidence: output.confidence,
      complexity: this.calculateComplexity(output)
    };
  }

  async scoreRelevance(
    query: string,
    document: Document
  ): Promise<number> {
    const features = this.extractPairFeatures(query, document);
    const output = await this.relevanceScorer.predict(features);
    return output.probabilities[0];
  }

  private extractFeatures(text: string): number[] {
    // TF-IDF, length, keyword presence, etc.
    return Array(128).fill(0).map((_, i) => {
      // Feature extraction logic
      return Math.random(); // Placeholder
    });
  }
}
```

### 2.6 Learning Layer (ReasoningBank)

**Responsibilities:**
- Trajectory tracking
- Pattern recognition
- Strategy optimization
- Memory consolidation

**Implementation:**

```typescript
// src/learning/reasoning-bank.service.ts
export class ReasoningBankService {
  private db: AgentDB;

  async recordInteraction(interaction: Interaction): Promise<void> {
    // Store trajectory for learning
    await this.db.recordTrajectory({
      state: {
        query_type: interaction.queryType,
        complexity: interaction.complexity,
        has_context: interaction.hasContext
      },
      action: {
        retrieval_strategy: interaction.strategy,
        num_agents: interaction.agentsUsed,
        topology: interaction.topology
      },
      reward: interaction.accuracy,
      nextState: {
        success: interaction.success,
        user_satisfaction: interaction.feedback
      }
    });

    // Update learning plugins
    await this.updatePlugins(['query_routing', 'relevance_scoring']);
  }

  async learnPatterns(sessionId: string): Promise<Pattern[]> {
    const session = await this.db.getSession(sessionId);

    const patterns = await this.db.analyzePatterns(sessionId, {
      minFrequency: 3,
      minConfidence: 0.7,
      patternTypes: [
        'sequential_query',
        'cross_reference',
        'concept_cluster'
      ]
    });

    // Store learned patterns
    for (const pattern of patterns) {
      await this.db.storePattern({
        patternId: uuid(),
        patternType: pattern.type,
        confidence: pattern.confidence,
        metadata: pattern.metadata
      });
    }

    // Memory consolidation
    await this.db.consolidateMemory(sessionId, {
      strategy: 'importance_based',
      frequencyWeight: 0.3,
      recencyWeight: 0.2,
      relevanceWeight: 0.5
    });

    return patterns;
  }
}
```

### 2.7 Monitoring Layer (Prometheus + Grafana)

**Responsibilities:**
- Performance metrics collection
- Real-time dashboards
- Alerting
- Log aggregation

**Metrics:**

```typescript
// src/monitoring/metrics.service.ts
import { Registry, Counter, Histogram, Gauge } from 'prom-client';

export class MetricsService {
  private registry: Registry;
  private queryCounter: Counter;
  private queryLatency: Histogram;
  private accuracyGauge: Gauge;

  constructor() {
    this.registry = new Registry();

    this.queryCounter = new Counter({
      name: 'rag_queries_total',
      help: 'Total number of queries processed',
      labelNames: ['status', 'intent'],
      registers: [this.registry]
    });

    this.queryLatency = new Histogram({
      name: 'rag_query_duration_ms',
      help: 'Query processing latency in milliseconds',
      labelNames: ['stage'],
      buckets: [10, 50, 100, 200, 500, 1000, 2000],
      registers: [this.registry]
    });

    this.accuracyGauge = new Gauge({
      name: 'rag_accuracy_score',
      help: 'Response accuracy score',
      registers: [this.registry]
    });
  }

  recordQuery(intent: string, status: string): void {
    this.queryCounter.inc({ status, intent });
  }

  recordLatency(stage: string, durationMs: number): void {
    this.queryLatency.observe({ stage }, durationMs);
  }

  recordAccuracy(score: number): void {
    this.accuracyGauge.set(score);
  }
}
```

---

## 3. Data Architecture

### 3.1 Data Models

```typescript
// src/models/query.model.ts
export interface QueryRequest {
  query: string;
  filters?: Record<string, string>;
  maxResults?: number;
  includeMetadata?: boolean;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  metadata: ResponseMetadata;
}

export interface Source {
  docId: string;
  section: string;
  page: number;
  text: string;
  relevance: number;
}

export interface ResponseMetadata {
  queryType: string;
  strategy: string;
  agentsUsed: string[];
  processingTimeMs: number;
  numSources: number;
}
```

```typescript
// src/models/document.model.ts
export interface Document {
  id: string;
  type: string;
  title: string;
  uploadedAt: Date;
  chunksCount: number;
  metadata: DocumentMetadata;
}

export interface Chunk {
  id: string;
  docId: string;
  index: number;
  text: string;
  embedding: number[];
  section: string;
  page: number;
  metadata: ChunkMetadata;
}

export interface ChunkMetadata {
  chunkType: 'requirement' | 'definition' | 'procedure' | 'exception';
  confidence: number;
  verified: boolean;
  crossReferences: string[];
}
```

### 3.2 AgentDB Collections

```typescript
// Collection: technical_standards
{
  id: "uuid",
  vector: [1536 dimensions], // ada-002 embedding
  metadata: {
    doc_id: "uuid",
    chunk_index: 0,
    doc_type: "PCI-DSS",
    section: "Section 3.4 - Encryption Requirements",
    page: 42,
    text: "All cardholder data must be encrypted...",
    chunk_type: "requirement",
    confidence: 0.95,
    verified: true,
    cross_references: ["3.4.1", "3.4.2"],
    timestamp: "2025-10-24T00:00:00Z"
  }
}
```

### 3.3 Redis Cache Strategy

```typescript
// src/cache/cache.service.ts
import Redis from 'ioredis';

export class CacheService {
  private redis: Redis;

  async cacheQuery(query: string, result: QueryResponse): Promise<void> {
    const key = `query:${this.hashQuery(query)}`;
    await this.redis.setex(key, 3600, JSON.stringify(result)); // 1 hour TTL
  }

  async getCachedQuery(query: string): Promise<QueryResponse | null> {
    const key = `query:${this.hashQuery(query)}`;
    const cached = await this.redis.get(key);
    return cached ? JSON.parse(cached) : null;
  }

  async cacheSession(sessionId: string, data: any): Promise<void> {
    const key = `session:${sessionId}`;
    await this.redis.setex(key, 86400, JSON.stringify(data)); // 24 hours
  }

  private hashQuery(query: string): string {
    // Simple hash for cache key
    return Buffer.from(query).toString('base64');
  }
}
```

---

## 4. Integration Patterns

### 4.1 AgentDB Client Integration

```typescript
// src/integrations/agentdb.client.ts
import { AgentDB } from 'agentdb';

export class AgentDBClient {
  private client: AgentDB;

  async connect(): Promise<void> {
    this.client = new AgentDB({
      host: process.env.AGENTDB_HOST,
      port: parseInt(process.env.AGENTDB_PORT),
      apiKey: process.env.AGENTDB_API_KEY,
      collection: 'technical_standards',
      vectorConfig: {
        size: 1536,
        distance: 'cosine',
        hnsw: { m: 16, efConstruction: 200, efSearch: 100 },
        quantization: { enabled: true, method: 'scalar', compressionRatio: 4 }
      }
    });

    await this.client.initialize();
  }

  async insertChunks(chunks: Chunk[]): Promise<string[]> {
    return this.client.insertBatch({
      vectors: chunks.map(c => c.embedding),
      metadata: chunks.map(c => c.metadata),
      payloads: chunks.map(c => c.text)
    });
  }

  async search(query: string, options: SearchOptions): Promise<Document[]> {
    const results = await this.client.search({
      queryVector: await this.embedQuery(query),
      limit: options.limit,
      useHNSW: true,
      filter: options.filter
    });

    return results.map(r => ({
      id: r.id,
      text: r.payload,
      metadata: r.metadata,
      score: r.score
    }));
  }
}
```

### 4.2 agentic-flow Coordinator Integration

```typescript
// src/integrations/agentic-flow.coordinator.ts
import { AgenticFlow, SwarmTopology } from 'agentic-flow';

export class AgenticFlowCoordinator {
  private swarm: AgenticFlow;

  async initialize(topology: SwarmTopology): Promise<string> {
    this.swarm = new AgenticFlow({
      topology,
      maxAgents: 8,
      strategy: 'adaptive',
      quic: { enabled: true, port: 4433 }
    });

    const swarmId = await this.swarm.init();
    return swarmId;
  }

  async spawnAgent(type: string, capabilities: string[]): Promise<string> {
    return this.swarm.spawn({
      type,
      capabilities,
      resources: { memory: '512MB', cpu: '0.5' }
    });
  }

  async orchestrateTask(task: Task): Promise<TaskResult> {
    return this.swarm.orchestrate({
      task: task.description,
      strategy: 'adaptive',
      priority: task.priority,
      maxAgents: task.maxAgents
    });
  }
}
```

### 4.3 ruv-FANN WASM Loader

```typescript
// src/integrations/ruv-fann.loader.ts
import { NeuralNet, initWasm } from 'ruv-swarm-wasm';

export class RuvFannLoader {
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Initialize WASM module with SIMD support
    await initWasm({
      simd: true,
      memory: { initial: 10, maximum: 100 } // Pages (1 page = 64KB)
    });

    this.initialized = true;
  }

  async loadModel(path: string): Promise<NeuralNet> {
    await this.initialize();

    return NeuralNet.load(path, {
      enableCache: true,
      useSIMD: true
    });
  }

  async createModel(config: NetworkConfig): Promise<NeuralNet> {
    await this.initialize();

    return NeuralNet.new({
      inputSize: config.inputSize,
      hiddenLayers: config.hiddenLayers,
      outputSize: config.outputSize,
      activation: config.activation
    });
  }
}
```

---

## 5. Security Architecture

### 5.1 Authentication (JWT)

```typescript
// src/security/auth.middleware.ts
import jwt from 'jsonwebtoken';

export const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};
```

### 5.2 Rate Limiting

```typescript
// src/security/rate-limit.middleware.ts
import rateLimit from 'express-rate-limit';

export const rateLimiter = rateLimit({
  windowMs: 60000, // 1 minute
  max: 100, // 100 requests per minute
  message: 'Too many requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false
});
```

### 5.3 Data Encryption

```typescript
// At-rest: AgentDB encrypted storage
// In-transit: TLS 1.3 for all API connections
// Secrets: Environment variables + AWS Secrets Manager

export const securityConfig = {
  tls: {
    enabled: true,
    minVersion: 'TLSv1.3',
    cert: process.env.TLS_CERT,
    key: process.env.TLS_KEY
  },
  encryption: {
    algorithm: 'aes-256-gcm',
    keyLength: 256
  }
};
```

---

## 6. Deployment Architecture

### 6.1 Docker Containers

```dockerfile
# Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application
COPY dist/ ./dist/
COPY models/ ./models/

# Run application
EXPOSE 3000
CMD ["node", "dist/main.js"]
```

### 6.2 Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production
      AGENTDB_HOST: agentdb
      REDIS_HOST: redis
    depends_on:
      - agentdb
      - redis

  agentdb:
    image: agentdb/agentdb:latest
    ports:
      - "6333:6333"
    volumes:
      - agentdb_data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin

volumes:
  agentdb_data:
  redis_data:
```

### 6.3 Environment Variables

```bash
# .env.production
NODE_ENV=production
PORT=3000

# AgentDB
AGENTDB_HOST=agentdb
AGENTDB_PORT=6333
AGENTDB_API_KEY=${AGENTDB_API_KEY}

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# JWT
JWT_SECRET=${JWT_SECRET}

# OpenAI (for embeddings)
OPENAI_API_KEY=${OPENAI_API_KEY}

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=info
```

---

## 7. Observability Architecture

### 7.1 Structured Logging (Pino)

```typescript
// src/monitoring/logger.ts
import pino from 'pino';

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label })
  },
  timestamp: () => `,"time":"${new Date().toISOString()}"`,
  serializers: {
    req: (req) => ({
      method: req.method,
      url: req.url,
      headers: req.headers
    }),
    res: (res) => ({
      statusCode: res.statusCode
    }),
    err: pino.stdSerializers.err
  }
});
```

### 7.2 Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['app:3000']
    metrics_path: '/api/v1/metrics'
```

**Key Metrics:**
- `rag_queries_total` - Total queries processed
- `rag_query_duration_ms` - Query latency histogram
- `rag_accuracy_score` - Response accuracy gauge
- `rag_cache_hits_total` - Cache hit rate
- `rag_errors_total` - Error count by type

### 7.3 Grafana Dashboards

**Dashboard 1: System Health**
- Request rate (queries/sec)
- Error rate (%)
- P50/P95/P99 latency
- Cache hit rate

**Dashboard 2: RAG Performance**
- Accuracy score over time
- Retrieval strategy distribution
- Agent utilization
- Neural inference time

**Dashboard 3: Infrastructure**
- CPU usage
- Memory usage
- AgentDB query performance
- Redis cache stats

---

## 8. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Accuracy** | >97% | User feedback + verification |
| **Latency (P50)** | <400ms | Prometheus histogram |
| **Latency (P95)** | <600ms | Prometheus histogram |
| **Throughput** | 150 q/s | Single-threaded Node.js |
| **Cache Hit Rate** | >60% | Redis stats |
| **Memory Usage** | <8GB | 1M documents indexed |
| **Neural Inference** | <20ms | WASM SIMD optimized |
| **HNSW Search** | <100ms | AgentDB metrics |

---

## 9. Scalability Considerations

### 9.1 Horizontal Scaling

```
┌─────────────────────────────────────────────────┐
│              Load Balancer (Nginx)              │
└────────┬─────────────────────────────┬──────────┘
         │                             │
    ┌────▼────┐                   ┌────▼────┐
    │ API     │                   │ API     │
    │ Node 1  │                   │ Node 2  │
    └────┬────┘                   └────┬────┘
         │                             │
         └─────────┬───────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Shared Resources  │
         │  - AgentDB         │
         │  - Redis Cache     │
         └────────────────────┘
```

### 9.2 Caching Strategy

**L1 Cache (In-memory):** Recent queries (100 MB)
**L2 Cache (Redis):** Query results (1 GB, 1 hour TTL)
**L3 Cache (AgentDB):** Vector cache (10 GB, 24 hour TTL)

### 9.3 Growth Plan

| Phase | Documents | Memory | Nodes | Throughput |
|-------|-----------|--------|-------|------------|
| MVP | 10K | 2GB | 1 | 50 q/s |
| Beta | 100K | 8GB | 2 | 150 q/s |
| v1.0 | 1M | 40GB | 4 | 500 q/s |
| v2.0 | 10M | 200GB | 8 | 1200 q/s |

---

## Appendix: Key Architectural Decisions

### A1. Why TypeScript?
- **Faster development** (8 weeks vs 12 weeks)
- **Lower cost** (20% cheaper 3-year TCO)
- **Larger talent pool** (17.5M developers)
- **Acceptable performance** (600ms P95 vs 490ms in Rust)

### A2. Why AgentDB?
- **150x faster** HNSW search vs naive
- **4x memory reduction** with quantization
- **Built-in learning** with 9 RL algorithms
- **TypeScript-native** API

### A3. Why agentic-flow?
- **Multi-agent orchestration** (54+ agents)
- **QUIC protocol** (<100ms coordination)
- **ReasoningBank learning** (adaptive optimization)
- **Proven track record** (84.8% SWE-Bench)

### A4. Why ruv-FANN WASM?
- **SIMD optimization** (2-4x faster)
- **Small bundle** (<800KB)
- **Fast inference** (<20ms)
- **Browser + Node.js** compatible

---

**Document Status:** COMPLETE
**Total Pages:** 15
**Focus:** Architecture design (WHAT to build, not HOW)
**Next Phase:** 04-REFINEMENT (TDD implementation)

---

*Architecture designed for >97% accuracy, <600ms P95 latency, TypeScript implementation*
*Date: October 24, 2025*
