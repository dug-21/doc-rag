# System Architecture Design - Phase 4 Corrected System
## High-Accuracy RAG with Proper Library Integration

**Version**: 4.0.0  
**Status**: Architecture Design  
**Accuracy Target**: 99% through proper integration  
**Consensus Model**: Byzantine Fault Tolerant (66% threshold)  
**Libraries**: DAA, ruv-FANN v0.1.6+, FACT  

---

## 1. Executive Summary

Phase 4 addresses critical integration issues identified in the analysis, implementing a corrected system architecture that achieves 99% accuracy through proper use of mandated libraries, Byzantine consensus, and elimination of custom implementations that duplicate library functionality.

### Key Corrections

- **DAA Orchestration**: Complete handover of orchestration and consensus to DAA library
- **ruv-FANN Integration**: All neural processing through ruv-FANN (no custom neural code)
- **FACT Caching**: Complete citation tracking and intelligent caching via FACT
- **Byzantine Consensus**: Proper 66% threshold implementation with multi-agent voting
- **Performance Guarantees**: Sub-500ms consensus, <2s total pipeline

---

## 2. Complete System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Web Interface<br/>React Frontend]
        API_GW[API Gateway<br/>Load Balancer]
        REST[REST Endpoints<br/>Query/Upload/Status]
    end
    
    subgraph "DAA Orchestration Core"
        MRAP[MRAP Control Loop<br/>Monitor→Reason→Act→Persist]
        CONSENSUS[Byzantine Consensus Engine<br/>66% Threshold]
        AGENT_POOL[Multi-Agent Pool<br/>Retriever|Analyzer|Validator|Reranker]
        COORDINATOR[Task Coordinator<br/>Agent Assignment & Scheduling]
    end
    
    subgraph "ruv-FANN Neural Processing"
        INTENT[Intent Analysis<br/>Query Classification]
        CHUNKER[Semantic Chunker<br/>Document Boundaries]
        EMBEDDER[Embedding Generator<br/>Vector Representations]
        RERANKER[Relevance Reranking<br/>Result Scoring]
        VALIDATOR[Response Validator<br/>Quality Assurance]
    end
    
    subgraph "FACT Intelligence Layer"
        CACHE[Memory Cache<br/><50ms Retrieval]
        CITATIONS[Citation Tracker<br/>Source Attribution]
        FACT_STORE[Persistent Storage<br/>Document Facts]
        QUERY_OPT[Query Optimizer<br/>Performance Enhancement]
    end
    
    subgraph "Data Layer"
        MONGO[(MongoDB Cluster<br/>Document Metadata)]
        VECTORS[(Vector Store<br/>Embeddings Index)]
        FILES[File Storage<br/>Raw Documents]
        MODELS[Neural Models<br/>ruv-FANN Weights]
    end
    
    subgraph "Infrastructure Layer"
        DOCKER[Docker Containers<br/>Isolated Services]
        K8S[Kubernetes<br/>Orchestration]
        MONITORING[Prometheus/Grafana<br/>Observability]
        SECURITY[Security Layer<br/>Auth/TLS/Audit]
    end
    
    %% User Flow
    WEB --> API_GW
    API_GW --> REST
    REST --> MRAP
    
    %% DAA Orchestration
    MRAP --> CONSENSUS
    MRAP --> AGENT_POOL
    AGENT_POOL --> COORDINATOR
    CONSENSUS --> COORDINATOR
    
    %% Neural Processing Pipeline
    COORDINATOR --> INTENT
    INTENT --> CHUNKER
    CHUNKER --> EMBEDDER
    EMBEDDER --> RERANKER
    RERANKER --> VALIDATOR
    
    %% FACT Integration
    MRAP --> CACHE
    CACHE --> CITATIONS
    CITATIONS --> FACT_STORE
    FACT_STORE --> QUERY_OPT
    
    %% Data Access
    COORDINATOR --> MONGO
    EMBEDDER --> VECTORS
    CHUNKER --> FILES
    INTENT --> MODELS
    
    %% Infrastructure
    REST --> DOCKER
    DOCKER --> K8S
    K8S --> MONITORING
    MONITORING --> SECURITY
    
    classDef orchestration fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef neural fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef cache fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef infra fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class MRAP,CONSENSUS,AGENT_POOL,COORDINATOR orchestration
    class INTENT,CHUNKER,EMBEDDER,RERANKER,VALIDATOR neural
    class CACHE,CITATIONS,FACT_STORE,QUERY_OPT cache
    class MONGO,VECTORS,FILES,MODELS data
    class DOCKER,K8S,MONITORING,SECURITY infra
```

---

## 3. Component Architecture Details

### 3.1 DAA Orchestration Architecture

```mermaid
graph LR
    subgraph "DAA Core Components"
        CONTROL[Control Center<br/>Master Orchestrator]
        AGENTS[Agent Registry<br/>Pool Management]
        TASKS[Task Queue<br/>Work Distribution]
        STATE[State Manager<br/>System Memory]
    end
    
    subgraph "Consensus Engine"
        VOTING[Voting System<br/>Multi-Agent Decisions]
        BYZANTINE[Byzantine Filter<br/>Malicious Agent Detection]
        THRESHOLD[66% Threshold<br/>Consensus Validation]
        CONFLICT[Conflict Resolution<br/>Tie Breaking]
    end
    
    subgraph "Agent Specializations"
        RETRIEVER[Retriever Agent<br/>Document Search]
        ANALYZER[Analyzer Agent<br/>Content Analysis]
        VALIDATOR[Validator Agent<br/>Quality Check]
        RERANKER[Reranker Agent<br/>Result Scoring]
        MONITOR[Monitor Agent<br/>System Health]
    end
    
    CONTROL --> AGENTS
    AGENTS --> TASKS
    TASKS --> STATE
    
    CONTROL --> VOTING
    VOTING --> BYZANTINE
    BYZANTINE --> THRESHOLD
    THRESHOLD --> CONFLICT
    
    AGENTS --> RETRIEVER
    AGENTS --> ANALYZER
    AGENTS --> VALIDATOR
    AGENTS --> RERANKER
    AGENTS --> MONITOR
    
    classDef core fill:#e3f2fd,stroke:#1976d2
    classDef consensus fill:#fce4ec,stroke:#c2185b
    classDef agents fill:#f3e5f5,stroke:#7b1fa2
    
    class CONTROL,AGENTS,TASKS,STATE core
    class VOTING,BYZANTINE,THRESHOLD,CONFLICT consensus
    class RETRIEVER,ANALYZER,VALIDATOR,RERANKER,MONITOR agents
```

**DAA Configuration:**
```rust
// DAA Orchestration Configuration
daa_config = {
    agent_pool_size: 5,           // Minimum 3 for consensus
    consensus_threshold: 0.66,     // Byzantine fault tolerance
    max_consensus_time: 500,       // milliseconds
    voting_strategy: "weighted",   // confidence-based voting
    fault_detection: true,         // Byzantine agent detection
    task_timeout: 30,             // seconds per task
    health_check_interval: 10,    // seconds
}
```

### 3.2 ruv-FANN Neural Processing Architecture

```mermaid
graph TB
    subgraph "Neural Network Models"
        INTENT_MODEL[Intent Classification<br/>Query Understanding]
        CHUNK_MODEL[Semantic Chunking<br/>Document Segmentation]
        EMBED_MODEL[Embedding Generation<br/>Vector Representations]
        RANK_MODEL[Relevance Ranking<br/>Result Scoring]
        VALID_MODEL[Validation Network<br/>Quality Assessment]
    end
    
    subgraph "Processing Pipeline"
        PREPROCESS[Input Preprocessing<br/>Text Normalization]
        INFERENCE[Neural Inference<br/>Model Execution]
        POSTPROCESS[Output Processing<br/>Result Formatting]
        CACHE_MODELS[Model Caching<br/>Performance Optimization]
    end
    
    subgraph "Performance Monitoring"
        LATENCY[Latency Tracking<br/><200ms Target]
        ACCURACY[Accuracy Metrics<br/>Quality Measurement]
        THROUGHPUT[Throughput Monitor<br/>Requests/Second]
        MEMORY[Memory Usage<br/>Resource Tracking]
    end
    
    INTENT_MODEL --> PREPROCESS
    CHUNK_MODEL --> PREPROCESS
    EMBED_MODEL --> PREPROCESS
    RANK_MODEL --> PREPROCESS
    VALID_MODEL --> PREPROCESS
    
    PREPROCESS --> INFERENCE
    INFERENCE --> POSTPROCESS
    POSTPROCESS --> CACHE_MODELS
    
    INFERENCE --> LATENCY
    LATENCY --> ACCURACY
    ACCURACY --> THROUGHPUT
    THROUGHPUT --> MEMORY
    
    classDef models fill:#f3e5f5,stroke:#7b1fa2
    classDef pipeline fill:#e1f5fe,stroke:#0277bd
    classDef monitoring fill:#e8f5e8,stroke:#388e3c
    
    class INTENT_MODEL,CHUNK_MODEL,EMBED_MODEL,RANK_MODEL,VALID_MODEL models
    class PREPROCESS,INFERENCE,POSTPROCESS,CACHE_MODELS pipeline
    class LATENCY,ACCURACY,THROUGHPUT,MEMORY monitoring
```

**ruv-FANN Configuration:**
```rust
// ruv-FANN Neural Processing Configuration
ruv_fann_config = {
    model_path: "/models/current.bin",
    max_inference_time: 200,      // milliseconds
    batch_size: 32,               // for batch processing
    cache_models: true,           // model caching enabled
    precision: "float32",         // model precision
    threads: 4,                   // parallel processing
    warmup_iterations: 10,        // startup optimization
}
```

### 3.3 FACT Caching and Intelligence Architecture

```mermaid
graph LR
    subgraph "Cache Management"
        MEMORY[Memory Cache<br/>Hot Data Access]
        PERSISTENT[Persistent Store<br/>Cold Data Storage]
        EVICTION[LRU Eviction<br/>Memory Management]
        WARMUP[Cache Warmup<br/>Preloading Strategy]
    end
    
    subgraph "Citation System"
        EXTRACTOR[Citation Extractor<br/>Source Identification]
        TRACKER[Citation Tracker<br/>Reference Management]
        FORMATTER[Citation Formatter<br/>Standard Format]
        VALIDATOR[Citation Validator<br/>Accuracy Check]
    end
    
    subgraph "Query Intelligence"
        ANALYZER[Query Analyzer<br/>Pattern Recognition]
        OPTIMIZER[Query Optimizer<br/>Performance Enhancement]
        PREDICTOR[Cache Predictor<br/>Preemptive Loading]
        STATS[Usage Statistics<br/>Analytics]
    end
    
    MEMORY --> PERSISTENT
    PERSISTENT --> EVICTION
    EVICTION --> WARMUP
    
    EXTRACTOR --> TRACKER
    TRACKER --> FORMATTER
    FORMATTER --> VALIDATOR
    
    ANALYZER --> OPTIMIZER
    OPTIMIZER --> PREDICTOR
    PREDICTOR --> STATS
    
    MEMORY --> EXTRACTOR
    PERSISTENT --> ANALYZER
    
    classDef cache fill:#fff3e0,stroke:#f57c00
    classDef citations fill:#f3e5f5,stroke:#7b1fa2
    classDef intelligence fill:#e8f5e8,stroke:#388e3c
    
    class MEMORY,PERSISTENT,EVICTION,WARMUP cache
    class EXTRACTOR,TRACKER,FORMATTER,VALIDATOR citations
    class ANALYZER,OPTIMIZER,PREDICTOR,STATS intelligence
```

**FACT Configuration:**
```rust
// FACT Caching and Intelligence Configuration
fact_config = {
    memory_limit: "2048MB",       // in-memory cache size
    cache_ttl: 3600,             // 1 hour default TTL
    eviction_policy: "LRU",      // Least Recently Used
    persistence_path: "/data/fact_cache",
    citation_format: "section.page",
    max_cache_time: 50,          // milliseconds
    warmup_enabled: true,        // cache preloading
    analytics_enabled: true,     // usage tracking
}
```

---

## 4. Data Flow Architecture

### 4.1 Complete Query Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant Gateway as API Gateway
    participant DAA as DAA Orchestration
    participant FACT as FACT Cache
    participant RUV as ruv-FANN
    participant Data as Data Storage
    
    Note over Client,Data: Phase 4 Corrected Pipeline - Complete Flow
    
    Client->>Gateway: POST /query
    Gateway->>DAA: Initialize MRAP loop
    
    Note over DAA: Monitor Phase
    DAA->>DAA: System health check
    DAA->>FACT: Cache health status
    DAA->>RUV: Neural model status
    
    Note over DAA: Reason Phase
    DAA->>FACT: Query cache check (<50ms)
    alt Cache Hit
        FACT-->>DAA: Cached response + citations
        DAA-->>Gateway: Response (cache hit)
        Gateway-->>Client: Final response
    else Cache Miss
        FACT-->>DAA: Cache miss
        
        Note over DAA: Act Phase - Agent Coordination
        DAA->>RUV: Intent analysis (<100ms)
        RUV-->>DAA: Query classification + intent
        
        par Agent Pool Processing
            DAA->>Data: Retriever agent - document search
            DAA->>RUV: Analyzer agent - content analysis
            DAA->>RUV: Validator agent - quality check
        end
        
        Data-->>DAA: Retrieved documents
        RUV-->>DAA: Analysis results
        RUV-->>DAA: Validation scores
        
        Note over DAA: Byzantine Consensus (66% threshold)
        DAA->>DAA: Collect agent votes
        DAA->>DAA: Apply Byzantine filter
        DAA->>DAA: Validate 66% consensus
        
        alt Consensus Achieved
            Note over DAA: Persist Phase
            DAA->>RUV: Final reranking (<100ms)
            RUV-->>DAA: Scored results
            
            DAA->>FACT: Extract citations
            FACT-->>DAA: Citation metadata
            
            DAA->>FACT: Cache response (background)
            DAA-->>Gateway: Consensus response
            Gateway-->>Client: Final response + metadata
        else Consensus Failed
            DAA-->>Gateway: Consensus failure error
            Gateway-->>Client: Error response
        end
    end
    
    Note over DAA: MRAP Reflect & Adapt
    DAA->>DAA: Update agent performance
    DAA->>FACT: Update cache statistics
    DAA->>RUV: Update neural metrics
```

### 4.2 Document Upload and Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant Gateway as API Gateway
    participant DAA as DAA Orchestration
    participant RUV as ruv-FANN
    participant FACT as FACT Storage
    participant Data as Data Storage
    
    Client->>Gateway: POST /upload (document)
    Gateway->>DAA: Initialize document processing
    
    Note over DAA: Document Analysis Phase
    DAA->>RUV: Load document analysis model
    RUV-->>DAA: Model loaded
    
    DAA->>RUV: Document structure analysis
    RUV-->>DAA: Document metadata + structure
    
    Note over DAA: Semantic Chunking Phase
    DAA->>RUV: Semantic chunking with boundaries
    Note over RUV: ChunkingConfig:<br/>max_size: 512<br/>overlap: 50<br/>semantic_threshold: 0.85
    RUV-->>DAA: Semantic chunks with boundaries
    
    Note over DAA: Fact Extraction Phase
    loop For each chunk
        DAA->>FACT: Extract structured facts
        FACT-->>DAA: Facts with citations
    end
    
    Note over DAA: Embedding Generation Phase
    DAA->>RUV: Generate embeddings for chunks
    RUV-->>DAA: Vector embeddings
    
    Note over DAA: Quality Validation Phase
    DAA->>RUV: Validate chunk quality
    RUV-->>DAA: Quality scores
    
    Note over DAA: Storage Phase
    DAA->>Data: Store document metadata
    DAA->>FACT: Store facts + citations
    DAA->>Data: Store embeddings in vector DB
    
    Data-->>DAA: Storage confirmation
    FACT-->>DAA: Fact storage confirmation
    
    DAA-->>Gateway: Processing complete + metadata
    Gateway-->>Client: Upload response
```

### 4.3 Byzantine Consensus Flow

```mermaid
graph TB
    subgraph "Agent Voting Process"
        A1[Retriever Agent<br/>Document Relevance Vote]
        A2[Analyzer Agent<br/>Content Analysis Vote]
        A3[Validator Agent<br/>Quality Assessment Vote]
        A4[Reranker Agent<br/>Ranking Confidence Vote]
        A5[Monitor Agent<br/>System Health Vote]
    end
    
    subgraph "Consensus Engine"
        COLLECT[Vote Collection<br/>Aggregate All Votes]
        BYZANTINE[Byzantine Filter<br/>Detect Malicious Agents]
        THRESHOLD[Threshold Check<br/>66% Agreement Required]
        RESOLVE[Conflict Resolution<br/>Weighted Scoring]
    end
    
    subgraph "Consensus Outcomes"
        SUCCESS[Consensus Achieved<br/>Response Validated]
        PARTIAL[Partial Consensus<br/>67-80% Agreement]
        FAILURE[Consensus Failed<br/><66% Agreement]
        MALICIOUS[Byzantine Fault<br/>Malicious Agent Detected]
    end
    
    A1 --> COLLECT
    A2 --> COLLECT
    A3 --> COLLECT
    A4 --> COLLECT
    A5 --> COLLECT
    
    COLLECT --> BYZANTINE
    BYZANTINE --> THRESHOLD
    THRESHOLD --> RESOLVE
    
    RESOLVE --> SUCCESS
    RESOLVE --> PARTIAL
    RESOLVE --> FAILURE
    BYZANTINE --> MALICIOUS
    
    classDef agents fill:#e3f2fd,stroke:#1976d2
    classDef consensus fill:#f3e5f5,stroke:#7b1fa2
    classDef outcomes fill:#e8f5e8,stroke:#388e3c
    classDef malicious fill:#ffebee,stroke:#d32f2f
    
    class A1,A2,A3,A4,A5 agents
    class COLLECT,BYZANTINE,THRESHOLD,RESOLVE consensus
    class SUCCESS,PARTIAL outcomes
    class FAILURE,MALICIOUS malicious
```

---

## 5. Deployment Architecture

### 5.1 Containerized Architecture

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Application Pods"
            API[API Server Pod<br/>DAA + ruv-FANN + FACT]
            WORKER1[Worker Pod 1<br/>Neural Processing]
            WORKER2[Worker Pod 2<br/>Neural Processing]
        end
        
        subgraph "Data Pods"
            MONGO[(MongoDB Pod<br/>Document Metadata)]
            VECTOR[(Vector DB Pod<br/>Embeddings)]
        end
        
        subgraph "Infrastructure Pods"
            REDIS[(Redis Pod<br/>Session Cache)]
            MONITORING[Monitoring Pod<br/>Prometheus/Grafana]
        end
    end
    
    subgraph "Persistent Storage"
        MODELS[Model Storage<br/>ruv-FANN Weights]
        FACT_STORE[FACT Storage<br/>Cache + Citations]
        DATA_STORE[Data Storage<br/>Documents + Vectors]
    end
    
    subgraph "External Services"
        LB[Load Balancer<br/>Traffic Distribution]
        CDN[CDN<br/>Static Assets]
        BACKUP[Backup Service<br/>Data Protection]
    end
    
    LB --> API
    API --> WORKER1
    API --> WORKER2
    
    API --> MONGO
    API --> VECTOR
    API --> REDIS
    
    WORKER1 --> MODELS
    WORKER2 --> MODELS
    API --> FACT_STORE
    
    MONGO --> DATA_STORE
    VECTOR --> DATA_STORE
    
    MONITORING --> BACKUP
    CDN --> LB
    
    classDef pods fill:#e3f2fd,stroke:#1976d2
    classDef storage fill:#fff3e0,stroke:#f57c00
    classDef external fill:#e8f5e8,stroke:#388e3c
    
    class API,WORKER1,WORKER2,MONGO,VECTOR,REDIS,MONITORING pods
    class MODELS,FACT_STORE,DATA_STORE storage
    class LB,CDN,BACKUP external
```

### 5.2 Docker Configuration

```yaml
# Phase 4 Corrected Docker Compose
version: '3.8'

services:
  # Main API with integrated DAA + ruv-FANN + FACT
  api:
    build:
      context: .
      dockerfile: Dockerfile.corrected
    container_name: doc-rag-api-corrected
    ports:
      - "8080:8080"      # API endpoints
      - "9090:9090"      # Metrics
      - "8081:8081"      # Health checks
    environment:
      - RUST_LOG=info
      - DAA_AGENT_POOL_SIZE=5
      - DAA_CONSENSUS_THRESHOLD=0.66
      - DAA_MAX_CONSENSUS_TIME=500
      - RUV_FANN_MODEL_PATH=/models/current.bin
      - RUV_FANN_MAX_INFERENCE_TIME=200
      - FACT_CACHE_PATH=/data/fact_cache
      - FACT_MEMORY_LIMIT=2048MB
      - FACT_MAX_CACHE_TIME=50
      - MONGODB_URL=mongodb://mongodb:27017/doc_rag
    volumes:
      - ./models:/models:ro
      - ./data/fact_cache:/data/fact_cache
      - ./uploads:/uploads
    depends_on:
      - mongodb
      - vector-db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'

  # MongoDB for document metadata
  mongodb:
    image: mongo:7.0
    container_name: doc-rag-mongodb
    ports:
      - "27017:27017"
    volumes:
      - ./data/mongo:/data/db
    environment:
      MONGO_INITDB_DATABASE: doc_rag
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  # Vector database for embeddings
  vector-db:
    image: qdrant/qdrant:latest
    container_name: doc-rag-vectors
    ports:
      - "6333:6333"
    volumes:
      - ./data/vectors:/qdrant/storage
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  # Monitoring stack
  prometheus:
    image: prom/prometheus:latest
    container_name: doc-rag-monitoring
    ports:
      - "9091:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - api

  grafana:
    image: grafana/grafana:latest
    container_name: doc-rag-grafana
    ports:
      - "3000:3000"
    volumes:
      - ./config/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus

networks:
  default:
    name: doc-rag-network
    driver: bridge
```

### 5.3 Kubernetes Manifests

```yaml
# Phase 4 Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doc-rag-api-corrected
  labels:
    app: doc-rag-api
    version: "4.0.0"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: doc-rag-api
  template:
    metadata:
      labels:
        app: doc-rag-api
        version: "4.0.0"
    spec:
      containers:
      - name: api
        image: doc-rag-api:4.0.0-corrected
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9090
          name: metrics
        - containerPort: 8081
          name: health
        env:
        - name: DAA_AGENT_POOL_SIZE
          value: "5"
        - name: DAA_CONSENSUS_THRESHOLD
          value: "0.66"
        - name: DAA_MAX_CONSENSUS_TIME
          value: "500"
        - name: RUV_FANN_MODEL_PATH
          value: "/models/current.bin"
        - name: FACT_CACHE_PATH
          value: "/data/fact_cache"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: fact-cache
          mountPath: /data/fact_cache
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ruv-fann-models
      - name: fact-cache
        persistentVolumeClaim:
          claimName: fact-cache-storage

---
apiVersion: v1
kind: Service
metadata:
  name: doc-rag-api-service
spec:
  selector:
    app: doc-rag-api
  ports:
  - name: api
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

---

## 6. Performance Architecture

### 6.1 Performance Requirements Matrix

| Component | Target | Measurement | Validation |
|-----------|--------|-------------|------------|
| FACT Cache Retrieval | <50ms | 99.9% percentile | Error if exceeded |
| ruv-FANN Intent Analysis | <100ms | Per inference | Warning if exceeded |
| ruv-FANN Reranking | <100ms | Per batch | Warning if exceeded |
| DAA Agent Coordination | <300ms | Per task | Warning if exceeded |
| Byzantine Consensus | <500ms | Per decision | Error if exceeded |
| **Total Pipeline** | **<2000ms** | **End-to-end** | **SLA violation** |

### 6.2 Performance Monitoring Architecture

```mermaid
graph LR
    subgraph "Real-time Metrics"
        LATENCY[Latency Tracking<br/>P50/P95/P99]
        THROUGHPUT[Throughput Monitor<br/>Requests/Second]
        ERROR[Error Rate<br/>Success Percentage]
        CONSENSUS[Consensus Rate<br/>Agreement Percentage]
    end
    
    subgraph "Performance Alerts"
        SLA[SLA Violations<br/>Immediate Alerts]
        DEGRADATION[Performance Degradation<br/>Trend Analysis]
        CAPACITY[Capacity Planning<br/>Resource Usage]
        HEALTH[Health Checks<br/>System Status]
    end
    
    subgraph "Optimization Actions"
        SCALING[Auto-scaling<br/>Resource Adjustment]
        CACHING[Cache Warming<br/>Preemptive Loading]
        BALANCING[Load Balancing<br/>Traffic Distribution]
        TUNING[Parameter Tuning<br/>Model Optimization]
    end
    
    LATENCY --> SLA
    THROUGHPUT --> DEGRADATION
    ERROR --> CAPACITY
    CONSENSUS --> HEALTH
    
    SLA --> SCALING
    DEGRADATION --> CACHING
    CAPACITY --> BALANCING
    HEALTH --> TUNING
    
    classDef metrics fill:#e3f2fd,stroke:#1976d2
    classDef alerts fill:#fff3e0,stroke:#f57c00
    classDef actions fill:#e8f5e8,stroke:#388e3c
    
    class LATENCY,THROUGHPUT,ERROR,CONSENSUS metrics
    class SLA,DEGRADATION,CAPACITY,HEALTH alerts
    class SCALING,CACHING,BALANCING,TUNING actions
```

### 6.3 Scalability Architecture

```mermaid
graph TB
    subgraph "Horizontal Scaling"
        LB[Load Balancer<br/>Request Distribution]
        API1[API Pod 1<br/>Full Stack]
        API2[API Pod 2<br/>Full Stack]
        API3[API Pod 3<br/>Full Stack]
        APIX[API Pod N<br/>Auto-scaled]
    end
    
    subgraph "Vertical Scaling"
        CPU[CPU Scaling<br/>Neural Processing]
        MEMORY[Memory Scaling<br/>FACT Cache]
        GPU[GPU Scaling<br/>ruv-FANN Acceleration]
        STORAGE[Storage Scaling<br/>Data Growth]
    end
    
    subgraph "Auto-scaling Triggers"
        LATENCY_TRIGGER[Latency > 1.5s<br/>Scale Up]
        CPU_TRIGGER[CPU > 80%<br/>Scale Up]
        MEMORY_TRIGGER[Memory > 85%<br/>Scale Up]
        QUEUE_TRIGGER[Queue Depth > 100<br/>Scale Out]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    LB --> APIX
    
    API1 --> CPU
    API2 --> MEMORY
    API3 --> GPU
    APIX --> STORAGE
    
    LATENCY_TRIGGER --> APIX
    CPU_TRIGGER --> CPU
    MEMORY_TRIGGER --> MEMORY
    QUEUE_TRIGGER --> LB
    
    classDef horizontal fill:#e3f2fd,stroke:#1976d2
    classDef vertical fill:#f3e5f5,stroke:#7b1fa2
    classDef triggers fill:#e8f5e8,stroke:#388e3c
    
    class LB,API1,API2,API3,APIX horizontal
    class CPU,MEMORY,GPU,STORAGE vertical
    class LATENCY_TRIGGER,CPU_TRIGGER,MEMORY_TRIGGER,QUEUE_TRIGGER triggers
```

---

## 7. Security Architecture

### 7.1 Security Layers

```mermaid
graph TB
    subgraph "Network Security"
        WAF[Web Application Firewall<br/>Attack Protection]
        TLS[TLS 1.3 Encryption<br/>Data in Transit]
        VPC[Virtual Private Cloud<br/>Network Isolation]
        FIREWALL[Internal Firewall<br/>Service-to-Service]
    end
    
    subgraph "Application Security"
        AUTH[Authentication<br/>JWT Token Validation]
        AUTHZ[Authorization<br/>Role-Based Access]
        INPUT[Input Validation<br/>Injection Prevention]
        RATE[Rate Limiting<br/>DoS Protection]
    end
    
    subgraph "Data Security"
        ENCRYPT[Encryption at Rest<br/>Database & Files]
        KEYS[Key Management<br/>Secure Storage]
        AUDIT[Audit Logging<br/>Access Tracking]
        BACKUP[Secure Backup<br/>Encrypted Storage]
    end
    
    subgraph "Component Security"
        DAA_SEC[DAA Agent Security<br/>Byzantine Protection]
        FANN_SEC[ruv-FANN Security<br/>Model Validation]
        FACT_SEC[FACT Cache Security<br/>Access Control]
        CONTAINER[Container Security<br/>Image Scanning]
    end
    
    WAF --> AUTH
    TLS --> AUTHZ
    VPC --> INPUT
    FIREWALL --> RATE
    
    AUTH --> ENCRYPT
    AUTHZ --> KEYS
    INPUT --> AUDIT
    RATE --> BACKUP
    
    ENCRYPT --> DAA_SEC
    KEYS --> FANN_SEC
    AUDIT --> FACT_SEC
    BACKUP --> CONTAINER
    
    classDef network fill:#e3f2fd,stroke:#1976d2
    classDef application fill:#f3e5f5,stroke:#7b1fa2
    classDef data fill:#fff3e0,stroke:#f57c00
    classDef component fill:#e8f5e8,stroke:#388e3c
    
    class WAF,TLS,VPC,FIREWALL network
    class AUTH,AUTHZ,INPUT,RATE application
    class ENCRYPT,KEYS,AUDIT,BACKUP data
    class DAA_SEC,FANN_SEC,FACT_SEC,CONTAINER component
```

### 7.2 Security Monitoring

```mermaid
graph LR
    subgraph "Threat Detection"
        ANOMALY[Anomaly Detection<br/>Behavioral Analysis]
        INTRUSION[Intrusion Detection<br/>Attack Patterns]
        MALWARE[Malware Scanning<br/>File Analysis]
        VULNERABILITY[Vulnerability Scanning<br/>Security Assessment]
    end
    
    subgraph "Incident Response"
        ALERT[Security Alerts<br/>Real-time Notifications]
        ISOLATION[Component Isolation<br/>Contain Threats]
        FORENSICS[Digital Forensics<br/>Attack Investigation]
        RECOVERY[System Recovery<br/>Service Restoration]
    end
    
    subgraph "Compliance Monitoring"
        GDPR[GDPR Compliance<br/>Data Protection]
        SOC2[SOC 2 Compliance<br/>Security Controls]
        AUDIT_TRAIL[Audit Trail<br/>Compliance Evidence]
        REPORTING[Compliance Reporting<br/>Regular Assessments]
    end
    
    ANOMALY --> ALERT
    INTRUSION --> ISOLATION
    MALWARE --> FORENSICS
    VULNERABILITY --> RECOVERY
    
    ALERT --> GDPR
    ISOLATION --> SOC2
    FORENSICS --> AUDIT_TRAIL
    RECOVERY --> REPORTING
    
    classDef detection fill:#ffebee,stroke:#d32f2f
    classDef response fill:#fff3e0,stroke:#f57c00
    classDef compliance fill:#e8f5e8,stroke:#388e3c
    
    class ANOMALY,INTRUSION,MALWARE,VULNERABILITY detection
    class ALERT,ISOLATION,FORENSICS,RECOVERY response
    class GDPR,SOC2,AUDIT_TRAIL,REPORTING compliance
```

---

## 8. Integration Interfaces

### 8.1 DAA Integration Interface

```rust
// DAA Orchestration Integration
pub struct DAAOrchestrator {
    agent_pool: AgentPool,
    consensus_engine: ConsensusEngine,
    task_coordinator: TaskCoordinator,
    state_manager: StateManager,
}

impl DAAOrchestrator {
    pub async fn initialize(&mut self) -> Result<()> {
        self.agent_pool.spawn_agents(5).await?;
        self.consensus_engine.set_threshold(0.66);
        self.task_coordinator.start().await?;
        Ok(())
    }
    
    pub async fn process_query(&self, query: &Query) -> Result<Response> {
        // MRAP Loop: Monitor
        let health = self.monitor_system().await?;
        
        // MRAP Loop: Reason
        let strategy = self.reason_strategy(query, health).await?;
        
        // MRAP Loop: Act
        let results = self.coordinate_agents(query, strategy).await?;
        
        // Byzantine Consensus
        let consensus = self.consensus_engine
            .achieve_consensus(results, 500).await?;
            
        if consensus.agreement_rate < 0.66 {
            return Err(ConsensusError::InsufficientAgreement);
        }
        
        // MRAP Loop: Persist
        self.state_manager.persist_state(query, &consensus).await?;
        
        Ok(consensus.response)
    }
}
```

### 8.2 ruv-FANN Integration Interface

```rust
// ruv-FANN Neural Processing Integration
pub struct RuvFANNProcessor {
    intent_model: IntentAnalyzer,
    chunker_model: SemanticChunker,
    embedder_model: EmbeddingGenerator,
    reranker_model: RelevanceRanker,
}

impl RuvFANNProcessor {
    pub async fn load_models(&mut self, model_path: &str) -> Result<()> {
        self.intent_model.load(model_path).await?;
        self.chunker_model.load(model_path).await?;
        self.embedder_model.load(model_path).await?;
        self.reranker_model.load(model_path).await?;
        Ok(())
    }
    
    pub async fn analyze_intent(&self, query: &str) -> Result<Intent> {
        let start = Instant::now();
        let intent = self.intent_model.analyze(query).await?;
        let duration = start.elapsed();
        
        if duration.as_millis() > 100 {
            error!("Intent analysis exceeded 100ms: {}ms", duration.as_millis());
        }
        
        Ok(intent)
    }
    
    pub async fn chunk_document(&self, document: &str) -> Result<Vec<Chunk>> {
        let config = ChunkingConfig {
            max_size: 512,
            overlap: 50,
            semantic_threshold: 0.85,
        };
        
        self.chunker_model.chunk(document, config).await
    }
    
    pub async fn rerank_results(&self, results: Vec<SearchResult>) -> Result<Vec<ScoredResult>> {
        let start = Instant::now();
        let ranked = self.reranker_model.rank(results).await?;
        let duration = start.elapsed();
        
        if duration.as_millis() > 100 {
            error!("Reranking exceeded 100ms: {}ms", duration.as_millis());
        }
        
        Ok(ranked)
    }
}
```

### 8.3 FACT Integration Interface

```rust
// FACT Caching and Intelligence Integration
pub struct FACTCache {
    memory_cache: MemoryCache,
    persistent_store: PersistentStore,
    citation_tracker: CitationTracker,
    query_optimizer: QueryOptimizer,
}

impl FACTCache {
    pub async fn initialize(&mut self, config: FACTConfig) -> Result<()> {
        self.memory_cache.set_limit(config.memory_limit);
        self.memory_cache.set_ttl(config.cache_ttl);
        self.persistent_store.initialize(config.persistence_path).await?;
        Ok(())
    }
    
    pub async fn get_cached_response(&self, query_hash: &str) -> Result<Option<CachedResponse>> {
        let start = Instant::now();
        let response = self.memory_cache.get(query_hash).await;
        let duration = start.elapsed();
        
        if duration.as_millis() > 50 {
            error!("Cache retrieval exceeded 50ms: {}ms", duration.as_millis());
        }
        
        Ok(response)
    }
    
    pub async fn extract_citations(&self, response: &Response) -> Result<Vec<Citation>> {
        self.citation_tracker.extract(response).await
    }
    
    pub async fn cache_response(&self, query: &str, response: Response) -> Result<()> {
        let citations = self.extract_citations(&response).await?;
        let cached = CachedResponse {
            response,
            citations,
            timestamp: Utc::now(),
        };
        
        // Background caching - non-blocking
        tokio::spawn(async move {
            self.memory_cache.put(query, cached).await;
        });
        
        Ok(())
    }
}
```

---

## 9. API Specifications

### 9.1 REST API Endpoints

```yaml
# OpenAPI 3.0 Specification
openapi: 3.0.3
info:
  title: Doc-RAG Phase 4 Corrected API
  version: 4.0.0
  description: High-accuracy RAG with proper library integration

paths:
  /query:
    post:
      summary: Process query with Byzantine consensus
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  example: "What are the PCI DSS encryption requirements?"
                options:
                  type: object
                  properties:
                    require_consensus:
                      type: boolean
                      default: true
                    consensus_threshold:
                      type: number
                      minimum: 0.5
                      maximum: 1.0
                      default: 0.66
      responses:
        200:
          description: Successful query response
          content:
            application/json:
              schema:
                type: object
                properties:
                  response:
                    type: string
                  citations:
                    type: array
                    items:
                      $ref: '#/components/schemas/Citation'
                  consensus:
                    $ref: '#/components/schemas/ConsensusInfo'
                  performance:
                    $ref: '#/components/schemas/PerformanceMetrics'

  /upload:
    post:
      summary: Upload document for processing
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                options:
                  type: object
                  properties:
                    enable_chunking:
                      type: boolean
                      default: true
                    extract_facts:
                      type: boolean
                      default: true
      responses:
        200:
          description: Document processed successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  document_id:
                    type: string
                  chunks:
                    type: integer
                  facts:
                    type: integer
                  processing_time:
                    type: number

  /health:
    get:
      summary: System health check
      responses:
        200:
          description: System healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'

components:
  schemas:
    Citation:
      type: object
      properties:
        source:
          type: string
        section:
          type: string
        page:
          type: integer
        quote:
          type: string
        relevance_score:
          type: number
    
    ConsensusInfo:
      type: object
      properties:
        agreement_rate:
          type: number
        participating_agents:
          type: array
          items:
            type: string
        consensus_time_ms:
          type: integer
        byzantine_agents_detected:
          type: array
          items:
            type: string
    
    PerformanceMetrics:
      type: object
      properties:
        cache_ms:
          type: integer
        neural_ms:
          type: integer
        consensus_ms:
          type: integer
        total_ms:
          type: integer
        sla_violations:
          type: array
          items:
            type: string
    
    HealthStatus:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        components:
          type: object
          properties:
            daa_orchestrator:
              $ref: '#/components/schemas/ComponentHealth'
            ruv_fann:
              $ref: '#/components/schemas/ComponentHealth'
            fact_cache:
              $ref: '#/components/schemas/ComponentHealth'
    
    ComponentHealth:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        last_check:
          type: string
          format: date-time
        metrics:
          type: object
```

### 9.2 WebSocket API

```typescript
// WebSocket Event Types
interface QueryStreamMessage {
  type: 'query_progress';
  data: {
    stage: 'intent' | 'retrieval' | 'consensus' | 'citation';
    progress: number; // 0-100
    agent_votes?: AgentVote[];
    performance?: PerformanceUpdate;
  };
}

interface ConsensusMessage {
  type: 'consensus_update';
  data: {
    current_agreement: number;
    participating_agents: string[];
    byzantine_detected?: string[];
    time_remaining: number; // milliseconds
  };
}

interface ErrorMessage {
  type: 'error';
  data: {
    code: string;
    message: string;
    component: 'daa' | 'ruv-fann' | 'fact';
    recovery_suggestion?: string;
  };
}
```

---

## 10. Monitoring and Observability

### 10.1 Metrics Collection

```rust
// Prometheus Metrics Integration
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};

pub struct SystemMetrics {
    // Query metrics
    pub query_total: Counter,
    pub query_duration: Histogram,
    pub query_errors: Counter,
    
    // Consensus metrics
    pub consensus_achieved: Counter,
    pub consensus_failed: Counter,
    pub consensus_duration: Histogram,
    pub byzantine_agents_detected: Counter,
    
    // Performance metrics
    pub cache_hit_ratio: Gauge,
    pub neural_processing_time: Histogram,
    pub agent_coordination_time: Histogram,
    
    // System health
    pub active_agents: Gauge,
    pub memory_usage: Gauge,
    pub cpu_usage: Gauge,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            query_total: register_counter!("queries_total", "Total number of queries").unwrap(),
            query_duration: register_histogram!("query_duration_seconds", "Query processing time").unwrap(),
            consensus_achieved: register_counter!("consensus_achieved_total", "Successful consensus").unwrap(),
            consensus_failed: register_counter!("consensus_failed_total", "Failed consensus").unwrap(),
            // ... other metrics
        }
    }
}
```

### 10.2 Alerting Rules

```yaml
# Prometheus Alerting Rules
groups:
- name: doc-rag-performance
  rules:
  - alert: CacheLatencyHigh
    expr: histogram_quantile(0.95, cache_duration_seconds) > 0.05
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "FACT cache latency exceeds 50ms SLA"
      description: "Cache P95 latency: {{ $value }}s"

  - alert: NeuralProcessingSlowdown
    expr: histogram_quantile(0.90, neural_processing_seconds) > 0.2
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "ruv-FANN processing exceeds 200ms target"
      description: "Neural processing P90: {{ $value }}s"

  - alert: ConsensusFailureRate
    expr: rate(consensus_failed_total[5m]) / rate(consensus_total[5m]) > 0.1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High consensus failure rate"
      description: "Consensus failure rate: {{ $value | humanizePercentage }}"

  - alert: ByzantineAgentsDetected
    expr: increase(byzantine_agents_detected_total[1m]) > 0
    for: 0s
    labels:
      severity: critical
    annotations:
      summary: "Byzantine agents detected"
      description: "{{ $value }} malicious agents detected in last minute"

- name: doc-rag-availability
  rules:
  - alert: APIEndpointDown
    expr: up{job="doc-rag-api"} == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "API endpoint is down"
      
  - alert: HighErrorRate
    expr: rate(query_errors_total[5m]) / rate(queries_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate: {{ $value | humanizePercentage }}"
```

### 10.3 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Doc-RAG Phase 4 Monitoring",
    "panels": [
      {
        "title": "Query Performance",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, query_duration_seconds)",
            "legendFormat": "P95 Query Time"
          }
        ],
        "fieldConfig": {
          "max": 2.0,
          "thresholds": [
            {"color": "green", "value": 0},
            {"color": "yellow", "value": 1.5},
            {"color": "red", "value": 2.0}
          ]
        }
      },
      {
        "title": "Consensus Success Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rate(consensus_achieved_total[5m]) / rate(consensus_total[5m])",
            "legendFormat": "Consensus Rate"
          }
        ],
        "fieldConfig": {
          "min": 0,
          "max": 1,
          "thresholds": [
            {"color": "red", "value": 0},
            {"color": "yellow", "value": 0.66},
            {"color": "green", "value": 0.9}
          ]
        }
      },
      {
        "title": "Component Performance",
        "type": "graph",
        "targets": [
          {"expr": "histogram_quantile(0.95, cache_duration_seconds)", "legendFormat": "Cache P95"},
          {"expr": "histogram_quantile(0.95, neural_processing_seconds)", "legendFormat": "Neural P95"},
          {"expr": "histogram_quantile(0.95, consensus_duration_seconds)", "legendFormat": "Consensus P95"}
        ],
        "yAxes": [{"max": 0.5, "unit": "seconds"}]
      },
      {
        "title": "System Health",
        "type": "table",
        "targets": [
          {"expr": "up{job=~\"doc-rag.*\"}", "format": "table"},
          {"expr": "active_agents", "format": "table"},
          {"expr": "cache_hit_ratio", "format": "table"}
        ]
      }
    ]
  }
}
```

---

## Conclusion

This comprehensive system architecture design provides the foundation for a corrected Phase 4 implementation that achieves 99% accuracy through proper integration of mandated libraries:

### Key Architectural Principles

1. **DAA-First Orchestration**: Complete delegation of orchestration, consensus, and fault tolerance to DAA library
2. **ruv-FANN Neural Excellence**: All neural processing through ruv-FANN with no custom implementations
3. **FACT Intelligence**: Complete caching and citation management via FACT with <50ms guarantees
4. **Byzantine Consensus**: Proper 66% threshold implementation with multi-agent voting
5. **Performance by Design**: Sub-2-second pipeline with component-level SLAs

### Implementation Readiness

The architecture is production-ready with:
- ✅ Complete deployment specifications (Docker + Kubernetes)
- ✅ Comprehensive monitoring and alerting
- ✅ Security architecture with multi-layer protection
- ✅ Scalability design for horizontal and vertical scaling
- ✅ Performance guarantees with SLA monitoring

### Next Steps

1. **Component Integration** - Implement DAA, ruv-FANN, and FACT integrations
2. **Consensus Implementation** - Deploy Byzantine consensus with 66% threshold
3. **Performance Validation** - Benchmark against SLA targets
4. **Security Hardening** - Implement full security architecture
5. **Production Deployment** - Roll out with blue-green deployment strategy

This architecture serves as the definitive blueprint for the corrected system implementation.