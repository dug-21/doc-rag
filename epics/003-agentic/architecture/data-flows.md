# Data Flows: AgentDB + Agentic-Flow + ruv-FANN
## Processing Pipeline and Data Movement Patterns

*Data Flow Diagrams and Processing Pipeline*
*Version 1.0*
*Date: October 23, 2025*

---

## 🎯 Overview

This document details all data flows through the pivot architecture, from document ingestion to query response generation, showing how data moves between AgentDB, agentic-flow, and ruv-FANN.

---

## 📥 Flow 1: Document Ingestion Pipeline

### 1.1 High-Level Ingestion Flow

```
┌──────────────┐
│  PDF Upload  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│    ruv-FANN Document Classifier              │
│    Input: Raw PDF bytes                      │
│    Output: doc_type, confidence              │
└──────┬───────────────────────────────────────┘
       │ doc_type="PCI-DSS", conf=0.94
       ▼
┌──────────────────────────────────────────────┐
│    agentic-flow Ingestion Swarm Init         │
│    • Spawn Extractor Agent                   │
│    • Spawn Chunker Agent                     │
│    • Spawn Embedder Agent                    │
└──────┬───────────────────────────────────────┘
       │
       ├─────────────┬─────────────┬────────────
       ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│Extractor │  │ Chunker  │  │ Embedder │
│  Agent   │  │  Agent   │  │  Agent   │
└──────┬───┘  └────┬─────┘  └────┬─────┘
       │           │             │
       │ sections  │ chunks      │ embeddings
       ▼           ▼             ▼
┌──────────────────────────────────────────────┐
│    Data Aggregation & Enrichment             │
│    • Merge extraction results                │
│    • ruv-FANN section classification         │
│    • Feature extraction                      │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│    AgentDB Storage                           │
│    • Vector storage (HNSW indexed)           │
│    • Metadata storage (filterable)           │
│    • Session creation (document context)     │
│    • Learning plugin init (domain specific)  │
└──────────────────────────────────────────────┘
```

### 1.2 Detailed Data Transformations

```rust
// Step 1: PDF → Raw Text
PDFBytes (binary)
  ↓ [pdf_extractor]
RawText {
  pages: Vec<PageContent>,
  metadata: PDFMetadata
}

// Step 2: Raw Text → Document Type
RawText
  ↓ [ruv-FANN doc_classifier]
Classification {
  doc_type: "PCI-DSS",
  confidence: 0.94,
  features: Vec<f32>[1536]
}

// Step 3: Document → Sections (Extractor Agent)
RawText
  ↓ [structure_extractor]
Hierarchy {
  sections: Vec<Section> {
    id: "3.2.1",
    title: "Encryption Requirements",
    content: "...",
    level: 3,
    page_range: (45, 48)
  }
}

// Step 4: Sections → Chunks (Chunker Agent)
Section
  ↓ [intelligent_chunker]
Chunks: Vec<Chunk> {
  id: UUID,
  text: String[500 chars],
  overlap: 50,
  metadata: {
    section_id: "3.2.1",
    page: 45,
    start_char: 0,
    end_char: 500
  }
}

// Step 5: Chunks → Classified Chunks (ruv-FANN)
Chunk
  ↓ [ruv-FANN section_classifier]
ClassifiedChunk {
  chunk: Chunk,
  chunk_type: "requirement",  // requirement|definition|procedure|exception
  confidence: 0.91,
  features: Vec<f32>[512]
}

// Step 6: Chunks → Embeddings (Embedder Agent)
ClassifiedChunk
  ↓ [embedding_model]
EmbeddedChunk {
  chunk: ClassifiedChunk,
  embedding: Vec<f32>[1536],
  model: "text-embedding-ada-002"
}

// Step 7: Embedded Chunks → AgentDB Storage
EmbeddedChunk
  ↓ [agentdb.insert]
StoredDocument {
  id: UUID,
  vector: Vec<f32>[1536],  // HNSW indexed
  metadata: {
    document: "PCI-DSS-v4.0.pdf",
    section: "3.2.1",
    chunk_type: "requirement",
    page: 45,
    confidence: 0.91,
    timestamp: DateTime
  },
  payload: String  // Original text
}
```

### 1.3 Parallel Processing Flow

```
Document Upload
      │
      ├──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
  Section 1      Section 2      Section 3      Section 4
      │              │              │              │
[Agent 1]      [Agent 2]      [Agent 3]      [Agent 4]
      │              │              │              │
   Chunk         Chunk          Chunk          Chunk
      │              │              │              │
   Embed         Embed          Embed          Embed
      │              │              │              │
   Store         Store          Store          Store
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                            │
                   Merge & Complete
                            │
                ┌───────────▼───────────┐
                │   AgentDB Collection   │
                │   1,247 chunks stored  │
                │   Session created      │
                │   Patterns initialized │
                └────────────────────────┘
```

---

## 🔍 Flow 2: Query Processing Pipeline

### 2.1 High-Level Query Flow

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │ "What encryption is required for stored cardholder data?"
       ▼
┌──────────────────────────────────────────────┐
│    ruv-FANN Query Classifier                 │
│    Input: Query text                         │
│    Output: query_type, complexity, routing   │
└──────┬───────────────────────────────────────┘
       │ type="requirement_lookup", complexity="moderate"
       ▼
┌──────────────────────────────────────────────┐
│    agentic-flow Query Swarm Init             │
│    • Topology: mesh (based on complexity)    │
│    • Spawn: retrieval, reasoning, synthesis  │
└──────┬───────────────────────────────────────┘
       │
       ├──────────────────┬─────────────────────
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  Retrieval   │   │  Retrieval   │
│  Agent #1    │   │  Agent #2    │
│  (HNSW)      │   │  (Hybrid)    │
└──────┬───────┘   └──────┬───────┘
       │                  │
       │ docs[20]         │ docs[18]
       │                  │
       └──────────┬───────┘
                  │ merged & deduped: docs[28]
                  ▼
           ┌──────────────┐
           │   ruv-FANN   │
           │   Relevance  │
           │   Scoring    │
           └──────┬───────┘
                  │ scored_docs[28]
                  ▼
           ┌──────────────┐
           │  Reasoning   │
           │   Agent      │
           │ (+ patterns) │
           └──────┬───────┘
                  │ reasoning_result
                  ▼
           ┌──────────────┐
           │  Synthesis   │
           │   Agent      │
           │ (templates)  │
           └──────┬───────┘
                  │ draft_response
                  ▼
           ┌──────────────┐
           │Verification  │
           │   Agent      │
           │ (accuracy)   │
           └──────┬───────┘
                  │ verified_response (0.98 accuracy)
                  ▼
┌──────────────────────────────────────────────┐
│    AgentDB Learning Update                   │
│    • Record trajectory                       │
│    • Update plugins                          │
│    • Train if batch ready                    │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Response   │
│   to User    │
└──────────────┘
```

### 2.2 Detailed Query Data Transformations

```rust
// Step 1: Query → Classification
Query: "What encryption is required for stored cardholder data?"
  ↓ [ruv-FANN query_classifier]
QueryAnalysis {
  query_type: RequirementLookup,
  complexity: Moderate,
  confidence: 0.92,
  features: Vec<f32>[768],
  recommended_agents: ["retrieval", "reasoning", "synthesis"],
  recommended_topology: Mesh
}

// Step 2: Query → Embeddings
Query
  ↓ [embedding_model]
QueryEmbedding: Vec<f32>[1536]

// Step 3: Query Embedding → AgentDB Search (Agent #1: HNSW)
QueryEmbedding
  ↓ [agentdb.search_hnsw]
SearchResults {
  documents: Vec<Document>[20],
  scores: Vec<f32>[20],
  strategy: "hnsw",
  duration_ms: 45
}

// Step 4: Query Embedding + Filters → AgentDB Search (Agent #2: Hybrid)
QueryEmbedding + Filter { chunk_type: "requirement", confidence > 0.8 }
  ↓ [agentdb.search_hybrid]
SearchResults {
  documents: Vec<Document>[18],
  scores: Vec<f32>[18],
  strategy: "hybrid",
  duration_ms: 78
}

// Step 5: Multiple Results → Merged Results
[SearchResults(HNSW), SearchResults(Hybrid)]
  ↓ [result_merger.dedupe]
MergedResults {
  documents: Vec<Document>[28],  // 10 overlapping deduplicated
  sources: ["hnsw", "hybrid"]
}

// Step 6: Query + Documents → Relevance Scores
(Query, MergedResults)
  ↓ [ruv-FANN relevance_scorer]
ScoredDocuments {
  docs: Vec<(Document, f32)>[28],
  top_10: Vec<(Document, f32)> {
    (doc_1, 0.96),
    (doc_2, 0.94),
    (doc_3, 0.92),
    ...
  }
}

// Step 7: Query + Scored Docs → Learned Patterns
(Query, ScoredDocuments)
  ↓ [agentdb.query_patterns]
LearnedPatterns {
  patterns: Vec<Pattern> {
    CrossReference("3.2.1" → "3.4.1", conf: 0.85),
    ConceptCluster("encryption" + "cardholder", conf: 0.91)
  }
}

// Step 8: All Context → Reasoning
(Query, ScoredDocuments, LearnedPatterns)
  ↓ [reasoning_agent + ruv-FANN inference]
ReasoningResult {
  summary: "PCI-DSS 3.2.1 requires AES-256 encryption for stored cardholder data",
  evidence: Vec<Evidence>[5],
  confidence: 0.95,
  reasoning_chain: Vec<Step> {
    "Requirement 3.2.1 explicitly states...",
    "This applies to stored data as per 3.1...",
    "Exceptions exist in 3.2.3 for temporary storage"
  }
}

// Step 9: Reasoning → Structured Response
ReasoningResult
  ↓ [synthesis_agent + templates]
DraftResponse {
  answer: "According to PCI-DSS 3.2.1...",
  citations: Vec<Citation>[3],
  confidence: 0.95,
  metadata: ResponseMetadata
}

// Step 10: Draft → Verification
DraftResponse
  ↓ [verification_agent]
VerificationResult {
  passed: true,
  accuracy: 0.98,
  checks: {
    citation_accuracy: 1.0,
    logical_consistency: 0.97,
    completeness: 0.96
  }
}

// Step 11: Verified Response + Context → Learning Trajectory
(Query, Strategy, Response, Accuracy)
  ↓ [agentdb.record_trajectory]
Trajectory {
  state: {
    query_type: "requirement_lookup",
    complexity: "moderate"
  },
  action: {
    strategy: "hnsw + hybrid",
    agents: ["retrieval_1", "retrieval_2", "reasoning", "synthesis"]
  },
  reward: 0.98,
  next_state: {
    success: true,
    user_feedback: 0.95
  }
}

// Step 12: Trajectory → Plugin Training
Trajectory
  ↓ [agentdb.train_plugin (if batch ready)]
TrainingResult {
  plugin_id: "query_routing",
  episodes_trained: 100,
  avg_reward_improvement: +0.03,
  new_policy: "prefer hybrid for moderate complexity requirements"
}
```

### 2.3 Parallel Retrieval Flow

```
Query Embedding
      │
      ├──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
   HNSW         Hybrid        ReRank       GraphWalk
   Search       Search        Search        Search
      │              │              │              │
[AgentDB]      [AgentDB]     [AgentDB]     [AgentDB]
 + HNSW        + Filters     + Broad      + Patterns
      │              │              │              │
  docs[20]       docs[18]       docs[15]       docs[12]
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                            │
                      Merge & Dedupe
                            │
                        docs[35]
                            │
                   ┌────────▼────────┐
                   │  ruv-FANN Score │
                   └────────┬────────┘
                            │
                     scored_docs[35]
                            │
                     top_k(10) → [10 best docs]
```

---

## 🔄 Flow 3: Learning Feedback Loop

### 3.1 Continuous Learning Flow

```
┌─────────────────────────────────────────────────────┐
│                  Query Execution                     │
│  Query → Classify → Retrieve → Reason → Synthesize  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  User Feedback │  (optional: rating, correction)
            └────────┬───────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         Trajectory Recording (AgentDB)               │
│  • State: query_type, complexity, context           │
│  • Action: strategy, agents, topology               │
│  • Reward: accuracy_score, user_feedback            │
│  • Next State: success, metrics                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ Trajectory Store│
            │   (buffer)      │
            └────────┬───────┘
                     │
                     ▼ (when batch size = 100)
┌─────────────────────────────────────────────────────┐
│         Plugin Training Trigger                      │
│  • Extract batch of 100 trajectories                │
│  • Prepare training data                            │
│  • Run RL algorithm (Decision Transformer, etc.)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         Updated Policy (AgentDB)                     │
│  • New routing strategy                             │
│  • Improved relevance scoring                       │
│  • Better context management                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Next Query    │  (uses improved policy)
            └────────────────┘
```

### 3.2 Learning Data Flow

```rust
// Interaction → Trajectory
Interaction {
  query: "What is required for password policies?",
  query_analysis: QueryAnalysis { ... },
  strategy_used: "hybrid",
  agents_used: ["retrieval", "reasoning"],
  response: Response { ... },
  accuracy_score: 0.97,
  user_feedback: Some(0.95)
}
  ↓ [trajectory_builder]
Trajectory {
  state: json!({
    "query_type": "requirement_lookup",
    "complexity": "simple",
    "has_context": false
  }),
  action: json!({
    "strategy": "hybrid",
    "num_agents": 2
  }),
  reward: 0.97,  // accuracy_score
  next_state: json!({
    "success": true,
    "user_satisfaction": 0.95
  })
}

// Batch of Trajectories → Training Data
Vec<Trajectory>[100]
  ↓ [training_data_preparer]
TrainingBatch {
  states: Tensor[100, state_dim],
  actions: Tensor[100, action_dim],
  rewards: Tensor[100, 1],
  next_states: Tensor[100, state_dim],
  dones: Tensor[100, 1]
}

// Training Data → Model Update
TrainingBatch
  ↓ [rl_algorithm.train]
ModelUpdate {
  old_policy_loss: 0.234,
  new_policy_loss: 0.187,
  value_loss: 0.145,
  improvement: +0.047
}

// Updated Model → New Policy
ModelUpdate
  ↓ [policy_extractor]
NewPolicy {
  query_routing_rules: {
    "simple requirement" → "hnsw",
    "moderate requirement" → "hybrid",
    "complex requirement" → "graph_walk"
  },
  confidence_threshold: 0.85
}
```

---

## 🧠 Flow 4: Memory Management

### 4.1 Session Memory Flow

```
┌──────────────┐
│  Session 1   │
│  (User 123)  │
└──────┬───────┘
       │
   Query 1: "What is PCI-DSS?"
       │
       ▼
┌──────────────────────────────────┐
│   AgentDB Session Creation        │
│   session_id: "user_123"         │
│   type: LongTerm                 │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Session Context Storage         │
│   context: {                     │
│     queries: ["What is PCI-DSS?"]│
│     concepts: ["pci-dss"]        │
│     timestamp: t0                │
│   }                              │
└──────┬───────────────────────────┘
       │
   Query 2: "What about encryption?"
       │
       ▼
┌──────────────────────────────────┐
│   Context-Enhanced Query          │
│   • Retrieve session context     │
│   • Infer: "encryption in PCI-DSS"│
│   • Expand query with context    │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Enhanced Search                 │
│   query: "PCI-DSS encryption"    │
│   filter: { domain: "pci-dss" }  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Update Session Context          │
│   context: {                     │
│     queries: [                   │
│       "What is PCI-DSS?",        │
│       "What about encryption?"   │
│     ],                           │
│     concepts: ["pci-dss",        │
│                "encryption"],    │
│     relationships: [             │
│       ("pci-dss", "encryption")  │
│     ]                            │
│   }                              │
└──────────────────────────────────┘
```

### 4.2 Memory Consolidation Flow

```
Session End (or periodic consolidation)
      │
      ▼
┌─────────────────────────────────────┐
│   Extract Session Interactions       │
│   • All queries                     │
│   • All retrieved documents         │
│   • All user feedback               │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Importance Scoring                 │
│   • Frequency weight: 0.3           │
│   • Recency weight: 0.2             │
│   • Relevance weight: 0.5           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Memory Selection                   │
│   • Keep high-importance items      │
│   • Discard low-importance items    │
│   • Compress medium-importance      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Consolidated Memory                │
│   • Essential concepts: 12          │
│   • Key relationships: 8            │
│   • Important queries: 4            │
│   Size: 2.3 KB (was 15 KB)         │
└─────────────────────────────────────┘
```

---

## 📊 Flow 5: Monitoring and Observability

### 5.1 Metrics Collection Flow

```
Every Query Execution
      │
      ├─────────────────┬─────────────────┬────────────────
      ▼                 ▼                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Latency  │    │Accuracy  │    │  Cost    │
│ Tracking │    │Tracking  │    │Tracking  │
└──────┬───┘    └────┬─────┘    └────┬─────┘
       │             │                │
       │ 420ms       │ 0.98           │ $0.0008
       │             │                │
       └─────────────┴────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Metrics Store  │
            │   (AgentDB)     │
            └────────┬───────┘
                     │
                     ▼ (every 100 queries)
            ┌────────────────┐
            │   Aggregation   │
            │   & Analysis    │
            └────────┬───────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│        Performance Dashboard             │
│  • Avg Latency: 385ms (P95: 512ms)     │
│  • Avg Accuracy: 0.976                  │
│  • Avg Cost: $0.00075/query            │
│  • Throughput: 145 queries/min         │
└─────────────────────────────────────────┘
                     │
                     ▼ (if threshold breached)
            ┌────────────────┐
            │     Alerts      │
            └────────────────┘
```

---

## 🎯 Data Flow Performance Metrics

| Flow | Latency (P50) | Latency (P95) | Throughput |
|------|---------------|---------------|------------|
| **Document Ingestion** | 2.5 pages/sec | 1.8 pages/sec | ~150 pages/min |
| **Query Classification** | 12ms | 18ms | 5000 queries/sec |
| **Vector Search (HNSW)** | 45ms | 78ms | 1200 queries/sec |
| **Relevance Scoring** | 85ms | 142ms | 650 queries/sec |
| **Reasoning** | 120ms | 205ms | 450 queries/sec |
| **Synthesis** | 35ms | 58ms | 1500 queries/sec |
| **Verification** | 65ms | 98ms | 850 queries/sec |
| **End-to-End Query** | 320ms | 490ms | 180 queries/sec |
| **Learning Update** | 15ms | 28ms | async (background) |

---

## 🔄 Data Flow Optimization Strategies

### 1. Caching Strategy

```
Query → [Cache Check] → Cache Hit? → Return Cached
                ↓ No
        [Full Pipeline] → Cache Result → Return
```

### 2. Batch Processing

```
Multiple Queries → [Batch] → Single Embedding Call → [Unbatch] → Individual Results
```

### 3. Pipeline Parallelism

```
Query → [Stage 1: Classify] ────┐
                                 ├→ [Stage 2: Retrieve]
        [Stage 1: Classify] ────┘
```

### 4. Result Streaming

```
Query → [Retrieve] → [Stream Results] → [Reason on Partial] → [Early Response]
                              ↓
                     [Continue Background Processing]
```

---

## ✅ Data Flow Validation Checkpoints

| Checkpoint | Validation | Threshold |
|------------|------------|-----------|
| **After Classification** | Confidence score | > 0.85 |
| **After Retrieval** | Num results | ≥ 5 docs |
| **After Scoring** | Top doc score | > 0.8 |
| **After Reasoning** | Evidence count | ≥ 3 sources |
| **After Synthesis** | Citation count | ≥ 1 citation |
| **After Verification** | Accuracy score | ≥ 0.97 |

---

## 🎬 Conclusion

The data flows in this architecture are designed for:

1. **Low Latency**: <500ms end-to-end through parallel processing
2. **High Accuracy**: >97% through multi-stage verification
3. **Continuous Learning**: Feedback loops improve over time
4. **Cost Efficiency**: Caching and batch processing reduce costs
5. **Observability**: Comprehensive metrics at every stage

All data flows are instrumented for monitoring and can be optimized based on production metrics.

---

*Data Flow Documentation by System Architecture Designer*
*Version 1.0 - Implementation Ready*
