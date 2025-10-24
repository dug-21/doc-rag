# SPARC SPECIFICATION: PCI-DSS RAG System v1.0
## Agentic Pivot Architecture with AgentDB + agentic-flow + ruv-FANN

**Project:** Technical Standards RAG System
**Version:** 1.0 (Pivot Architecture)
**Date:** October 24, 2025
**Document Type:** SPARC Phase 1 - Specification
**Status:** Draft for Review

---

## 1. Executive Summary

### 1.1 Project Overview

Build a TypeScript-based Retrieval-Augmented Generation (RAG) system achieving **>97% accuracy** for PCI-DSS compliance queries through adaptive learning and multi-agent orchestration.

**Core Stack:**
- **AgentDB**: Vector database with HNSW indexing and 9 RL algorithms
- **agentic-flow**: Multi-agent orchestration framework
- **ruv-FANN**: WASM neural networks for classification

**Key Differentiators:**
- Self-learning system (improves from every query)
- Single unified database (vs. 4 separate systems)
- 150x faster search with HNSW
- <500ms P95 latency
- <$0.001 per query cost

### 1.2 Budget & Timeline

| Category | Estimate |
|----------|----------|
| **Implementation** | $216K-$288K |
| **Infrastructure Setup** | $20K |
| **Training Data** | $3K |
| **Total Budget** | **$239K-$311K** |
| **Timeline** | **12 weeks** (18-22 weeks w/ buffer) |
| **Annual Operating Cost** | $15K/year |

### 1.3 Success Metrics

| Metric | Baseline (No RL) | Target (With RL) |
|--------|------------------|------------------|
| Accuracy | >85% | >97% |
| P95 Latency | <1000ms | <500ms |
| Cost per Query | <$0.01 | <$0.001 |
| Learning Convergence | N/A | <1,500 queries |

---

## 2. Functional Requirements

### FR1: Document Ingestion
**Priority:** Critical

**Description:** Process technical standard PDFs (PCI-DSS, HIPAA, SOC2) into searchable vector embeddings with rich metadata.

**Capabilities:**
- FR1.1: PDF parsing with structure extraction
- FR1.2: Intelligent chunking (semantic boundaries)
- FR1.3: Embedding generation (1536-dim vectors)
- FR1.4: Metadata extraction (requirements, sections, cross-refs)
- FR1.5: Storage in AgentDB with HNSW indexing

**Acceptance Criteria:**
```gherkin
Given a PCI-DSS PDF (300+ pages)
When ingestion pipeline processes document
Then system shall:
  - Extract all sections with hierarchy
  - Generate 2,000-3,000 chunks
  - Produce embeddings within 5 minutes
  - Store with 100% metadata accuracy
  - Enable HNSW search with <100ms latency
```

**Performance:**
- Throughput: 2-5 pages/sec
- Storage: ~4MB per 1,000 chunks (with quantization)
- Success rate: >99% chunk extraction

---

### FR2: Query Processing
**Priority:** Critical

**Description:** Multi-agent query processing with adaptive routing and strategy selection.

**Capabilities:**
- FR2.1: Query analysis (classification via ruv-FANN)
- FR2.2: Multi-strategy retrieval (HNSW, hybrid, rerank, graph-walk)
- FR2.3: Parallel agent coordination (agentic-flow)
- FR2.4: Context-aware search (session memory)
- FR2.5: Real-time performance monitoring

**Query Types Supported:**
| Type | Example | Strategy |
|------|---------|----------|
| **Simple** | "What is PCI-DSS requirement 3.2?" | HNSW direct |
| **Moderate** | "How do encryption requirements differ?" | Hybrid search |
| **Complex** | "Map HIPAA to PCI-DSS requirements" | Graph-walk + ensemble |

**Acceptance Criteria:**
```gherkin
Scenario: Simple query processing
  Given query "What is PCI-DSS requirement 3.2?"
  When query processing swarm executes
  Then system shall:
    - Classify query type within 50ms
    - Retrieve top 20 chunks within 200ms
    - Generate response within 500ms (P95)
    - Achieve >97% accuracy

Scenario: Complex cross-standard query
  Given query "Compare encryption requirements: PCI-DSS vs HIPAA"
  When multi-agent swarm processes query
  Then system shall:
    - Use adaptive topology (hierarchical)
    - Spawn 4-6 specialized agents
    - Execute graph-walk retrieval
    - Complete within 800ms (P95)
    - Provide citations from both standards
```

**Performance Targets:**
- P50 latency: <300ms
- P95 latency: <500ms
- P99 latency: <1000ms
- Concurrent queries: 100+

---

### FR3: Multi-Agent Orchestration
**Priority:** Critical

**Description:** Dynamic swarm coordination using agentic-flow with adaptive topology selection.

**Agent Types:**
| Agent | Role | Framework |
|-------|------|-----------|
| **Query Analyzer** | Classify queries, extract intent | ruv-FANN |
| **Retrieval Agent** | Vector search, filtering | AgentDB |
| **Reasoning Agent** | Pattern matching, inference | ruv-FANN + AgentDB |
| **Synthesis Agent** | Evidence aggregation | Template-based |
| **Verification Agent** | Accuracy validation | Multi-check |

**Topologies:**
- **Mesh** (default): Peer-to-peer for simple queries
- **Hierarchical**: Coordinator + workers for complex queries
- **Star**: Single coordinator for moderate queries
- **Adaptive**: Auto-select based on complexity

**Acceptance Criteria:**
```gherkin
Scenario: Topology adaptation
  Given query complexity score > 0.7
  When swarm initializes
  Then topology shall be "hierarchical"
  And agent count shall be 4-6

Scenario: Agent coordination
  Given 3 parallel retrieval tasks
  When orchestration begins
  Then all tasks execute concurrently
  And results aggregate within 500ms
```

---

### FR4: Vector Search (HNSW)
**Priority:** Critical

**Description:** High-performance vector similarity search using HNSW algorithm in AgentDB.

**Capabilities:**
- FR4.1: HNSW index construction (M=16, ef_construction=200)
- FR4.2: Fast approximate nearest neighbor search
- FR4.3: Hybrid search (vector + metadata filters)
- FR4.4: Scalar quantization (4x memory reduction)
- FR4.5: Cache-optimized lookup

**Performance Specifications:**
| Metric | Target | Method |
|--------|--------|--------|
| **Search Speed** | <100µs | HNSW indexing |
| **Recall@20** | >97% | Optimized ef_search=100 |
| **Memory Usage** | 4x reduction | Scalar quantization |
| **Index Build Time** | <10min/million | Parallel construction |

**Acceptance Criteria:**
```gherkin
Scenario: HNSW search performance
  Given AgentDB with 100,000 vectors
  When query vector searches collection
  Then search completes within 100µs
  And recall@20 exceeds 97%
  And memory usage < 400MB
```

---

### FR5: Reinforcement Learning
**Priority:** High

**Description:** Continuous learning through 9 RL algorithms in AgentDB learning plugins.

**Learning Plugins:**

| Plugin | Algorithm | Purpose | State Space | Action Space |
|--------|-----------|---------|-------------|--------------|
| **Query Routing** | Decision Transformer | Route queries optimally | Query type, complexity | Strategy, topology, agents |
| **Relevance Scoring** | Actor-Critic | Rank document relevance | Query+doc embeddings | Relevance score [0,1] |
| **Context Management** | Q-Learning | Manage session memory | Session length, similarity | Use memory, window size |

**Learning Process:**
1. **Trajectory Recording**: Capture state → action → reward → next_state
2. **Batch Training**: Train after 100 interactions (or real-time)
3. **Policy Update**: Update routing/scoring policies
4. **Evaluation**: A/B test against baseline

**Acceptance Criteria:**
```gherkin
Scenario: Learning improvement
  Given system processes 1,000 queries
  When RL plugins train on trajectories
  Then accuracy improvement shall be +2-5%
  And latency improvement shall be +10-20%
  And convergence achieved within 1,500 queries

Scenario: Trajectory recording
  Given successful query interaction
  When interaction completes
  Then trajectory stored in AgentDB
  And reward calculated from accuracy score
  And plugin training triggered if batch full
```

**Expected Learning Curve:**
- Week 1: 88-90% accuracy
- Week 4: 92-94% accuracy
- Week 12: 95-97% accuracy
- Month 3: >97% accuracy (stable)

---

### FR6: Response Generation with Citations
**Priority:** Critical

**Description:** Evidence-based response synthesis with full citation chains and confidence scoring.

**Capabilities:**
- FR6.1: Evidence aggregation (top-k retrieval)
- FR6.2: Citation extraction (document, page, section)
- FR6.3: Confidence scoring (neural + metadata)
- FR6.4: Template-based formatting
- FR6.5: Cross-reference validation

**Response Structure:**
```json
{
  "summary": "PCI-DSS Requirement 3.2 mandates...",
  "evidence": [
    {
      "text": "Cardholder data must be encrypted...",
      "source": "PCI-DSS v4.0",
      "section": "Requirement 3.2",
      "page": 45,
      "confidence": 0.98
    }
  ],
  "citations": [
    "[1] PCI-DSS v4.0, Requirement 3.2, p.45",
    "[2] PCI-DSS v4.0, Requirement 3.3, p.47"
  ],
  "confidence_score": 0.98,
  "metadata": {
    "query_type": "requirement_lookup",
    "retrieval_strategy": "hnsw",
    "processing_time_ms": 342
  }
}
```

**Acceptance Criteria:**
```gherkin
Scenario: Response with citations
  Given query retrieves 20 relevant chunks
  When synthesis agent generates response
  Then response shall include:
    - Summary (1-3 sentences)
    - Top 3-5 evidence excerpts
    - Citations with [document, section, page]
    - Confidence score >0.95
    - Metadata (query type, strategy, timing)
```

---

### FR7: Accuracy Validation
**Priority:** Critical

**Description:** Multi-layered verification ensuring >97% accuracy before response delivery.

**Verification Checks:**
| Check | Method | Weight | Threshold |
|-------|--------|--------|-----------|
| **Citation Accuracy** | Cross-reference validation | 30% | >95% |
| **Logical Consistency** | Contradiction detection | 25% | No conflicts |
| **Completeness** | Coverage analysis | 25% | All aspects addressed |
| **Cross-Reference** | Relationship validation | 20% | Links verified |

**Verification Flow:**
```
Response → 4 Parallel Checks → Aggregate Scores → Pass/Fail → Retry or Deliver
```

**Acceptance Criteria:**
```gherkin
Scenario: Verification pass
  Given response with confidence 0.98
  When verification agent runs checks
  Then all checks score >0.95
  And aggregate accuracy >0.97
  And response delivered to user

Scenario: Verification fail
  Given response with confidence 0.92
  When verification agent runs checks
  Then aggregate accuracy <0.97
  And alternative strategy suggested
  And retry triggered with different approach
```

---

## 3. Non-Functional Requirements

### NFR1: Performance

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| **Latency P50** | <300ms | Query start → response delivery |
| **Latency P95** | <500ms | 95th percentile queries |
| **Latency P99** | <1000ms | 99th percentile queries |
| **Throughput** | 100+ concurrent | Concurrent query handling |
| **Cost** | <$0.001/query | Infrastructure + API costs |
| **Memory** | <4GB/million vectors | With scalar quantization |

**Load Testing:**
- Baseline: 10 concurrent queries @ <300ms
- Stress: 100 concurrent queries @ <500ms
- Peak: 200 concurrent queries @ <1000ms

---

### NFR2: Scalability

**Horizontal Scaling:**
- AgentDB: Sharding support for >10M vectors
- agentic-flow: Dynamic agent spawning (1-20 agents)
- Stateless design: Load balancer compatible

**Vertical Scaling:**
- Memory: 8GB → 32GB (4x capacity)
- CPU: 4 cores → 16 cores (4x throughput)
- Storage: 100GB → 1TB (10x documents)

**Growth Targets:**
| Metric | Current | 1 Year | 3 Years |
|--------|---------|--------|---------|
| Documents | 5 standards | 20 standards | 100 standards |
| Vectors | 100K | 1M | 10M |
| Users | 10 | 100 | 1,000 |
| Queries/day | 1,000 | 10,000 | 100,000 |

---

### NFR3: Reliability

**Uptime:** 99.9% (8.76 hours downtime/year)

**Failure Handling:**
- **Database Failure**: Read replica failover (<30s)
- **Agent Failure**: Retry with alternative agent (<5s)
- **API Failure**: Exponential backoff with circuit breaker
- **Network Failure**: Request queue with replay

**Monitoring & Alerting:**
- Health checks: Every 30 seconds
- Performance metrics: Real-time dashboard
- Error tracking: Full stack traces
- Alerts: Slack/PagerDuty on critical errors

---

### NFR4: Security

**Data Protection:**
- Encryption at rest: AES-256
- Encryption in transit: TLS 1.3
- API authentication: JWT tokens
- Role-based access control (RBAC)

**Compliance:**
- No PII storage (document content only)
- Audit logging (all queries recorded)
- Data retention: 90 days (configurable)

---

### NFR5: Observability

**Metrics Collection:**
- Query latency (P50/P95/P99)
- Accuracy scores (per query type)
- Agent performance (by agent type)
- Learning convergence (RL plugin metrics)
- Cost per query (infrastructure)

**Logging:**
- Structured JSON logs
- Query traces (OpenTelemetry)
- Error tracking (Sentry)
- Performance profiling

**Dashboards:**
- Real-time query metrics
- Learning progress visualization
- Cost breakdown
- System health overview

---

## 4. System Context

### 4.1 External Systems

**AgentDB** (Vector Database)
- **Purpose**: Unified storage for vectors, memory, learning
- **Interface**: Rust/TypeScript client library
- **Data Flow**: Embeddings → HNSW index → Search results

**agentic-flow** (Orchestration)
- **Purpose**: Multi-agent swarm coordination
- **Interface**: TypeScript SDK
- **Data Flow**: Query → Agent tasks → Aggregated results

**ruv-FANN** (Neural Networks)
- **Purpose**: Fast classification and inference
- **Interface**: WASM module
- **Data Flow**: Input features → Neural net → Predictions

**OpenAI API** (Embeddings)
- **Purpose**: Generate 1536-dim embeddings
- **Interface**: REST API
- **Data Flow**: Text → Embedding vector

### 4.2 User Personas

**Compliance Officer**
- **Goal**: Quickly find PCI-DSS requirements
- **Frequency**: 10-20 queries/day
- **Expertise**: Domain expert, technical

**Security Auditor**
- **Goal**: Cross-reference requirements across standards
- **Frequency**: 5-10 queries/day
- **Expertise**: High technical, multi-standard

**Developer**
- **Goal**: Understand implementation requirements
- **Frequency**: 3-5 queries/day
- **Expertise**: Technical, needs examples

### 4.3 Use Cases

**UC1: Simple Requirement Lookup**
```
User: "What is PCI-DSS requirement 3.2?"
System: Direct HNSW search → Template response
Time: <300ms
```

**UC2: Comparative Analysis**
```
User: "Compare encryption requirements in PCI-DSS vs HIPAA"
System: Multi-agent coordination → Graph-walk → Synthesis
Time: <800ms
```

**UC3: Contextual Follow-up**
```
User: "What are the encryption algorithms?"
Context: Previous query about PCI-DSS 3.2
System: Session memory → Context-aware search → Response
Time: <400ms
```

---

## 5. Constraints

### 5.1 Technology Constraints
- **Language**: TypeScript only (no Rust implementation)
- **Database**: AgentDB required (no alternatives)
- **Framework**: agentic-flow for orchestration
- **Neural Nets**: ruv-FANN WASM (no TensorFlow/PyTorch)
- **Embeddings**: OpenAI API (or compatible)

### 5.2 Resource Constraints
- **Budget**: $239K-$311K implementation + $15K/year operating
- **Timeline**: 12 weeks base, 18-22 weeks with buffer
- **Team**: 3-4 engineers (2 TypeScript, 1 ML, 1 DevOps)
- **Infrastructure**: Single-region AWS/GCP deployment

### 5.3 Business Constraints
- **Primary Use Case**: PCI-DSS (other standards secondary)
- **Accuracy**: >97% non-negotiable for compliance queries
- **Explainability**: Full citation chains required
- **No SaaS**: Self-hosted deployment preferred

### 5.4 Operational Constraints
- **Uptime**: 99.9% during business hours
- **Maintenance Window**: Weekends only
- **Support**: Email support, no 24/7 on-call

---

## 6. Success Criteria

### 6.1 Baseline Metrics (Without RL)

**Phase 1 Target** (Week 4)
| Metric | Target | Validation |
|--------|--------|------------|
| Accuracy | >85% | 200 test queries |
| P95 Latency | <1000ms | Load test |
| Cost | <$0.01/query | Cost tracking |
| Recall@20 | >95% | Retrieval eval |

**Go/No-Go Decision:** If baseline <85%, pivot strategy.

---

### 6.2 Final Metrics (With RL)

**Phase 6 Target** (Week 12)
| Metric | Target | Validation |
|--------|--------|------------|
| **Accuracy** | **>97%** | 980-1,490 test questions |
| **P95 Latency** | **<500ms** | 1,000 query load test |
| **Cost** | **<$0.001/query** | 1 month production data |
| **Learning** | **+2-5% improvement** | Before/after RL comparison |
| **Uptime** | **99.9%** | 1 month monitoring |

**Success Criteria:**
- ✅ All metrics met → Production deployment
- ⚠️ 1 metric missed → 2-week remediation
- ❌ 2+ metrics missed → Architecture review

---

### 6.3 Learning Convergence

**Convergence Definition:** Accuracy improvement plateaus (<0.5% gain over 500 queries)

**Expected Convergence:**
- **Optimistic**: 1,000 queries (2-3 weeks)
- **Realistic**: 1,500 queries (3-4 weeks)
- **Pessimistic**: 2,500 queries (5-6 weeks)

**Monitoring:**
- Track accuracy per 100-query batch
- Plot learning curve
- Detect plateau with moving average

---

## 7. Acceptance Criteria

### 7.1 Test Question Bank

**Size:** 980-1,490 questions (stratified by complexity)

| Complexity | Count | Pass Rate |
|------------|-------|-----------|
| Simple | 400-600 | >98% |
| Moderate | 400-600 | >97% |
| Complex | 180-290 | >95% |
| **Total** | **980-1,490** | **>97%** |

**Question Types:**
- **Requirement Lookup** (40%): "What is requirement X?"
- **Comparative** (30%): "Compare X and Y"
- **Procedural** (20%): "How do I implement X?"
- **Exception** (10%): "When doesn't X apply?"

---

### 7.2 Testing Phases

**Phase 1: Unit Tests** (Week 6)
- Component-level validation
- 100+ unit tests per module
- Coverage: >80%

**Phase 2: Integration Tests** (Week 8)
- End-to-end query flow
- 50+ integration scenarios
- All agent types tested

**Phase 3: Accuracy Validation** (Week 10)
- Baseline test (200 questions)
- Target: >85% without RL
- Go/No-Go decision point

**Phase 4: Production Validation** (Week 12)
- Full test bank (980-1,490 questions)
- Target: >97% with RL
- Load testing (100 concurrent)

**Go/No-Go Gates:**
| Gate | Criteria | Action if Failed |
|------|----------|------------------|
| **Phase 3** | >85% accuracy | 2-week remediation or pivot |
| **Phase 4** | >97% accuracy | 2-week remediation or soft launch |

---

### 7.3 Production Readiness Checklist

**Technical:**
- [ ] All tests passing (>97% accuracy)
- [ ] Performance validated (<500ms P95)
- [ ] Load testing complete (100 concurrent)
- [ ] Security audit passed
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery tested

**Operational:**
- [ ] Documentation complete
- [ ] Runbooks created
- [ ] Training provided to users
- [ ] Support process defined
- [ ] Rollback plan documented

**Business:**
- [ ] Stakeholder approval
- [ ] Budget approved
- [ ] SLA defined
- [ ] Success metrics tracked

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Accuracy <97%** | Low (20%) | Critical | Phased testing with go/no-go gates |
| **Latency >500ms** | Medium (30%) | High | HNSW optimization + caching |
| **Learning doesn't converge** | Low (15%) | High | Start with supervised learning baseline |
| **AgentDB stability** | Low (10%) | High | 84.8% SWE-Bench proves production-ready |

---

### 8.2 Resource Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Budget overrun** | Medium (30%) | Medium | 20% contingency buffer ($50K) |
| **Timeline slip** | High (50%) | Medium | 10-week buffer (22 weeks total) |
| **Team availability** | Low (15%) | High | Cross-training, documentation |

---

### 8.3 Overall Risk Level

**Assessment:** 🟡 **MEDIUM RISK**

**Justification:**
- Proven baseline: 84.8% SWE-Bench accuracy
- Established technologies: AgentDB, agentic-flow, ruv-FANN
- Phased approach with validation gates
- 20% probability of failure (acceptable for R&D)

---

## 9. Appendices

### 9.1 Glossary

- **HNSW**: Hierarchical Navigable Small World (graph-based ANN)
- **RL**: Reinforcement Learning
- **MRAP**: Multi-Round Agentic Protocol
- **P95 Latency**: 95th percentile response time
- **Quantization**: Compression technique (4x memory reduction)
- **Session Memory**: Context from previous queries in session
- **Trajectory**: RL tuple (state, action, reward, next_state)

### 9.2 References

- **Current Architecture**: `/workspaces/doc-rag/epics/002-Redesign/architecture/MASTER-ARCHITECTURE-v3.md`
- **Pivot Architecture**: `/workspaces/doc-rag/epics/003-agentic/architecture/pivot-architecture-v1.md`
- **Strategic Recommendation**: `/workspaces/doc-rag/epics/003-agentic/recommendations/STRATEGIC-RECOMMENDATION.md`
- **AgentDB Docs**: https://agentdb.ruv.io/
- **agentic-flow**: https://github.com/ruvnet/agentic-flow
- **ruv-FANN**: https://github.com/ruvnet/ruv-FANN

### 9.3 Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-10-24 | Specification Architect | Initial draft |

---

**END OF SPECIFICATION**

*This document serves as the foundation for SPARC Phase 2 (Pseudocode) and Phase 3 (Architecture).*
