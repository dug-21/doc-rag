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

### 1.2 Budget & Timeline (MCP-Optimized)

| Category | Estimate | MCP Impact |
|----------|----------|------------|
| **Implementation** | $216K-$288K | No change |
| **Infrastructure Setup** | $20K | -$5K (no API gateway needed) |
| **Training Data** | $3K | No change |
| **MCP Server Development** | +$10K | New component |
| **Total Budget** | **$244K-$316K** | +$5K (minimal increase) |
| **Timeline** | **12 weeks** (18-22 weeks w/ buffer) | No change |
| **Annual Operating Cost** | **$12K/year** | -$3K/year (no API infrastructure) |

**Cost Savings from MCP:**
- **API Gateway eliminated**: -$2K setup, -$1.5K/year operating
- **No API management tools**: -$1K setup, -$0.5K/year
- **Reduced networking overhead**: -$1K setup, -$1K/year bandwidth
- **Simplified authentication**: -$1K setup (JWT only, no OAuth server)
- **Total 3-year savings**: ~$15K (infrastructure) + operational efficiency

### 1.3 Success Metrics (MCP-Enhanced)

| Metric | Baseline (No RL) | Target (With RL) | MCP Advantage |
|--------|------------------|------------------|---------------|
| Accuracy | >85% | >97% | No change |
| P95 MCP Tool Latency | <1000ms | <500ms | -100ms vs REST API |
| Cost per Query | <$0.01 | <$0.0005 | 50% reduction (zero API overhead) |
| Learning Convergence | N/A | <1,500 queries | No change |
| MCP Tool Success Rate | >99% | >99.5% | Built-in error handling |
| Session State Accuracy | N/A | >99% | MCP session management |

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

### FR2: Query Processing via MCP Tools
**Priority:** Critical

**Description:** Multi-agent query processing with adaptive routing and strategy selection, exposed via Model Context Protocol (MCP) server for direct Claude integration.

**Capabilities:**
- FR2.1: Query analysis (classification via ruv-FANN) - Exposed as MCP tool
- FR2.2: Multi-strategy retrieval (HNSW, hybrid, rerank, graph-walk) - MCP resource access
- FR2.3: Parallel agent coordination (agentic-flow) - MCP-orchestrated
- FR2.4: Context-aware search (session memory) - MCP session state
- FR2.5: Real-time performance monitoring - MCP streaming responses

**Query Types Supported:**
| Type | Example | Strategy |
|------|---------|----------|
| **Simple** | "What is PCI-DSS requirement 3.2?" | HNSW direct |
| **Moderate** | "How do encryption requirements differ?" | Hybrid search |
| **Complex** | "Map HIPAA to PCI-DSS requirements" | Graph-walk + ensemble |

**Acceptance Criteria:**
```gherkin
Scenario: Simple query via MCP tool call
  Given Claude calls MCP tool "query_pci_dss"
  And parameters: {"query": "What is PCI-DSS requirement 3.2?"}
  When MCP server processes tool request
  Then system shall:
    - Classify query type within 50ms
    - Retrieve top 20 chunks within 200ms
    - Return MCP tool response within 500ms (P95)
    - Achieve >97% accuracy
    - Include full citation chain in response

Scenario: Complex cross-standard query via MCP
  Given Claude calls MCP tool "compare_standards"
  And parameters: {"standards": ["PCI-DSS", "HIPAA"], "topic": "encryption"}
  When multi-agent swarm processes MCP request
  Then system shall:
    - Use adaptive topology (hierarchical)
    - Spawn 4-6 specialized agents
    - Execute graph-walk retrieval
    - Complete within 800ms (P95)
    - Stream progress updates via MCP
    - Provide citations from both standards

Scenario: MCP resource access for document retrieval
  Given Claude requests MCP resource "pci-dss://requirement/3.2"
  When MCP server resolves resource URI
  Then system shall:
    - Return document chunk with metadata
    - Include MIME type and encoding
    - Complete within 100ms
```

**Performance Targets:**
- MCP tool call latency P50: <300ms (includes network overhead eliminated for local MCP)
- MCP tool call latency P95: <500ms
- MCP tool call latency P99: <1000ms
- MCP resource access: <100ms
- MCP streaming: <50ms first chunk
- Concurrent MCP sessions: 100+
- Zero API overhead for local queries (vs REST API ~50-100ms)

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

### FR8: MCP Server Implementation
**Priority:** Critical

**Description:** Model Context Protocol (MCP) server providing Claude-native integration with tools, resources, and prompts for RAG system access.

**Capabilities:**
- FR8.1: MCP tool exposure (query, search, analyze, compare)
- FR8.2: MCP resource providers (document access via URI schemes)
- FR8.3: MCP prompt templates (pre-configured query patterns)
- FR8.4: Streaming responses for long-running operations
- FR8.5: Session state management across tool calls
- FR8.6: Authentication and authorization (JWT-based)
- FR8.7: Rate limiting and quota enforcement
- FR8.8: MCP protocol compliance (stdio and HTTP transports)

**MCP Tools Specification:**

| Tool Name | Purpose | Input Schema | Output | Latency Target |
|-----------|---------|--------------|--------|----------------|
| `query_pci_dss` | Simple requirement lookup | `{query: string}` | Citation + answer | <500ms |
| `compare_standards` | Cross-standard analysis | `{standards: string[], topic: string}` | Comparative table | <800ms |
| `search_requirements` | Semantic search | `{query: string, filters: object}` | Ranked results | <300ms |
| `explain_requirement` | Detailed explanation | `{requirement_id: string}` | Rich explanation | <400ms |
| `validate_compliance` | Compliance checking | `{scenario: string, standard: string}` | Validation report | <1000ms |

**MCP Resources Specification:**

| Resource URI Scheme | Purpose | Example | Response Type |
|---------------------|---------|---------|---------------|
| `pci-dss://requirement/{id}` | Direct requirement access | `pci-dss://requirement/3.2` | text/markdown |
| `hipaa://section/{id}` | HIPAA section access | `hipaa://section/164.312` | text/markdown |
| `compliance://graph/{node}` | Knowledge graph access | `compliance://graph/encryption` | application/json |
| `document://{standard}/{page}` | Page-level access | `document://pci-dss/45` | application/pdf |

**MCP Prompts Specification:**

| Prompt Name | Purpose | Variables | Use Case |
|-------------|---------|-----------|----------|
| `requirement-lookup` | Standard requirement query | `{standard, requirement}` | Quick reference |
| `compliance-check` | Validation workflow | `{scenario, standards}` | Audit preparation |
| `gap-analysis` | Identify missing controls | `{current_state, target_standard}` | Risk assessment |
| `implementation-guide` | Step-by-step implementation | `{requirement_id}` | Developer guidance |

**Acceptance Criteria:**
```gherkin
Scenario: MCP server initialization
  Given TypeScript MCP server package installed
  When server starts with config
  Then server shall:
    - Register all 5 tools with schemas
    - Expose 4 resource URI schemes
    - Load 4 prompt templates
    - Listen on stdio transport (default)
    - Support HTTP transport (optional)
    - Initialize AgentDB connection pool
    - Complete startup within 2 seconds

Scenario: Tool call execution
  Given Claude Code invokes MCP tool "query_pci_dss"
  And valid authentication token provided
  When MCP server receives tool call
  Then server shall:
    - Validate input schema
    - Check rate limits (100 calls/minute)
    - Execute query processing pipeline
    - Return structured response with citations
    - Log tool call metrics
    - Complete within latency target

Scenario: Resource access
  Given Claude requests resource "pci-dss://requirement/3.2"
  When MCP server resolves URI
  Then server shall:
    - Parse URI scheme and identifier
    - Query AgentDB for document chunk
    - Format response with proper MIME type
    - Include metadata (version, last_updated)
    - Cache response for 5 minutes
    - Return within 100ms

Scenario: Streaming response for long query
  Given complex query requiring >500ms processing
  When MCP tool called with streaming enabled
  Then server shall:
    - Send initial acknowledgment within 50ms
    - Stream progress updates every 100ms
    - Include partial results as available
    - Send final response with complete data
    - Close stream properly

Scenario: Session state management
  Given Claude makes follow-up query
  And session context from previous call exists
  When MCP server processes request
  Then server shall:
    - Retrieve session state from memory
    - Apply contextual filters
    - Maintain conversation history
    - Update session with new interaction
    - Expire session after 30 minutes idle
```

**Performance Requirements:**
- Tool registration: <100ms
- Tool call overhead: <10ms (MCP protocol processing)
- Resource lookup: <100ms
- Streaming first chunk: <50ms
- Session state access: <5ms
- Concurrent sessions: 100+
- Memory per session: <1MB

**Security Requirements:**
- JWT token validation on every call
- Role-based access control (RBAC) for tools
- Rate limiting per user/session (100 calls/min)
- Input sanitization for all parameters
- No sensitive data in logs
- Audit trail for all tool calls
- TLS 1.3 for HTTP transport

**Integration Points:**
```typescript
// MCP Server → AgentDB
interface MCPAgentDBIntegration {
  // Tool calls trigger AgentDB operations
  queryPciDss(query: string): Promise<AgentDBSearchResult>

  // Resources access AgentDB storage
  getResource(uri: string): Promise<AgentDBDocument>

  // Session state stored in AgentDB memory
  sessionStore: AgentDBMemoryPlugin
}

// MCP Server → agentic-flow
interface MCPSwarmIntegration {
  // Complex queries spawn agent swarms
  orchestrateQuery(params: QueryParams): Promise<SwarmResult>

  // Stream progress from swarm execution
  streamSwarmProgress(): AsyncGenerator<ProgressUpdate>
}

// MCP Server → ruv-FANN
interface MCPNeuralIntegration {
  // Classification exposed as tool
  classifyQuery(text: string): Promise<QueryClassification>

  // Intent detection for routing
  detectIntent(query: string): Promise<IntentVector>
}
```

**Error Handling:**
```typescript
// MCP-specific error codes
enum MCPErrorCode {
  TOOL_NOT_FOUND = 'tool_not_found',
  INVALID_PARAMS = 'invalid_parameters',
  RATE_LIMITED = 'rate_limit_exceeded',
  UNAUTHORIZED = 'unauthorized',
  RESOURCE_NOT_FOUND = 'resource_not_found',
  TIMEOUT = 'execution_timeout',
  INTERNAL_ERROR = 'internal_server_error'
}

// Error response format (MCP standard)
interface MCPError {
  code: MCPErrorCode
  message: string
  data?: {
    retryAfter?: number  // For rate limits
    suggestion?: string   // User-friendly help
    traceId?: string     // For debugging
  }
}
```

**Observability:**
- Prometheus metrics for tool calls, latency, errors
- OpenTelemetry traces for request flow
- Structured JSON logs with trace correlation
- Real-time dashboard (Grafana)
- Alert on error rate >1% or latency >2x target

---

## 3. Non-Functional Requirements

### NFR1: Performance (MCP-Enhanced)

| Metric | Requirement | Measurement | MCP Advantage |
|--------|-------------|-------------|---------------|
| **MCP Tool Call Latency P50** | <300ms | Tool invocation → response | -50ms vs REST API |
| **MCP Tool Call Latency P95** | <500ms | 95th percentile | -100ms vs REST API |
| **MCP Tool Call Latency P99** | <1000ms | 99th percentile | -150ms vs REST API |
| **MCP Resource Access** | <100ms | URI resolution → data | -50ms vs HTTP |
| **MCP Streaming First Chunk** | <50ms | Initial response | Real-time progress |
| **MCP Session State Access** | <5ms | Context retrieval | Built-in caching |
| **Throughput** | 100+ concurrent sessions | Concurrent MCP connections | stdio multiplexing |
| **Cost per Query** | <$0.0005 | Infrastructure only | Zero API overhead |
| **Memory** | <4GB/million vectors | With scalar quantization | AgentDB optimization |

**MCP Performance Breakdown:**
- Protocol overhead: ~10ms (JSON-RPC parsing)
- Authentication: ~5ms (JWT validation, cached)
- Tool dispatch: ~5ms (route to handler)
- Session lookup: ~5ms (AgentDB memory plugin)
- **Total MCP overhead: ~25ms** (vs 50-100ms for REST API)

**Load Testing:**
- Baseline: 10 concurrent MCP sessions @ <300ms per tool call
- Stress: 100 concurrent MCP sessions @ <500ms per tool call
- Peak: 200 concurrent MCP sessions @ <1000ms per tool call
- Streaming: 50 concurrent long-running queries with progress updates

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

### NFR4: Security (MCP-Specific)

**MCP Authentication & Authorization:**
- JWT tokens for MCP tool calls (issued per session)
- Role-based access control (RBAC) at tool level
- Session-based authorization (30-minute TTL)
- Rate limiting per user/session (100 calls/minute)
- Tool-level permissions (read/write/admin)

**Data Protection:**
- Encryption at rest: AES-256 (AgentDB storage)
- Encryption in transit: TLS 1.3 (HTTP transport only, stdio is local)
- MCP message signing: HMAC-SHA256 for integrity
- Input sanitization: All tool parameters validated via JSON Schema

**MCP-Specific Security:**
- Tool discovery: Only expose authorized tools per user role
- Resource access: URI-based permissions (e.g., `pci-dss://` requires compliance role)
- Prompt templates: Sanitized variable substitution
- Streaming: Secure channel with early termination on auth failure
- Audit trail: All MCP tool calls logged with trace IDs

**Compliance:**
- No PII storage (document content only)
- Audit logging (all MCP tool calls recorded with parameters)
- Data retention: 90 days (configurable)
- MCP session logs: Encrypted and anonymized

---

### NFR5: Observability (MCP-Enhanced)

**MCP-Specific Metrics:**
- MCP tool call latency (P50/P95/P99) - per tool
- MCP resource access latency - per URI scheme
- MCP session duration and tool call count
- MCP streaming performance (time to first chunk)
- MCP error rates (by error code)
- Tool usage distribution (which tools used most)
- Session state cache hit rate

**Core Metrics Collection:**
- Query latency (P50/P95/P99) - end-to-end including MCP overhead
- Accuracy scores (per query type)
- Agent performance (by agent type in swarm)
- Learning convergence (RL plugin metrics)
- Cost per MCP tool call (infrastructure only, no API costs)

**Logging:**
- Structured JSON logs with MCP trace IDs
- Query traces (OpenTelemetry) - includes MCP request flow
- MCP tool call audit trail (parameters, results, errors)
- Error tracking (Sentry) - MCP error categorization
- Performance profiling - MCP overhead breakdown

**Dashboards:**
- Real-time MCP tool metrics (calls/sec, latency, errors)
- MCP session analytics (duration, tool usage patterns)
- Learning progress visualization
- Cost breakdown (MCP vs traditional API savings)
- System health overview (including MCP server status)

---

## 4. System Context

### 4.1 Primary Client: Claude via MCP

**Claude Code / Claude Desktop**
- **Integration**: Model Context Protocol (MCP) server
- **Transport**: stdio (default) or HTTP
- **Authentication**: JWT tokens via MCP auth flow
- **Capabilities**:
  - Direct tool calls (no REST API needed)
  - Resource access via custom URI schemes
  - Streaming responses for long operations
  - Session state preservation
  - Zero network overhead for local queries
- **Use Cases**:
  - Compliance officers querying requirements
  - Developers validating implementation
  - Auditors cross-referencing standards
  - AI assistants providing guided compliance

**Advantages of MCP over REST API:**
- ✅ **Zero API overhead**: Local tool calls vs HTTP requests (~50-100ms saved)
- ✅ **Native Claude integration**: First-class tool support
- ✅ **Streaming built-in**: Progressive results for long queries
- ✅ **Session management**: Automatic context preservation
- ✅ **Type safety**: JSON Schema validation at protocol level
- ✅ **Discovery**: Tools/resources self-describing

### 4.2 External Systems

**AgentDB** (Vector Database)
- **Purpose**: Unified storage for vectors, memory, learning
- **Interface**: TypeScript client library
- **Data Flow**: Embeddings → HNSW index → Search results
- **MCP Integration**: Accessed via MCP tools and resources

**agentic-flow** (Orchestration)
- **Purpose**: Multi-agent swarm coordination
- **Interface**: TypeScript SDK
- **Data Flow**: Query → Agent tasks → Aggregated results
- **MCP Integration**: Swarm execution triggered by MCP tool calls

**ruv-FANN** (Neural Networks)
- **Purpose**: Fast classification and inference
- **Interface**: WASM module
- **Data Flow**: Input features → Neural net → Predictions
- **MCP Integration**: Classification exposed as MCP tools

**OpenAI API** (Embeddings)
- **Purpose**: Generate 1536-dim embeddings
- **Interface**: REST API
- **Data Flow**: Text → Embedding vector
- **MCP Integration**: Transparent to MCP clients

### 4.3 User Personas & MCP Workflows

**Compliance Officer** (via Claude Code)
- **Goal**: Quickly find PCI-DSS requirements
- **Frequency**: 10-20 queries/day via MCP tools
- **Expertise**: Domain expert, technical
- **MCP Usage**:
  - Calls `query_pci_dss` tool directly
  - Accesses resources via `pci-dss://` URIs
  - Uses `requirement-lookup` prompt template

**Security Auditor** (via Claude Desktop)
- **Goal**: Cross-reference requirements across standards
- **Frequency**: 5-10 queries/day via MCP
- **Expertise**: High technical, multi-standard
- **MCP Usage**:
  - Calls `compare_standards` tool for analysis
  - Uses `gap-analysis` prompt template
  - Accesses multiple standard resources in single session

**Developer** (via Claude API with MCP)
- **Goal**: Understand implementation requirements
- **Frequency**: 3-5 queries/day via MCP
- **Expertise**: Technical, needs examples
- **MCP Usage**:
  - Calls `explain_requirement` tool
  - Uses `implementation-guide` prompt template
  - Accesses code examples via resources

### 4.4 MCP Use Cases

**UC1: Simple Requirement Lookup via MCP**
```
Claude Code: calls tool "query_pci_dss"
Parameters: {"query": "What is PCI-DSS requirement 3.2?"}
MCP Server: Direct HNSW search → Template response
Response: Citation + answer in <300ms
Network overhead: 0ms (local stdio transport)
```

**UC2: Comparative Analysis via MCP**
```
Claude Desktop: calls tool "compare_standards"
Parameters: {"standards": ["PCI-DSS", "HIPAA"], "topic": "encryption"}
MCP Server: Multi-agent coordination → Graph-walk → Synthesis
Response: Streaming updates every 100ms → Final result <800ms
```

**UC3: Contextual Follow-up with MCP Session**
```
Claude: "What are the encryption algorithms?"
Context: MCP session retains previous "PCI-DSS 3.2" query
MCP Server: Session memory → Context-aware search → Response
Response: <400ms with automatic context application
Session TTL: 30 minutes idle timeout
```

**UC4: Resource Access via MCP URI**
```
Claude: requests resource "pci-dss://requirement/3.2"
MCP Server: Parse URI → AgentDB lookup → Format markdown
Response: Direct requirement text with metadata <100ms
Cache: 5-minute TTL for frequently accessed resources
```

**UC5: Batch Query via MCP Prompt Template**
```
Claude: uses prompt "gap-analysis"
Variables: {current_state: "basic-encryption", target_standard: "PCI-DSS"}
MCP Server: Execute multi-step workflow → Gap report
Response: Structured analysis with actionable recommendations <2000ms
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

### 6.2 Final Metrics (With RL & MCP)

**Phase 6 Target** (Week 12)
| Metric | Target | Validation | MCP Impact |
|--------|--------|------------|------------|
| **Accuracy** | **>97%** | 980-1,490 test questions | No change |
| **P95 MCP Tool Call Latency** | **<500ms** | 1,000 MCP tool call load test | -100ms vs REST |
| **Cost** | **<$0.0005/query** | 1 month MCP usage data | 50% reduction (no API overhead) |
| **Learning** | **+2-5% improvement** | Before/after RL comparison | No change |
| **Uptime** | **99.9%** | 1 month monitoring | Improved (local stdio resilient) |
| **MCP Tool Call Success Rate** | **>99.5%** | MCP audit logs | N/A |
| **MCP Session State Accuracy** | **>99%** | Context preservation tests | N/A |

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

### 7.2 Testing Phases (MCP-Integrated)

**Phase 1: Unit Tests** (Week 6)
- Component-level validation
- 100+ unit tests per module
- **MCP-specific:** Tool handler unit tests, resource resolver tests
- Coverage: >80%

**Phase 2: MCP Integration Tests** (Week 8)
- End-to-end MCP tool call flow
- 50+ integration scenarios including:
  - All 5 MCP tools with various parameters
  - All 4 resource URI schemes
  - All 4 prompt templates
  - Session state preservation across calls
  - Streaming response handling
- All agent types tested via MCP orchestration

**Phase 3: MCP Protocol Compliance** (Week 9)
- MCP protocol validator (stdio and HTTP transports)
- JSON Schema validation for all tool inputs/outputs
- Error handling per MCP specification
- Authentication/authorization flows
- Rate limiting enforcement

**Phase 4: Accuracy Validation** (Week 10)
- Baseline test (200 questions via MCP tools)
- Target: >85% without RL
- **MCP-specific:** Tool call success rate >99%
- Go/No-Go decision point

**Phase 5: Production Validation** (Week 12)
- Full test bank (980-1,490 questions via MCP)
- Target: >97% with RL
- Load testing (100 concurrent MCP sessions)
- **MCP-specific:** Session state accuracy >99%

**Go/No-Go Gates:**
| Gate | Criteria | Action if Failed |
|------|----------|------------------|
| **Phase 3** | >85% accuracy | 2-week remediation or pivot |
| **Phase 4** | >97% accuracy | 2-week remediation or soft launch |

---

### 7.3 Production Readiness Checklist

**Technical:**
- [ ] All tests passing (>97% accuracy via MCP tools)
- [ ] Performance validated (<500ms P95 MCP tool call latency)
- [ ] Load testing complete (100 concurrent MCP sessions)
- [ ] Security audit passed (including MCP authentication)
- [ ] Monitoring and alerting configured (MCP metrics)
- [ ] Backup and recovery tested

**MCP-Specific:**
- [ ] All 5 MCP tools registered and functional
- [ ] All 4 resource URI schemes working
- [ ] All 4 prompt templates tested
- [ ] stdio transport validated (default)
- [ ] HTTP transport tested (optional)
- [ ] Session state persistence verified
- [ ] Streaming responses working correctly
- [ ] MCP protocol compliance validated
- [ ] Tool discovery working for all user roles
- [ ] Rate limiting enforced and tested
- [ ] MCP error handling complete

**Operational:**
- [ ] Documentation complete (including MCP tool reference)
- [ ] Runbooks created (MCP server operations)
- [ ] Training provided to users (Claude integration)
- [ ] Support process defined (MCP troubleshooting)
- [ ] Rollback plan documented

**Business:**
- [ ] Stakeholder approval
- [ ] Budget approved
- [ ] SLA defined (including MCP uptime)
- [ ] Success metrics tracked (MCP vs REST comparison)

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
- **MCP**: Model Context Protocol - Standard for AI-native tool integration
- **MCP Server**: TypeScript server exposing tools, resources, and prompts via MCP
- **MCP Tool**: Callable function exposed via MCP (e.g., `query_pci_dss`)
- **MCP Resource**: URI-accessible content (e.g., `pci-dss://requirement/3.2`)
- **MCP Prompt**: Pre-configured prompt template with variables
- **stdio Transport**: MCP communication over standard input/output (local, zero network overhead)
- **HTTP Transport**: MCP communication over HTTP (remote access)
- **P95 Latency**: 95th percentile response time (includes MCP overhead)
- **Quantization**: Compression technique (4x memory reduction)
- **Session Memory**: Context from previous queries in MCP session
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
