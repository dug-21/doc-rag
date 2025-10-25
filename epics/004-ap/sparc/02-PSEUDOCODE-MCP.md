# SPARC PSEUDOCODE: MCP-based RAG System
## Phase 2 - Algorithm Design for Agentic Pivot Architecture

**Project:** PCI-DSS RAG System v1.0
**Version:** 1.0 (MCP-First Architecture)
**Date:** October 25, 2025
**Document Type:** SPARC Phase 2 - Pseudocode
**Status:** Design Review

---

## Table of Contents

1. [MCP Server Lifecycle](#1-mcp-server-lifecycle)
2. [MCP Resources](#2-mcp-resources)
3. [MCP Tools](#3-mcp-tools)
4. [MCP Prompts](#4-mcp-prompts)
5. [Query Processing Flow](#5-query-processing-flow)
6. [Integration Layers](#6-integration-layers)
7. [Error Handling](#7-error-handling)
8. [Performance Optimization](#8-performance-optimization)

---

## 1. MCP Server Lifecycle

### 1.1 Server Initialization

```
ALGORITHM: InitializeMCPServer
INPUT: config (ServerConfig)
OUTPUT: server (MCPServer) or error

TYPES:
  ServerConfig = {
    name: string,
    version: string,
    agentDbPath: string,
    embeddingModel: string,
    maxConcurrentQueries: integer,
    port: integer
  }

  MCPServer = {
    transport: StdioTransport,
    agentDb: AgentDBInstance,
    resources: Map<string, ResourceHandler>,
    tools: Map<string, ToolHandler>,
    prompts: Map<string, PromptHandler>,
    state: ServerState
  }

  ServerState = {
    status: "initializing" | "ready" | "error" | "shutdown",
    activeSessions: Map<string, Session>,
    queryQueue: PriorityQueue<Query>,
    metrics: PerformanceMetrics
  }

BEGIN
  Log("🚀 Initializing MCP Server for RAG System")

  // Step 1: Initialize AgentDB
  TRY
    agentDb ← InitializeAgentDB(config)
    IF agentDb is NULL THEN
      RETURN error("Failed to initialize AgentDB")
    END IF
  CATCH error
    Log("❌ AgentDB initialization failed", error)
    RETURN error("AgentDB initialization failed: " + error.message)
  END TRY

  // Step 2: Initialize HNSW indexes
  TRY
    indexes ← InitializeHNSWIndexes(agentDb, config)
    Log("✅ HNSW indexes initialized", indexes.stats)
  CATCH error
    Log("⚠️ HNSW index initialization failed", error)
    // Continue with degraded performance
  END TRY

  // Step 3: Register MCP Resources
  resources ← RegisterMCPResources(agentDb)
  Log("📚 Registered " + resources.size + " MCP resources")

  // Step 4: Register MCP Tools
  tools ← RegisterMCPTools(agentDb)
  Log("🔧 Registered " + tools.size + " MCP tools")

  // Step 5: Register MCP Prompts
  prompts ← RegisterMCPPrompts(agentDb)
  Log("💬 Registered " + prompts.size + " MCP prompts")

  // Step 6: Initialize transport layer
  transport ← CreateStdioTransport()

  // Step 7: Create server instance
  server ← {
    transport: transport,
    agentDb: agentDb,
    resources: resources,
    tools: tools,
    prompts: prompts,
    state: {
      status: "ready",
      activeSessions: EmptyMap(),
      queryQueue: CreatePriorityQueue(),
      metrics: InitializeMetrics()
    }
  }

  // Step 8: Setup error handlers
  SetupGlobalErrorHandlers(server)

  // Step 9: Start health monitoring
  StartHealthMonitor(server)

  Log("✅ MCP Server initialized successfully")
  RETURN server
END
```

**Complexity Analysis:**
- **Time:** O(n log n) where n = number of vectors (HNSW index construction)
- **Space:** O(n) for storing vectors and index
- **Bottleneck:** HNSW index construction during cold start

---

### 1.2 AgentDB Initialization

```
ALGORITHM: InitializeAgentDB
INPUT: config (ServerConfig)
OUTPUT: agentDb (AgentDBInstance)

TYPES:
  AgentDBConfig = {
    path: string,
    dimensions: integer,          // 1536 for OpenAI embeddings
    metric: "cosine" | "euclidean" | "dot",
    indexParams: HNSWParams,
    quantization: QuantizationConfig
  }

  HNSWParams = {
    M: integer,                   // 16 (connections per node)
    efConstruction: integer,      // 200 (build-time search depth)
    efSearch: integer,            // 100 (query-time search depth)
    maxElements: integer          // 1,000,000
  }

  QuantizationConfig = {
    enabled: boolean,
    type: "scalar" | "product",   // scalar for 4x compression
    bits: integer                 // 8 bits for scalar quantization
  }

BEGIN
  Log("🔧 Initializing AgentDB instance")

  // Create AgentDB configuration
  agentDbConfig ← {
    path: config.agentDbPath,
    dimensions: 1536,             // OpenAI ada-002 embedding size
    metric: "cosine",             // Cosine similarity for embeddings
    indexParams: {
      M: 16,                      // Balanced for recall/speed
      efConstruction: 200,        // High quality index
      efSearch: 100,              // Fast queries
      maxElements: 1000000        // Support 1M vectors
    },
    quantization: {
      enabled: TRUE,              // Enable for memory efficiency
      type: "scalar",             // 4x memory reduction
      bits: 8                     // 8-bit quantization
    }
  }

  // Initialize AgentDB client
  agentDb ← AgentDB.initialize(agentDbConfig)

  // Create collections if they don't exist
  collections ← [
    "document_chunks",            // Main vector collection
    "query_trajectories",         // RL training data
    "session_memory",             // Conversation context
    "learning_patterns"           // Learned patterns
  ]

  FOR EACH collection IN collections DO
    IF NOT agentDb.hasCollection(collection) THEN
      agentDb.createCollection(collection, {
        dimensions: 1536,
        metric: "cosine",
        index: "hnsw"
      })
      Log("✅ Created collection: " + collection)
    ELSE
      Log("📂 Collection exists: " + collection)
    END IF
  END FOR

  // Load existing indexes into memory
  FOR EACH collection IN collections DO
    indexStats ← agentDb.loadIndex(collection)
    Log("📊 Loaded " + collection + ": " + indexStats.vectorCount + " vectors")
  END FOR

  RETURN agentDb
END
```

**Data Structure: AgentDB HNSW Index**
```
Structure:
  - Entry layer: All nodes connected with probability 1
  - Layer 0: Dense graph with M connections per node
  - Layer 1+: Hierarchical routing layers

Operations:
  - insert(vector, metadata): O(log n) average
  - search(query, k): O(log n) average
  - delete(id): O(log n)
  - update(id, vector): O(log n)

Memory:
  - Without quantization: n * dimensions * 4 bytes (float32)
  - With scalar quantization: n * dimensions * 1 byte (8-bit)
  - Reduction: 4x memory savings
```

---

### 1.3 HNSW Index Initialization

```
ALGORITHM: InitializeHNSWIndexes
INPUT: agentDb (AgentDBInstance), config (ServerConfig)
OUTPUT: indexes (Map<string, IndexStats>)

TYPES:
  IndexStats = {
    collection: string,
    vectorCount: integer,
    indexSize: integer,           // bytes
    buildTime: integer,           // milliseconds
    avgSearchTime: float,         // microseconds
    recall: float                 // @k=20
  }

BEGIN
  indexes ← EmptyMap()

  // Index 1: Document Chunks (Primary Index)
  Log("🔨 Building HNSW index for document_chunks")
  startTime ← CurrentTimeMillis()

  chunkIndex ← agentDb.buildIndex("document_chunks", {
    M: 16,                        // Balanced connectivity
    efConstruction: 200,          // High-quality build
    numThreads: 4                 // Parallel construction
  })

  buildTime ← CurrentTimeMillis() - startTime

  // Benchmark index performance
  benchmarkResults ← BenchmarkIndex(agentDb, "document_chunks", {
    numQueries: 100,
    k: 20
  })

  indexes.set("document_chunks", {
    collection: "document_chunks",
    vectorCount: chunkIndex.size,
    indexSize: chunkIndex.memoryUsage,
    buildTime: buildTime,
    avgSearchTime: benchmarkResults.avgLatency,
    recall: benchmarkResults.recall
  })

  Log("✅ document_chunks index built in " + buildTime + "ms")
  Log("📊 Recall@20: " + benchmarkResults.recall)

  // Index 2: Query Trajectories (RL Training)
  trajectoryIndex ← agentDb.buildIndex("query_trajectories", {
    M: 8,                         // Smaller for trajectories
    efConstruction: 100
  })

  indexes.set("query_trajectories", {
    collection: "query_trajectories",
    vectorCount: trajectoryIndex.size,
    indexSize: trajectoryIndex.memoryUsage,
    buildTime: 0,
    avgSearchTime: 0,
    recall: 0
  })

  Log("✅ All HNSW indexes initialized")
  RETURN indexes
END

SUBROUTINE: BenchmarkIndex
INPUT: agentDb, collection (string), params (object)
OUTPUT: results (BenchmarkResults)

BEGIN
  totalLatency ← 0
  correctResults ← 0

  // Generate random query vectors
  FOR i FROM 1 TO params.numQueries DO
    queryVector ← GenerateRandomVector(1536)

    // Measure search time
    startTime ← CurrentTimeMicros()
    results ← agentDb.search(collection, queryVector, params.k)
    endTime ← CurrentTimeMicros()

    totalLatency ← totalLatency + (endTime - startTime)

    // Check recall (requires ground truth)
    groundTruth ← BruteForceSearch(collection, queryVector, params.k)
    overlap ← CountOverlap(results, groundTruth)
    correctResults ← correctResults + overlap
  END FOR

  avgLatency ← totalLatency / params.numQueries
  recall ← correctResults / (params.numQueries * params.k)

  RETURN {
    avgLatency: avgLatency,
    recall: recall,
    numQueries: params.numQueries
  }
END
```

**Performance Targets:**
- Build time: <10 minutes for 1M vectors
- Search time: <100µs per query
- Recall@20: >97%
- Memory: <400MB with quantization

---

## 2. MCP Resources

### 2.1 Resource Registration

```
ALGORITHM: RegisterMCPResources
INPUT: agentDb (AgentDBInstance)
OUTPUT: resources (Map<string, ResourceHandler>)

BEGIN
  resources ← EmptyMap()

  // Resource 1: PCI-DSS Document Collection
  resources.set("pci-dss://documents", {
    name: "pci-dss://documents",
    description: "PCI-DSS compliance documentation collection",
    mimeType: "application/json",
    handler: CreateDocumentCollectionHandler(agentDb)
  })

  // Resource 2: Document Chunks
  resources.set("pci-dss://chunks", {
    name: "pci-dss://chunks",
    description: "Searchable document chunks with embeddings",
    mimeType: "application/json",
    handler: CreateChunksHandler(agentDb)
  })

  // Resource 3: Embeddings Metadata
  resources.set("pci-dss://embeddings/metadata", {
    name: "pci-dss://embeddings/metadata",
    description: "Embedding model and configuration metadata",
    mimeType: "application/json",
    handler: CreateEmbeddingsMetadataHandler(agentDb)
  })

  // Resource 4: Session Context
  resources.set("pci-dss://sessions/{sessionId}", {
    name: "pci-dss://sessions/{sessionId}",
    description: "Session-specific conversation context",
    mimeType: "application/json",
    handler: CreateSessionHandler(agentDb)
  })

  // Resource 5: Learning Statistics
  resources.set("pci-dss://learning/stats", {
    name: "pci-dss://learning/stats",
    description: "Reinforcement learning progress statistics",
    mimeType: "application/json",
    handler: CreateLearningStatsHandler(agentDb)
  })

  Log("📚 Registered " + resources.size + " MCP resources")
  RETURN resources
END
```

---

### 2.2 Document Collection Resource

```
ALGORITHM: HandleDocumentCollectionResource
INPUT: request (ResourceRequest)
OUTPUT: response (ResourceResponse)

TYPES:
  ResourceRequest = {
    uri: string,                  // "pci-dss://documents"
    params: Map<string, string>
  }

  ResourceResponse = {
    contents: Array<ResourceContent>,
    metadata: object
  }

  ResourceContent = {
    uri: string,
    mimeType: string,
    text: string                  // JSON-serialized content
  }

BEGIN
  Log("📖 Handling resource request: " + request.uri)

  // Query AgentDB for all documents
  documents ← agentDb.query("document_chunks", {
    filter: {},                   // No filter = all documents
    groupBy: "document_id",
    aggregate: TRUE
  })

  // Build response structure
  documentList ← []

  FOR EACH doc IN documents DO
    documentMetadata ← {
      id: doc.document_id,
      name: doc.metadata.document_name,
      version: doc.metadata.version,
      standard: doc.metadata.standard,
      pageCount: doc.metadata.page_count,
      chunkCount: doc.chunk_count,
      lastUpdated: doc.metadata.last_updated,
      embedding: {
        model: "text-embedding-ada-002",
        dimensions: 1536,
        totalVectors: doc.chunk_count
      }
    }

    documentList.append(documentMetadata)
  END FOR

  // Create response
  response ← {
    contents: [{
      uri: "pci-dss://documents",
      mimeType: "application/json",
      text: JSON.stringify({
        documents: documentList,
        totalDocuments: documentList.length,
        totalChunks: SUM(doc.chunkCount FOR doc IN documentList),
        generatedAt: CurrentTimestamp()
      })
    }],
    metadata: {
      source: "AgentDB",
      cacheKey: "documents:all",
      ttl: 3600                   // Cache for 1 hour
    }
  }

  RETURN response
END
```

---

### 2.3 Document Chunks Resource

```
ALGORITHM: HandleChunksResource
INPUT: request (ResourceRequest)
OUTPUT: response (ResourceResponse)

BEGIN
  Log("🧩 Handling chunks resource: " + request.uri)

  // Parse query parameters
  documentId ← request.params.get("document_id")
  section ← request.params.get("section")
  pageRange ← request.params.get("page_range")
  limit ← ParseInt(request.params.get("limit"), 100)

  // Build filter
  filter ← {}
  IF documentId IS NOT NULL THEN
    filter.document_id ← documentId
  END IF
  IF section IS NOT NULL THEN
    filter.section ← section
  END IF
  IF pageRange IS NOT NULL THEN
    pages ← ParsePageRange(pageRange)  // "10-20" → [10, 11, ..., 20]
    filter.page_number ← {"$in": pages}
  END IF

  // Query chunks from AgentDB
  chunks ← agentDb.query("document_chunks", {
    filter: filter,
    limit: limit,
    sort: {page_number: 1, chunk_index: 1}
  })

  // Format chunks for response
  formattedChunks ← []
  FOR EACH chunk IN chunks DO
    formattedChunk ← {
      id: chunk.id,
      text: chunk.text,
      metadata: {
        document_id: chunk.metadata.document_id,
        document_name: chunk.metadata.document_name,
        section: chunk.metadata.section,
        page_number: chunk.metadata.page_number,
        chunk_index: chunk.metadata.chunk_index,
        token_count: chunk.metadata.token_count
      },
      embedding: {
        available: TRUE,
        dimensions: 1536
      }
    }
    formattedChunks.append(formattedChunk)
  END FOR

  response ← {
    contents: [{
      uri: request.uri,
      mimeType: "application/json",
      text: JSON.stringify({
        chunks: formattedChunks,
        totalMatches: chunks.length,
        filter: filter,
        generatedAt: CurrentTimestamp()
      })
    }]
  }

  RETURN response
END
```

---

## 3. MCP Tools

### 3.1 Tool Registration

```
ALGORITHM: RegisterMCPTools
INPUT: agentDb (AgentDBInstance)
OUTPUT: tools (Map<string, ToolHandler>)

BEGIN
  tools ← EmptyMap()

  // Tool 1: Semantic Search
  tools.set("semantic_search", {
    name: "semantic_search",
    description: "Perform vector similarity search on document chunks",
    inputSchema: {
      type: "object",
      properties: {
        query: {type: "string", description: "Search query"},
        top_k: {type: "integer", default: 20},
        filters: {type: "object", optional: true},
        threshold: {type: "number", default: 0.7}
      },
      required: ["query"]
    },
    handler: CreateSemanticSearchHandler(agentDb)
  })

  // Tool 2: Hybrid Search
  tools.set("hybrid_search", {
    name: "hybrid_search",
    description: "Combine semantic and keyword search",
    inputSchema: {
      type: "object",
      properties: {
        query: {type: "string"},
        semantic_weight: {type: "number", default: 0.7},
        keyword_weight: {type: "number", default: 0.3},
        top_k: {type: "integer", default: 20}
      },
      required: ["query"]
    },
    handler: CreateHybridSearchHandler(agentDb)
  })

  // Tool 3: Retrieve Document
  tools.set("retrieve_document", {
    name: "retrieve_document",
    description: "Retrieve full document or specific sections",
    inputSchema: {
      type: "object",
      properties: {
        document_id: {type: "string"},
        sections: {type: "array", items: {type: "string"}, optional: true}
      },
      required: ["document_id"]
    },
    handler: CreateRetrieveDocumentHandler(agentDb)
  })

  // Tool 4: Verify Citations
  tools.set("verify_citations", {
    name: "verify_citations",
    description: "Verify accuracy of citations and references",
    inputSchema: {
      type: "object",
      properties: {
        citations: {type: "array", items: {type: "object"}},
        query: {type: "string"}
      },
      required: ["citations"]
    },
    handler: CreateVerifyCitationsHandler(agentDb)
  })

  // Tool 5: Classify Query
  tools.set("classify_query", {
    name: "classify_query",
    description: "Classify query complexity using ruv-FANN",
    inputSchema: {
      type: "object",
      properties: {
        query: {type: "string"}
      },
      required: ["query"]
    },
    handler: CreateClassifyQueryHandler(agentDb)
  })

  // Tool 6: Store Trajectory
  tools.set("store_trajectory", {
    name: "store_trajectory",
    description: "Store RL trajectory for learning",
    inputSchema: {
      type: "object",
      properties: {
        state: {type: "object"},
        action: {type: "object"},
        reward: {type: "number"},
        next_state: {type: "object"}
      },
      required: ["state", "action", "reward", "next_state"]
    },
    handler: CreateStoreTrajectoryHandler(agentDb)
  })

  // Tool 7: Get Session Context
  tools.set("get_session_context", {
    name: "get_session_context",
    description: "Retrieve session conversation history",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {type: "string"},
        window_size: {type: "integer", default: 5}
      },
      required: ["session_id"]
    },
    handler: CreateSessionContextHandler(agentDb)
  })

  Log("🔧 Registered " + tools.size + " MCP tools")
  RETURN tools
END
```

---

### 3.2 Semantic Search Tool

```
ALGORITHM: HandleSemanticSearch
INPUT: request (ToolRequest)
OUTPUT: response (ToolResponse)

TYPES:
  ToolRequest = {
    params: {
      query: string,
      top_k: integer,
      filters: object,
      threshold: float
    }
  }

  ToolResponse = {
    content: Array<ToolContent>,
    isError: boolean
  }

  ToolContent = {
    type: "text" | "resource",
    text: string,
    resource: ResourceReference
  }

  SearchResult = {
    id: string,
    text: string,
    score: float,
    metadata: object
  }

BEGIN
  Log("🔍 Semantic search: " + request.params.query)

  // Step 1: Record start time for metrics
  startTime ← CurrentTimeMillis()

  // Step 2: Generate query embedding
  TRY
    queryEmbedding ← GenerateEmbedding(request.params.query)
    embeddingTime ← CurrentTimeMillis() - startTime
    Log("📊 Embedding generated in " + embeddingTime + "ms")
  CATCH error
    Log("❌ Embedding generation failed", error)
    RETURN {
      content: [{
        type: "text",
        text: JSON.stringify({
          error: "Failed to generate embedding",
          message: error.message
        })
      }],
      isError: TRUE
    }
  END TRY

  // Step 3: Perform HNSW search
  searchStartTime ← CurrentTimeMillis()

  searchResults ← agentDb.search("document_chunks", queryEmbedding, {
    k: request.params.top_k,
    ef: 100,                      // Search depth (recall vs speed)
    filter: request.params.filters,
    returnVectors: FALSE          // Don't return embeddings
  })

  searchTime ← CurrentTimeMillis() - searchStartTime
  Log("⚡ HNSW search completed in " + searchTime + "µs")

  // Step 4: Filter by similarity threshold
  filteredResults ← []
  FOR EACH result IN searchResults DO
    IF result.score >= request.params.threshold THEN
      filteredResults.append(result)
    END IF
  END FOR

  // Step 5: Enrich results with metadata
  enrichedResults ← []
  FOR EACH result IN filteredResults DO
    enriched ← {
      id: result.id,
      text: result.metadata.text,
      score: result.score,
      confidence: result.score,   // Alias for score
      metadata: {
        document_id: result.metadata.document_id,
        document_name: result.metadata.document_name,
        section: result.metadata.section,
        page_number: result.metadata.page_number,
        chunk_index: result.metadata.chunk_index,
        requirement_id: result.metadata.requirement_id
      },
      citation: FormatCitation(result.metadata)
    }
    enrichedResults.append(enriched)
  END FOR

  // Step 6: Calculate total latency
  totalLatency ← CurrentTimeMillis() - startTime

  // Step 7: Build response
  response ← {
    content: [{
      type: "text",
      text: JSON.stringify({
        results: enrichedResults,
        metadata: {
          total_results: enrichedResults.length,
          query: request.params.query,
          top_k: request.params.top_k,
          threshold: request.params.threshold,
          latency_ms: totalLatency,
          embedding_time_ms: embeddingTime,
          search_time_us: searchTime,
          filters_applied: request.params.filters
        }
      })
    }],
    isError: FALSE
  }

  // Step 8: Log metrics
  LogMetrics("semantic_search", {
    latency: totalLatency,
    embeddingTime: embeddingTime,
    searchTime: searchTime,
    resultsReturned: enrichedResults.length
  })

  RETURN response
END

SUBROUTINE: GenerateEmbedding
INPUT: text (string)
OUTPUT: embedding (Array<float>)

BEGIN
  // Call OpenAI embedding API
  apiResponse ← CallOpenAIEmbeddings({
    model: "text-embedding-ada-002",
    input: text
  })

  IF apiResponse.error THEN
    THROW Error("Embedding API failed: " + apiResponse.error)
  END IF

  embedding ← apiResponse.data[0].embedding

  // Validate embedding
  IF embedding.length != 1536 THEN
    THROW Error("Invalid embedding dimensions: " + embedding.length)
  END IF

  RETURN embedding
END

SUBROUTINE: FormatCitation
INPUT: metadata (object)
OUTPUT: citation (string)

BEGIN
  citation ← metadata.document_name + " v" + metadata.version

  IF metadata.section IS NOT NULL THEN
    citation ← citation + ", " + metadata.section
  END IF

  IF metadata.requirement_id IS NOT NULL THEN
    citation ← citation + " (Requirement " + metadata.requirement_id + ")"
  END IF

  IF metadata.page_number IS NOT NULL THEN
    citation ← citation + ", p." + metadata.page_number
  END IF

  RETURN citation
END
```

**Complexity Analysis:**
- **Embedding Generation:** O(1) - API call, ~50-200ms
- **HNSW Search:** O(log n) - where n = total vectors, ~50-100µs
- **Filtering:** O(k) - where k = top_k results
- **Total:** O(log n + k) ≈ <300ms for P50

---

### 3.3 Hybrid Search Tool

```
ALGORITHM: HandleHybridSearch
INPUT: request (ToolRequest)
OUTPUT: response (ToolResponse)

TYPES:
  HybridParams = {
    query: string,
    semantic_weight: float,       // 0.0 - 1.0
    keyword_weight: float,        // 0.0 - 1.0
    top_k: integer
  }

BEGIN
  Log("🔀 Hybrid search: " + request.params.query)

  startTime ← CurrentTimeMillis()

  // Step 1: Parallel execution of semantic and keyword search
  PARALLEL_START
    // Thread 1: Semantic search
    semanticTask ← SPAWN_TASK {
      embedding ← GenerateEmbedding(request.params.query)
      semanticResults ← agentDb.search("document_chunks", embedding, {
        k: request.params.top_k * 2,  // Over-fetch for fusion
        ef: 100
      })
      RETURN semanticResults
    }

    // Thread 2: Keyword search
    keywordTask ← SPAWN_TASK {
      keywords ← ExtractKeywords(request.params.query)
      keywordResults ← agentDb.fullTextSearch("document_chunks", keywords, {
        limit: request.params.top_k * 2
      })
      RETURN keywordResults
    }
  PARALLEL_END

  // Wait for both tasks to complete
  semanticResults ← AWAIT semanticTask
  keywordResults ← AWAIT keywordTask

  parallelTime ← CurrentTimeMillis() - startTime
  Log("⚡ Parallel search completed in " + parallelTime + "ms")

  // Step 2: Reciprocal Rank Fusion (RRF)
  fusedResults ← ReciprocalRankFusion(
    semanticResults,
    keywordResults,
    request.params.semantic_weight,
    request.params.keyword_weight
  )

  // Step 3: Take top k results
  topResults ← fusedResults.slice(0, request.params.top_k)

  // Step 4: Enrich and format
  enrichedResults ← []
  FOR EACH result IN topResults DO
    enriched ← {
      id: result.id,
      text: result.metadata.text,
      score: result.fusedScore,
      scores: {
        semantic: result.semanticScore,
        keyword: result.keywordScore,
        fused: result.fusedScore
      },
      metadata: result.metadata,
      citation: FormatCitation(result.metadata)
    }
    enrichedResults.append(enriched)
  END FOR

  totalLatency ← CurrentTimeMillis() - startTime

  response ← {
    content: [{
      type: "text",
      text: JSON.stringify({
        results: enrichedResults,
        metadata: {
          total_results: enrichedResults.length,
          semantic_weight: request.params.semantic_weight,
          keyword_weight: request.params.keyword_weight,
          latency_ms: totalLatency,
          fusion_method: "reciprocal_rank_fusion"
        }
      })
    }],
    isError: FALSE
  }

  RETURN response
END

SUBROUTINE: ReciprocalRankFusion
INPUT: semanticResults, keywordResults, semanticWeight, keywordWeight
OUTPUT: fusedResults (Array<Result>)

ALGORITHM:
  // RRF Formula: score = Σ (weight / (rank + k))
  // k = 60 (standard RRF constant)

  k ← 60
  resultMap ← EmptyMap()  // Map<id, FusedResult>

  // Process semantic results
  FOR rank, result IN ENUMERATE(semanticResults) DO
    semanticScore ← semanticWeight / (rank + k)

    IF resultMap.has(result.id) THEN
      resultMap.get(result.id).fusedScore += semanticScore
      resultMap.get(result.id).semanticScore ← semanticScore
    ELSE
      resultMap.set(result.id, {
        id: result.id,
        metadata: result.metadata,
        fusedScore: semanticScore,
        semanticScore: semanticScore,
        keywordScore: 0
      })
    END IF
  END FOR

  // Process keyword results
  FOR rank, result IN ENUMERATE(keywordResults) DO
    keywordScore ← keywordWeight / (rank + k)

    IF resultMap.has(result.id) THEN
      resultMap.get(result.id).fusedScore += keywordScore
      resultMap.get(result.id).keywordScore ← keywordScore
    ELSE
      resultMap.set(result.id, {
        id: result.id,
        metadata: result.metadata,
        fusedScore: keywordScore,
        semanticScore: 0,
        keywordScore: keywordScore
      })
    END IF
  END FOR

  // Convert map to sorted array
  fusedResults ← Array.from(resultMap.values())
  fusedResults.sortByDescending(result → result.fusedScore)

  RETURN fusedResults
END

SUBROUTINE: ExtractKeywords
INPUT: query (string)
OUTPUT: keywords (Array<string>)

BEGIN
  // Simple keyword extraction (can be enhanced with NLP)

  // 1. Lowercase and tokenize
  tokens ← query.toLowerCase().split(/\s+/)

  // 2. Remove stop words
  stopWords ← ["the", "is", "at", "which", "on", "a", "an", "and", "or", "but"]
  keywords ← []

  FOR EACH token IN tokens DO
    IF NOT stopWords.contains(token) AND token.length > 2 THEN
      keywords.append(token)
    END IF
  END FOR

  // 3. Extract phrases (bigrams)
  FOR i FROM 0 TO keywords.length - 2 DO
    bigram ← keywords[i] + " " + keywords[i+1]
    keywords.append(bigram)
  END FOR

  RETURN keywords
END
```

**Performance:**
- Parallel search: ~150-250ms (max of both searches)
- Fusion: O(n log n) where n = combined results, ~5-10ms
- Total: <300ms for P50

---

### 3.4 Query Classification Tool (ruv-FANN Integration)

```
ALGORITHM: HandleClassifyQuery
INPUT: request (ToolRequest)
OUTPUT: response (ToolResponse)

TYPES:
  QueryClassification = {
    complexity: "simple" | "moderate" | "complex",
    complexityScore: float,       // 0.0 - 1.0
    intent: string,               // "requirement_lookup" | "comparison" | etc.
    suggestedStrategy: string,    // "hnsw" | "hybrid" | "graph_walk"
    suggestedTopology: string,    // "mesh" | "star" | "hierarchical"
    suggestedAgents: integer      // 1-6
  }

BEGIN
  Log("🧠 Classifying query: " + request.params.query)

  startTime ← CurrentTimeMillis()

  // Step 1: Extract query features
  features ← ExtractQueryFeatures(request.params.query)

  // Step 2: Load ruv-FANN classifier
  TRY
    classifier ← LoadRuvFANNClassifier("query_complexity_classifier")
  CATCH error
    // Fallback to rule-based classification
    Log("⚠️ ruv-FANN classifier unavailable, using fallback")
    classification ← FallbackClassification(request.params.query)
    RETURN CreateClassificationResponse(classification)
  END TRY

  // Step 3: Run WASM neural network inference
  neuralInput ← [
    features.tokenCount,
    features.questionWords,
    features.comparisonWords,
    features.requirementReferences,
    features.standardReferences,
    features.avgWordLength,
    features.sentenceCount,
    features.technicalTerms
  ]

  neuralOutput ← classifier.predict(neuralInput)

  inferenceTime ← CurrentTimeMillis() - startTime
  Log("⚡ Classification completed in " + inferenceTime + "ms")

  // Step 4: Interpret neural output
  complexityScore ← neuralOutput[0]      // 0.0 - 1.0
  intentVector ← neuralOutput.slice(1, 5)  // One-hot encoded intent

  complexity ← InterpretComplexity(complexityScore)
  intent ← InterpretIntent(intentVector)

  // Step 5: Determine optimal strategy
  strategy ← DetermineStrategy(complexity, intent)
  topology ← DetermineTopology(complexityScore)
  agentCount ← DetermineAgentCount(complexityScore)

  classification ← {
    complexity: complexity,
    complexityScore: complexityScore,
    intent: intent,
    suggestedStrategy: strategy,
    suggestedTopology: topology,
    suggestedAgents: agentCount,
    features: features,
    confidence: neuralOutput[5]  // Classifier confidence
  }

  response ← {
    content: [{
      type: "text",
      text: JSON.stringify({
        classification: classification,
        metadata: {
          query: request.params.query,
          inference_time_ms: inferenceTime,
          model: "ruv-fann-query-classifier-v1"
        }
      })
    }],
    isError: FALSE
  }

  RETURN response
END

SUBROUTINE: ExtractQueryFeatures
INPUT: query (string)
OUTPUT: features (object)

BEGIN
  tokens ← Tokenize(query.toLowerCase())

  // Count question words
  questionWords ← ["what", "how", "why", "when", "where", "which", "who"]
  questionWordCount ← 0
  FOR EACH token IN tokens DO
    IF questionWords.contains(token) THEN
      questionWordCount ← questionWordCount + 1
    END IF
  END FOR

  // Count comparison indicators
  comparisonWords ← ["compare", "difference", "versus", "vs", "between", "and"]
  comparisonWordCount ← 0
  FOR EACH token IN tokens DO
    IF comparisonWords.contains(token) THEN
      comparisonWordCount ← comparisonWordCount + 1
    END IF
  END FOR

  // Count requirement references (e.g., "3.2", "requirement 4")
  requirementPattern ← /\d+\.\d+|requirement\s+\d+/gi
  requirementReferences ← query.match(requirementPattern).length

  // Count standard references
  standardPattern ← /pci-dss|hipaa|soc2|gdpr/gi
  standardReferences ← query.match(standardPattern).length

  // Calculate complexity indicators
  tokenCount ← tokens.length
  avgWordLength ← AVERAGE(token.length FOR token IN tokens)
  sentenceCount ← query.split(/[.!?]+/).length

  // Count technical terms (simple dictionary lookup)
  technicalTerms ← CountTechnicalTerms(tokens)

  features ← {
    tokenCount: tokenCount,
    questionWords: questionWordCount,
    comparisonWords: comparisonWordCount,
    requirementReferences: requirementReferences,
    standardReferences: standardReferences,
    avgWordLength: avgWordLength,
    sentenceCount: sentenceCount,
    technicalTerms: technicalTerms
  }

  RETURN features
END

SUBROUTINE: InterpretComplexity
INPUT: score (float)
OUTPUT: complexity (string)

BEGIN
  IF score < 0.33 THEN
    RETURN "simple"
  ELSE IF score < 0.66 THEN
    RETURN "moderate"
  ELSE
    RETURN "complex"
  END IF
END

SUBROUTINE: DetermineStrategy
INPUT: complexity (string), intent (string)
OUTPUT: strategy (string)

BEGIN
  // Strategy decision table

  IF complexity == "simple" AND intent == "requirement_lookup" THEN
    RETURN "hnsw"              // Direct vector search

  ELSE IF complexity == "moderate" AND intent == "comparison" THEN
    RETURN "hybrid"            // Hybrid search

  ELSE IF complexity == "complex" THEN
    RETURN "graph_walk"        // Graph-based retrieval

  ELSE IF intent == "procedural" THEN
    RETURN "rerank"            // Search + rerank

  ELSE
    RETURN "hybrid"            // Default to hybrid
  END IF
END

SUBROUTINE: DetermineTopology
INPUT: complexityScore (float)
OUTPUT: topology (string)

BEGIN
  IF complexityScore < 0.33 THEN
    RETURN "mesh"              // Peer-to-peer for simple

  ELSE IF complexityScore < 0.66 THEN
    RETURN "star"              // Coordinator for moderate

  ELSE
    RETURN "hierarchical"      // Tree structure for complex
  END IF
END

SUBROUTINE: DetermineAgentCount
INPUT: complexityScore (float)
OUTPUT: agentCount (integer)

BEGIN
  // Map complexity to agent count
  // Simple: 1-2 agents
  // Moderate: 2-4 agents
  // Complex: 4-6 agents

  agentCount ← CEIL(1 + complexityScore * 5)

  // Clamp to [1, 6]
  IF agentCount < 1 THEN agentCount ← 1 END IF
  IF agentCount > 6 THEN agentCount ← 6 END IF

  RETURN agentCount
END
```

**ruv-FANN Neural Network Architecture:**
```
Input Layer: 8 neurons (features)
  ↓
Hidden Layer 1: 16 neurons (ReLU activation)
  ↓
Hidden Layer 2: 8 neurons (ReLU activation)
  ↓
Output Layer: 6 neurons
  - complexityScore (sigmoid, 0-1)
  - intent_1 to intent_4 (softmax, one-hot)
  - confidence (sigmoid, 0-1)

Training Data: 5,000 labeled query examples
Accuracy: >92% on test set
Inference Time: <5ms (WASM-optimized)
```

---

### 3.5 Store Trajectory Tool (RL Integration)

```
ALGORITHM: HandleStoreTrajectory
INPUT: request (ToolRequest)
OUTPUT: response (ToolResponse)

TYPES:
  Trajectory = {
    state: StateVector,
    action: ActionVector,
    reward: float,
    next_state: StateVector,
    metadata: object
  }

  StateVector = {
    query_complexity: float,
    query_intent: string,
    session_length: integer,
    context_similarity: float,
    time_of_day: integer,
    user_feedback_history: Array<float>
  }

  ActionVector = {
    strategy_chosen: string,
    topology_used: string,
    num_agents: integer,
    top_k: integer,
    rerank_applied: boolean
  }

BEGIN
  Log("💾 Storing RL trajectory")

  startTime ← CurrentTimeMillis()

  // Step 1: Validate trajectory data
  TRY
    ValidateTrajectory(request.params)
  CATCH error
    RETURN {
      content: [{
        type: "text",
        text: JSON.stringify({error: "Invalid trajectory: " + error.message})
      }],
      isError: TRUE
    }
  END TRY

  // Step 2: Generate trajectory embedding
  // Combine state and action into a single vector for retrieval
  trajectoryText ← SerializeTrajectory(request.params)
  trajectoryEmbedding ← GenerateEmbedding(trajectoryText)

  // Step 3: Store in AgentDB
  trajectoryId ← GenerateUUID()

  agentDb.insert("query_trajectories", {
    id: trajectoryId,
    embedding: trajectoryEmbedding,
    metadata: {
      state: request.params.state,
      action: request.params.action,
      reward: request.params.reward,
      next_state: request.params.next_state,
      timestamp: CurrentTimestamp(),
      extra: request.params.metadata
    }
  })

  // Step 4: Check if batch training should trigger
  trajectoryCount ← agentDb.count("query_trajectories", {
    filter: {
      "metadata.trained": FALSE
    }
  })

  shouldTrain ← trajectoryCount >= 100  // Batch size

  IF shouldTrain THEN
    Log("🎓 Triggering batch RL training")
    TriggerRLTraining()
  END IF

  storageTime ← CurrentTimeMillis() - startTime

  response ← {
    content: [{
      type: "text",
      text: JSON.stringify({
        trajectory_id: trajectoryId,
        stored: TRUE,
        batch_training_triggered: shouldTrain,
        pending_trajectories: trajectoryCount,
        storage_time_ms: storageTime
      })
    }],
    isError: FALSE
  }

  RETURN response
END

SUBROUTINE: SerializeTrajectory
INPUT: trajectory (Trajectory)
OUTPUT: text (string)

BEGIN
  // Create text representation for embedding

  parts ← []

  // State description
  parts.append("Query complexity: " + trajectory.state.query_complexity)
  parts.append("Intent: " + trajectory.state.query_intent)
  parts.append("Session length: " + trajectory.state.session_length)

  // Action description
  parts.append("Strategy: " + trajectory.action.strategy_chosen)
  parts.append("Topology: " + trajectory.action.topology_used)
  parts.append("Agents: " + trajectory.action.num_agents)

  // Reward
  parts.append("Reward: " + trajectory.reward)

  text ← parts.join(". ")

  RETURN text
END

SUBROUTINE: TriggerRLTraining
OUTPUT: none

BEGIN
  // Asynchronous batch training

  SPAWN_BACKGROUND_TASK {
    Log("🧠 Starting batch RL training")

    // Fetch untrained trajectories
    trajectories ← agentDb.query("query_trajectories", {
      filter: {"metadata.trained": FALSE},
      limit: 100
    })

    // Train Decision Transformer
    decisionTransformer ← LoadLearningPlugin("decision_transformer")
    decisionTransformer.train(trajectories)

    // Train Actor-Critic
    actorCritic ← LoadLearningPlugin("actor_critic")
    actorCritic.train(trajectories)

    // Mark trajectories as trained
    FOR EACH trajectory IN trajectories DO
      agentDb.update("query_trajectories", trajectory.id, {
        "metadata.trained": TRUE,
        "metadata.trained_at": CurrentTimestamp()
      })
    END FOR

    Log("✅ Batch RL training complete")
  }
END
```

---

## 4. MCP Prompts

### 4.1 Prompt Registration

```
ALGORITHM: RegisterMCPPrompts
INPUT: agentDb (AgentDBInstance)
OUTPUT: prompts (Map<string, PromptHandler>)

BEGIN
  prompts ← EmptyMap()

  // Prompt 1: Inject RAG Context
  prompts.set("pci_dss_context", {
    name: "pci_dss_context",
    description: "Inject relevant PCI-DSS context into user query",
    arguments: [
      {
        name: "user_query",
        description: "User's original query",
        required: TRUE
      },
      {
        name: "session_id",
        description: "Session ID for conversation context",
        required: FALSE
      },
      {
        name: "top_k",
        description: "Number of context chunks to inject",
        required: FALSE,
        default: 5
      }
    ],
    handler: CreateContextInjectionHandler(agentDb)
  })

  // Prompt 2: Suggest Related Queries
  prompts.set("suggest_queries", {
    name: "suggest_queries",
    description: "Suggest related queries based on current context",
    arguments: [
      {
        name: "current_query",
        description: "Current user query",
        required: TRUE
      },
      {
        name: "num_suggestions",
        description: "Number of suggestions to generate",
        required: FALSE,
        default: 3
      }
    ],
    handler: CreateQuerySuggestionHandler(agentDb)
  })

  // Prompt 3: Citation Formatting
  prompts.set("format_citations", {
    name: "format_citations",
    description: "Format search results as citations",
    arguments: [
      {
        name: "results",
        description: "Search results to format",
        required: TRUE
      },
      {
        name: "style",
        description: "Citation style (inline, footnote, bibliography)",
        required: FALSE,
        default: "inline"
      }
    ],
    handler: CreateCitationFormatterHandler(agentDb)
  })

  Log("💬 Registered " + prompts.size + " MCP prompts")
  RETURN prompts
END
```

---

### 4.2 Context Injection Prompt

```
ALGORITHM: HandleContextInjection
INPUT: request (PromptRequest)
OUTPUT: response (PromptResponse)

TYPES:
  PromptRequest = {
    params: {
      user_query: string,
      session_id: string,
      top_k: integer
    }
  }

  PromptResponse = {
    description: string,
    messages: Array<PromptMessage>
  }

  PromptMessage = {
    role: "user" | "assistant",
    content: {
      type: "text" | "resource",
      text: string
    }
  }

BEGIN
  Log("💬 Injecting context for query: " + request.params.user_query)

  startTime ← CurrentTimeMillis()

  // Step 1: Retrieve relevant context chunks
  embedding ← GenerateEmbedding(request.params.user_query)

  contextChunks ← agentDb.search("document_chunks", embedding, {
    k: request.params.top_k,
    ef: 100,
    threshold: 0.7
  })

  // Step 2: Retrieve session history (if available)
  sessionHistory ← []
  IF request.params.session_id IS NOT NULL THEN
    sessionHistory ← agentDb.query("session_memory", {
      filter: {session_id: request.params.session_id},
      sort: {timestamp: -1},
      limit: 3
    })
  END IF

  // Step 3: Build context message
  contextParts ← []

  // Add session context
  IF sessionHistory.length > 0 THEN
    contextParts.append("# Previous Conversation Context\n")
    FOR EACH entry IN sessionHistory DO
      contextParts.append("Q: " + entry.query + "\nA: " + entry.response + "\n")
    END FOR
  END IF

  // Add relevant document chunks
  contextParts.append("\n# Relevant PCI-DSS Documentation\n")

  FOR i, chunk IN ENUMERATE(contextChunks) DO
    citation ← FormatCitation(chunk.metadata)
    contextParts.append("\n## Source " + (i + 1) + ": " + citation + "\n")
    contextParts.append(chunk.metadata.text)
    contextParts.append("\n(Relevance: " + ROUND(chunk.score * 100) + "%)\n")
  END FOR

  contextMessage ← contextParts.join("\n")

  // Step 4: Build enhanced prompt
  enhancedPrompt ← BuildEnhancedPrompt(
    request.params.user_query,
    contextMessage
  )

  latency ← CurrentTimeMillis() - startTime

  response ← {
    description: "PCI-DSS context injected with " + contextChunks.length + " sources",
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: enhancedPrompt
        }
      }
    ],
    metadata: {
      sources_count: contextChunks.length,
      session_history_count: sessionHistory.length,
      latency_ms: latency
    }
  }

  RETURN response
END

SUBROUTINE: BuildEnhancedPrompt
INPUT: userQuery (string), context (string)
OUTPUT: enhancedPrompt (string)

BEGIN
  template ← """
You are a PCI-DSS compliance expert. Use the following context to answer the user's question accurately and provide citations.

{context}

# User Question
{user_query}

# Instructions
1. Answer based ONLY on the provided context
2. Cite all sources using [Source N] notation
3. If the context doesn't contain enough information, say so
4. Provide specific requirement numbers and page references
5. Format your response with clear sections

Answer:
"""

  enhancedPrompt ← template
    .replace("{context}", context)
    .replace("{user_query}", userQuery)

  RETURN enhancedPrompt
END
```

---

### 4.3 Query Suggestion Prompt

```
ALGORITHM: HandleQuerySuggestion
INPUT: request (PromptRequest)
OUTPUT: response (PromptResponse)

BEGIN
  Log("💡 Generating query suggestions for: " + request.params.current_query)

  startTime ← CurrentTimeMillis()

  // Step 1: Analyze current query
  queryEmbedding ← GenerateEmbedding(request.params.current_query)
  queryFeatures ← ExtractQueryFeatures(request.params.current_query)

  // Step 2: Find similar historical queries
  similarQueries ← agentDb.search("query_trajectories", queryEmbedding, {
    k: 20,
    filter: {
      "metadata.reward": {$gte: 0.8}  // Only successful queries
    }
  })

  // Step 3: Extract query patterns
  relatedTopics ← ExtractRelatedTopics(similarQueries)

  // Step 4: Generate suggestions using templates
  suggestions ← []

  // Suggestion 1: Drill-down
  IF queryFeatures.requirementReferences > 0 THEN
    reqId ← ExtractFirstRequirement(request.params.current_query)
    suggestions.append("What are the sub-requirements of " + reqId + "?")
  END IF

  // Suggestion 2: Comparison
  IF queryFeatures.standardReferences == 1 THEN
    standard ← ExtractStandard(request.params.current_query)
    suggestions.append("How does " + standard + " compare to HIPAA?")
  END IF

  // Suggestion 3: Implementation
  IF queryFeatures.intent == "requirement_lookup" THEN
    suggestions.append("How do I implement this requirement?")
  END IF

  // Suggestion 4: Related topics
  FOR EACH topic IN relatedTopics.slice(0, 2) DO
    suggestions.append("Tell me about " + topic)
  END FOR

  // Step 5: Limit to requested number
  suggestions ← suggestions.slice(0, request.params.num_suggestions)

  latency ← CurrentTimeMillis() - startTime

  // Step 6: Build response
  suggestionText ← "Based on your query, you might also be interested in:\n\n"
  FOR i, suggestion IN ENUMERATE(suggestions) DO
    suggestionText ← suggestionText + (i + 1) + ". " + suggestion + "\n"
  END FOR

  response ← {
    description: "Generated " + suggestions.length + " related query suggestions",
    messages: [
      {
        role: "assistant",
        content: {
          type: "text",
          text: suggestionText
        }
      }
    ],
    metadata: {
      current_query: request.params.current_query,
      suggestions: suggestions,
      latency_ms: latency
    }
  }

  RETURN response
END

SUBROUTINE: ExtractRelatedTopics
INPUT: similarQueries (Array<Result>)
OUTPUT: topics (Array<string>)

BEGIN
  topicMap ← EmptyMap()  // Map<topic, count>

  FOR EACH query IN similarQueries DO
    // Extract technical terms from query text
    terms ← ExtractTechnicalTerms(query.metadata.state.query_text)

    FOR EACH term IN terms DO
      IF topicMap.has(term) THEN
        topicMap.set(term, topicMap.get(term) + 1)
      ELSE
        topicMap.set(term, 1)
      END IF
    END FOR
  END FOR

  // Sort by frequency
  topics ← Array.from(topicMap.keys())
  topics.sort((a, b) → topicMap.get(b) - topicMap.get(a))

  RETURN topics
END
```

---

## 5. Query Processing Flow

### 5.1 End-to-End Query Flow

```
ALGORITHM: ProcessQuery
INPUT: query (string), sessionId (string), options (object)
OUTPUT: response (QueryResponse)

TYPES:
  QueryResponse = {
    answer: string,
    sources: Array<Source>,
    citations: Array<string>,
    confidence: float,
    metadata: object
  }

  Source = {
    id: string,
    text: string,
    citation: string,
    score: float,
    metadata: object
  }

BEGIN
  Log("🔄 Processing query: " + query)

  queryStartTime ← CurrentTimeMillis()

  // === PHASE 1: Query Analysis ===
  Log("📊 Phase 1: Query Analysis")

  // Classify query using ruv-FANN
  classification ← CallMCPTool("classify_query", {query: query})

  Log("Classification: " + classification.complexity)
  Log("Suggested strategy: " + classification.suggestedStrategy)

  // === PHASE 2: agentic-flow Swarm Initialization ===
  Log("🤖 Phase 2: Swarm Initialization")

  swarm ← InitializeAgenticFlowSwarm({
    topology: classification.suggestedTopology,
    maxAgents: classification.suggestedAgents,
    strategy: "adaptive"
  })

  // === PHASE 3: Parallel Retrieval ===
  Log("🔍 Phase 3: Parallel Retrieval")

  retrievalTasks ← []

  // Spawn retrieval agents based on strategy
  IF classification.suggestedStrategy == "hybrid" THEN
    retrievalTasks ← [
      SpawnAgent(swarm, "retrieval", "semantic_search", {
        query: query,
        top_k: 20
      }),
      SpawnAgent(swarm, "retrieval", "keyword_search", {
        query: query,
        top_k: 20
      })
    ]
  ELSE IF classification.suggestedStrategy == "hnsw" THEN
    retrievalTasks ← [
      SpawnAgent(swarm, "retrieval", "semantic_search", {
        query: query,
        top_k: 20
      })
    ]
  ELSE
    // Complex strategy: multi-stage retrieval
    retrievalTasks ← [
      SpawnAgent(swarm, "retrieval", "semantic_search", {
        query: query,
        top_k: 30
      }),
      SpawnAgent(swarm, "reasoning", "graph_walk", {
        query: query,
        depth: 2
      })
    ]
  END IF

  // Wait for all retrieval tasks
  retrievalResults ← AWAIT_ALL(retrievalTasks)

  // Merge results
  allChunks ← MergeRetrievalResults(retrievalResults)

  retrievalTime ← CurrentTimeMillis() - queryStartTime
  Log("⚡ Retrieval completed in " + retrievalTime + "ms")

  // === PHASE 4: Response Synthesis ===
  Log("✍️ Phase 4: Response Synthesis")

  // Spawn synthesis agent
  synthesisAgent ← SpawnAgent(swarm, "synthesis", "generate_response", {
    query: query,
    chunks: allChunks,
    sessionId: sessionId
  })

  synthesisResult ← AWAIT synthesisAgent

  // === PHASE 5: Verification ===
  Log("✅ Phase 5: Verification")

  // Parallel verification checks
  verificationTasks ← [
    SpawnAgent(swarm, "verification", "check_citations", {
      response: synthesisResult,
      chunks: allChunks
    }),
    SpawnAgent(swarm, "verification", "check_consistency", {
      response: synthesisResult
    }),
    SpawnAgent(swarm, "verification", "check_completeness", {
      response: synthesisResult,
      query: query
    })
  ]

  verificationResults ← AWAIT_ALL(verificationTasks)

  // Aggregate verification scores
  overallConfidence ← AggregateVerificationScores(verificationResults)

  // === PHASE 6: RL Trajectory Storage ===
  Log("💾 Phase 6: Learning")

  // Calculate reward (accuracy proxy)
  reward ← CalculateReward(overallConfidence, retrievalTime, classification)

  // Store trajectory for learning
  trajectory ← {
    state: {
      query_complexity: classification.complexityScore,
      query_intent: classification.intent,
      session_length: GetSessionLength(sessionId)
    },
    action: {
      strategy_chosen: classification.suggestedStrategy,
      topology_used: classification.suggestedTopology,
      num_agents: swarm.agentCount,
      top_k: 20
    },
    reward: reward,
    next_state: {
      // Next state captured on next query
    }
  }

  CallMCPTool("store_trajectory", trajectory)

  // === PHASE 7: Response Delivery ===
  totalLatency ← CurrentTimeMillis() - queryStartTime

  Log("🎉 Query processing complete in " + totalLatency + "ms")

  response ← {
    answer: synthesisResult.answer,
    sources: FormatSources(allChunks),
    citations: ExtractCitations(synthesisResult),
    confidence: overallConfidence,
    metadata: {
      query: query,
      classification: classification,
      retrieval_time_ms: retrievalTime,
      total_latency_ms: totalLatency,
      agents_used: swarm.agentCount,
      topology: swarm.topology,
      reward: reward
    }
  }

  // Cleanup swarm
  DestroySwarm(swarm)

  RETURN response
END
```

**Performance Breakdown:**
- Phase 1 (Classification): ~5-20ms
- Phase 2 (Swarm Init): ~10-30ms
- Phase 3 (Retrieval): ~100-200ms (parallel)
- Phase 4 (Synthesis): ~50-100ms
- Phase 5 (Verification): ~30-50ms (parallel)
- Phase 6 (RL Storage): ~5-10ms
- **Total P95:** <500ms

---

### 5.2 agentic-flow Swarm Integration

```
ALGORITHM: InitializeAgenticFlowSwarm
INPUT: config (SwarmConfig)
OUTPUT: swarm (SwarmInstance)

TYPES:
  SwarmConfig = {
    topology: "mesh" | "star" | "hierarchical",
    maxAgents: integer,
    strategy: "balanced" | "specialized" | "adaptive"
  }

  SwarmInstance = {
    id: string,
    topology: string,
    agents: Map<string, Agent>,
    agentCount: integer,
    messageQueue: Queue<Message>
  }

BEGIN
  Log("🤖 Initializing agentic-flow swarm")

  // Call agentic-flow initialization
  swarmId ← agenticFlow.init({
    topology: config.topology,
    maxAgents: config.maxAgents,
    strategy: config.strategy
  })

  swarm ← {
    id: swarmId,
    topology: config.topology,
    agents: EmptyMap(),
    agentCount: 0,
    messageQueue: CreateQueue()
  }

  Log("✅ Swarm initialized: " + swarmId)
  RETURN swarm
END

ALGORITHM: SpawnAgent
INPUT: swarm (SwarmInstance), agentType (string), task (string), params (object)
OUTPUT: agentTask (Promise)

BEGIN
  Log("🔄 Spawning " + agentType + " agent for: " + task)

  // Define agent capabilities based on type
  capabilities ← DefineAgentCapabilities(agentType, task)

  // Spawn agent via agentic-flow
  agent ← agenticFlow.spawnAgent({
    swarmId: swarm.id,
    type: agentType,
    capabilities: capabilities,
    task: task,
    params: params
  })

  // Store agent reference
  swarm.agents.set(agent.id, agent)
  swarm.agentCount ← swarm.agentCount + 1

  // Return promise for agent completion
  agentTask ← agent.execute()

  RETURN agentTask
END

SUBROUTINE: DefineAgentCapabilities
INPUT: agentType (string), task (string)
OUTPUT: capabilities (Array<string>)

BEGIN
  IF agentType == "retrieval" THEN
    IF task == "semantic_search" THEN
      RETURN ["mcp_tool:semantic_search", "embedding_generation"]
    ELSE IF task == "keyword_search" THEN
      RETURN ["full_text_search", "keyword_extraction"]
    ELSE IF task == "graph_walk" THEN
      RETURN ["graph_traversal", "relationship_extraction"]
    END IF

  ELSE IF agentType == "reasoning" THEN
    RETURN ["pattern_matching", "inference", "ruv_fann_classification"]

  ELSE IF agentType == "synthesis" THEN
    RETURN ["text_generation", "citation_formatting", "template_rendering"]

  ELSE IF agentType == "verification" THEN
    IF task == "check_citations" THEN
      RETURN ["citation_validation", "cross_referencing"]
    ELSE IF task == "check_consistency" THEN
      RETURN ["contradiction_detection", "logical_analysis"]
    ELSE IF task == "check_completeness" THEN
      RETURN ["coverage_analysis", "gap_detection"]
    END IF
  END IF

  RETURN []
END
```

---

### 5.3 Result Merging and Reranking

```
ALGORITHM: MergeRetrievalResults
INPUT: retrievalResults (Array<AgentResult>)
OUTPUT: mergedChunks (Array<Chunk>)

BEGIN
  Log("🔀 Merging retrieval results from " + retrievalResults.length + " agents")

  chunkMap ← EmptyMap()  // Map<chunkId, Chunk>

  // Combine results from all agents
  FOR EACH agentResult IN retrievalResults DO
    FOR EACH chunk IN agentResult.chunks DO
      IF chunkMap.has(chunk.id) THEN
        // Chunk already seen, boost score
        existing ← chunkMap.get(chunk.id)
        existing.score ← MAX(existing.score, chunk.score)
        existing.sourceCount ← existing.sourceCount + 1
      ELSE
        // New chunk
        chunk.sourceCount ← 1
        chunkMap.set(chunk.id, chunk)
      END IF
    END FOR
  END FOR

  // Convert to array
  mergedChunks ← Array.from(chunkMap.values())

  // Rerank using cross-encoder (optional, expensive)
  IF mergedChunks.length > 30 THEN
    Log("🔄 Reranking top 30 results")
    topChunks ← mergedChunks.slice(0, 30)
    rerankedChunks ← RerankWithCrossEncoder(topChunks, query)
    mergedChunks ← rerankedChunks.concat(mergedChunks.slice(30))
  END IF

  // Sort by score descending
  mergedChunks.sort((a, b) → b.score - a.score)

  Log("✅ Merged " + mergedChunks.length + " unique chunks")

  RETURN mergedChunks
END

SUBROUTINE: RerankWithCrossEncoder
INPUT: chunks (Array<Chunk>), query (string)
OUTPUT: rerankedChunks (Array<Chunk>)

BEGIN
  // Cross-encoder reranking (optional enhancement)
  // Uses a more expensive model for better accuracy

  Log("🧠 Cross-encoder reranking")

  // Batch process for efficiency
  batchSize ← 10
  batches ← SplitIntoBatches(chunks, batchSize)

  rerankedChunks ← []

  FOR EACH batch IN batches DO
    // Prepare input pairs
    pairs ← []
    FOR EACH chunk IN batch DO
      pairs.append({
        query: query,
        document: chunk.text
      })
    END FOR

    // Call cross-encoder API (e.g., Cohere rerank)
    scores ← CallCrossEncoderAPI(pairs)

    // Update chunk scores
    FOR i, chunk IN ENUMERATE(batch) DO
      chunk.rerankScore ← scores[i]
      chunk.originalScore ← chunk.score
      chunk.score ← (chunk.score * 0.4) + (scores[i] * 0.6)  // Weighted blend
    END FOR

    rerankedChunks.append(batch)
  END FOR

  // Flatten and sort
  rerankedChunks ← FLATTEN(rerankedChunks)
  rerankedChunks.sort((a, b) → b.score - a.score)

  RETURN rerankedChunks
END
```

---

## 6. Integration Layers

### 6.1 MCP ↔ AgentDB Integration

```
ALGORITHM: AgentDBMCPBridge
PURPOSE: Bridge between MCP tools and AgentDB operations

BEGIN
  // Singleton pattern for AgentDB connection
  GLOBAL agentDbInstance ← NULL

  FUNCTION GetAgentDBInstance()
    IF agentDbInstance IS NULL THEN
      agentDbInstance ← AgentDB.initialize(globalConfig)
    END IF
    RETURN agentDbInstance
  END FUNCTION

  // MCP Tool → AgentDB mapping
  FUNCTION ExecuteMCPToolOnAgentDB(toolName, params)
    agentDb ← GetAgentDBInstance()

    SWITCH toolName
      CASE "semantic_search":
        embedding ← GenerateEmbedding(params.query)
        RETURN agentDb.search("document_chunks", embedding, params)

      CASE "hybrid_search":
        RETURN PerformHybridSearch(agentDb, params)

      CASE "retrieve_document":
        RETURN agentDb.query("document_chunks", {
          filter: {document_id: params.document_id}
        })

      CASE "store_trajectory":
        embedding ← GenerateEmbedding(SerializeTrajectory(params))
        RETURN agentDb.insert("query_trajectories", {
          embedding: embedding,
          metadata: params
        })

      DEFAULT:
        THROW Error("Unknown MCP tool: " + toolName)
    END SWITCH
  END FUNCTION
END
```

---

### 6.2 MCP ↔ ruv-FANN Integration

```
ALGORITHM: RuvFANNMCPBridge
PURPOSE: Integrate ruv-FANN WASM neural networks with MCP tools

TYPES:
  FANNModel = {
    name: string,
    wasmModule: WebAssembly.Module,
    inputSize: integer,
    outputSize: integer,
    layers: Array<integer>
  }

BEGIN
  // Load WASM models on startup
  GLOBAL classifierModel ← NULL
  GLOBAL scoringModel ← NULL

  FUNCTION LoadRuvFANNModels()
    Log("🧠 Loading ruv-FANN WASM models")

    // Load query complexity classifier
    classifierModel ← LoadWASMModel("models/query_classifier.wasm", {
      inputSize: 8,
      outputSize: 6,
      layers: [8, 16, 8, 6]
    })

    // Load relevance scoring model
    scoringModel ← LoadWASMModel("models/relevance_scorer.wasm", {
      inputSize: 1536 * 2,  // Query + document embeddings
      outputSize: 1,
      layers: [3072, 512, 128, 1]
    })

    Log("✅ ruv-FANN models loaded")
  END FUNCTION

  FUNCTION LoadWASMModel(path, config)
    // Load WASM file
    wasmBytes ← ReadFile(path)
    wasmModule ← WebAssembly.compile(wasmBytes)

    // Instantiate with imports
    instance ← WebAssembly.instantiate(wasmModule, {
      env: {
        memory: new WebAssembly.Memory({initial: 256, maximum: 512})
      }
    })

    RETURN {
      name: path,
      wasmModule: wasmModule,
      instance: instance,
      inputSize: config.inputSize,
      outputSize: config.outputSize,
      layers: config.layers
    }
  END FUNCTION

  // MCP Tool: classify_query implementation
  FUNCTION ClassifyQueryWithFANN(query)
    IF classifierModel IS NULL THEN
      LoadRuvFANNModels()
    END IF

    // Extract features
    features ← ExtractQueryFeatures(query)

    // Prepare input array
    input ← [
      features.tokenCount / 100,          // Normalize
      features.questionWords / 5,
      features.comparisonWords / 3,
      features.requirementReferences / 10,
      features.standardReferences / 3,
      features.avgWordLength / 10,
      features.sentenceCount / 5,
      features.technicalTerms / 20
    ]

    // Run WASM inference
    output ← classifierModel.instance.exports.predict(input)

    // Parse output
    classification ← {
      complexityScore: output[0],
      complexity: InterpretComplexity(output[0]),
      intent: InterpretIntent(output.slice(1, 5)),
      confidence: output[5]
    }

    RETURN classification
  END FUNCTION

  // MCP Tool: score_relevance implementation
  FUNCTION ScoreRelevanceWithFANN(queryEmbedding, docEmbedding)
    IF scoringModel IS NULL THEN
      LoadRuvFANNModels()
    END IF

    // Concatenate embeddings
    input ← queryEmbedding.concat(docEmbedding)

    // Run inference
    score ← scoringModel.instance.exports.predict(input)[0]

    RETURN score
  END FUNCTION
END
```

**ruv-FANN Performance:**
- WASM compilation: ~50-100ms (one-time)
- Inference: <5ms per query
- Memory: ~10-20MB per model
- Thread-safe: Yes (separate instances)

---

### 6.3 MCP ↔ agentic-flow Integration

```
ALGORITHM: AgenticFlowMCPBridge
PURPOSE: Orchestrate multi-agent queries via agentic-flow

BEGIN
  GLOBAL activeSwarms ← EmptyMap()  // Map<swarmId, SwarmInstance>

  // Initialize swarm for query
  FUNCTION InitializeQuerySwarm(classification)
    swarmConfig ← {
      topology: classification.suggestedTopology,
      maxAgents: classification.suggestedAgents,
      strategy: "adaptive"
    }

    swarmId ← agenticFlow.init(swarmConfig)

    swarm ← {
      id: swarmId,
      topology: swarmConfig.topology,
      agents: [],
      startTime: CurrentTimeMillis()
    }

    activeSwarms.set(swarmId, swarm)

    RETURN swarm
  END FUNCTION

  // Spawn agents in parallel
  FUNCTION SpawnParallelAgents(swarm, tasks)
    agentPromises ← []

    FOR EACH task IN tasks DO
      agentId ← agenticFlow.spawnAgent({
        swarmId: swarm.id,
        type: task.type,
        capabilities: task.capabilities,
        task: task.name,
        params: task.params
      })

      swarm.agents.append(agentId)

      agentPromise ← agenticFlow.executeAgent(agentId)
      agentPromises.append(agentPromise)
    END FOR

    // Wait for all agents to complete
    results ← AWAIT Promise.all(agentPromises)

    RETURN results
  END FUNCTION

  // Cleanup swarm
  FUNCTION DestroySwarm(swarm)
    agenticFlow.destroy(swarm.id)
    activeSwarms.delete(swarm.id)

    duration ← CurrentTimeMillis() - swarm.startTime
    Log("✅ Swarm destroyed: " + swarm.id + " (duration: " + duration + "ms)")
  END FUNCTION
END
```

**agentic-flow Topologies:**

1. **Mesh** (Peer-to-Peer)
   - Use case: Simple queries, 1-2 agents
   - Latency: Lowest
   - Coordination: Minimal

2. **Star** (Coordinator + Workers)
   - Use case: Moderate queries, 2-4 agents
   - Latency: Medium
   - Coordination: Central coordinator

3. **Hierarchical** (Tree)
   - Use case: Complex queries, 4-6 agents
   - Latency: Higher
   - Coordination: Multi-level (coordinator → supervisors → workers)

---

## 7. Error Handling

### 7.1 Graceful Degradation Strategy

```
ALGORITHM: HandleQueryError
INPUT: query (string), error (Error), context (object)
OUTPUT: response (QueryResponse)

BEGIN
  Log("❌ Error during query processing", error)

  // Determine error severity
  severity ← ClassifyErrorSeverity(error)

  SWITCH severity
    CASE "critical":
      // Cannot proceed, return error response
      RETURN {
        error: TRUE,
        message: "Query processing failed: " + error.message,
        fallback: NULL,
        metadata: {
          query: query,
          error_type: error.type,
          timestamp: CurrentTimestamp()
        }
      }

    CASE "high":
      // Try fallback strategy
      Log("⚠️ Attempting fallback strategy")
      fallbackResponse ← FallbackQueryProcessing(query, context)
      RETURN fallbackResponse

    CASE "medium":
      // Retry with degraded performance
      Log("🔄 Retrying with degraded configuration")
      degradedResponse ← RetryWithDegradedConfig(query, context)
      RETURN degradedResponse

    CASE "low":
      // Log and continue
      Log("⚠️ Minor error, continuing", error)
      RETURN ContinueWithBestEffort(query, context)
  END SWITCH
END

SUBROUTINE: ClassifyErrorSeverity
INPUT: error (Error)
OUTPUT: severity (string)

BEGIN
  IF error.type == "AgentDBConnectionError" THEN
    RETURN "critical"

  ELSE IF error.type == "EmbeddingAPIError" THEN
    RETURN "high"

  ELSE IF error.type == "SwarmInitializationError" THEN
    RETURN "high"

  ELSE IF error.type == "TimeoutError" THEN
    RETURN "medium"

  ELSE IF error.type == "PartialResultsError" THEN
    RETURN "low"

  ELSE
    RETURN "medium"
  END IF
END

SUBROUTINE: FallbackQueryProcessing
INPUT: query (string), context (object)
OUTPUT: response (QueryResponse)

BEGIN
  Log("🔄 Executing fallback query processing")

  TRY
    // Fallback 1: Try simpler strategy
    // Use direct HNSW search instead of hybrid

    embedding ← GenerateEmbedding(query)

    IF embedding IS NULL THEN
      // Fallback 2: Use keyword search only
      keywords ← ExtractKeywords(query)
      results ← agentDb.fullTextSearch("document_chunks", keywords, {
        limit: 20
      })
    ELSE
      results ← agentDb.search("document_chunks", embedding, {
        k: 20,
        ef: 50  // Lower search depth for speed
      })
    END IF

    // Simple template-based response
    response ← GenerateSimpleResponse(query, results)

    response.metadata.fallback_used ← TRUE
    response.metadata.fallback_reason ← "Primary strategy failed"

    RETURN response

  CATCH fallbackError
    Log("❌ Fallback also failed", fallbackError)
    RETURN {
      error: TRUE,
      message: "All query strategies failed",
      query: query,
      metadata: {
        primary_error: context.primaryError,
        fallback_error: fallbackError.message
      }
    }
  END TRY
END
```

---

### 7.2 Retry Logic with Exponential Backoff

```
ALGORITHM: RetryWithBackoff
INPUT: operation (Function), maxRetries (integer), baseDelay (integer)
OUTPUT: result or error

BEGIN
  retryCount ← 0
  lastError ← NULL

  WHILE retryCount < maxRetries DO
    TRY
      result ← operation()
      RETURN result

    CATCH error
      lastError ← error
      retryCount ← retryCount + 1

      IF retryCount >= maxRetries THEN
        BREAK
      END IF

      // Calculate exponential backoff delay
      delay ← baseDelay * POWER(2, retryCount - 1)

      // Add jitter (±20%)
      jitter ← delay * (0.8 + RANDOM() * 0.4)

      Log("⏰ Retry " + retryCount + "/" + maxRetries + " after " + jitter + "ms")

      Sleep(jitter)
    END TRY
  END WHILE

  // All retries exhausted
  Log("❌ All retries exhausted for operation")
  THROW Error("Operation failed after " + maxRetries + " retries: " + lastError.message)
END

// Example usage:
FUNCTION CallEmbeddingAPIWithRetry(text)
  RETURN RetryWithBackoff(
    () → CallOpenAIEmbeddings({model: "ada-002", input: text}),
    maxRetries: 3,
    baseDelay: 1000  // Start with 1 second
  )
END
```

**Retry Strategy:**
- **Attempt 1**: Immediate
- **Attempt 2**: 1s delay (±200ms jitter)
- **Attempt 3**: 2s delay (±400ms jitter)
- **Attempt 4**: 4s delay (±800ms jitter)

---

### 7.3 Circuit Breaker Pattern

```
ALGORITHM: CircuitBreaker
PURPOSE: Prevent cascading failures by temporarily blocking failing operations

TYPES:
  CircuitState = "CLOSED" | "OPEN" | "HALF_OPEN"

  CircuitBreakerConfig = {
    failureThreshold: integer,    // Open after N failures
    successThreshold: integer,    // Close after N successes in HALF_OPEN
    timeout: integer,             // Time to wait before HALF_OPEN (ms)
    monitoringPeriod: integer     // Time window for failures (ms)
  }

  CircuitBreakerInstance = {
    state: CircuitState,
    failureCount: integer,
    successCount: integer,
    lastFailureTime: integer,
    config: CircuitBreakerConfig
  }

BEGIN
  FUNCTION CreateCircuitBreaker(config)
    RETURN {
      state: "CLOSED",
      failureCount: 0,
      successCount: 0,
      lastFailureTime: 0,
      config: config
    }
  END FUNCTION

  FUNCTION ExecuteWithCircuitBreaker(breaker, operation)
    // Check circuit state
    IF breaker.state == "OPEN" THEN
      timeSinceLastFailure ← CurrentTimeMillis() - breaker.lastFailureTime

      IF timeSinceLastFailure >= breaker.config.timeout THEN
        // Transition to HALF_OPEN
        Log("🔄 Circuit breaker: OPEN → HALF_OPEN")
        breaker.state ← "HALF_OPEN"
        breaker.successCount ← 0
      ELSE
        // Reject request
        THROW Error("Circuit breaker is OPEN, rejecting request")
      END IF
    END IF

    // Execute operation
    TRY
      result ← operation()

      // Success
      OnSuccess(breaker)

      RETURN result

    CATCH error
      // Failure
      OnFailure(breaker)

      THROW error
    END TRY
  END FUNCTION

  FUNCTION OnSuccess(breaker)
    IF breaker.state == "HALF_OPEN" THEN
      breaker.successCount ← breaker.successCount + 1

      IF breaker.successCount >= breaker.config.successThreshold THEN
        // Close circuit
        Log("✅ Circuit breaker: HALF_OPEN → CLOSED")
        breaker.state ← "CLOSED"
        breaker.failureCount ← 0
      END IF

    ELSE IF breaker.state == "CLOSED" THEN
      // Reset failure count on success
      breaker.failureCount ← 0
    END IF
  END FUNCTION

  FUNCTION OnFailure(breaker)
    breaker.lastFailureTime ← CurrentTimeMillis()

    IF breaker.state == "HALF_OPEN" THEN
      // Immediately reopen
      Log("❌ Circuit breaker: HALF_OPEN → OPEN")
      breaker.state ← "OPEN"

    ELSE IF breaker.state == "CLOSED" THEN
      breaker.failureCount ← breaker.failureCount + 1

      IF breaker.failureCount >= breaker.config.failureThreshold THEN
        // Open circuit
        Log("❌ Circuit breaker: CLOSED → OPEN")
        breaker.state ← "OPEN"
      END IF
    END IF
  END FUNCTION
END

// Global circuit breakers
GLOBAL embeddingAPIBreaker ← CreateCircuitBreaker({
  failureThreshold: 5,
  successThreshold: 2,
  timeout: 30000,  // 30 seconds
  monitoringPeriod: 60000
})

GLOBAL agentDBBreaker ← CreateCircuitBreaker({
  failureThreshold: 3,
  successThreshold: 2,
  timeout: 10000,  // 10 seconds
  monitoringPeriod: 30000
})
```

---

## 8. Performance Optimization

### 8.1 Caching Strategy

```
ALGORITHM: MultiLevelCache
PURPOSE: Optimize query performance with multi-level caching

TYPES:
  CacheLevel = {
    name: string,
    storage: Map<string, CacheEntry>,
    ttl: integer,
    maxSize: integer,
    hitRate: float
  }

  CacheEntry = {
    key: string,
    value: any,
    timestamp: integer,
    accessCount: integer,
    size: integer
  }

BEGIN
  // Level 1: Embedding Cache (in-memory)
  embeddingCache ← {
    name: "embedding_cache",
    storage: new LRUCache({maxSize: 10000}),
    ttl: 3600000,  // 1 hour
    maxSize: 10000,
    hitRate: 0
  }

  // Level 2: Query Results Cache (in-memory)
  queryCache ← {
    name: "query_cache",
    storage: new LRUCache({maxSize: 1000}),
    ttl: 1800000,  // 30 minutes
    maxSize: 1000,
    hitRate: 0
  }

  // Level 3: Session Context Cache (AgentDB)
  sessionCache ← {
    name: "session_cache",
    storage: "agentdb:session_memory",
    ttl: 86400000,  // 24 hours
    maxSize: Infinity
  }

  FUNCTION GetCachedEmbedding(text)
    cacheKey ← HashText(text)

    // Check L1 cache
    cached ← embeddingCache.storage.get(cacheKey)

    IF cached IS NOT NULL AND NOT IsExpired(cached) THEN
      Log("✅ Embedding cache HIT")
      embeddingCache.hitRate ← UpdateHitRate(embeddingCache, TRUE)
      RETURN cached.value
    END IF

    Log("❌ Embedding cache MISS")
    embeddingCache.hitRate ← UpdateHitRate(embeddingCache, FALSE)

    // Generate embedding
    embedding ← GenerateEmbedding(text)

    // Store in cache
    embeddingCache.storage.set(cacheKey, {
      key: cacheKey,
      value: embedding,
      timestamp: CurrentTimeMillis(),
      accessCount: 1,
      size: embedding.length * 4  // 4 bytes per float32
    })

    RETURN embedding
  END FUNCTION

  FUNCTION GetCachedQueryResults(query, params)
    cacheKey ← HashQueryWithParams(query, params)

    cached ← queryCache.storage.get(cacheKey)

    IF cached IS NOT NULL AND NOT IsExpired(cached) THEN
      Log("✅ Query cache HIT")
      queryCache.hitRate ← UpdateHitRate(queryCache, TRUE)
      RETURN cached.value
    END IF

    Log("❌ Query cache MISS")
    queryCache.hitRate ← UpdateHitRate(queryCache, FALSE)

    RETURN NULL  // Cache miss, execute query
  END FUNCTION

  FUNCTION CacheQueryResults(query, params, results)
    cacheKey ← HashQueryWithParams(query, params)

    queryCache.storage.set(cacheKey, {
      key: cacheKey,
      value: results,
      timestamp: CurrentTimeMillis(),
      accessCount: 1,
      size: EstimateSize(results)
    })
  END FUNCTION

  FUNCTION IsExpired(entry)
    age ← CurrentTimeMillis() - entry.timestamp
    RETURN age > embeddingCache.ttl
  END FUNCTION
END
```

**Cache Performance:**
- Embedding cache hit rate: ~70-80%
- Query cache hit rate: ~30-40%
- Latency reduction: ~100-200ms on cache hit

---

### 8.2 Batch Processing

```
ALGORITHM: BatchEmbeddingGeneration
INPUT: texts (Array<string>)
OUTPUT: embeddings (Array<Array<float>>)

PURPOSE: Reduce API calls by batching embedding requests

BEGIN
  Log("🔄 Batch embedding generation for " + texts.length + " texts")

  embeddings ← []
  batchSize ← 100  // OpenAI API limit

  // Check cache first
  uncachedTexts ← []
  cachedEmbeddings ← EmptyMap()  // Map<index, embedding>

  FOR i, text IN ENUMERATE(texts) DO
    cached ← GetCachedEmbedding(text)
    IF cached IS NOT NULL THEN
      cachedEmbeddings.set(i, cached)
    ELSE
      uncachedTexts.append({index: i, text: text})
    END IF
  END FOR

  Log("📊 Cache hits: " + cachedEmbeddings.size + "/" + texts.length)

  // Batch process uncached texts
  batches ← SplitIntoBatches(uncachedTexts, batchSize)

  FOR EACH batch IN batches DO
    batchTexts ← batch.map(item → item.text)

    // Call OpenAI batch API
    batchEmbeddings ← CallOpenAIEmbeddings({
      model: "text-embedding-ada-002",
      input: batchTexts
    })

    // Store in cache and result array
    FOR i, item IN ENUMERATE(batch) DO
      embedding ← batchEmbeddings[i]

      // Cache
      CacheEmbedding(item.text, embedding)

      // Store in result map
      cachedEmbeddings.set(item.index, embedding)
    END FOR
  END FOR

  // Reconstruct in original order
  FOR i FROM 0 TO texts.length - 1 DO
    embeddings.append(cachedEmbeddings.get(i))
  END FOR

  RETURN embeddings
END
```

**Batch Performance:**
- API calls: 100x reduction (1 call for 100 texts)
- Latency: ~500ms for 100 embeddings (vs ~50s sequential)
- Cost: Same per-token cost, fewer request overheads

---

### 8.3 Parallel Query Processing

```
ALGORITHM: ProcessQueriesInParallel
INPUT: queries (Array<string>), concurrencyLimit (integer)
OUTPUT: responses (Array<QueryResponse>)

BEGIN
  Log("🔄 Processing " + queries.length + " queries with concurrency " + concurrencyLimit)

  // Create semaphore to limit concurrency
  semaphore ← CreateSemaphore(concurrencyLimit)

  // Create promise for each query
  queryPromises ← []

  FOR EACH query IN queries DO
    queryPromise ← ASYNC {
      // Acquire semaphore
      AWAIT semaphore.acquire()

      TRY
        response ← ProcessQuery(query, NULL, {})
        RETURN response
      FINALLY
        semaphore.release()
      END TRY
    }

    queryPromises.append(queryPromise)
  END FOR

  // Wait for all queries to complete
  responses ← AWAIT Promise.all(queryPromises)

  Log("✅ All " + queries.length + " queries processed")

  RETURN responses
END

SUBROUTINE: CreateSemaphore
INPUT: limit (integer)
OUTPUT: semaphore (Semaphore)

BEGIN
  semaphore ← {
    limit: limit,
    current: 0,
    queue: []
  }

  semaphore.acquire ← ASYNC FUNCTION() {
    IF semaphore.current < semaphore.limit THEN
      semaphore.current ← semaphore.current + 1
      RETURN
    ELSE
      // Wait in queue
      RETURN NEW Promise((resolve) → {
        semaphore.queue.append(resolve)
      })
    END IF
  }

  semaphore.release ← FUNCTION() {
    IF semaphore.queue.length > 0 THEN
      resolve ← semaphore.queue.shift()
      resolve()
    ELSE
      semaphore.current ← semaphore.current - 1
    END IF
  }

  RETURN semaphore
END
```

---

## 9. Complexity Analysis Summary

### 9.1 Time Complexity

| Operation | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| **Server Init** | O(n log n) | O(n log n) | O(n log n) |
| **Embedding Generation** | O(1) | O(1) | O(1) |
| **HNSW Search** | O(log n) | O(log n) | O(n) |
| **Hybrid Search** | O(log n) | O(log n) | O(n) |
| **Query Classification** | O(1) | O(1) | O(1) |
| **Trajectory Storage** | O(1) | O(log n) | O(log n) |
| **End-to-End Query** | O(log n) | O(log n) | O(n) |

**n** = number of vectors in AgentDB

---

### 9.2 Space Complexity

| Component | Memory Usage |
|-----------|--------------|
| **HNSW Index (100K vectors, no quantization)** | ~600MB |
| **HNSW Index (100K vectors, scalar quantization)** | ~150MB |
| **Embedding Cache (10K entries)** | ~60MB |
| **Query Cache (1K entries)** | ~20MB |
| **ruv-FANN Models (2 models)** | ~30MB |
| **AgentDB Connection Pool** | ~50MB |
| **Total Runtime Memory** | ~300-400MB |

---

### 9.3 Performance Targets

| Metric | Target | Achieved (Expected) |
|--------|--------|---------------------|
| **Cold Start** | <30s | ~15-20s |
| **P50 Latency** | <300ms | ~250ms |
| **P95 Latency** | <500ms | ~450ms |
| **P99 Latency** | <1000ms | ~800ms |
| **Throughput** | 100 qps | ~120 qps |
| **HNSW Search** | <100µs | ~50-80µs |
| **Embedding Generation** | <200ms | ~100-150ms |
| **Cache Hit Rate** | >60% | ~70% |

---

## 10. Conclusion

This pseudocode specification provides a complete algorithm design for the MCP-based RAG system. Key design decisions include:

1. **MCP-First Architecture**: All functionality exposed via MCP resources, tools, and prompts
2. **AgentDB Integration**: HNSW indexing for 150x faster search
3. **ruv-FANN Classification**: <5ms query classification via WASM
4. **agentic-flow Orchestration**: Dynamic multi-agent coordination
5. **Graceful Degradation**: Multi-level error handling and fallbacks
6. **Performance Optimization**: Multi-level caching, batching, parallel processing

**Next Phase**: SPARC Phase 3 - Architecture will translate these algorithms into detailed system design with deployment diagrams.

---

**Document Version:** 1.0
**Last Updated:** October 25, 2025
**Status:** Ready for Architecture Phase
