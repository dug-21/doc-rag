# SPARC Pseudocode: FACT Integration Logic
## Phase 5 - Implementation Algorithms

**Document Version**: 1.0  
**Date**: January 8, 2025  
**Dependencies**: 01-SPARC-Specification.md

---

## 1. Core Integration Flow

```pseudocode
FUNCTION integrate_fact_system():
    // Phase 1: Cleanup
    REMOVE_DIRECTORY("src/fact/")
    UPDATE_DEPENDENCIES("Cargo.toml", add_fact_external)
    
    // Phase 2: Initialize
    fact_client = INITIALIZE_FACT_CLIENT(config)
    mcp_server = INITIALIZE_MCP_SERVER(fact_client)
    
    // Phase 3: Register Tools
    FOR EACH tool IN required_tools:
        mcp_server.REGISTER_TOOL(tool)
    
    // Phase 4: Update Query Processor
    UPDATE_QUERY_PROCESSOR(fact_client)
    
    // Phase 5: Validate
    RUN_INTEGRATION_TESTS()
    VERIFY_PERFORMANCE_METRICS()
    
    RETURN success
```

## 2. Query Processing with FACT

```pseudocode
FUNCTION process_query_with_fact(query: String) -> ProcessedQuery:
    START_TIMER()
    
    // Step 1: Cache-first check (Target: <23ms)
    cache_key = GENERATE_CACHE_KEY(query)
    cached_result = fact_client.GET(cache_key)
    
    IF cached_result EXISTS:
        RECORD_METRIC("cache_hit", timer.elapsed())
        RETURN DESERIALIZE(cached_result)
    
    // Step 2: Cache miss - execute retrieval (Target: <95ms)
    RECORD_METRIC("cache_miss", 1)
    
    // Step 3: MRAP Control Loop (DAA compliance)
    result = MRAP_LOOP:
        MONITOR: Analyze query intent
        REASON: Select optimal tools
        ACT: Execute tool-based retrieval
        REFLECT: Validate results
        ADAPT: Update cache strategy
    
    // Step 4: Cache the result
    serialized = SERIALIZE(result)
    ttl = CALCULATE_TTL(result.confidence, result.volatility)
    fact_client.SET(cache_key, serialized, ttl)
    
    // Step 5: Return result
    RECORD_METRIC("total_latency", timer.elapsed())
    RETURN result
```

## 3. MCP Tool Registration

```pseudocode
FUNCTION register_mcp_tools(mcp_server: MCPServer):
    // Define tools for RAG operations
    tools = [
        TOOL("search_documents", {
            description: "Search compliance documents",
            parameters: {query: String, limit: Integer},
            handler: search_handler
        }),
        
        TOOL("extract_citations", {
            description: "Extract citations from sources",
            parameters: {document_id: String, section: String},
            handler: citation_handler
        }),
        
        TOOL("validate_facts", {
            description: "Validate facts against sources",
            parameters: {claim: String, sources: Array},
            handler: validation_handler
        })
    ]
    
    FOR EACH tool IN tools:
        mcp_server.REGISTER(tool)
        LOG("Registered tool: " + tool.name)
```

## 4. Cache Key Generation

```pseudocode
FUNCTION generate_cache_key(query: String) -> String:
    // Normalize query for consistent caching
    normalized = query.LOWERCASE().TRIM()
    
    // Remove stop words
    tokens = TOKENIZE(normalized)
    meaningful_tokens = FILTER(tokens, NOT is_stopword)
    
    // Sort for consistency
    sorted_tokens = SORT(meaningful_tokens)
    
    // Generate hash
    key_base = JOIN(sorted_tokens, "_")
    hash = SHA256(key_base)
    
    // Add version prefix for cache invalidation
    RETURN "fact_v1_" + hash
```

## 5. Intelligent TTL Calculation

```pseudocode
FUNCTION calculate_ttl(confidence: Float, volatility: Float) -> Duration:
    // Base TTL in seconds
    BASE_TTL = 3600  // 1 hour
    
    // Adjust based on confidence (higher confidence = longer TTL)
    confidence_multiplier = confidence * 2.0
    
    // Adjust based on volatility (higher volatility = shorter TTL)
    volatility_divisor = 1.0 + (volatility * 3.0)
    
    // Calculate final TTL
    ttl = (BASE_TTL * confidence_multiplier) / volatility_divisor
    
    // Apply bounds
    MIN_TTL = 60      // 1 minute
    MAX_TTL = 86400   // 24 hours
    
    RETURN CLAMP(ttl, MIN_TTL, MAX_TTL)
```

## 6. Tool-Based Retrieval

```pseudocode
FUNCTION execute_tool_retrieval(query: Query) -> Result:
    // Step 1: Analyze query intent
    intent = analyze_intent(query)
    
    // Step 2: Select appropriate tools
    tools = SELECT_TOOLS(intent)
    
    // Step 3: Execute tools in parallel
    tool_results = PARALLEL_EXECUTE:
        FOR EACH tool IN tools:
            params = PREPARE_PARAMS(query, tool.requirements)
            result = tool.EXECUTE(params)
            YIELD result
    
    // Step 4: Aggregate results
    aggregated = AGGREGATE_RESULTS(tool_results)
    
    // Step 5: Apply Byzantine consensus (66% threshold)
    IF consensus_score(aggregated) >= 0.66:
        RETURN aggregated
    ELSE:
        RETURN RETRY_WITH_DIFFERENT_TOOLS()
```

## 7. Citation Tracking Integration

```pseudocode
FUNCTION track_citations(result: QueryResult) -> CitedResult:
    citations = []
    
    FOR EACH claim IN result.claims:
        // Find supporting sources
        sources = FIND_SOURCES(claim.text)
        
        FOR EACH source IN sources:
            citation = CREATE_CITATION:
                source_id: source.id
                document_uri: source.uri
                section: source.section
                page: source.page
                confidence: CALCULATE_RELEVANCE(claim, source)
                exact_match: CHECK_EXACT_MATCH(claim, source)
                context: EXTRACT_CONTEXT(source, 100)
            
            citations.APPEND(citation)
    
    // Sort by relevance
    citations.SORT_BY(confidence, DESCENDING)
    
    // Attach to result
    result.citations = citations
    RETURN result
```

## 8. Performance Monitoring

```pseudocode
FUNCTION monitor_fact_performance():
    metrics = {
        cache_hits: 0,
        cache_misses: 0,
        total_requests: 0,
        latencies: [],
        hit_rate: 0.0
    }
    
    WHILE system_running:
        // Collect metrics every second
        SLEEP(1000)
        
        current_stats = fact_client.GET_STATS()
        
        // Update metrics
        metrics.cache_hits = current_stats.hits
        metrics.cache_misses = current_stats.misses
        metrics.total_requests = metrics.cache_hits + metrics.cache_misses
        
        // Calculate hit rate
        IF metrics.total_requests > 0:
            metrics.hit_rate = metrics.cache_hits / metrics.total_requests
        
        // Check SLA compliance
        IF metrics.hit_rate < 0.873:  // 87.3% threshold
            ALERT("Cache hit rate below threshold")
        
        avg_latency = AVERAGE(metrics.latencies[-100:])
        IF avg_latency > 23:  // 23ms threshold
            ALERT("Cache latency exceeds SLA")
        
        // Export to Prometheus
        EXPORT_METRICS(metrics)
```

## 9. Fallback Strategy

```pseudocode
FUNCTION fallback_to_direct_retrieval(query: Query) -> Result:
    // When FACT is unavailable
    TRY:
        // Attempt direct database query
        result = database.QUERY(query)
        
        // Add warning about degraded mode
        result.ADD_WARNING("FACT unavailable - using fallback")
        
        RETURN result
        
    CATCH error:
        // Last resort - return cached stale data if available
        stale_cache = local_cache.GET_STALE(query)
        
        IF stale_cache EXISTS:
            stale_cache.ADD_WARNING("Using stale cache - FACT unavailable")
            RETURN stale_cache
        ELSE:
            THROW "Service temporarily unavailable"
```

## 10. Migration Sequence

```pseudocode
FUNCTION migrate_to_fact():
    // Phase 1: Preparation
    BACKUP_EXISTING_CACHE()
    CREATE_FEATURE_FLAG("use_real_fact", false)
    
    // Phase 2: Dual-mode operation
    SET_FEATURE_FLAG("use_real_fact", true, percentage=10)
    MONITOR_PERFORMANCE(duration=24_hours)
    
    IF performance_acceptable:
        // Phase 3: Gradual rollout
        FOR percentage IN [25, 50, 75, 100]:
            SET_FEATURE_FLAG("use_real_fact", true, percentage)
            MONITOR_PERFORMANCE(duration=6_hours)
            
            IF NOT performance_acceptable:
                ROLLBACK()
                BREAK
    
    // Phase 4: Cleanup
    IF feature_flag_percentage == 100:
        REMOVE_PLACEHOLDER_CODE()
        REMOVE_FEATURE_FLAG()
        COMMIT_CHANGES()
```

## 11. Byzantine Consensus Integration

```pseudocode
FUNCTION validate_with_consensus(result: QueryResult) -> ValidatedResult:
    // Create multiple validation nodes
    validators = []
    FOR i IN RANGE(3):  // Minimum 3 for Byzantine tolerance
        validator = CREATE_VALIDATOR_NODE(i)
        validators.APPEND(validator)
    
    // Collect votes
    votes = []
    FOR EACH validator IN validators:
        vote = validator.VALIDATE(result)
        votes.APPEND(vote)
    
    // Apply Byzantine consensus (2/3 + 1 rule)
    agreement_count = COUNT_AGREEMENTS(votes)
    total_validators = LENGTH(validators)
    
    consensus_threshold = (total_validators * 2 / 3) + 1
    
    IF agreement_count >= consensus_threshold:
        result.consensus_achieved = true
        result.consensus_confidence = agreement_count / total_validators
    ELSE:
        result.consensus_achieved = false
        RETRY_WITH_MORE_VALIDATORS()
    
    RETURN result
```

## 12. Rollback Procedure

```pseudocode
FUNCTION rollback_fact_integration():
    // Step 1: Switch to fallback
    SET_FEATURE_FLAG("use_real_fact", false)
    
    // Step 2: Restore placeholder (temporary)
    RESTORE_FROM_BACKUP("src/fact/")
    
    // Step 3: Revert Cargo.toml
    REVERT_DEPENDENCIES("Cargo.toml")
    
    // Step 4: Clear FACT cache
    fact_client.CLEAR_ALL()
    
    // Step 5: Notify stakeholders
    SEND_ALERT("FACT integration rolled back")
    
    // Step 6: Document issues
    CREATE_INCIDENT_REPORT()
```

---

## Summary

This pseudocode provides the algorithmic foundation for integrating the real FACT system. Key aspects:

1. **Cache-first approach** with <23ms target for hits
2. **MCP protocol integration** for tool-based retrieval
3. **Byzantine consensus** with 66% threshold
4. **Intelligent TTL calculation** based on confidence
5. **Comprehensive monitoring** and alerting
6. **Graceful fallback** strategies
7. **Phased migration** with feature flags

**Next Steps**: Proceed to SPARC Architecture for system design details.