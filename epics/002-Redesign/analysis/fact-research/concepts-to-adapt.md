# FACT Analysis: Transferable Concepts for Rust Implementation

## Executive Summary

FACT (Fast Augmented Context Tools) represents a paradigm shift from RAG to intelligent caching + deterministic tool execution. Key insight: **Replace fuzzy vector similarity with exact tool results and intelligent caching strategies**.

## Core Architectural Concepts

### 1. Cache-First Query Processing

**FACT Pattern:**
```
Query → Cache Check → [Hit: Return] | [Miss: Tool Execution → Cache Store → Return]
```

**Rust Implementation Strategy:**
```rust
pub struct CacheFirstProcessor {
    cache: Arc<IntelligentCache>,
    tools: Arc<ToolRegistry>,
}

impl CacheFirstProcessor {
    async fn process_query(&self, query: &str) -> Result<Response> {
        let query_hash = self.generate_hash(query);
        
        // Cache-first lookup
        if let Some(cached) = self.cache.get(&query_hash).await? {
            return Ok(cached);
        }
        
        // Cache miss - execute tools
        let response = self.execute_tools(query).await?;
        
        // Store for future use
        self.cache.store(query_hash, response.clone()).await?;
        
        Ok(response)
    }
}
```

### 2. Intelligent Multi-Tier Caching Strategies

**FACT Insight:** Different content types need different caching strategies:

**Strategies Identified:**
- **LRU**: Least Recently Used (time-based)
- **LFU**: Least Frequently Used (access-based) 
- **Token-Optimized**: Cost efficiency (tokens per KB)
- **Adaptive**: Switches strategies based on performance metrics

**Rust Implementation:**
```rust
#[derive(Debug, Clone)]
pub enum CacheStrategy {
    Lru,
    Lfu, 
    TokenOptimized { target_efficiency: f64 },
    Adaptive { evaluation_interval: Duration },
}

pub struct IntelligentCache {
    strategy: CacheStrategy,
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    metrics: Arc<Mutex<CacheMetrics>>,
    circuit_breaker: Option<CircuitBreaker>,
}
```

### 3. Cache Warming and Preloading

**FACT Innovation:** Proactive cache population with high-value queries

**Key Concepts:**
- **Query Pattern Analysis**: Identify frequently requested patterns
- **Template-Based Generation**: Generate variants of common queries
- **Concurrent Warming**: Batch warming for performance
- **Adaptive Prioritization**: Adjust warming based on hit rates

**Rust Implementation:**
```rust
pub struct CacheWarmer {
    cache: Arc<IntelligentCache>,
    query_analyzer: QueryPatternAnalyzer,
    warming_templates: Vec<QueryTemplate>,
}

impl CacheWarmer {
    async fn warm_intelligently(&self, query_log: &[String]) -> WarmupResult {
        let candidates = self.query_analyzer.analyze_patterns(query_log);
        let prioritized = self.prioritize_queries(candidates);
        
        // Concurrent warming in batches
        let batches = prioritized.chunks(5);
        for batch in batches {
            let tasks: Vec<_> = batch.iter()
                .map(|query| self.warm_single_query(query))
                .collect();
            
            let _ = futures::future::join_all(tasks).await;
            tokio::time::sleep(Duration::from_millis(50)).await; // Throttling
        }
        
        Ok(WarmupResult::new())
    }
}
```

### 4. Circuit Breaker for Cache Resilience

**FACT Pattern:** Graceful degradation when cache system fails

**Key Features:**
- **State Management**: Closed/Open/Half-Open states
- **Failure Threshold**: Trip after N failures
- **Recovery Testing**: Gradually test recovery
- **Fallback Execution**: Continue without cache

**Rust Implementation:**
```rust
pub struct CacheCircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    config: CircuitBreakerConfig,
    failure_count: Arc<AtomicU64>,
    last_failure: Arc<Mutex<Option<Instant>>>,
}

#[derive(Debug)]
pub enum CircuitState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen { test_count: u32 },
}

impl CacheCircuitBreaker {
    async fn call_with_protection<T, F>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match self.get_state().await {
            CircuitState::Closed => {
                match operation.await {
                    Ok(result) => {
                        self.record_success().await;
                        Ok(result)
                    }
                    Err(e) => {
                        self.record_failure().await;
                        Err(e)
                    }
                }
            }
            CircuitState::Open { .. } => {
                Err(CacheError::CircuitBreakerOpen)
            }
            CircuitState::HalfOpen { .. } => {
                // Test recovery
                self.try_recovery(operation).await
            }
        }
    }
}
```

### 5. Token-Based Cache Optimization

**FACT Innovation:** Cache decisions based on token efficiency (tokens per KB)

**Core Metrics:**
- **Token Efficiency**: `tokens / (content_size_kb)`
- **Cost-Benefit Analysis**: Storage cost vs token savings
- **Dynamic Thresholds**: Adjust based on cache utilization

**Rust Implementation:**
```rust
#[derive(Debug, Clone)]
pub struct CacheEntry {
    content: String,
    token_count: u32,
    created_at: Instant,
    access_count: u64,
    last_accessed: Option<Instant>,
}

impl CacheEntry {
    fn calculate_efficiency(&self) -> f64 {
        let size_kb = self.content.len() as f64 / 1024.0;
        self.token_count as f64 / size_kb
    }
    
    fn should_cache(&self, min_efficiency: f64, min_tokens: u32) -> bool {
        self.token_count >= min_tokens && 
        self.calculate_efficiency() >= min_efficiency
    }
}

pub struct TokenOptimizedStrategy {
    target_efficiency: f64,
    min_tokens: u32,
}

impl CacheStrategy for TokenOptimizedStrategy {
    fn should_evict(&self, entries: &[CacheEntry]) -> Vec<String> {
        entries.iter()
            .filter(|entry| entry.calculate_efficiency() < self.target_efficiency)
            .map(|entry| entry.key.clone())
            .collect()
    }
}
```

### 6. Content Quality and Security Validation

**FACT Features:**
- **Sensitive Data Detection**: Regex patterns for PII/secrets
- **Content Quality Analysis**: Error detection, content integrity
- **Security Assessment**: Risk scoring and mitigation

**Rust Implementation:**
```rust
pub struct CacheValidator {
    sensitive_patterns: Vec<Regex>,
    quality_checkers: Vec<Box<dyn ContentQualityChecker>>,
}

impl CacheValidator {
    async fn validate_entry(&self, entry: &CacheEntry) -> ValidationResult {
        let mut issues = Vec::new();
        
        // Security validation
        for pattern in &self.sensitive_patterns {
            if pattern.is_match(&entry.content) {
                issues.push(SecurityIssue::SensitiveData);
            }
        }
        
        // Quality validation
        for checker in &self.quality_checkers {
            if let Some(issue) = checker.check(&entry.content) {
                issues.push(issue);
            }
        }
        
        ValidationResult { issues, ..Default::default() }
    }
}

trait ContentQualityChecker: Send + Sync {
    fn check(&self, content: &str) -> Option<QualityIssue>;
}

struct ErrorContentChecker;

impl ContentQualityChecker for ErrorContentChecker {
    fn check(&self, content: &str) -> Option<QualityIssue> {
        if content.to_lowercase().contains("error:") || 
           content.to_lowercase().contains("failed:") {
            Some(QualityIssue::ErrorContent)
        } else {
            None
        }
    }
}
```

### 7. Adaptive Strategy Selection

**FACT Innovation:** Automatically switch caching strategies based on performance

**Metrics Tracked:**
- Hit rate
- Token efficiency
- Memory utilization
- Query latency

**Rust Implementation:**
```rust
pub struct AdaptiveStrategy {
    strategies: Vec<Box<dyn CacheStrategy>>,
    current_strategy: usize,
    metrics_history: VecDeque<StrategyMetrics>,
    evaluation_interval: Duration,
    last_evaluation: Instant,
}

impl AdaptiveStrategy {
    async fn evaluate_and_adapt(&mut self, cache: &IntelligentCache) {
        if self.last_evaluation.elapsed() < self.evaluation_interval {
            return;
        }
        
        let current_metrics = cache.get_metrics().await;
        
        // Record performance of current strategy
        self.metrics_history.push_back(StrategyMetrics {
            strategy_index: self.current_strategy,
            hit_rate: current_metrics.hit_rate,
            token_efficiency: current_metrics.token_efficiency,
            timestamp: Instant::now(),
        });
        
        // Evaluate all strategies and select best performer
        let best_strategy = self.find_best_strategy();
        if best_strategy != self.current_strategy {
            info!("Switching cache strategy: {} → {}", 
                  self.current_strategy, best_strategy);
            self.current_strategy = best_strategy;
        }
        
        self.last_evaluation = Instant::now();
    }
}
```

## Tool Execution Patterns

### 8. Secure Tool Execution Framework

**FACT Features:**
- **Parameter Validation**: Schema-based input validation
- **Security Checks**: SQL injection prevention, input sanitization  
- **Rate Limiting**: Token bucket algorithm
- **Timeout Protection**: Prevent hanging operations

**Rust Implementation:**
```rust
pub struct ToolExecutor {
    tools: HashMap<String, Box<dyn Tool>>,
    rate_limiter: TokenBucket,
    security_validator: SecurityValidator,
    parameter_validator: ParameterValidator,
}

#[async_trait]
impl ToolExecutor {
    async fn execute_tool(&self, 
                         tool_name: &str, 
                         params: Value) -> Result<ToolResult> {
        
        // Rate limiting
        self.rate_limiter.acquire().await?;
        
        // Security validation
        self.security_validator.validate_request(tool_name, &params)?;
        
        // Parameter validation
        self.parameter_validator.validate(tool_name, &params)?;
        
        // Get tool and execute with timeout
        let tool = self.tools.get(tool_name)
            .ok_or(ToolError::NotFound)?;
            
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            tool.execute(params)
        ).await??;
        
        Ok(result)
    }
}
```

### 9. Database Query Optimization

**FACT Pattern:** Read-only SQL with comprehensive validation

**Key Features:**
- **Query Whitelisting**: Only SELECT statements allowed
- **SQL Injection Prevention**: Parameterized queries only
- **Table Name Validation**: Prevent schema injection
- **Result Caching**: Cache query results based on patterns

**Rust Implementation:**
```rust
pub struct SqlTool {
    db_pool: Pool<Sqlite>,
    validator: SqlValidator,
    result_cache: Arc<QueryResultCache>,
}

impl Tool for SqlTool {
    async fn execute(&self, params: Value) -> Result<ToolResult> {
        let query = params["statement"].as_str()
            .ok_or(ToolError::InvalidParameter)?;
        
        // Validate SQL is read-only
        self.validator.validate_readonly(query)?;
        
        // Check result cache first
        let query_hash = self.hash_query(query);
        if let Some(cached) = self.result_cache.get(&query_hash).await? {
            return Ok(cached);
        }
        
        // Execute query
        let result = sqlx::query(query)
            .fetch_all(&self.db_pool)
            .await?;
        
        let tool_result = ToolResult {
            data: self.format_rows(result),
            execution_time: start.elapsed(),
            status: "success".to_string(),
        };
        
        // Cache the result
        self.result_cache.store(query_hash, tool_result.clone()).await?;
        
        Ok(tool_result)
    }
}
```

## Performance and Monitoring

### 10. Comprehensive Metrics Collection

**FACT Metrics:**
- Cache hit/miss rates
- Token efficiency
- Query latency (P50, P95, P99)
- Memory utilization
- Circuit breaker states
- Tool execution times

**Rust Implementation:**
```rust
#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub total_requests: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_tokens_cached: AtomicU64,
    pub total_size_bytes: AtomicU64,
    pub hit_latencies: Arc<Mutex<Vec<Duration>>>,
    pub miss_latencies: Arc<Mutex<Vec<Duration>>>,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let total = self.total_requests.load(Ordering::Relaxed) as f64;
        
        if total > 0.0 { hits / total * 100.0 } else { 0.0 }
    }
    
    pub fn token_efficiency(&self) -> f64 {
        let tokens = self.total_tokens_cached.load(Ordering::Relaxed) as f64;
        let size_kb = self.total_size_bytes.load(Ordering::Relaxed) as f64 / 1024.0;
        
        if size_kb > 0.0 { tokens / size_kb } else { 0.0 }
    }
    
    pub fn p95_latency(&self, cache_hit: bool) -> Duration {
        let latencies = if cache_hit { 
            &self.hit_latencies 
        } else { 
            &self.miss_latencies 
        };
        
        let mut latencies = latencies.lock().unwrap().clone();
        latencies.sort();
        
        let index = (latencies.len() as f64 * 0.95) as usize;
        latencies.get(index).copied().unwrap_or(Duration::ZERO)
    }
}
```

## Key Adaptations for doc-rag Project

### 1. Fact/Claim Caching Strategy

```rust
pub struct FactCacheEntry {
    pub fact: String,
    pub sources: Vec<SourceReference>,
    pub confidence_score: f64,
    pub validation_status: ValidationStatus,
    pub token_count: u32,
    pub created_at: Instant,
    pub last_verified: Option<Instant>,
}

impl FactCacheEntry {
    fn should_cache(&self) -> bool {
        self.confidence_score >= 0.8 && 
        self.validation_status == ValidationStatus::Verified &&
        self.token_count >= 100
    }
    
    fn needs_revalidation(&self, ttl: Duration) -> bool {
        self.last_verified
            .map(|last| last.elapsed() > ttl)
            .unwrap_or(true)
    }
}
```

### 2. Document Chunking Intelligence

```rust
pub struct IntelligentChunker {
    chunk_cache: Arc<ChunkCache>,
    strategies: Vec<Box<dyn ChunkingStrategy>>,
}

#[async_trait]
pub trait ChunkingStrategy: Send + Sync {
    async fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;
    fn name(&self) -> &str;
    fn efficiency_score(&self, chunks: &[Chunk]) -> f64;
}

impl IntelligentChunker {
    async fn chunk_with_caching(&self, document: &Document) -> Result<Vec<Chunk>> {
        let doc_hash = self.hash_document(document);
        
        // Check cache first
        if let Some(cached_chunks) = self.chunk_cache.get(&doc_hash).await? {
            return Ok(cached_chunks);
        }
        
        // Try different chunking strategies and pick best
        let mut best_chunks = Vec::new();
        let mut best_score = 0.0;
        
        for strategy in &self.strategies {
            let chunks = strategy.chunk(document).await?;
            let score = strategy.efficiency_score(&chunks);
            
            if score > best_score {
                best_score = score;
                best_chunks = chunks;
            }
        }
        
        // Cache the best result
        self.chunk_cache.store(doc_hash, best_chunks.clone()).await?;
        
        Ok(best_chunks)
    }
}
```

### 3. Fact Validation Pipeline

```rust
pub struct FactValidator {
    validation_cache: Arc<ValidationCache>,
    sources: Arc<SourceManager>,
    confidence_threshold: f64,
}

impl FactValidator {
    async fn validate_fact(&self, fact: &Fact) -> Result<ValidationResult> {
        let fact_hash = self.hash_fact(fact);
        
        // Check validation cache
        if let Some(cached) = self.validation_cache.get(&fact_hash).await? {
            if !cached.needs_revalidation(Duration::from_hours(24)) {
                return Ok(cached);
            }
        }
        
        // Perform validation
        let sources = self.sources.find_sources_for_fact(fact).await?;
        let confidence = self.calculate_confidence(fact, &sources)?;
        
        let result = ValidationResult {
            fact: fact.clone(),
            confidence_score: confidence,
            supporting_sources: sources,
            validation_timestamp: Instant::now(),
            status: if confidence >= self.confidence_threshold { 
                ValidationStatus::Verified 
            } else { 
                ValidationStatus::Unverified 
            },
        };
        
        // Cache validation result
        self.validation_cache.store(fact_hash, result.clone()).await?;
        
        Ok(result)
    }
}
```

## Implementation Priority

### Phase 1: Core Caching Infrastructure
1. **IntelligentCache with multi-tier strategies**
2. **Cache-first query processing**
3. **Basic metrics collection**
4. **Token-based optimization**

### Phase 2: Resilience and Optimization  
1. **Circuit breaker pattern**
2. **Cache warming system**
3. **Adaptive strategy selection**
4. **Comprehensive validation**

### Phase 3: Domain-Specific Features
1. **Fact/claim caching**
2. **Intelligent document chunking**
3. **Source validation caching** 
4. **Advanced query optimization**

## Conclusion

FACT's core insight is **replacing probabilistic vector similarity with deterministic tool execution + intelligent caching**. The multi-tier caching strategies, circuit breaker resilience, token optimization, and cache warming techniques provide a robust foundation for building high-performance, cost-effective fact retrieval systems.

The key transfer to our Rust doc-rag system is implementing cache-first processing with intelligent eviction policies based on content quality, access patterns, and token efficiency rather than simple recency or frequency metrics.