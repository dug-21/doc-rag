# SPARC ARCHITECTURE: Phase 3 MVP Prototype Development
## Neurosymbolic RAG System - Integrated Docker Architecture

**Document Version**: 2.0  
**Date**: January 12, 2025  
**Phase**: 3 (MVP Prototype Development)  
**Dependencies**: SPARC-SPECIFICATION.md, SPARC-PSEUDOCODE.md  

---

## 🏗️ ARCHITECTURAL OVERVIEW

### System Architecture Philosophy

Phase 3 implements a **working prototype neurosymbolic architecture** that combines symbolic reasoning with neural processing in an integrated, container-based environment focused on demonstrating core functionality rather than production scale.

**Core Design Principles:**
1. **Working First**: Get basic functionality operational before optimization
2. **Container-Based Integration**: Docker Compose for simple orchestration
3. **Sequential Processing**: Clear, debuggable processing pipeline
4. **Minimal Complexity**: Simple communication patterns
5. **Validation Focus**: Prove concept before scaling
6. **Development-Friendly**: Easy to test, debug, and iterate

---

## 🌐 HIGH-LEVEL SYSTEM ARCHITECTURE

### Docker Compose Development Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCKER COMPOSE ENVIRONMENT                       │
│                      Single Host Development Setup                      │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                      MAIN APPLICATION CONTAINER                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                 Integrated Query Processor                      │     │
│  │  • Sequential pipeline processing                               │     │
│  │  • Direct function calls between components                     │     │
│  │  • Simple error handling and logging                            │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬─────────────────────────────┐
        │                 │                 │                             │
┌───────▼──────┐  ┌───────▼──────┐  ┌──────▼─────┐  ┌──────────────────▼──┐
│   SYMBOLIC   │  │    GRAPH     │  │   NEURAL   │  │     VECTOR          │
│  PROCESSING  │  │  CONTAINER   │  │ CONTAINER  │  │   CONTAINER         │
│  INTEGRATED  │  │              │  │            │  │                     │
│              │  │              │  │            │  │                     │
│ ┌──────────┐ │  │ ┌──────────┐ │  │┌──────────┐│  │ ┌─────────────────┐ │
│ │ Datalog  │ │  │ │  Neo4j   │ │  ││ruv-fann  ││  │ │    Qdrant       │ │
│ │  Logic   │ │  │ │ Single   │ │  ││Training  ││  │ │   Single        │ │
│ │ Embedded │ │  │ │Instance  │ │  ││Pipeline  ││  │ │  Instance       │ │
│ └──────────┘ │  │ └──────────┘ │  │└──────────┘│  │ └─────────────────┘ │
│ ┌──────────┐ │  │              │  │            │  │                     │
│ │ Prolog   │ │  │              │  │            │  │                     │
│ │ Embedded │ │  │              │  │            │  │                     │
│ └──────────┘ │  │              │  │            │  │                     │
└──────────────┘  └──────────────┘  └────────────┘  └─────────────────────┘
        │                 │                 │                             │
        └─────────────────┼─────────────────┼─────────────────────────────┘
                          │                 │
┌─────────────────────────▼─────────────────▼─────────────────────────────┐
│                     DEVELOPMENT DATA LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐    │
│  │    Neo4j    │  │    Redis    │  │         Local Files             │    │
│  │  Container  │  │  Container  │  │     • Sample documents          │    │
│  │    :7474    │  │    :6379    │  │     • Test data                 │    │
│  │    :7687    │  │             │  │     • Configuration             │    │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 INTEGRATED CONTAINER ARCHITECTURE

### Component Integration Strategy

#### Core Processing Components

**1. Main Application Container**
```yaml
Component: main-app
Purpose: Integrated query processing pipeline
Technology: Rust with embedded components
Scaling: Single container for development
Dependencies: Database containers
Goal: Working end-to-end functionality
```

**2. Neural Classification Module**
```yaml
Component: neural-classifier  
Purpose: Basic query classification
Technology: Rust + ruv-fann (embedded)
Scaling: Single instance, embedded in main app
Dependencies: Training data files
Goal: Basic routing and classification working
```

**3. Symbolic Reasoning Module**
```yaml
Component: symbolic-engine
Purpose: Datalog/Prolog query execution
Technology: Rust + Crepe + Scryer-Prolog (embedded)
Scaling: Single instance, embedded in main app
Dependencies: Logic rules files
Goal: Core symbolic reasoning functional
```

**4. Graph Database Container**
```yaml
Component: neo4j-dev
Purpose: Graph data storage and queries
Technology: Neo4j Community Edition
Scaling: Single container instance
Dependencies: Volume for data persistence
Goal: Basic graph operations working
```

**5. Response Generation Module**
```yaml
Component: response-generator
Purpose: Template-based response formatting
Technology: Rust + Handlebars (embedded)
Scaling: Single instance, embedded in main app
Dependencies: Template files
Goal: Basic response formatting working
```

#### Supporting Components

**6. Redis Cache Container**
```yaml
Component: redis-cache
Purpose: Simple caching for development
Technology: Redis single instance
Scaling: Single container
Goal: Basic caching working
```

**7. Vector Database Container**
```yaml
Component: qdrant-vector
Purpose: Vector search fallback
Technology: Qdrant single instance
Scaling: Single container
Dependencies: Volume for vector data
Goal: Basic vector search available
```

**8. Development Tools**
```yaml
Component: dev-tools
Purpose: Health checks and basic monitoring
Technology: Simple HTTP endpoints
Scaling: Embedded in main app
Goal: Basic observability for debugging
```

---

## 📡 COMPONENT COMMUNICATION

### Communication Patterns

#### Direct Function Calls
```rust
// Integrated Query Processing
pub struct QueryProcessor {
    symbolic_engine: SymbolicEngine,
    neural_classifier: NeuralClassifier,
    graph_client: Neo4jClient,
    response_generator: ResponseGenerator,
}

impl QueryProcessor {
    pub async fn process_query(&self, query: &str) -> Result<Response> {
        // Direct, sequential processing
        let classification = self.neural_classifier.classify(query).await?;
        
        let results = match classification.query_type {
            QueryType::Symbolic => self.symbolic_engine.process(query).await?,
            QueryType::Graph => self.graph_client.query(query).await?,
            _ => self.vector_fallback.search(query).await?
        };
        
        self.response_generator.generate(results).await
    }
}
```

#### Container Communication (HTTP/REST)
```yaml
Communication: Simple HTTP REST APIs
Patterns:
  - Direct HTTP calls for database operations
  - JSON request/response format
  - Basic error handling with HTTP status codes
  - Simple retry logic for transient failures

Endpoints:
  - Neo4j: Cypher queries via HTTP API
  - Redis: Simple key-value operations
  - Qdrant: Vector search via REST API
```

### Docker Compose Networking
```yaml
Networking: Docker Compose bridge network
Service Discovery: Docker DNS resolution
Load Balancing: Not applicable (single instances)
Retry Policy: Simple retry with backoff
Timeout: Component-specific (1s-30s)
```

---

## 💾 DEVELOPMENT DATA ARCHITECTURE

### Simple Data Strategy

#### Development Data Stores

**1. Neo4j Development Container**
```cypher
-- Basic Schema for Development
CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE;

-- Simple Node Types
(:Document {id, title, content})
(:Section {id, title, content})
(:Requirement {id, text, type})
(:Definition {id, term, definition})

-- Basic Relationships
(:Document)-[:HAS_SECTION]->(:Section)
(:Section)-[:CONTAINS_REQUIREMENT]->(:Requirement)
(:Requirement)-[:REFERENCES]->(:Requirement)
```

**2. Redis Development Container**
```yaml
Single Redis Instance: 
  - Technology: Redis 7 container
  - Size: Default memory allocation
  - TTL: Simple time-to-live settings
  - Purpose: Basic caching for development
  - Port: 6379
  - Persistence: Optional for development
```

**3. Local File Storage**
```yaml
File Storage:
  - Sample documents: ./data/docs/
  - Test data: ./data/test/
  - Configuration: ./config/
  - Logs: ./logs/
  - Model data: ./models/

Structure:
  /data
    /docs          # Sample documents for testing
    /test          # Test queries and expected results
    /training      # Neural training data
  /config          # Application configuration
  /logs            # Application logs
```

**4. Qdrant Development Container**
```yaml
Single Qdrant Instance:
  - Technology: Qdrant container
  - Collections: Basic document embeddings
  - Configuration: Default settings
  - Vector dimension: 384 (smaller for development)
  - Distance metric: Cosine similarity
  - Port: 6333
```

### Development Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sample Data   │───▶│   Simple         │───▶│   Container     │
│   Loading       │    │   Processing     │    │   Storage       │
│                 │    │                  │    │                 │
│ • Test Files    │    │ • Basic          │    │ • Neo4j Single │
│ • Sample Docs   │    │   Extraction     │    │ • Redis Cache   │
│ • Config Files  │    │ • Simple         │    │ • Qdrant Vector │
│ • Validation    │    │   Parsing        │    │ • Local Files   │
└─────────────────┘    │ • Direct Storage │    └─────────────────┘
                       │ • No Complex     │
                       │   Transformations│
                       └──────────────────┘
```

---

## 🔄 INTEGRATED PROCESSING PIPELINE

### Sequential Processing Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Query     │───▶│   Neural     │───▶│  Simple     │───▶│  Sequential  │
│ Validation  │    │Classification│    │  Routing    │    │ Processing   │
│ & Cleaning  │    │              │    │  Logic      │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                           │                                       │
                           ▼                                       ▼
                   ┌──────────────┐                       ┌──────────────┐
                   │ Basic        │                       │ Single       │
                   │ Confidence   │                       │ Component    │
                   │ Check        │                       │ Processing   │
                   └──────────────┘                       └──────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Response   │◄───│   Simple     │◄───│  Basic      │◄───│   Result     │
│ Formatting  │    │ Integration  │    │ Citation    │    │ Collection   │
│             │    │              │    │ Extraction  │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Integrated Processing Architecture

```rust
// Integrated Processing Pipeline
pub struct IntegratedProcessor {
    datalog_engine: CrepeEngine,
    neo4j_client: Neo4jClient,
    neural_classifier: NeuralClassifier,
    response_generator: ResponseGenerator,
}

impl IntegratedProcessor {
    pub async fn process(&self, query: &str) -> Result<ProcessingResult> {
        // Step 1: Basic query validation
        let clean_query = self.validate_and_clean(query)?;
        
        // Step 2: Simple classification
        let classification = self.neural_classifier.classify(&clean_query).await?;
        
        // Step 3: Route to appropriate processor
        let results = match classification.query_type {
            QueryType::Symbolic => {
                self.datalog_engine.query(&clean_query).await?
            },
            QueryType::Graph => {
                self.neo4j_client.execute_cypher(&clean_query).await?
            },
            _ => {
                // Simple fallback
                self.basic_search(&clean_query).await?
            }
        };
        
        // Step 4: Generate response
        let response = self.response_generator.generate(&results).await?;
        
        Ok(ProcessingResult {
            response,
            source: classification.query_type,
            processing_time: Instant::now() - start_time,
        })
    }
}
```

### Simple Graph Processing

```rust
// Simple Graph Client
pub struct SimpleGraphClient {
    neo4j_client: Graph,
}

impl SimpleGraphClient {
    pub async fn process_query(&self, query: &str) -> Result<GraphResult> {
        // Step 1: Basic query parsing
        let cypher_query = self.build_simple_cypher(query)?;
        
        // Step 2: Execute query directly
        let mut result = self.neo4j_client.execute(cypher_query).await?;
        
        // Step 3: Simple result processing
        let mut records = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            records.push(self.process_row(row)?);
        }
        
        // Step 4: Basic formatting
        Ok(GraphResult {
            records,
            query_type: "graph".to_string(),
            processing_time: Instant::now() - start_time,
        })
    }
    
    fn build_simple_cypher(&self, query: &str) -> Result<Query> {
        // Simple keyword-based Cypher generation
        let cypher = format!(
            "MATCH (n) WHERE n.text CONTAINS '{}' RETURN n LIMIT 10",
            query
        );
        Ok(query(&cypher))
    }
}
```

---

## 🚀 DOCKER COMPOSE SETUP

### Development Environment Configuration

```yaml
# Docker Compose Configuration
version: '3.8'

services:
  # Main application
  neurosymbolic-app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=debug
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - neo4j
      - redis
      - qdrant
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped

  # Neo4j database
  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - ./neo4j/import:/import
    restart: unless-stopped

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Qdrant vector database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  neo4j_data:
  redis_data:
  qdrant_data:

networks:
  default:
    name: neurosymbolic-dev
```

### Docker Development Scripts

```bash
#!/bin/bash
# Development scripts for Docker environment

# dev-start.sh - Start development environment
#!/bin/bash
echo "Starting neurosymbolic development environment..."
docker-compose up -d
echo "Waiting for services to be ready..."
sleep 30
./scripts/health-check.sh
echo "Development environment ready!"

# dev-stop.sh - Stop development environment
#!/bin/bash
echo "Stopping neurosymbolic development environment..."
docker-compose down
echo "Environment stopped."

# dev-reset.sh - Reset development environment
#!/bin/bash
echo "Resetting development environment..."
docker-compose down -v
docker system prune -f
docker-compose up -d
echo "Environment reset complete."

# health-check.sh - Check service health
#!/bin/bash
echo "Checking service health..."
curl -f http://localhost:8080/health || echo "Main app not ready"
curl -f http://localhost:7474/ || echo "Neo4j not ready"
redis-cli -h localhost ping || echo "Redis not ready"
curl -f http://localhost:6333/health || echo "Qdrant not ready"
```

---

## 🔍 BASIC MONITORING & DEBUGGING

### Development Monitoring

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│    Simple        │───▶│   Console &     │
│    Logging      │    │   Logging        │    │   File Output   │
│                 │    │                  │    │                 │
│ • Request/      │    │ • Structured     │    │ • Terminal      │
│   Response      │    │   Logging        │    │ • Log Files     │
│ • Processing    │    │ • JSON Format    │    │ • Debug Info    │
│   Steps         │    │ • Error Details  │    │ • Manual Review │
└─────────────────┘    └──────────────────┘    └─────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Component     │───▶│     Health       │───▶│   Development   │
│    Status       │    │    Endpoints     │    │   Dashboard     │
│                 │    │                  │    │                 │
│ • Service       │    │ • HTTP Health    │    │ • Simple HTML   │
│   Availability  │    │   Checks         │    │ • Status Page   │
│ • Basic Metrics │    │ • Component      │    │ • Manual        │
│ • Error Counts  │    │   Status         │    │   Testing       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Development Status Dashboard

```html
<!-- Simple HTML status page -->
<!DOCTYPE html>
<html>
<head>
    <title>Neurosymbolic RAG - Development Status</title>
</head>
<body>
    <h1>Development Environment Status</h1>
    
    <div id="services-status">
        <h2>Services</h2>
        <ul>
            <li id="main-app">Main Application: <span class="status">Checking...</span></li>
            <li id="neo4j">Neo4j: <span class="status">Checking...</span></li>
            <li id="redis">Redis: <span class="status">Checking...</span></li>
            <li id="qdrant">Qdrant: <span class="status">Checking...</span></li>
        </ul>
    </div>
    
    <div id="recent-queries">
        <h2>Recent Test Queries</h2>
        <ul id="query-list">
            <!-- Populated by JavaScript -->
        </ul>
    </div>
    
    <div id="logs">
        <h2>Recent Logs</h2>
        <pre id="log-output">
            <!-- Real-time log output -->
        </pre>
    </div>
    
    <script src="/static/status.js"></script>
</body>
</html>
```

---

## 🔐 DEVELOPMENT SECURITY

### Basic Security Measures

```
┌─────────────────────────────────────────────────────────────────┐
│                   DEVELOPMENT SECURITY BASICS                   │
├─────────────────────────────────────────────────────────────────┤
│ Application Level                                               │
│ • Basic input validation                                        │
│ • Query sanitization                                           │
│ • Error handling without information leakage                   │
│ • Simple authentication for development access                  │
├─────────────────────────────────────────────────────────────────┤
│ Container Level                                                │
│ • Non-root users in containers                                 │
│ • Basic secrets management (environment variables)             │
│ • Container resource limits                                    │
│ • Regular image updates                                        │
├─────────────────────────────────────────────────────────────────┤
│ Network Level                                                  │
│ • Docker network isolation                                     │
│ • Expose only necessary ports                                  │
│ • Local development access only                                │
│ • Basic firewall rules                                         │
├─────────────────────────────────────────────────────────────────┤
│ Data Level                                                     │
│ • Default database passwords changed                           │
│ • Sensitive data not in logs                                   │
│ • Sample data only (no real sensitive data)                    │
│ • Regular container data cleanup                               │
└─────────────────────────────────────────────────────────────────┘
```

### Basic Security Implementation

```rust
// Basic Security for Development
pub struct BasicSecurityValidator {
    input_sanitizer: InputSanitizer,
}

impl BasicSecurityValidator {
    pub fn new() -> Self {
        Self {
            input_sanitizer: InputSanitizer::new(),
        }
    }
    
    pub fn validate_query(&self, query: &str) -> Result<String> {
        // Basic input validation
        if query.is_empty() || query.len() > 1000 {
            return Err(SecurityError::InvalidInput("Query length invalid".to_string()));
        }
        
        // Basic sanitization
        let sanitized = self.input_sanitizer.sanitize(query)?;
        
        // Simple injection prevention
        if self.contains_suspicious_patterns(&sanitized) {
            return Err(SecurityError::SuspiciousInput);
        }
        
        Ok(sanitized)
    }
    
    fn contains_suspicious_patterns(&self, input: &str) -> bool {
        let suspicious_patterns = [
            "<script",
            "javascript:",
            "DROP TABLE",
            "DELETE FROM",
            "--",
            "/*",
            "xp_",
        ];
        
        suspicious_patterns.iter()
            .any(|pattern| input.to_lowercase().contains(&pattern.to_lowercase()))
    }
    
    pub fn log_security_event(&self, event: &str, details: &str) {
        // Simple security logging
        log::warn!("Security Event: {} - {}", event, details);
    }
}

// Simple authentication for development
pub fn check_dev_auth(auth_header: Option<&str>) -> Result<()> {
    match auth_header {
        Some(header) if header == "Bearer dev-token-123" => Ok(()),
        Some(_) => Err(SecurityError::InvalidAuth),
        None => Err(SecurityError::MissingAuth),
    }
}
```

This integrated architecture specification provides the technical foundation for implementing a working prototype neurosymbolic RAG system using Docker Compose, focused on demonstrating core functionality and establishing a foundation for future scaling to production.