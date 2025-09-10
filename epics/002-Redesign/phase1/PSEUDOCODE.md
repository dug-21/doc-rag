# Phase 1 Pseudocode: Neurosymbolic Foundation
## SPARC Methodology - Pseudocode Phase

**Document Version**: 1.0  
**Date**: January 10, 2025  
**Phase**: Neurosymbolic Foundation Implementation Logic  
**Methodology**: SPARC (Specification → Pseudocode → Architecture → Refinement → Completion)  

---

## 🎯 PSEUDOCODE OVERVIEW

This document provides detailed algorithmic logic for implementing the three core neurosymbolic components:
1. **Symbolic Reasoning Engine** (Datalog + Prolog)
2. **Graph Database Integration** (Neo4j)  
3. **Smart Ingestion Pipeline** (Neural Classification + Logic Extraction)

**Implementation Language**: Rust  
**Key Libraries**: crepe (Datalog), scryer-prolog (Prolog), neo4j (Graph DB), ruv-fann (Neural Networks)

---

## 📚 COMPONENT 1: SYMBOLIC REASONING ENGINE

### 1.1 Datalog Engine Implementation

```rust
// File: src/symbolic/datalog/engine.rs

struct DatalogEngine {
    engine: crepe::CrepeEngine,
    rule_cache: DashMap<String, CompiledRule>,
    fact_store: Arc<RwLock<FactStore>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl DatalogEngine {
    // Initialize Datalog engine with empty rule set
    async fn new() -> Result<Self> {
        let engine = crepe::CrepeEngine::new()
            .with_config(EngineConfig {
                max_memory_mb: 512,
                timeout_ms: 100, // <100ms requirement
                optimization_level: OptimizationLevel::High,
            })?;
        
        Ok(Self {
            engine,
            rule_cache: DashMap::new(),
            fact_store: Arc::new(RwLock::new(FactStore::new())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
        })
    }
    
    // Compile requirement text to Datalog rule
    async fn compile_requirement_to_rule(requirement_text: &str) -> Result<DatalogRule> {
        // Step 1: Parse requirement structure
        let parsed_req = self.parse_requirement_structure(requirement_text).await?;
        
        // Step 2: Extract logical components
        let subject = parsed_req.extract_subject()?; // e.g., "cardholder_data"
        let predicate = parsed_req.extract_predicate()?; // e.g., "requires_encryption"
        let conditions = parsed_req.extract_conditions()?; // e.g., "stored_at_rest"
        let requirement_type = parsed_req.extract_type()?; // MUST, SHALL, SHOULD
        
        // Step 3: Generate Datalog rule syntax
        let rule = match requirement_type {
            RequirementType::Must | RequirementType::Shall => {
                format!("{}({}) :- {}.", predicate, subject, conditions.join(", "))
            },
            RequirementType::Should => {
                format!("recommended_{}({}) :- {}.", predicate, subject, conditions.join(", "))
            },
            RequirementType::May => {
                format!("optional_{}({}) :- {}.", predicate, subject, conditions.join(", "))
            },
        };
        
        // Step 4: Validate rule syntax
        self.validate_rule_syntax(&rule).await?;
        
        // Step 5: Create rule object with metadata
        Ok(DatalogRule {
            id: generate_rule_id(&parsed_req),
            text: rule,
            source_requirement: requirement_text.to_string(),
            rule_type: requirement_type,
            created_at: Utc::now(),
            dependencies: parsed_req.extract_dependencies(),
        })
    }
    
    // Add compiled rule to engine
    async fn add_rule(rule: DatalogRule) -> Result<()> {
        // Step 1: Check for rule conflicts
        if self.has_conflicting_rule(&rule).await? {
            return Err(DatalogError::RuleConflict(rule.id.clone()));
        }
        
        // Step 2: Compile rule for execution
        let compiled_rule = self.engine.compile(&rule.text)?;
        
        // Step 3: Add to engine
        self.engine.add_rule(compiled_rule)?;
        
        // Step 4: Cache for fast lookup
        self.rule_cache.insert(rule.id.clone(), compiled_rule);
        
        // Step 5: Update fact store
        let mut fact_store = self.fact_store.write().await;
        fact_store.add_rule_facts(&rule).await?;
        
        info!("Added Datalog rule: {}", rule.id);
        Ok(())
    }
    
    // Execute query with performance monitoring
    async fn query(query_text: &str) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // Step 1: Parse and validate query
        let parsed_query = self.parse_query(query_text).await?;
        
        // Step 2: Check cache for recent results
        if let Some(cached_result) = self.check_query_cache(&parsed_query).await? {
            self.update_cache_hit_metrics().await;
            return Ok(cached_result);
        }
        
        // Step 3: Execute query against engine
        let raw_results = self.engine.query(&parsed_query.to_datalog_syntax())?;
        
        // Step 4: Process and format results
        let formatted_results = self.format_query_results(raw_results).await?;
        
        // Step 5: Extract citations and proof information
        let citations = self.extract_citations(&formatted_results).await?;
        let proof_steps = self.build_proof_chain(&parsed_query, &formatted_results).await?;
        
        // Step 6: Create query result
        let execution_time = start_time.elapsed();
        let result = QueryResult {
            query: parsed_query,
            results: formatted_results,
            citations,
            proof_chain: proof_steps,
            execution_time_ms: execution_time.as_millis() as u64,
            confidence: self.calculate_confidence(&formatted_results),
            used_rules: self.get_used_rules(&formatted_results),
        };
        
        // Step 7: Cache result for future queries
        self.cache_query_result(&parsed_query, &result).await?;
        
        // Step 8: Update performance metrics
        self.update_query_metrics(execution_time).await;
        
        // Step 9: Validate performance requirement
        if execution_time.as_millis() > 100 {
            warn!("Query exceeded 100ms performance target: {}ms", execution_time.as_millis());
        }
        
        Ok(result)
    }
    
    // Parse natural language requirement to structured components
    async fn parse_requirement_structure(requirement_text: &str) -> Result<ParsedRequirement> {
        // Step 1: Identify requirement type indicators
        let requirement_type = if requirement_text.to_lowercase().contains("must") ||
                                 requirement_text.to_lowercase().contains("shall") {
            RequirementType::Must
        } else if requirement_text.to_lowercase().contains("should") {
            RequirementType::Should  
        } else if requirement_text.to_lowercase().contains("may") {
            RequirementType::May
        } else {
            RequirementType::Must // Default for compliance requirements
        };
        
        // Step 2: Extract entities (subjects and objects)
        let entities = self.extract_entities(requirement_text).await?;
        
        // Step 3: Identify action verbs and predicates
        let actions = self.extract_actions(requirement_text).await?;
        
        // Step 4: Extract conditions and constraints
        let conditions = self.extract_conditions(requirement_text).await?;
        
        // Step 5: Identify cross-references
        let cross_references = self.extract_cross_references(requirement_text).await?;
        
        Ok(ParsedRequirement {
            original_text: requirement_text.to_string(),
            requirement_type,
            entities,
            actions,
            conditions,
            cross_references,
            confidence: self.calculate_parsing_confidence(&entities, &actions),
        })
    }
}

// Supporting data structures
struct ParsedRequirement {
    original_text: String,
    requirement_type: RequirementType,
    entities: Vec<Entity>,
    actions: Vec<Action>,
    conditions: Vec<Condition>,
    cross_references: Vec<CrossReference>,
    confidence: f64,
}

enum RequirementType {
    Must,    // MUST/SHALL - mandatory
    Should,  // SHOULD - recommended  
    May,     // MAY - optional
}

struct DatalogRule {
    id: String,
    text: String,
    source_requirement: String,
    rule_type: RequirementType,
    created_at: DateTime<Utc>,
    dependencies: Vec<String>,
}

struct QueryResult {
    query: ParsedQuery,
    results: Vec<QueryResultItem>,
    citations: Vec<Citation>,
    proof_chain: Vec<ProofStep>,
    execution_time_ms: u64,
    confidence: f64,
    used_rules: Vec<String>,
}
```

### 1.2 Prolog Engine Implementation

```rust
// File: src/symbolic/prolog/engine.rs

struct PrologEngine {
    engine: scryer_prolog::Machine,
    knowledge_base: Arc<RwLock<KnowledgeBase>>,
    inference_cache: DashMap<String, InferenceResult>,
    proof_tracer: Arc<ProofTracer>,
}

impl PrologEngine {
    // Initialize Prolog engine with domain ontology
    async fn new() -> Result<Self> {
        let mut engine = scryer_prolog::Machine::new();
        
        // Load domain ontology and base facts
        let ontology_facts = Self::load_domain_ontology().await?;
        for fact in ontology_facts {
            engine.consult_text(&fact)?;
        }
        
        Ok(Self {
            engine,
            knowledge_base: Arc::new(RwLock::new(KnowledgeBase::new())),
            inference_cache: DashMap::new(),
            proof_tracer: Arc::new(ProofTracer::new()),
        })
    }
    
    // Execute complex reasoning query with proof chain
    async fn query_with_proof(query_text: &str) -> Result<ProofResult> {
        let start_time = Instant::now();
        
        // Step 1: Parse query into Prolog syntax
        let prolog_query = self.parse_to_prolog_query(query_text).await?;
        
        // Step 2: Initialize proof tracer
        self.proof_tracer.begin_trace(&prolog_query).await;
        
        // Step 3: Execute query with tracing
        let query_result = self.engine.run_query_with_tracing(&prolog_query)?;
        
        // Step 4: Extract proof steps from trace
        let proof_steps = self.proof_tracer.extract_proof_steps().await?;
        
        // Step 5: Validate proof completeness
        let proof_validation = self.validate_proof_chain(&proof_steps).await?;
        
        // Step 6: Generate citations from proof steps
        let citations = self.generate_citations_from_proof(&proof_steps).await?;
        
        // Step 7: Calculate confidence based on proof strength
        let confidence = self.calculate_proof_confidence(&proof_steps, &proof_validation);
        
        let execution_time = start_time.elapsed();
        
        Ok(ProofResult {
            query: prolog_query,
            result: query_result,
            proof_steps,
            citations,
            confidence,
            execution_time_ms: execution_time.as_millis() as u64,
            validation: proof_validation,
        })
    }
    
    // Add compliance rule to knowledge base
    async fn add_compliance_rule(rule_text: &str, source_document: &str) -> Result<()> {
        // Step 1: Parse rule into Prolog clauses
        let prolog_clauses = self.parse_compliance_rule(rule_text).await?;
        
        // Step 2: Validate clause syntax and logic
        for clause in &prolog_clauses {
            self.validate_prolog_clause(clause)?;
        }
        
        // Step 3: Add to engine
        for clause in &prolog_clauses {
            self.engine.consult_text(clause)?;
        }
        
        // Step 4: Update knowledge base metadata
        let mut kb = self.knowledge_base.write().await;
        kb.add_rule_metadata(RuleMetadata {
            clauses: prolog_clauses,
            source_text: rule_text.to_string(),
            source_document: source_document.to_string(),
            added_at: Utc::now(),
        }).await?;
        
        info!("Added Prolog rule from {}: {}", source_document, rule_text);
        Ok(())
    }
    
    // Convert natural language query to Prolog syntax
    async fn parse_to_prolog_query(query_text: &str) -> Result<PrologQuery> {
        // Step 1: Identify query type (existence, relationship, inference)
        let query_type = self.classify_query_type(query_text).await?;
        
        // Step 2: Extract query components
        let components = match query_type {
            QueryType::Existence => {
                // "Does X exist?" -> exists(X)
                let entity = self.extract_query_entity(query_text).await?;
                vec![format!("exists({})", entity)]
            },
            QueryType::Relationship => {
                // "What relates to X?" -> related(X, Y)
                let (subject, relation_type) = self.extract_relationship_query(query_text).await?;
                vec![format!("{}({}, Y)", relation_type, subject)]
            },
            QueryType::Inference => {
                // "What are the implications of X?" -> implies(X, Y)
                let premise = self.extract_inference_premise(query_text).await?;
                vec![format!("implies({}, Y)", premise)]
            },
            QueryType::Compliance => {
                // "Is X compliant with Y?" -> compliant(X, Y)
                let (system, standard) = self.extract_compliance_query(query_text).await?;
                vec![format!("compliant({}, {})", system, standard)]
            },
        };
        
        Ok(PrologQuery {
            original_text: query_text.to_string(),
            query_type,
            prolog_clauses: components,
            variables: self.extract_query_variables(&components),
        })
    }
    
    // Load domain-specific ontology for compliance reasoning
    async fn load_domain_ontology() -> Result<Vec<String>> {
        vec![
            // Basic compliance concepts
            "compliance_framework(pci_dss).".to_string(),
            "compliance_framework(iso_27001).".to_string(),
            "compliance_framework(soc2).".to_string(),
            "compliance_framework(nist).".to_string(),
            
            // Data classification rules
            "sensitive_data(cardholder_data).".to_string(),
            "sensitive_data(personal_information).".to_string(),
            "sensitive_data(authentication_data).".to_string(),
            
            // Security control categories
            "security_control(encryption).".to_string(),
            "security_control(access_control).".to_string(),
            "security_control(network_security).".to_string(),
            "security_control(monitoring).".to_string(),
            
            // Inference rules for compliance
            "requires_protection(Data) :- sensitive_data(Data).".to_string(),
            "requires_encryption(Data) :- cardholder_data(Data).".to_string(),
            "requires_access_control(System) :- processes_sensitive_data(System).".to_string(),
            
            // Compliance derivation rules
            "compliant(System, Framework) :- ".to_string() +
                "compliance_framework(Framework), " +
                "implements_all_controls(System, Framework).",
                
            "implements_all_controls(System, Framework) :- ".to_string() +
                "findall(Control, required_control(Framework, Control), Controls), " +
                "forall(member(Control, Controls), implements(System, Control)).",
        ]
    }
}

struct ProofResult {
    query: PrologQuery,
    result: QueryResult,
    proof_steps: Vec<ProofStep>,
    citations: Vec<Citation>,
    confidence: f64,
    execution_time_ms: u64,
    validation: ProofValidation,
}

struct ProofStep {
    step_number: usize,
    rule_applied: String,
    premises: Vec<String>,
    conclusion: String,
    source_citation: Option<Citation>,
    confidence: f64,
}
```

---

## 📊 COMPONENT 2: GRAPH DATABASE INTEGRATION

### 2.1 Neo4j Client Implementation

```rust
// File: src/graph/neo4j/client.rs

struct Neo4jClient {
    driver: neo4j::Driver,
    session_pool: Arc<SessionPool>,
    schema_manager: Arc<SchemaManager>,
    query_cache: DashMap<String, CachedQuery>,
    performance_monitor: Arc<PerformanceMonitor>,
}

impl Neo4jClient {
    // Initialize Neo4j client with connection pool
    async fn new(connection_uri: &str, auth: BasicAuth) -> Result<Self> {
        let driver = neo4j::Driver::new(connection_uri, auth).await?;
        
        // Test connection
        let session = driver.session(&SessionConfig::default()).await?;
        session.run("RETURN 1").await?;
        
        let client = Self {
            driver,
            session_pool: Arc::new(SessionPool::new(16).await?),
            schema_manager: Arc::new(SchemaManager::new()),
            query_cache: DashMap::new(),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
        };
        
        // Initialize database schema
        client.initialize_schema().await?;
        
        Ok(client)
    }
    
    // Create document hierarchy in graph
    async fn create_document_hierarchy(document: &ProcessedDocument) -> Result<DocumentGraph> {
        let session = self.session_pool.get().await?;
        let mut tx = session.begin_transaction().await?;
        
        // Step 1: Create document node
        let doc_query = r#"
            CREATE (d:Document {
                id: $doc_id,
                title: $title,
                version: $version,
                doc_type: $doc_type,
                created_at: $created_at
            })
            RETURN d
        "#;
        
        let doc_result = tx.run(doc_query)
            .with_parameters([
                ("doc_id", document.id.clone()),
                ("title", document.title.clone()),
                ("version", document.version.clone()),
                ("doc_type", document.doc_type.to_string()),
                ("created_at", document.created_at.to_rfc3339()),
            ])
            .await?;
        
        let doc_node_id = doc_result.single()?.get::<String>("d.id")?;
        
        // Step 2: Create section hierarchy
        let mut section_nodes = Vec::new();
        for section in &document.hierarchy.sections {
            let section_node = self.create_section_node(&mut tx, section, &doc_node_id).await?;
            section_nodes.push(section_node);
        }
        
        // Step 3: Create requirement nodes
        let mut requirement_nodes = Vec::new();
        for requirement in &document.requirements {
            let req_node = self.create_requirement_node(&mut tx, requirement).await?;
            requirement_nodes.push(req_node);
            
            // Link requirement to its section
            let section_id = self.find_section_for_requirement(requirement, &section_nodes).await?;
            self.create_contains_relationship(&mut tx, &section_id, &req_node.id).await?;
        }
        
        // Step 4: Create cross-reference relationships
        for cross_ref in &document.cross_references {
            self.create_cross_reference_relationship(&mut tx, cross_ref).await?;
        }
        
        // Step 5: Commit transaction
        tx.commit().await?;
        
        Ok(DocumentGraph {
            document_node: doc_node_id,
            section_nodes,
            requirement_nodes,
            relationships: self.get_document_relationships(&document.id).await?,
        })
    }
    
    // Create requirement node with properties
    async fn create_requirement_node(tx: &mut Transaction, requirement: &Requirement) -> Result<RequirementNode> {
        let query = r#"
            CREATE (r:Requirement {
                id: $req_id,
                text: $text,
                section: $section,
                requirement_type: $req_type,
                domain: $domain,
                priority: $priority,
                created_at: $created_at
            })
            RETURN r
        "#;
        
        let result = tx.run(query)
            .with_parameters([
                ("req_id", requirement.id.clone()),
                ("text", requirement.text.clone()),
                ("section", requirement.section.clone()),
                ("req_type", requirement.requirement_type.to_string()),
                ("domain", requirement.domain.clone()),
                ("priority", requirement.priority.to_string()),
                ("created_at", requirement.created_at.to_rfc3339()),
            ])
            .await?;
        
        let node = result.single()?;
        Ok(RequirementNode {
            id: node.get::<String>("r.id")?,
            neo4j_id: node.get::<i64>("id(r)")?,
            properties: requirement.clone(),
        })
    }
    
    // Create typed relationship between nodes
    async fn create_relationship(
        from_id: &str,
        to_id: &str,
        relationship_type: RelationshipType,
    ) -> Result<RelationshipEdge> {
        let session = self.session_pool.get().await?;
        
        let relationship_label = match relationship_type {
            RelationshipType::References => "REFERENCES",
            RelationshipType::DependsOn => "DEPENDS_ON",
            RelationshipType::Exception => "EXCEPTION",
            RelationshipType::Implements => "IMPLEMENTS",
            RelationshipType::Contains => "CONTAINS",
        };
        
        let query = format!(r#"
            MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
            CREATE (a)-[r:{}]->(b)
            RETURN r, id(r) as rel_id
        "#, relationship_label);
        
        let result = session.run(&query)
            .with_parameters([
                ("from_id", from_id),
                ("to_id", to_id),
            ])
            .await?;
        
        let record = result.single()?;
        Ok(RelationshipEdge {
            id: record.get::<i64>("rel_id")?,
            from_node: from_id.to_string(),
            to_node: to_id.to_string(),
            relationship_type,
            properties: HashMap::new(),
            created_at: Utc::now(),
        })
    }
    
    // Execute relationship traversal query
    async fn traverse_requirements(
        start_id: &str,
        max_depth: usize,
        relationship_types: Vec<RelationshipType>,
    ) -> Result<TraversalResult> {
        let start_time = Instant::now();
        
        // Build relationship type filter
        let rel_filter = relationship_types.iter()
            .map(|rt| match rt {
                RelationshipType::References => "REFERENCES",
                RelationshipType::DependsOn => "DEPENDS_ON", 
                RelationshipType::Exception => "EXCEPTION",
                RelationshipType::Implements => "IMPLEMENTS",
                RelationshipType::Contains => "CONTAINS",
            })
            .collect::<Vec<_>>()
            .join("|");
        
        let query = format!(r#"
            MATCH path = (start:Requirement {{id: $start_id}})
                -[:{}*1..{}]-(related:Requirement)
            RETURN path,
                   start,
                   related,
                   length(path) as depth,
                   [r in relationships(path) | type(r)] as relationship_chain
            ORDER BY depth, related.section
        "#, rel_filter, max_depth);
        
        let session = self.session_pool.get().await?;
        let result = session.run(&query)
            .with_parameter("start_id", start_id)
            .await?;
        
        let mut traversal_paths = Vec::new();
        let mut related_requirements = Vec::new();
        
        for record in result {
            let depth = record.get::<i64>("depth")? as usize;
            let related_node = record.get::<neo4j::Node>("related")?;
            let relationship_chain = record.get::<Vec<String>>("relationship_chain")?;
            
            let related_req = Requirement {
                id: related_node.get::<String>("id")?,
                text: related_node.get::<String>("text")?,
                section: related_node.get::<String>("section")?,
                requirement_type: RequirementType::from_string(&related_node.get::<String>("requirement_type")?)?,
                domain: related_node.get::<String>("domain")?,
                priority: Priority::from_string(&related_node.get::<String>("priority")?)?,
                created_at: DateTime::parse_from_rfc3339(&related_node.get::<String>("created_at")?)?.into(),
            };
            
            related_requirements.push(related_req.clone());
            
            traversal_paths.push(TraversalPath {
                start_id: start_id.to_string(),
                end_id: related_req.id,
                depth,
                relationship_chain: relationship_chain.iter()
                    .map(|r| RelationshipType::from_string(r).unwrap())
                    .collect(),
                path_strength: self.calculate_path_strength(&relationship_chain),
            });
        }
        
        let execution_time = start_time.elapsed();
        
        // Validate performance requirement (<200ms for 3-hop)
        if max_depth <= 3 && execution_time.as_millis() > 200 {
            warn!("Graph traversal exceeded 200ms performance target: {}ms", execution_time.as_millis());
        }
        
        Ok(TraversalResult {
            start_requirement_id: start_id.to_string(),
            related_requirements,
            traversal_paths,
            execution_time_ms: execution_time.as_millis() as u64,
            total_paths: traversal_paths.len(),
        })
    }
    
    // Initialize database schema with indexes and constraints
    async fn initialize_schema() -> Result<()> {
        let session = self.session_pool.get().await?;
        
        // Create uniqueness constraints
        let constraints = vec![
            "CREATE CONSTRAINT requirement_id_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        ];
        
        for constraint in constraints {
            session.run(constraint).await?;
        }
        
        // Create performance indexes
        let indexes = vec![
            "CREATE INDEX requirement_section_idx IF NOT EXISTS FOR (r:Requirement) ON (r.section)",
            "CREATE INDEX requirement_type_idx IF NOT EXISTS FOR (r:Requirement) ON (r.requirement_type)",
            "CREATE INDEX requirement_domain_idx IF NOT EXISTS FOR (r:Requirement) ON (r.domain)",
            "CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
        ];
        
        for index in indexes {
            session.run(index).await?;
        }
        
        info!("Neo4j schema initialized successfully");
        Ok(())
    }
}

// Supporting data structures
struct DocumentGraph {
    document_node: String,
    section_nodes: Vec<SectionNode>,
    requirement_nodes: Vec<RequirementNode>,
    relationships: Vec<RelationshipEdge>,
}

struct TraversalResult {
    start_requirement_id: String,
    related_requirements: Vec<Requirement>,
    traversal_paths: Vec<TraversalPath>,
    execution_time_ms: u64,
    total_paths: usize,
}

struct TraversalPath {
    start_id: String,
    end_id: String,
    depth: usize,
    relationship_chain: Vec<RelationshipType>,
    path_strength: f64,
}

enum RelationshipType {
    References,   // Direct citation
    DependsOn,    // Logical dependency
    Exception,    // Override relationship
    Implements,   // Implementation relationship
    Contains,     // Hierarchical containment
}
```

---

## 🧠 COMPONENT 3: SMART INGESTION PIPELINE

### 3.1 Document Classification System

```rust
// File: src/ingestion/classification/document_classifier.rs

struct DocumentClassifier {
    doc_type_network: Network<f32>,
    section_type_network: Network<f32>,
    feature_extractor: FeatureExtractor,
    classification_cache: DashMap<String, ClassificationResult>,
    performance_metrics: Arc<RwLock<ClassificationMetrics>>,
}

impl DocumentClassifier {
    // Initialize neural networks for document classification
    async fn new() -> Result<Self> {
        // Document type classifier network (4 output classes)
        let doc_network = Network::new(&[
            512,  // Input features (document metadata + text features)
            256,  // Hidden layer 1
            128,  // Hidden layer 2
            4,    // Output classes (PCI-DSS, ISO-27001, SOC2, NIST)
        ])?;
        
        // Section type classifier network (6 output classes)
        let section_network = Network::new(&[
            256,  // Input features (section metadata + text features)
            128,  // Hidden layer 1
            64,   // Hidden layer 2
            6,    // Output classes (Requirements, Definitions, Procedures, Appendices, Examples, References)
        ])?;
        
        Ok(Self {
            doc_type_network: doc_network,
            section_type_network: section_network,
            feature_extractor: FeatureExtractor::new().await?,
            classification_cache: DashMap::new(),
            performance_metrics: Arc::new(RwLock::new(ClassificationMetrics::new())),
        })
    }
    
    // Classify complete document type and extract structure
    async fn classify_document(document_bytes: &[u8]) -> Result<DocumentClassification> {
        let start_time = Instant::now();
        
        // Step 1: Extract document text and metadata
        let parsed_doc = self.parse_document(document_bytes).await?;
        
        // Step 2: Extract features for document-level classification
        let doc_features = self.feature_extractor.extract_document_features(&parsed_doc).await?;
        
        // Step 3: Run document type classification
        let doc_type_output = self.doc_type_network.run(&doc_features)?;
        let doc_type_result = self.interpret_doc_type_output(doc_type_output)?;
        
        // Step 4: Extract table of contents and build hierarchy
        let toc = self.extract_table_of_contents(&parsed_doc).await?;
        let hierarchy = self.build_document_hierarchy(&toc).await?;
        
        // Step 5: Classify each section in the hierarchy
        let mut section_classifications = Vec::new();
        for section in &hierarchy.sections {
            let section_class = self.classify_section(section).await?;
            section_classifications.push(section_class);
        }
        
        // Step 6: Extract metadata and document properties
        let metadata = DocumentMetadata {
            title: parsed_doc.extract_title(),
            version: parsed_doc.extract_version(),
            publication_date: parsed_doc.extract_date(),
            author: parsed_doc.extract_author(),
            page_count: parsed_doc.page_count(),
            word_count: parsed_doc.word_count(),
        };
        
        let classification_time = start_time.elapsed();
        
        // Validate performance requirement (>90% accuracy, reasonable time)
        if doc_type_result.confidence < 0.90 {
            warn!("Document classification confidence below 90%: {:.2}%", 
                  doc_type_result.confidence * 100.0);
        }
        
        Ok(DocumentClassification {
            document_type: doc_type_result.document_type,
            confidence: doc_type_result.confidence,
            hierarchy,
            section_classifications,
            metadata,
            classification_time_ms: classification_time.as_millis() as u64,
            features_used: doc_features.feature_names(),
        })
    }
    
    // Classify individual document section
    async fn classify_section(section: &DocumentSection) -> Result<SectionClassification> {
        let start_time = Instant::now();
        
        // Step 1: Extract section features
        let section_features = self.feature_extractor.extract_section_features(section).await?;
        
        // Step 2: Run section type classification
        let section_output = self.section_type_network.run(&section_features)?;
        let section_result = self.interpret_section_type_output(section_output)?;
        
        // Step 3: Extract section-specific metadata
        let section_metadata = SectionMetadata {
            section_number: section.extract_section_number(),
            title: section.title.clone(),
            page_numbers: section.page_range.clone(),
            subsection_count: section.subsections.len(),
            paragraph_count: section.paragraphs.len(),
        };
        
        let classification_time = start_time.elapsed();
        
        Ok(SectionClassification {
            section_id: section.id.clone(),
            section_type: section_result.section_type,
            confidence: section_result.confidence,
            metadata: section_metadata,
            classification_time_ms: classification_time.as_millis() as u64,
        })
    }
    
    // Extract document features for neural network input
    async fn extract_document_features(parsed_doc: &ParsedDocument) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(512);
        
        // Text-based features (first 256 features)
        features.extend(self.extract_text_features(&parsed_doc.full_text).await?);
        
        // Structure-based features (next 128 features)
        features.extend(self.extract_structure_features(parsed_doc).await?);
        
        // Metadata-based features (next 64 features)
        features.extend(self.extract_metadata_features(parsed_doc).await?);
        
        // Domain-specific features (final 64 features)
        features.extend(self.extract_domain_features(parsed_doc).await?);
        
        // Normalize features to [0,1] range
        self.normalize_features(&mut features);
        
        Ok(features)
    }
    
    // Extract text-based features using NLP techniques
    async fn extract_text_features(text: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(256);
        
        // Word frequency features
        let word_counts = self.count_domain_keywords(text);
        features.extend(self.normalize_word_counts(&word_counts));
        
        // Sentence structure features
        features.push(self.calculate_average_sentence_length(text));
        features.push(self.calculate_complex_sentence_ratio(text));
        features.push(self.calculate_passive_voice_ratio(text));
        
        // Compliance-specific terminology
        let compliance_terms = vec![
            "requirement", "must", "shall", "should", "compliance",
            "security", "encryption", "access", "control", "audit",
            "policy", "procedure", "documentation", "assessment",
        ];
        
        for term in compliance_terms {
            let term_frequency = text.to_lowercase().matches(term).count() as f32;
            let normalized_freq = term_frequency / text.len() as f32 * 10000.0; // per 10k chars
            features.push(normalized_freq);
        }
        
        // Fill remaining features with text statistics
        while features.len() < 256 {
            features.push(0.0); // Placeholder for additional features
        }
        
        Ok(features)
    }
    
    // Interpret neural network output for document type
    fn interpret_doc_type_output(output: Vec<f32>) -> Result<DocumentTypeResult> {
        let document_types = vec![
            DocumentType::PciDss,
            DocumentType::Iso27001,
            DocumentType::Soc2,
            DocumentType::Nist,
        ];
        
        // Find maximum activation
        let (max_index, max_confidence) = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        Ok(DocumentTypeResult {
            document_type: document_types[max_index].clone(),
            confidence: *max_confidence as f64,
            all_scores: output.iter()
                .zip(document_types.iter())
                .map(|(score, doc_type)| (doc_type.clone(), *score as f64))
                .collect(),
        })
    }
}

// Supporting data structures
struct DocumentClassification {
    document_type: DocumentType,
    confidence: f64,
    hierarchy: DocumentHierarchy,
    section_classifications: Vec<SectionClassification>,
    metadata: DocumentMetadata,
    classification_time_ms: u64,
    features_used: Vec<String>,
}

enum DocumentType {
    PciDss,
    Iso27001,
    Soc2,
    Nist,
    Unknown,
}

enum SectionType {
    Requirements,
    Definitions,
    Procedures,
    Appendices,
    Examples,
    References,
}

struct SectionClassification {
    section_id: String,
    section_type: SectionType,
    confidence: f64,
    metadata: SectionMetadata,
    classification_time_ms: u64,
}
```

### 3.2 Requirement Extraction Engine

```rust
// File: src/ingestion/extraction/requirement_extractor.rs

struct RequirementExtractor {
    extraction_network: Network<f32>,
    requirement_parser: RequirementParser,
    cross_reference_detector: CrossReferenceDetector,
    extraction_cache: DashMap<String, ExtractionResult>,
}

impl RequirementExtractor {
    // Initialize requirement extraction system
    async fn new() -> Result<Self> {
        // Binary classification network (requirement vs non-requirement)
        let extraction_network = Network::new(&[
            128,  // Input features (sentence/paragraph features)
            64,   // Hidden layer 1  
            32,   // Hidden layer 2
            1,    // Output (probability of being a requirement)
        ])?;
        
        Ok(Self {
            extraction_network,
            requirement_parser: RequirementParser::new().await?,
            cross_reference_detector: CrossReferenceDetector::new().await?,
            extraction_cache: DashMap::new(),
        })
    }
    
    // Extract all requirements from a classified document section
    async fn extract_requirements(section: &DocumentSection) -> Result<Vec<ExtractedRequirement>> {
        let start_time = Instant::now();
        
        // Step 1: Split section into individual sentences/paragraphs
        let text_units = self.segment_section_text(section).await?;
        
        // Step 2: Classify each text unit as requirement or non-requirement
        let mut requirement_candidates = Vec::new();
        for (index, text_unit) in text_units.iter().enumerate() {
            let is_requirement = self.classify_as_requirement(text_unit).await?;
            if is_requirement.probability > 0.5 {
                requirement_candidates.push((index, text_unit, is_requirement));
            }
        }
        
        // Step 3: Parse each requirement candidate
        let mut extracted_requirements = Vec::new();
        for (index, text_unit, classification) in requirement_candidates {
            let parsed_req = self.parse_requirement_text(text_unit).await?;
            
            // Step 4: Extract cross-references within the requirement
            let cross_refs = self.cross_reference_detector.find_references(text_unit).await?;
            
            // Step 5: Determine requirement type and priority
            let req_type = self.determine_requirement_type(text_unit);
            let priority = self.calculate_requirement_priority(text_unit, &req_type);
            
            // Step 6: Create extracted requirement object
            let extracted_req = ExtractedRequirement {
                id: self.generate_requirement_id(section, index),
                text: text_unit.text.clone(),
                source_section: section.id.clone(),
                paragraph_index: index,
                requirement_type: req_type,
                priority,
                domain: self.classify_requirement_domain(text_unit).await?,
                cross_references: cross_refs,
                extraction_confidence: classification.probability,
                parsed_components: parsed_req,
                extracted_at: Utc::now(),
            };
            
            extracted_requirements.push(extracted_req);
        }
        
        let extraction_time = start_time.elapsed();
        
        // Validate extraction completeness (100% requirement coverage goal)
        self.validate_extraction_completeness(section, &extracted_requirements).await?;
        
        info!("Extracted {} requirements from section {} in {}ms", 
              extracted_requirements.len(), section.id, extraction_time.as_millis());
        
        Ok(extracted_requirements)
    }
    
    // Classify text unit as requirement or non-requirement
    async fn classify_as_requirement(text_unit: &TextUnit) -> Result<RequirementClassification> {
        // Step 1: Extract features from text unit
        let features = self.extract_requirement_features(text_unit).await?;
        
        // Step 2: Run neural network classification
        let output = self.extraction_network.run(&features)?;
        let probability = output[0] as f64;
        
        // Step 3: Add rule-based validation
        let rule_based_score = self.apply_requirement_rules(text_unit).await?;
        
        // Step 4: Combine neural and rule-based scores
        let combined_probability = (probability + rule_based_score) / 2.0;
        
        Ok(RequirementClassification {
            probability: combined_probability,
            neural_score: probability,
            rule_based_score,
            confidence: (probability - 0.5).abs() * 2.0, // Distance from decision boundary
        })
    }
    
    // Parse requirement text into structured components
    async fn parse_requirement_text(text_unit: &TextUnit) -> Result<ParsedRequirement> {
        let text = &text_unit.text;
        
        // Step 1: Identify requirement indicator words
        let indicators = self.find_requirement_indicators(text);
        
        // Step 2: Extract subject entities
        let subjects = self.extract_requirement_subjects(text).await?;
        
        // Step 3: Extract action verbs and predicates
        let actions = self.extract_requirement_actions(text).await?;
        
        // Step 4: Extract conditions and constraints
        let conditions = self.extract_requirement_conditions(text).await?;
        
        // Step 5: Extract exceptions and qualifiers
        let exceptions = self.extract_requirement_exceptions(text).await?;
        
        Ok(ParsedRequirement {
            indicators,
            subjects,
            actions,
            conditions,
            exceptions,
            original_text: text.clone(),
            parse_confidence: self.calculate_parse_confidence(&subjects, &actions),
        })
    }
    
    // Apply rule-based requirement identification
    async fn apply_requirement_rules(text_unit: &TextUnit) -> Result<f64> {
        let text = &text_unit.text.to_lowercase();
        let mut score = 0.0;
        let mut rule_count = 0;
        
        // Rule 1: Strong requirement indicators
        let strong_indicators = vec!["must", "shall", "required", "mandatory"];
        for indicator in strong_indicators {
            if text.contains(indicator) {
                score += 0.9;
                rule_count += 1;
            }
        }
        
        // Rule 2: Moderate requirement indicators  
        let moderate_indicators = vec!["should", "ought to", "needs to", "has to"];
        for indicator in moderate_indicators {
            if text.contains(indicator) {
                score += 0.7;
                rule_count += 1;
            }
        }
        
        // Rule 3: Optional requirement indicators
        let optional_indicators = vec!["may", "can", "could", "might"];
        for indicator in optional_indicators {
            if text.contains(indicator) {
                score += 0.3;
                rule_count += 1;
            }
        }
        
        // Rule 4: Compliance-specific patterns
        let compliance_patterns = vec![
            r"\d+\.\d+\.\d+", // Section references (e.g., 3.2.1)
            r"requirement \d+", // Explicit requirement numbering
            r"control \d+",     // Control references
        ];
        
        for pattern in compliance_patterns {
            if regex::Regex::new(pattern)?.is_match(text) {
                score += 0.8;
                rule_count += 1;
            }
        }
        
        // Rule 5: Sentence structure indicators
        if text.contains("in order to") || text.contains("to ensure") {
            score += 0.6;
            rule_count += 1;
        }
        
        // Average the scores if any rules matched
        if rule_count > 0 {
            Ok((score / rule_count as f64).min(1.0))
        } else {
            Ok(0.1) // Low default score if no rules matched
        }
    }
    
    // Validate that all requirements in section were extracted
    async fn validate_extraction_completeness(
        section: &DocumentSection,
        extracted_requirements: &[ExtractedRequirement],
    ) -> Result<ExtractionValidation> {
        // Step 1: Count explicit requirement indicators in text
        let explicit_indicators = self.count_explicit_requirement_indicators(&section.text);
        
        // Step 2: Count extracted requirements
        let extracted_count = extracted_requirements.len();
        
        // Step 3: Calculate coverage ratio
        let coverage_ratio = if explicit_indicators > 0 {
            extracted_count as f64 / explicit_indicators as f64
        } else {
            1.0 // If no explicit indicators, assume complete coverage
        };
        
        // Step 4: Check for missed patterns
        let potentially_missed = self.find_potentially_missed_requirements(
            &section.text,
            extracted_requirements,
        ).await?;
        
        let validation = ExtractionValidation {
            section_id: section.id.clone(),
            explicit_indicator_count: explicit_indicators,
            extracted_requirement_count: extracted_count,
            coverage_ratio,
            potentially_missed_count: potentially_missed.len(),
            potentially_missed_patterns: potentially_missed,
            completeness_score: if coverage_ratio >= 0.95 && potentially_missed.is_empty() {
                1.0
            } else {
                (coverage_ratio * 0.8 + (1.0 - potentially_missed.len() as f64 * 0.1)).max(0.0)
            },
        };
        
        // Log validation results
        if validation.completeness_score < 1.0 {
            warn!("Extraction completeness below 100% for section {}: {:.2}%", 
                  section.id, validation.completeness_score * 100.0);
        }
        
        Ok(validation)
    }
}

// Supporting data structures
struct ExtractedRequirement {
    id: String,
    text: String,
    source_section: String,
    paragraph_index: usize,
    requirement_type: RequirementType,
    priority: Priority,
    domain: String,
    cross_references: Vec<CrossReference>,
    extraction_confidence: f64,
    parsed_components: ParsedRequirement,
    extracted_at: DateTime<Utc>,
}

struct ExtractionValidation {
    section_id: String,
    explicit_indicator_count: usize,
    extracted_requirement_count: usize,
    coverage_ratio: f64,
    potentially_missed_count: usize,
    potentially_missed_patterns: Vec<String>,
    completeness_score: f64,
}

enum RequirementType {
    Must,       // Mandatory (MUST/SHALL)
    Should,     // Recommended (SHOULD)
    May,        // Optional (MAY/CAN)
    Guideline,  // Best practice guidance
}

enum Priority {
    Critical,   // Security-critical requirements
    High,       // Important compliance requirements
    Medium,     // Standard requirements
    Low,        // Nice-to-have requirements
}
```

---

## 🔗 COMPONENT INTEGRATION PSEUDOCODE

### 4.1 Smart Ingestion Pipeline Coordinator

```rust
// File: src/ingestion/pipeline/coordinator.rs

struct SmartIngestionPipeline {
    document_classifier: Arc<DocumentClassifier>,
    requirement_extractor: Arc<RequirementExtractor>,
    logic_generator: Arc<LogicRuleGenerator>,
    graph_builder: Arc<GraphBuilder>,
    symbolic_engine: Arc<SymbolicEngine>,
    pipeline_metrics: Arc<RwLock<PipelineMetrics>>,
}

impl SmartIngestionPipeline {
    // Process complete document through neurosymbolic pipeline
    async fn process_document(document_bytes: &[u8]) -> Result<ProcessedDocument> {
        let start_time = Instant::now();
        
        // Step 1: Document classification and structure extraction
        let classification = self.document_classifier.classify_document(document_bytes).await?;
        info!("Document classified as {:?} with confidence {:.2}%", 
              classification.document_type, classification.confidence * 100.0);
        
        // Step 2: Requirement extraction from all sections
        let mut all_requirements = Vec::new();
        for section_class in &classification.section_classifications {
            if section_class.section_type == SectionType::Requirements {
                let section = classification.hierarchy.get_section(&section_class.section_id)?;
                let requirements = self.requirement_extractor.extract_requirements(section).await?;
                all_requirements.extend(requirements);
            }
        }
        
        info!("Extracted {} total requirements", all_requirements.len());
        
        // Step 3: Generate logic rules from requirements
        let mut logic_rules = Vec::new();
        for requirement in &all_requirements {
            let datalog_rule = self.logic_generator.generate_datalog_rule(requirement).await?;
            let prolog_clauses = self.logic_generator.generate_prolog_clauses(requirement).await?;
            
            logic_rules.push(LogicRules {
                requirement_id: requirement.id.clone(),
                datalog_rule,
                prolog_clauses,
                generated_at: Utc::now(),
            });
        }
        
        // Step 4: Build knowledge graph
        let knowledge_graph = self.graph_builder.build_document_graph(
            &classification,
            &all_requirements,
        ).await?;
        
        // Step 5: Load rules into symbolic reasoning engine
        for logic_rule in &logic_rules {
            self.symbolic_engine.add_datalog_rule(&logic_rule.datalog_rule).await?;
            
            for prolog_clause in &logic_rule.prolog_clauses {
                self.symbolic_engine.add_prolog_clause(prolog_clause).await?;
            }
        }
        
        // Step 6: Create cross-reference relationships in graph
        let cross_references = self.extract_all_cross_references(&all_requirements).await?;
        for cross_ref in &cross_references {
            self.graph_builder.create_cross_reference_relationship(cross_ref).await?;
        }
        
        let processing_time = start_time.elapsed();
        
        // Validate performance requirements
        let pages_processed = classification.metadata.page_count as f64;
        let pages_per_second = pages_processed / processing_time.as_secs_f64();
        
        if pages_per_second < 2.0 {
            warn!("Processing speed below 2 pages/second target: {:.2}", pages_per_second);
        }
        
        // Step 7: Create final processed document
        Ok(ProcessedDocument {
            id: Uuid::new_v4(),
            source_document: document_bytes.to_vec(),
            classification,
            requirements: all_requirements,
            logic_rules,
            knowledge_graph,
            cross_references,
            processing_metrics: ProcessingMetrics {
                total_processing_time_ms: processing_time.as_millis() as u64,
                pages_per_second,
                requirements_extracted: logic_rules.len(),
                logic_rules_generated: logic_rules.len(),
                graph_nodes_created: knowledge_graph.total_nodes(),
                graph_relationships_created: knowledge_graph.total_relationships(),
            },
            processed_at: Utc::now(),
        })
    }
    
    // Validate complete pipeline performance and accuracy
    async fn validate_pipeline_performance(document: &ProcessedDocument) -> Result<PipelineValidation> {
        // Performance validation
        let performance_valid = document.processing_metrics.pages_per_second >= 2.0;
        
        // Accuracy validation
        let classification_accurate = document.classification.confidence >= 0.90;
        
        // Completeness validation
        let requirements_complete = document.requirements.iter()
            .all(|req| req.extraction_confidence >= 0.8);
        
        // Logic rule validation
        let logic_rules_valid = document.logic_rules.iter()
            .all(|rule| self.validate_logic_rule_syntax(&rule.datalog_rule));
        
        // Graph validation
        let graph_complete = document.knowledge_graph.validate_completeness().await?;
        
        Ok(PipelineValidation {
            performance_valid,
            classification_accurate,
            requirements_complete,
            logic_rules_valid,
            graph_complete,
            overall_valid: performance_valid && 
                          classification_accurate && 
                          requirements_complete && 
                          logic_rules_valid && 
                          graph_complete,
        })
    }
}

struct ProcessedDocument {
    id: Uuid,
    source_document: Vec<u8>,
    classification: DocumentClassification,
    requirements: Vec<ExtractedRequirement>,
    logic_rules: Vec<LogicRules>,
    knowledge_graph: KnowledgeGraph,
    cross_references: Vec<CrossReference>,
    processing_metrics: ProcessingMetrics,
    processed_at: DateTime<Utc>,
}

struct PipelineValidation {
    performance_valid: bool,
    classification_accurate: bool,
    requirements_complete: bool,
    logic_rules_valid: bool,
    graph_complete: bool,
    overall_valid: bool,
}
```

---

## 🎯 PHASE 1 IMPLEMENTATION SUMMARY

### Key Implementation Files Structure

```
src/symbolic/
├── datalog/
│   ├── engine.rs           # Core Datalog engine implementation
│   ├── rule_compiler.rs    # Natural language to Datalog conversion
│   └── query_processor.rs  # Query execution and optimization
├── prolog/
│   ├── engine.rs           # Core Prolog engine implementation  
│   ├── inference.rs        # Complex reasoning and proof chains
│   └── knowledge_base.rs   # Domain ontology management

src/graph/
├── neo4j/
│   ├── client.rs           # Neo4j database client
│   ├── schema.rs           # Graph schema management
│   └── traversal.rs        # Relationship traversal queries
└── models.rs               # Graph data structures

src/ingestion/
├── classification/
│   ├── document_classifier.rs  # Document type classification
│   └── feature_extractor.rs    # Feature extraction for neural networks
├── extraction/
│   ├── requirement_extractor.rs # Requirement extraction from text
│   └── cross_reference_detector.rs # Cross-reference identification
└── pipeline/
    ├── coordinator.rs      # Main ingestion pipeline
    └── logic_generator.rs  # Logic rule generation
```

### Performance Targets Summary

| Component | Target | Implementation Strategy |
|-----------|--------|------------------------|
| Document Processing | 2-5 pages/sec | Parallel processing, optimized parsers |
| Datalog Queries | <100ms | Pre-compiled rules, query optimization |
| Graph Traversal | <200ms (3-hop) | Indexed relationships, query caching |
| Classification | >90% accuracy | Trained neural networks, rule validation |
| Requirement Extraction | 100% coverage | Neural + rule-based hybrid approach |

### Integration with Existing System

- **DAA Orchestrator**: Register all neurosymbolic components
- **FACT Cache**: Cache logic query results and classifications
- **MRAP Loops**: Integrate symbolic reasoning into control flow
- **MongoDB**: Maintain existing document storage alongside graph
- **Existing Neural Networks**: Extend ruv-fann usage for classification

---

*Phase 1 Pseudocode complete - Ready for Architecture and Implementation phases*