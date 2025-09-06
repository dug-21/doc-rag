//! Response Generator CLI application

use response_generator::{
    Config, ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
    error::Result,
};
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use tokio;
use tracing::{info, error};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "response-generator")]
#[command(about = "High-accuracy response generation system with citation tracking")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,
    
    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a response for a query
    Generate {
        /// Query to process
        #[arg(short, long)]
        query: String,
        
        /// Context file paths
        #[arg(short, long)]
        context: Vec<String>,
        
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: String,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
        
        /// Minimum confidence threshold
        #[arg(long)]
        min_confidence: Option<f64>,
    },
    
    /// Benchmark response generation performance
    Benchmark {
        /// Number of test queries to run
        #[arg(short, long, default_value = "100")]
        queries: usize,
        
        /// Output benchmark results file
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Validate configuration file
    ValidateConfig {
        /// Configuration file to validate
        config_file: String,
    },
    
    /// Run interactive mode
    Interactive,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    init_logging(&cli.log_level, cli.verbose)?;
    
    // Load configuration
    let config = load_config(cli.config).await?;
    
    // Execute command
    match cli.command {
        Commands::Generate { query, context, format, output, min_confidence } => {
            generate_response(config, query, context, format, output, min_confidence).await?;
        }
        Commands::Benchmark { queries, output } => {
            run_benchmark(config, queries, output).await?;
        }
        Commands::ValidateConfig { config_file } => {
            validate_config_file(config_file).await?;
        }
        Commands::Interactive => {
            run_interactive_mode(config).await?;
        }
    }
    
    Ok(())
}

/// Initialize logging system
fn init_logging(level: &str, verbose: bool) -> Result<()> {
    let level = if verbose { "debug" } else { level };
    
    tracing_subscriber::fmt()
        .with_env_filter(level)
        .with_target(false)
        .init();
    
    info!("Response Generator initialized with log level: {}", level);
    Ok(())
}

/// Load configuration from file or defaults
async fn load_config(config_path: Option<String>) -> Result<Config> {
    match config_path {
        Some(path) => {
            info!("Loading configuration from: {}", path);
            Config::from_file(&path).await
        }
        None => {
            // Try to load from environment, fallback to defaults
            info!("Using default configuration");
            Ok(Config::from_env().unwrap_or_default())
        }
    }
}

/// Generate response for a query
async fn generate_response(
    config: Config,
    query: String,
    context_files: Vec<String>,
    format: String,
    output_file: Option<String>,
    min_confidence: Option<f64>,
) -> Result<()> {
    info!("Generating response for query: {}", query);
    
    let mut generator = ResponseGenerator::new(config).await;
    
    // Parse output format
    let output_format = match format.to_lowercase().as_str() {
        "json" => OutputFormat::Json,
        "markdown" | "md" => OutputFormat::Markdown,
        "text" | "txt" => OutputFormat::Text,
        "html" => OutputFormat::Html,
        "xml" => OutputFormat::Xml,
        "yaml" | "yml" => OutputFormat::Yaml,
        _ => {
            error!("Unsupported output format: {}", format);
            return Ok(());
        }
    };
    
    // Load context from files
    let mut context_chunks = Vec::new();
    for context_file in context_files {
        let content = tokio::fs::read_to_string(&context_file).await
            .map_err(|e| response_generator::error::ResponseError::external_service("file_system", &e.to_string()))?;
        
        context_chunks.push(ContextChunk {
            content,
            source: Source {
                id: Uuid::new_v4(),
                title: context_file.clone(),
                url: None,
                document_type: "file".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 1.0, // Default high relevance for provided context
            position: Some(0),
            metadata: HashMap::new(),
        });
    }
    
    // Build request
    let mut request_builder = GenerationRequest::builder()
        .query(query)
        .context(context_chunks)
        .format(output_format);
    
    if let Some(confidence) = min_confidence {
        request_builder = request_builder.min_confidence(confidence);
    }
    
    let request = request_builder.build()?;
    
    // Generate response
    let start_time = std::time::Instant::now();
    let response = generator.generate(request).await?;
    let generation_time = start_time.elapsed();
    
    info!("Response generated in {}ms with confidence {:.2}", 
          generation_time.as_millis(), response.confidence_score);
    
    // Output response
    match output_file {
        Some(path) => {
            tokio::fs::write(&path, &response.content).await
                .map_err(|e| response_generator::error::ResponseError::external_service("file_system", &e.to_string()))?;
            info!("Response saved to: {}", path);
        }
        None => {
            println!("{}", response.content);
        }
    }
    
    // Print metadata if verbose
    if !response.citations.is_empty() {
        info!("Citations: {}", response.citations.len());
    }
    if !response.warnings.is_empty() {
        info!("Warnings: {:?}", response.warnings);
    }
    
    Ok(())
}

/// Run benchmark tests
async fn run_benchmark(
    config: Config,
    num_queries: usize,
    output_file: Option<String>,
) -> Result<()> {
    info!("Running benchmark with {} queries", num_queries);
    
    let mut generator = ResponseGenerator::new(config).await;
    let test_queries = generate_test_queries(num_queries);
    
    let mut results = Vec::new();
    let start_time = std::time::Instant::now();
    
    for (i, query) in test_queries.iter().enumerate() {
        let request = GenerationRequest::builder()
            .query(query.clone())
            .build()?;
        
        let query_start = std::time::Instant::now();
        match generator.generate(request).await {
            Ok(response) => {
                let query_time = query_start.elapsed();
                results.push(BenchmarkResult {
                    query_id: i,
                    success: true,
                    duration_ms: query_time.as_millis() as u64,
                    confidence: response.confidence_score,
                    response_length: response.content.len(),
                });
            }
            Err(e) => {
                error!("Query {} failed: {}", i, e);
                results.push(BenchmarkResult {
                    query_id: i,
                    success: false,
                    duration_ms: query_start.elapsed().as_millis() as u64,
                    confidence: 0.0,
                    response_length: 0,
                });
            }
        }
        
        if (i + 1) % 10 == 0 {
            info!("Completed {} / {} queries", i + 1, num_queries);
        }
    }
    
    let total_time = start_time.elapsed();
    
    // Calculate statistics
    let successful_queries: Vec<_> = results.iter().filter(|r| r.success).collect();
    let success_rate = successful_queries.len() as f64 / results.len() as f64;
    let avg_duration = successful_queries.iter().map(|r| r.duration_ms).sum::<u64>() as f64 / successful_queries.len() as f64;
    let avg_confidence = successful_queries.iter().map(|r| r.confidence).sum::<f64>() / successful_queries.len() as f64;
    let under_100ms = successful_queries.iter().filter(|r| r.duration_ms < 100).count() as f64 / successful_queries.len() as f64;
    
    let benchmark_summary = BenchmarkSummary {
        total_queries: num_queries,
        successful_queries: successful_queries.len(),
        success_rate,
        avg_duration_ms: avg_duration,
        avg_confidence,
        under_100ms_rate: under_100ms,
        total_time_ms: total_time.as_millis() as u64,
    };
    
    // Output results
    let output = serde_json::to_string_pretty(&benchmark_summary)?;
    
    match output_file {
        Some(path) => {
            tokio::fs::write(&path, &output).await
                .map_err(|e| response_generator::error::ResponseError::external_service("file_system", &e.to_string()))?;
            info!("Benchmark results saved to: {}", path);
        }
        None => {
            println!("{}", output);
        }
    }
    
    info!("Benchmark completed: {:.1}% success rate, {:.1}ms avg duration", 
          success_rate * 100.0, avg_duration);
    
    Ok(())
}

/// Validate a configuration file
async fn validate_config_file(config_file: String) -> Result<()> {
    info!("Validating configuration file: {}", config_file);
    
    match Config::from_file(&config_file).await {
        Ok(config) => {
            match config.validate() {
                Ok(()) => {
                    info!("Configuration file is valid");
                    println!("✓ Configuration file is valid");
                    Ok(())
                }
                Err(e) => {
                    error!("Configuration validation failed: {}", e);
                    println!("✗ Configuration validation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            error!("Failed to load configuration file: {}", e);
            println!("✗ Failed to load configuration file: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run interactive mode
async fn run_interactive_mode(config: Config) -> Result<()> {
    info!("Starting interactive mode");
    println!("Response Generator Interactive Mode");
    println!("Type 'exit' to quit, 'help' for commands");
    
    let mut generator = ResponseGenerator::new(config).await;
    let stdin = std::io::stdin();
    
    loop {
        print!("query> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        let mut input = String::new();
        stdin.read_line(&mut input).unwrap();
        let input = input.trim();
        
        match input {
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            "help" => {
                println!("Commands:");
                println!("  <query>     - Generate response for query");
                println!("  exit/quit   - Exit interactive mode");
                println!("  help        - Show this help");
            }
            "" => continue,
            query => {
                let request = GenerationRequest::builder()
                    .query(query.to_string())
                    .format(OutputFormat::Markdown)
                    .build()?;
                
                match generator.generate(request).await {
                    Ok(response) => {
                        println!("\n--- Response ---");
                        println!("{}", response.content);
                        println!("\n--- Metadata ---");
                        println!("Confidence: {:.2}", response.confidence_score);
                        println!("Generation time: {}ms", response.metrics.total_duration.as_millis());
                        if !response.citations.is_empty() {
                            println!("Citations: {}", response.citations.len());
                        }
                        println!();
                    }
                    Err(e) => {
                        error!("Generation failed: {}", e);
                        println!("Error: {}", e);
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Generate test queries for benchmarking
fn generate_test_queries(count: usize) -> Vec<String> {
    let base_queries = vec![
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain the concept of neural networks",
        "What are the benefits of cloud computing?",
        "Describe the process of software development",
        "How do databases store and retrieve data?",
        "What is cybersecurity and why is it important?",
        "Explain the fundamentals of web development",
        "What are microservices in software architecture?",
        "How does version control work in software development?",
    ];
    
    let mut queries = Vec::new();
    for i in 0..count {
        let base_query = &base_queries[i % base_queries.len()];
        queries.push(format!("{} (Query {})", base_query, i + 1));
    }
    
    queries
}

#[derive(Debug, serde::Serialize)]
struct BenchmarkResult {
    query_id: usize,
    success: bool,
    duration_ms: u64,
    confidence: f64,
    response_length: usize,
}

#[derive(Debug, serde::Serialize)]
struct BenchmarkSummary {
    total_queries: usize,
    successful_queries: usize,
    success_rate: f64,
    avg_duration_ms: f64,
    avg_confidence: f64,
    under_100ms_rate: f64,
    total_time_ms: u64,
}