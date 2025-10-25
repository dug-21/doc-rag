/**
 * MCP Server Implementation for TypeScript RAG System
 *
 * This server exposes AgentDB semantic search and RAG capabilities
 * through the Model Context Protocol (MCP) for Claude integration.
 *
 * Architecture:
 * - Resources: Access to documents and collections
 * - Tools: Semantic search, ingestion, query processing
 * - Prompts: Pre-configured RAG templates
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { AgentDB } from 'agentdb';
import { AgenticFlow } from 'agentic-flow';
import { NeuralNet, initWasm } from 'ruv-swarm-wasm';
import pino from 'pino';

// Initialize logger
const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: { colorize: true }
  }
});

/**
 * Configuration interface for MCP server
 */
interface MCPServerConfig {
  agentdb: {
    host: string;
    port: number;
    collection: string;
  };
  agentic_flow: {
    enabled: boolean;
    default_topology: 'mesh' | 'hierarchical' | 'ring' | 'star';
  };
  neural: {
    enabled: boolean;
    model_path?: string;
  };
  server: {
    name: string;
    version: string;
  };
}

/**
 * Main MCP Server class
 */
class DocRAGMCPServer {
  private server: Server;
  private agentDB: AgentDB;
  private swarm?: AgenticFlow;
  private neuralNet?: NeuralNet;
  private config: MCPServerConfig;

  constructor(config: MCPServerConfig) {
    this.config = config;

    // Initialize MCP server
    this.server = new Server(
      {
        name: config.server.name,
        version: config.server.version,
      },
      {
        capabilities: {
          resources: {},
          tools: {},
          prompts: {},
        },
      }
    );

    // Initialize AgentDB
    this.agentDB = new AgentDB({
      host: config.agentdb.host,
      port: config.agentdb.port,
      collection: config.agentdb.collection,
    });

    this.setupHandlers();
  }

  /**
   * Initialize all components
   */
  async initialize(): Promise<void> {
    logger.info('Initializing MCP server...');

    // Connect to AgentDB
    await this.agentDB.connect();
    logger.info('AgentDB connected');

    // Initialize neural networks if enabled
    if (this.config.neural.enabled) {
      await initWasm({ simd: true });
      if (this.config.neural.model_path) {
        this.neuralNet = await NeuralNet.load(this.config.neural.model_path);
      }
      logger.info('Neural network initialized');
    }

    // Initialize agentic-flow swarm if enabled
    if (this.config.agentic_flow.enabled) {
      this.swarm = new AgenticFlow({
        topology: this.config.agentic_flow.default_topology,
        maxAgents: 6,
        strategy: 'adaptive',
      });
      await this.swarm.init();
      logger.info('Agentic-flow swarm initialized');
    }

    logger.info('MCP server ready');
  }

  /**
   * Setup all MCP protocol handlers
   */
  private setupHandlers(): void {
    // List available resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      const collections = await this.agentDB.listCollections();

      return {
        resources: collections.map(collection => ({
          uri: `agentdb://collection/${collection.name}`,
          name: `Collection: ${collection.name}`,
          description: `AgentDB collection with ${collection.count} documents`,
          mimeType: 'application/json',
        })),
      };
    });

    // Read a specific resource
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const uri = request.params.uri;
      const match = uri.match(/^agentdb:\/\/collection\/(.+)$/);

      if (!match) {
        throw new Error(`Invalid resource URI: ${uri}`);
      }

      const collectionName = match[1];
      const stats = await this.agentDB.getCollectionStats(collectionName);

      return {
        contents: [
          {
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(stats, null, 2),
          },
        ],
      };
    });

    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'semantic_search',
            description: 'Semantic vector search using AgentDB HNSW index (150x faster)',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Search query text',
                },
                top_k: {
                  type: 'number',
                  description: 'Number of results to return (default: 20)',
                  default: 20,
                },
                filters: {
                  type: 'object',
                  description: 'Metadata filters (e.g., {"doc_type": "PCI-DSS"})',
                },
              },
              required: ['query'],
            },
          },
          {
            name: 'hybrid_search',
            description: 'Hybrid search combining vector similarity and metadata filtering',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Search query text',
                },
                top_k: {
                  type: 'number',
                  description: 'Number of results',
                  default: 20,
                },
                metadata_filters: {
                  type: 'object',
                  description: 'Required metadata matches',
                },
                confidence_threshold: {
                  type: 'number',
                  description: 'Minimum confidence score (0-1)',
                  default: 0.8,
                },
              },
              required: ['query'],
            },
          },
          {
            name: 'rag_query',
            description: 'Full RAG pipeline with multi-agent orchestration and verification',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'User query',
                },
                session_id: {
                  type: 'string',
                  description: 'Optional session ID for context',
                },
                max_results: {
                  type: 'number',
                  description: 'Maximum sources to retrieve',
                  default: 10,
                },
                accuracy_threshold: {
                  type: 'number',
                  description: 'Minimum accuracy required (default: 0.97)',
                  default: 0.97,
                },
              },
              required: ['query'],
            },
          },
          {
            name: 'ingest_document',
            description: 'Ingest a PDF document with intelligent chunking and indexing',
            inputSchema: {
              type: 'object',
              properties: {
                pdf_path: {
                  type: 'string',
                  description: 'Path to PDF file',
                },
                doc_type: {
                  type: 'string',
                  description: 'Document type hint (e.g., "PCI-DSS", "HIPAA")',
                },
                chunk_size: {
                  type: 'number',
                  description: 'Chunk size in tokens',
                  default: 500,
                },
                chunk_overlap: {
                  type: 'number',
                  description: 'Overlap between chunks',
                  default: 50,
                },
              },
              required: ['pdf_path'],
            },
          },
          {
            name: 'get_document',
            description: 'Retrieve document metadata and chunks by ID',
            inputSchema: {
              type: 'object',
              properties: {
                document_id: {
                  type: 'string',
                  description: 'Document UUID',
                },
                include_chunks: {
                  type: 'boolean',
                  description: 'Include all chunks',
                  default: false,
                },
              },
              required: ['document_id'],
            },
          },
          {
            name: 'learn_from_feedback',
            description: 'Record user feedback for ReasoningBank learning',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Original query',
                },
                response_id: {
                  type: 'string',
                  description: 'Response UUID',
                },
                feedback_score: {
                  type: 'number',
                  description: 'User feedback (0-1)',
                },
                feedback_text: {
                  type: 'string',
                  description: 'Optional feedback text',
                },
              },
              required: ['query', 'response_id', 'feedback_score'],
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        let result: unknown;

        switch (name) {
          case 'semantic_search':
            result = await this.handleSemanticSearch(args);
            break;
          case 'hybrid_search':
            result = await this.handleHybridSearch(args);
            break;
          case 'rag_query':
            result = await this.handleRAGQuery(args);
            break;
          case 'ingest_document':
            result = await this.handleIngestDocument(args);
            break;
          case 'get_document':
            result = await this.handleGetDocument(args);
            break;
          case 'learn_from_feedback':
            result = await this.handleLearnFromFeedback(args);
            break;
          default:
            throw new Error(`Unknown tool: ${name}`);
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      } catch (error) {
        logger.error({ error, tool: name }, 'Tool execution failed');
        throw error;
      }
    });

    // List available prompts
    this.server.setRequestHandler(ListPromptsRequestSchema, async () => {
      return {
        prompts: [
          {
            name: 'rag_query_template',
            description: 'Template for RAG queries with context',
            arguments: [
              {
                name: 'query',
                description: 'User query text',
                required: true,
              },
              {
                name: 'context',
                description: 'Additional context',
                required: false,
              },
            ],
          },
          {
            name: 'document_analysis',
            description: 'Analyze document content and structure',
            arguments: [
              {
                name: 'document_id',
                description: 'Document UUID',
                required: true,
              },
            ],
          },
        ],
      };
    });

    // Get prompt
    this.server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      if (name === 'rag_query_template') {
        const query = args?.query as string;
        const context = args?.context as string | undefined;

        return {
          messages: [
            {
              role: 'user',
              content: {
                type: 'text',
                text: `Answer this query using the semantic search results:\n\nQuery: ${query}\n${context ? `\nContext: ${context}` : ''}`,
              },
            },
          ],
        };
      }

      throw new Error(`Unknown prompt: ${name}`);
    });
  }

  /**
   * Tool handler: Semantic search
   */
  private async handleSemanticSearch(args: Record<string, unknown>) {
    const { query, top_k = 20, filters } = args;

    const results = await this.agentDB.search({
      query: query as string,
      limit: top_k as number,
      useHNSW: true,
      efSearch: 100,
      filter: filters as Record<string, unknown>,
    });

    return {
      results: results.map(r => ({
        id: r.id,
        score: r.score,
        text: r.payload.text,
        metadata: r.payload,
      })),
      count: results.length,
      query: query as string,
    };
  }

  /**
   * Tool handler: Hybrid search
   */
  private async handleHybridSearch(args: Record<string, unknown>) {
    const {
      query,
      top_k = 20,
      metadata_filters,
      confidence_threshold = 0.8,
    } = args;

    const results = await this.agentDB.search({
      query: query as string,
      limit: (top_k as number) * 2, // Retrieve more for filtering
      filter: {
        must: [
          ...(metadata_filters ? Object.entries(metadata_filters as Record<string, unknown>).map(([key, value]) => ({
            field: key,
            match: value,
          })) : []),
          {
            field: 'confidence',
            range: { gte: confidence_threshold as number },
          },
        ],
      },
    });

    // Combine vector score with metadata relevance
    const scored = results.map(r => {
      const vectorScore = r.score;
      const metadataScore = this.calculateMetadataRelevance(r.payload, metadata_filters as Record<string, unknown>);
      const combinedScore = vectorScore * 0.7 + metadataScore * 0.3;

      return {
        id: r.id,
        score: combinedScore,
        vector_score: vectorScore,
        metadata_score: metadataScore,
        text: r.payload.text,
        metadata: r.payload,
      };
    });

    // Sort and limit
    scored.sort((a, b) => b.score - a.score);
    const topResults = scored.slice(0, top_k as number);

    return {
      results: topResults,
      count: topResults.length,
      query: query as string,
    };
  }

  /**
   * Tool handler: Full RAG query
   */
  private async handleRAGQuery(args: Record<string, unknown>) {
    const {
      query,
      session_id,
      max_results = 10,
      accuracy_threshold = 0.97,
    } = args;

    if (!this.swarm) {
      throw new Error('Agentic-flow not enabled');
    }

    logger.info({ query, session_id }, 'Processing RAG query');

    // Step 1: Classify query intent
    const intent = await this.classifyQuery(query as string);

    // Step 2: Retrieve relevant documents
    const retrievalResults = await this.agentDB.search({
      query: query as string,
      limit: max_results as number,
      useHNSW: true,
    });

    // Step 3: Orchestrate reasoning agents
    const reasoningResult = await this.swarm.spawn('reasoning', {
      query: query as string,
      documents: retrievalResults,
      intent,
      use_memory: !!session_id,
    });

    // Step 4: Synthesize response
    const synthesisResult = await this.swarm.spawn('synthesis', {
      query: query as string,
      evidence: retrievalResults,
      reasoning: reasoningResult,
      format: 'structured',
    });

    // Step 5: Verify accuracy
    const verificationResult = await this.swarm.spawn('verification', {
      response: synthesisResult,
      query: query as string,
      threshold: accuracy_threshold as number,
    });

    // Step 6: Record trajectory for learning
    await this.recordTrajectory({
      query: query as string,
      intent,
      response: verificationResult,
    });

    return {
      answer: verificationResult.answer,
      citations: verificationResult.citations,
      accuracy: verificationResult.accuracy,
      confidence: verificationResult.confidence,
      metadata: {
        intent,
        sources_used: retrievalResults.length,
        processing_time: verificationResult.processingTime,
      },
    };
  }

  /**
   * Tool handler: Ingest document
   */
  private async handleIngestDocument(args: Record<string, unknown>) {
    const { pdf_path, doc_type, chunk_size = 500, chunk_overlap = 50 } = args;

    logger.info({ pdf_path, doc_type }, 'Ingesting document');

    // Implementation would use pdf-parse and chunking logic
    // For now, return a mock response
    const documentId = crypto.randomUUID();

    return {
      document_id: documentId,
      chunks_stored: 0,
      status: 'pending',
      message: 'Document ingestion not yet implemented in example',
    };
  }

  /**
   * Tool handler: Get document
   */
  private async handleGetDocument(args: Record<string, unknown>) {
    const { document_id, include_chunks = false } = args;

    const metadata = await this.agentDB.getDocumentMetadata(document_id as string);

    let chunks = undefined;
    if (include_chunks) {
      chunks = await this.agentDB.getDocumentChunks(document_id as string);
    }

    return {
      document_id,
      metadata,
      chunks,
      chunk_count: metadata.chunk_count,
    };
  }

  /**
   * Tool handler: Learn from feedback
   */
  private async handleLearnFromFeedback(args: Record<string, unknown>) {
    const { query, response_id, feedback_score, feedback_text } = args;

    await this.agentDB.recordTrajectory({
      state: { query: query as string },
      action: { response_id: response_id as string },
      reward: feedback_score as number,
      nextState: { user_feedback: feedback_text as string },
    });

    return {
      status: 'recorded',
      message: 'Feedback recorded for learning',
    };
  }

  /**
   * Helper: Calculate metadata relevance score
   */
  private calculateMetadataRelevance(
    payload: Record<string, unknown>,
    filters?: Record<string, unknown>
  ): number {
    if (!filters) return 0.5;

    let matches = 0;
    const total = Object.keys(filters).length;

    for (const [key, value] of Object.entries(filters)) {
      if (payload[key] === value) {
        matches++;
      }
    }

    return matches / total;
  }

  /**
   * Helper: Classify query intent
   */
  private async classifyQuery(query: string) {
    if (this.neuralNet) {
      const features = await this.extractQueryFeatures(query);
      const result = await this.neuralNet.predict(features);
      return {
        type: this.mapClassId(result.classId),
        confidence: result.confidence,
        complexity: result.complexity,
      };
    }

    // Fallback to keyword-based classification
    return {
      type: 'general_question',
      confidence: 0.5,
      complexity: 'moderate',
    };
  }

  /**
   * Helper: Extract query features
   */
  private async extractQueryFeatures(query: string): Promise<number[]> {
    // Simple feature extraction (would be more sophisticated in production)
    const features = new Array(128).fill(0);
    features[0] = query.length / 100;
    features[1] = query.split(' ').length / 10;
    return features;
  }

  /**
   * Helper: Map class ID to query type
   */
  private mapClassId(classId: number): string {
    const types = [
      'requirement_lookup',
      'definition_search',
      'procedure_query',
      'exception_inquiry',
      'general_question',
    ];
    return types[classId] || 'general_question';
  }

  /**
   * Helper: Record trajectory for learning
   */
  private async recordTrajectory(data: {
    query: string;
    intent: { type: string; confidence: number; complexity: string };
    response: { accuracy: number };
  }) {
    await this.agentDB.recordTrajectory({
      state: {
        queryType: data.intent.type,
        complexity: data.intent.complexity,
      },
      action: {
        retrievalStrategy: ['hnsw'],
      },
      reward: data.response.accuracy,
      nextState: {
        success: data.response.accuracy >= 0.97,
      },
    });
  }

  /**
   * Start the MCP server
   */
  async start(): Promise<void> {
    await this.initialize();

    const transport = new StdioServerTransport();
    await this.server.connect(transport);

    logger.info('MCP server started on stdio');
  }

  /**
   * Shutdown the server gracefully
   */
  async shutdown(): Promise<void> {
    logger.info('Shutting down MCP server...');

    if (this.swarm) {
      await this.swarm.destroy();
    }

    await this.agentDB.disconnect();
    await this.server.close();

    logger.info('MCP server shutdown complete');
  }
}

// Main entry point
async function main() {
  const config: MCPServerConfig = {
    agentdb: {
      host: process.env.AGENTDB_HOST || 'localhost',
      port: parseInt(process.env.AGENTDB_PORT || '6333'),
      collection: process.env.AGENTDB_COLLECTION || 'technical_standards',
    },
    agentic_flow: {
      enabled: process.env.AGENTIC_FLOW_ENABLED === 'true',
      default_topology: (process.env.AGENTIC_FLOW_TOPOLOGY as 'mesh' | 'hierarchical' | 'ring' | 'star') || 'mesh',
    },
    neural: {
      enabled: process.env.NEURAL_ENABLED === 'true',
      model_path: process.env.NEURAL_MODEL_PATH,
    },
    server: {
      name: 'doc-rag-mcp',
      version: '1.0.0',
    },
  };

  const server = new DocRAGMCPServer(config);

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    await server.shutdown();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    await server.shutdown();
    process.exit(0);
  });

  await server.start();
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    logger.error({ error }, 'Fatal error');
    process.exit(1);
  });
}

export { DocRAGMCPServer, MCPServerConfig };
