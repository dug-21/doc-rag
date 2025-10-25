/**
 * MCP Resource Handlers
 *
 * Provides access to AgentDB collections, documents, and metadata
 * through MCP resource protocol.
 */

import { AgentDB } from 'agentdb';
import type { Resource } from '@modelcontextprotocol/sdk/types.js';

export interface ResourceManager {
  listResources(): Promise<Resource[]>;
  readResource(uri: string): Promise<{ contents: { uri: string; mimeType: string; text: string }[] }>;
}

/**
 * AgentDB resource manager implementation
 */
export class AgentDBResourceManager implements ResourceManager {
  constructor(private agentDB: AgentDB) {}

  /**
   * List all available resources
   */
  async listResources(): Promise<Resource[]> {
    const resources: Resource[] = [];

    // List collections
    const collections = await this.agentDB.listCollections();
    for (const collection of collections) {
      resources.push({
        uri: `agentdb://collection/${collection.name}`,
        name: `Collection: ${collection.name}`,
        description: `Vector collection with ${collection.count} documents`,
        mimeType: 'application/json',
      });

      // Add collection stats as sub-resource
      resources.push({
        uri: `agentdb://collection/${collection.name}/stats`,
        name: `${collection.name} Statistics`,
        description: 'Collection performance and usage statistics',
        mimeType: 'application/json',
      });
    }

    // List recent documents
    const recentDocs = await this.agentDB.listRecentDocuments(10);
    for (const doc of recentDocs) {
      resources.push({
        uri: `agentdb://document/${doc.id}`,
        name: `Document: ${doc.title || doc.id}`,
        description: `Document with ${doc.chunkCount} chunks`,
        mimeType: 'application/json',
      });
    }

    // List learning sessions
    const sessions = await this.agentDB.listSessions({ limit: 5 });
    for (const session of sessions) {
      resources.push({
        uri: `agentdb://session/${session.id}`,
        name: `Session: ${session.id}`,
        description: `Learning session with ${session.interactionCount} interactions`,
        mimeType: 'application/json',
      });
    }

    // Add patterns as resources
    resources.push({
      uri: 'agentdb://patterns',
      name: 'Learned Patterns',
      description: 'Cross-reference and interaction patterns learned by the system',
      mimeType: 'application/json',
    });

    return resources;
  }

  /**
   * Read a specific resource
   */
  async readResource(uri: string): Promise<{
    contents: { uri: string; mimeType: string; text: string }[];
  }> {
    // Parse URI
    const parsed = this.parseURI(uri);

    switch (parsed.type) {
      case 'collection':
        return this.readCollection(uri, parsed.id, parsed.subResource);

      case 'document':
        return this.readDocument(uri, parsed.id);

      case 'session':
        return this.readSession(uri, parsed.id);

      case 'patterns':
        return this.readPatterns(uri);

      default:
        throw new Error(`Unknown resource type: ${parsed.type}`);
    }
  }

  /**
   * Read collection resource
   */
  private async readCollection(
    uri: string,
    collectionName: string,
    subResource?: string
  ) {
    if (subResource === 'stats') {
      const stats = await this.agentDB.getCollectionStats(collectionName);
      return {
        contents: [
          {
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(
              {
                collection: collectionName,
                statistics: stats,
                timestamp: new Date().toISOString(),
              },
              null,
              2
            ),
          },
        ],
      };
    }

    // Return collection info
    const info = await this.agentDB.getCollectionInfo(collectionName);
    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(
            {
              collection: collectionName,
              info,
              timestamp: new Date().toISOString(),
            },
            null,
            2
          ),
        },
      ],
    };
  }

  /**
   * Read document resource
   */
  private async readDocument(uri: string, documentId: string) {
    const metadata = await this.agentDB.getDocumentMetadata(documentId);
    const chunks = await this.agentDB.getDocumentChunks(documentId);

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(
            {
              document_id: documentId,
              metadata,
              chunks: chunks.map(chunk => ({
                id: chunk.id,
                index: chunk.index,
                text: chunk.text,
                section: chunk.metadata.section,
                page: chunk.metadata.page,
              })),
              chunk_count: chunks.length,
            },
            null,
            2
          ),
        },
      ],
    };
  }

  /**
   * Read session resource
   */
  private async readSession(uri: string, sessionId: string) {
    const session = await this.agentDB.getSession(sessionId);
    const patterns = await this.agentDB.getSessionPatterns(sessionId);

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(
            {
              session_id: sessionId,
              session,
              patterns,
              timestamp: new Date().toISOString(),
            },
            null,
            2
          ),
        },
      ],
    };
  }

  /**
   * Read patterns resource
   */
  private async readPatterns(uri: string) {
    const patterns = await this.agentDB.listPatterns({
      limit: 50,
      minConfidence: 0.7,
    });

    return {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(
            {
              patterns: patterns.map(p => ({
                id: p.id,
                type: p.type,
                confidence: p.confidence,
                frequency: p.frequency,
                source_id: p.sourceId,
                target_id: p.targetId,
              })),
              count: patterns.length,
              timestamp: new Date().toISOString(),
            },
            null,
            2
          ),
        },
      ],
    };
  }

  /**
   * Parse resource URI
   */
  private parseURI(uri: string): {
    type: string;
    id: string;
    subResource?: string;
  } {
    const match = uri.match(/^agentdb:\/\/([^/]+)\/([^/]+)(?:\/(.+))?$/);
    if (!match) {
      throw new Error(`Invalid AgentDB URI: ${uri}`);
    }

    return {
      type: match[1],
      id: match[2],
      subResource: match[3],
    };
  }
}

/**
 * Example: Custom resource for system metrics
 */
export class SystemMetricsResource implements ResourceManager {
  private metrics: Map<string, number> = new Map();

  constructor() {
    // Initialize metrics
    this.updateMetrics();

    // Update metrics every minute
    setInterval(() => this.updateMetrics(), 60000);
  }

  private updateMetrics() {
    this.metrics.set('queries_total', Math.random() * 10000);
    this.metrics.set('avg_latency_ms', Math.random() * 500);
    this.metrics.set('accuracy_score', 0.95 + Math.random() * 0.05);
    this.metrics.set('cache_hit_rate', 0.6 + Math.random() * 0.2);
  }

  async listResources(): Promise<Resource[]> {
    return [
      {
        uri: 'metrics://system/performance',
        name: 'System Performance Metrics',
        description: 'Real-time performance metrics',
        mimeType: 'application/json',
      },
      {
        uri: 'metrics://system/health',
        name: 'System Health',
        description: 'System health status',
        mimeType: 'application/json',
      },
    ];
  }

  async readResource(uri: string) {
    if (uri === 'metrics://system/performance') {
      return {
        contents: [
          {
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(
              {
                metrics: Object.fromEntries(this.metrics),
                timestamp: new Date().toISOString(),
              },
              null,
              2
            ),
          },
        ],
      };
    }

    if (uri === 'metrics://system/health') {
      return {
        contents: [
          {
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(
              {
                status: 'healthy',
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                timestamp: new Date().toISOString(),
              },
              null,
              2
            ),
          },
        ],
      };
    }

    throw new Error(`Unknown metrics URI: ${uri}`);
  }
}
