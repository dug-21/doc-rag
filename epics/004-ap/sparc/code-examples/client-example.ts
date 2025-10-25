/**
 * MCP Client Usage Examples
 *
 * Demonstrates how Claude would interact with the RAG MCP server
 * using the Model Context Protocol.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

/**
 * Example 1: Simple semantic search
 */
async function exampleSemanticSearch() {
  console.log('\n=== Example 1: Semantic Search ===\n');

  const client = new Client(
    {
      name: 'rag-client',
      version: '1.0.0',
    },
    {
      capabilities: {},
    }
  );

  // Connect to MCP server
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // Call semantic_search tool
  const result = await client.callTool({
    name: 'semantic_search',
    arguments: {
      query: 'What are the PCI-DSS encryption requirements?',
      top_k: 10,
    },
  });

  console.log('Search Results:');
  console.log(JSON.stringify(result, null, 2));

  await client.close();
}

/**
 * Example 2: Hybrid search with filters
 */
async function exampleHybridSearch() {
  console.log('\n=== Example 2: Hybrid Search ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // Search with metadata filters
  const result = await client.callTool({
    name: 'hybrid_search',
    arguments: {
      query: 'encryption standards',
      top_k: 15,
      metadata_filters: {
        doc_type: 'PCI-DSS',
        chunk_type: 'requirement',
      },
      confidence_threshold: 0.85,
    },
  });

  console.log('Hybrid Search Results:');
  console.log(JSON.stringify(result, null, 2));

  await client.close();
}

/**
 * Example 3: Full RAG query with context
 */
async function exampleRAGQuery() {
  console.log('\n=== Example 3: RAG Query ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // Execute full RAG pipeline
  const result = await client.callTool({
    name: 'rag_query',
    arguments: {
      query: 'What encryption algorithms are required for cardholder data?',
      session_id: 'session-123',
      max_results: 20,
      accuracy_threshold: 0.97,
    },
  });

  console.log('RAG Query Response:');
  const response = JSON.parse(result.content[0].text);
  console.log('Answer:', response.answer);
  console.log('\nCitations:', response.citations);
  console.log('\nMetadata:', response.metadata);

  await client.close();
}

/**
 * Example 4: Document ingestion
 */
async function exampleDocumentIngestion() {
  console.log('\n=== Example 4: Document Ingestion ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // Ingest a PDF document
  const result = await client.callTool({
    name: 'ingest_document',
    arguments: {
      pdf_path: '/path/to/pci-dss-v4.pdf',
      doc_type: 'PCI-DSS',
      chunk_size: 500,
      chunk_overlap: 50,
    },
  });

  console.log('Ingestion Result:');
  console.log(JSON.stringify(result, null, 2));

  await client.close();
}

/**
 * Example 5: Access resources
 */
async function exampleAccessResources() {
  console.log('\n=== Example 5: Access Resources ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // List available resources
  const resourcesList = await client.listResources();
  console.log('Available Resources:');
  console.log(JSON.stringify(resourcesList.resources, null, 2));

  // Read a specific resource
  if (resourcesList.resources.length > 0) {
    const resource = resourcesList.resources[0];
    const content = await client.readResource({ uri: resource.uri });
    console.log(`\nContent of ${resource.name}:`);
    console.log(content.contents[0].text);
  }

  await client.close();
}

/**
 * Example 6: Use prompt templates
 */
async function examplePromptTemplates() {
  console.log('\n=== Example 6: Prompt Templates ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // List available prompts
  const promptsList = await client.listPrompts();
  console.log('Available Prompts:');
  console.log(JSON.stringify(promptsList.prompts, null, 2));

  // Get a specific prompt
  const prompt = await client.getPrompt({
    name: 'rag_query_template',
    arguments: {
      query: 'What are the password requirements?',
      context: 'For a financial application',
      format: 'bullet_points',
    },
  });

  console.log('\nGenerated Prompt:');
  console.log(JSON.stringify(prompt, null, 2));

  await client.close();
}

/**
 * Example 7: Provide learning feedback
 */
async function exampleLearningFeedback() {
  console.log('\n=== Example 7: Learning Feedback ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // Provide feedback on a query response
  const result = await client.callTool({
    name: 'learn_from_feedback',
    arguments: {
      query: 'What are encryption requirements?',
      response_id: 'response-456',
      feedback_score: 0.95,
      feedback_text: 'Very accurate and helpful response',
    },
  });

  console.log('Feedback Result:');
  console.log(JSON.stringify(result, null, 2));

  await client.close();
}

/**
 * Example 8: Complete workflow - Query with verification
 */
async function exampleCompleteWorkflow() {
  console.log('\n=== Example 8: Complete Workflow ===\n');

  const client = new Client(
    { name: 'rag-client', version: '1.0.0' },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/mcp-server.js'],
  });

  await client.connect(transport);

  // Step 1: Execute RAG query
  console.log('Step 1: Executing RAG query...');
  const queryResult = await client.callTool({
    name: 'rag_query',
    arguments: {
      query: 'What are the requirements for secure password storage?',
      max_results: 15,
    },
  });

  const response = JSON.parse(queryResult.content[0].text);
  console.log('Answer:', response.answer);
  console.log('Accuracy:', response.accuracy);

  // Step 2: Verify citations (if needed)
  if (response.accuracy < 0.98) {
    console.log('\nStep 2: Verifying citations...');
    // Use get_document to verify each citation
    for (const citation of response.citations.slice(0, 3)) {
      const docResult = await client.callTool({
        name: 'get_document',
        arguments: {
          document_id: citation.doc_id,
          include_chunks: false,
        },
      });
      console.log(`Verified citation from: ${JSON.parse(docResult.content[0].text).metadata.title}`);
    }
  }

  // Step 3: Provide feedback
  console.log('\nStep 3: Providing feedback...');
  await client.callTool({
    name: 'learn_from_feedback',
    arguments: {
      query: 'What are the requirements for secure password storage?',
      response_id: response.metadata.response_id || 'resp-' + Date.now(),
      feedback_score: 0.95,
      feedback_text: 'Complete and well-cited answer',
    },
  });

  console.log('Workflow complete!');

  await client.close();
}

/**
 * Run all examples
 */
async function runAllExamples() {
  try {
    await exampleSemanticSearch();
    await exampleHybridSearch();
    await exampleRAGQuery();
    // await exampleDocumentIngestion(); // Requires actual PDF
    await exampleAccessResources();
    await examplePromptTemplates();
    await exampleLearningFeedback();
    await exampleCompleteWorkflow();
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Export for use in other modules
export {
  exampleSemanticSearch,
  exampleHybridSearch,
  exampleRAGQuery,
  exampleDocumentIngestion,
  exampleAccessResources,
  examplePromptTemplates,
  exampleLearningFeedback,
  exampleCompleteWorkflow,
};

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples().catch(console.error);
}
