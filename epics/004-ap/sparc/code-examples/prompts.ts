/**
 * MCP Prompt Templates
 *
 * Pre-configured prompt templates for common RAG workflows.
 * These help Claude use the MCP server effectively.
 */

import type { PromptMessage } from '@modelcontextprotocol/sdk/types.js';

export interface PromptTemplate {
  name: string;
  description: string;
  arguments: Array<{
    name: string;
    description: string;
    required: boolean;
  }>;
  getMessages: (args: Record<string, string>) => PromptMessage[];
}

/**
 * RAG query template
 */
export const ragQueryTemplate: PromptTemplate = {
  name: 'rag_query_template',
  description: 'Template for RAG queries with optional context',
  arguments: [
    {
      name: 'query',
      description: 'User query text',
      required: true,
    },
    {
      name: 'context',
      description: 'Additional context or constraints',
      required: false,
    },
    {
      name: 'format',
      description: 'Response format (detailed, concise, bullet_points)',
      required: false,
    },
  ],
  getMessages: (args) => {
    const { query, context, format = 'detailed' } = args;

    const systemPrompt = `You are a RAG assistant with access to technical documentation.

Instructions:
1. Use the semantic_search or hybrid_search tool to find relevant information
2. Analyze the retrieved documents carefully
3. Provide an accurate answer with proper citations
4. Format your response as: ${format}
${context ? `\nAdditional context: ${context}` : ''}

Always cite your sources using [doc_id:page] format.`;

    return [
      {
        role: 'user',
        content: {
          type: 'text',
          text: systemPrompt + `\n\nUser query: ${query}`,
        },
      },
    ];
  },
};

/**
 * Document analysis template
 */
export const documentAnalysisTemplate: PromptTemplate = {
  name: 'document_analysis',
  description: 'Analyze a document\'s content and structure',
  arguments: [
    {
      name: 'document_id',
      description: 'Document UUID to analyze',
      required: true,
    },
    {
      name: 'analysis_type',
      description: 'Type of analysis (summary, structure, requirements, compliance)',
      required: false,
    },
  ],
  getMessages: (args) => {
    const { document_id, analysis_type = 'summary' } = args;

    const analysisPrompts = {
      summary: 'Provide a comprehensive summary of the document',
      structure: 'Analyze the document structure, sections, and organization',
      requirements: 'Extract and list all requirements and obligations',
      compliance: 'Identify compliance requirements and standards mentioned',
    };

    return [
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Use the get_document tool with document_id="${document_id}" and include_chunks=true.

Then ${analysisPrompts[analysis_type as keyof typeof analysisPrompts] || analysisPrompts.summary}.

Provide detailed insights based on the document content.`,
        },
      },
    ];
  },
};

/**
 * Multi-document comparison template
 */
export const multiDocumentComparisonTemplate: PromptTemplate = {
  name: 'multi_document_comparison',
  description: 'Compare information across multiple documents',
  arguments: [
    {
      name: 'query',
      description: 'Topic or requirement to compare',
      required: true,
    },
    {
      name: 'doc_types',
      description: 'Comma-separated list of document types to compare',
      required: false,
    },
  ],
  getMessages: (args) => {
    const { query, doc_types } = args;

    const filterInstruction = doc_types
      ? `Focus on these document types: ${doc_types}`
      : 'Search across all document types';

    return [
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Compare how different documents address: "${query}"

Steps:
1. Use hybrid_search with appropriate metadata filters
2. ${filterInstruction}
3. Group results by document type
4. Identify similarities and differences
5. Highlight any conflicts or contradictions
6. Provide a comparative analysis

Present your findings in a structured format with clear citations.`,
        },
      },
    ];
  },
};

/**
 * Compliance check template
 */
export const complianceCheckTemplate: PromptTemplate = {
  name: 'compliance_check',
  description: 'Check if a practice complies with documented requirements',
  arguments: [
    {
      name: 'practice',
      description: 'Practice or procedure to check',
      required: true,
    },
    {
      name: 'standard',
      description: 'Standard or regulation to check against (e.g., PCI-DSS, HIPAA)',
      required: true,
    },
  ],
  getMessages: (args) => {
    const { practice, standard } = args;

    return [
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Check if this practice complies with ${standard}:

"${practice}"

Steps:
1. Use hybrid_search with filters: {"doc_type": "${standard}"}
2. Find relevant requirements
3. Compare the practice against each requirement
4. Identify compliance gaps or violations
5. Provide recommendations for compliance

Format your response as:
- Compliance Status: [Compliant / Non-Compliant / Partial]
- Relevant Requirements: [List with citations]
- Gaps Identified: [List any gaps]
- Recommendations: [Steps to achieve compliance]`,
        },
      },
    ];
  },
};

/**
 * Learning session review template
 */
export const learningSessionReviewTemplate: PromptTemplate = {
  name: 'learning_session_review',
  description: 'Review patterns learned during a session',
  arguments: [
    {
      name: 'session_id',
      description: 'Session UUID to review',
      required: true,
    },
  ],
  getMessages: (args) => {
    const { session_id } = args;

    return [
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Review the learning session: ${session_id}

Access the session resource at: agentdb://session/${session_id}

Analyze:
1. Interaction patterns
2. Learned cross-references
3. Query types and strategies used
4. Performance improvements
5. Recommendations for future queries

Provide insights into what the system learned during this session.`,
        },
      },
    ];
  },
};

/**
 * Citation verification template
 */
export const citationVerificationTemplate: PromptTemplate = {
  name: 'citation_verification',
  description: 'Verify that citations support a given statement',
  arguments: [
    {
      name: 'statement',
      description: 'Statement to verify',
      required: true,
    },
    {
      name: 'citations',
      description: 'Citations to check (JSON array)',
      required: true,
    },
  ],
  getMessages: (args) => {
    const { statement, citations } = args;

    return [
      {
        role: 'user',
        content: {
          type: 'text',
          text: `Verify if these citations properly support the statement:

Statement: "${statement}"

Citations: ${citations}

For each citation:
1. Retrieve the actual content using get_document tool
2. Check if the content supports the statement
3. Rate the relevance (0-1)
4. Identify any contradictions

Provide verification results:
- Overall Verification Score: [0-1]
- Per-Citation Analysis: [List]
- Recommendation: [Accept / Reject / Partial]`,
        },
      },
    ];
  },
};

/**
 * Export all templates
 */
export const promptTemplates: PromptTemplate[] = [
  ragQueryTemplate,
  documentAnalysisTemplate,
  multiDocumentComparisonTemplate,
  complianceCheckTemplate,
  learningSessionReviewTemplate,
  citationVerificationTemplate,
];

/**
 * Get a specific prompt template
 */
export function getPromptTemplate(name: string): PromptTemplate | undefined {
  return promptTemplates.find(t => t.name === name);
}

/**
 * Generate prompt messages from template
 */
export function generatePrompt(
  templateName: string,
  args: Record<string, string>
): PromptMessage[] {
  const template = getPromptTemplate(templateName);
  if (!template) {
    throw new Error(`Unknown prompt template: ${templateName}`);
  }

  // Validate required arguments
  for (const arg of template.arguments) {
    if (arg.required && !args[arg.name]) {
      throw new Error(`Missing required argument: ${arg.name}`);
    }
  }

  return template.getMessages(args);
}
