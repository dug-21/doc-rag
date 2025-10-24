# SPARC Phase 4: Refinement - Test-Driven Development Strategy

**Project:** PCI-DSS Compliance RAG System
**Version:** 1.0
**Date:** October 24, 2025
**Phase:** Refinement & Quality Assurance

---

## Executive Summary

### Refinement Mission
Achieve >97% accuracy through rigorous Test-Driven Development (TDD), iterative testing, and quality assurance processes.

### Key Metrics
- **Coverage Target:** >90% code coverage
- **Accuracy Target:** >97% after RL training (>85% baseline)
- **Performance Target:** <500ms P95 latency
- **Test Suite Size:** 980-1,490 questions
- **Sprint Cycle:** 2-week iterations

---

## 1. TDD Strategy (Red-Green-Refactor)

### 1.1 Core TDD Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  RED: Write Failing Test                                    │
│  • Define expected behavior                                 │
│  • Write test that captures requirements                    │
│  • Run test and verify it fails                            │
│  • Document failure reason                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  GREEN: Make Test Pass                                      │
│  • Implement minimum code to pass test                      │
│  • Focus on correctness, not optimization                   │
│  • Run test and verify it passes                           │
│  • Commit working code                                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  REFACTOR: Improve Code Quality                             │
│  • Optimize performance                                     │
│  • Enhance readability                                      │
│  • Remove duplication                                       │
│  • Ensure tests still pass                                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Test-First Principles

**Golden Rules:**
1. ✅ **Write test before implementation**
2. ✅ **One test at a time** - focus on single behavior
3. ✅ **Test fails for right reason** - verify test logic
4. ✅ **Minimum code to pass** - avoid over-engineering
5. ✅ **Refactor with green tests** - maintain test suite

### 1.3 Coverage Targets

| Component | Unit Coverage | Integration Coverage | Total Target |
|-----------|--------------|---------------------|--------------|
| Ingestion | >95% | >85% | >90% |
| Query Processing | >95% | >90% | >92% |
| AgentDB Integration | >90% | >85% | >88% |
| Response Generation | >90% | >85% | >88% |
| RL Training | >85% | >80% | >82% |
| **Overall System** | **>92%** | **>85%** | **>90%** |

### 1.4 Testing Pyramid

```
         ┌──────────────┐
         │    E2E       │  10% - User journeys
         │   (Slow)     │       Full system tests
         ├──────────────┤
         │              │
         │ Integration  │  20% - Component integration
         │   (Medium)   │       API tests, DB tests
         │              │
         ├──────────────┤
         │              │
         │              │
         │   Unit       │  70% - Function/class tests
         │   (Fast)     │       Logic validation
         │              │
         │              │
         └──────────────┘
```

**Distribution Rationale:**
- **70% Unit:** Fast feedback, isolated logic
- **20% Integration:** Component interaction validation
- **10% E2E:** Critical user path verification

---

## 2. Unit Testing Strategy

### 2.1 Testing Framework: Vitest

**Why Vitest:**
- ⚡ Fast execution with native ESM support
- 🔄 Hot module replacement
- 📊 Built-in coverage with c8
- 🎯 Jest-compatible API
- 🦀 Compatible with Rust via WASM

**Configuration:**
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'c8',
      reporter: ['text', 'json', 'html', 'lcov'],
      lines: 90,
      functions: 90,
      branches: 85,
      statements: 90,
      exclude: [
        'node_modules/',
        'dist/',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/mocks/**'
      ]
    },
    testTimeout: 10000,
    hookTimeout: 10000,
    globals: true,
    environment: 'node',
    setupFiles: ['./tests/setup.ts']
  }
});
```

### 2.2 Unit Test Patterns by Component

#### A. Document Ingestion Tests

```typescript
// tests/unit/ingestion/chunker.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { SemanticChunker } from '@/ingestion/chunker';

describe('SemanticChunker', () => {
  let chunker: SemanticChunker;

  beforeEach(() => {
    chunker = new SemanticChunker({
      maxChunkSize: 512,
      overlapSize: 50
    });
  });

  it('should split document into semantic chunks', () => {
    const document = 'PCI-DSS Requirement 1.1.1 states...';
    const chunks = chunker.chunk(document);

    expect(chunks).toHaveLength(3);
    expect(chunks[0].text).toContain('PCI-DSS');
    expect(chunks[0].metadata.section).toBe('1.1.1');
  });

  it('should maintain context with overlap', () => {
    const document = 'Long document...';
    const chunks = chunker.chunk(document);

    const overlap = findOverlap(chunks[0], chunks[1]);
    expect(overlap.length).toBeGreaterThanOrEqual(50);
  });

  it('should handle edge case: empty document', () => {
    expect(() => chunker.chunk('')).toThrow('Empty document');
  });

  it('should extract metadata from PCI-DSS sections', () => {
    const text = 'Requirement 3.2.1: Store cardholder data...';
    const chunks = chunker.chunk(text);

    expect(chunks[0].metadata).toMatchObject({
      requirement: '3.2.1',
      category: 'data-protection',
      version: 'v4.0'
    });
  });
});
```

#### B. Query Processing Tests

```typescript
// tests/unit/query/processor.test.ts
import { describe, it, expect, vi } from 'vitest';
import { QueryProcessor } from '@/query/processor';

describe('QueryProcessor', () => {
  it('should classify query intent', async () => {
    const processor = new QueryProcessor();
    const query = 'What are the requirements for storing CVV?';

    const result = await processor.analyze(query);

    expect(result.intent).toBe('requirement-lookup');
    expect(result.entities).toContain('CVV');
    expect(result.confidence).toBeGreaterThan(0.8);
  });

  it('should expand query with synonyms', async () => {
    const processor = new QueryProcessor();
    const query = 'CVV storage rules';

    const expanded = await processor.expand(query);

    expect(expanded.terms).toContain('CVV');
    expect(expanded.terms).toContain('CVV2');
    expect(expanded.terms).toContain('card verification value');
  });

  it('should route to appropriate retrieval strategy', async () => {
    const processor = new QueryProcessor();

    const queries = [
      { text: 'What is requirement 1.1?', expected: 'direct-lookup' },
      { text: 'Explain encryption', expected: 'semantic-search' },
      { text: 'Compare requirements', expected: 'multi-hop' }
    ];

    for (const { text, expected } of queries) {
      const result = await processor.analyze(text);
      expect(result.strategy).toBe(expected);
    }
  });
});
```

#### C. AgentDB Integration Tests

```typescript
// tests/unit/storage/agentdb.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AgentDBStore } from '@/storage/agentdb';

describe('AgentDBStore', () => {
  let store: AgentDBStore;

  beforeEach(async () => {
    store = new AgentDBStore({
      path: ':memory:',
      dimension: 1536
    });
    await store.initialize();
  });

  afterEach(async () => {
    await store.close();
  });

  it('should insert and retrieve vectors', async () => {
    const vector = Array(1536).fill(0.1);
    const metadata = { section: '1.1.1', text: 'Test chunk' };

    const id = await store.insert(vector, metadata);
    const retrieved = await store.get(id);

    expect(retrieved.metadata.section).toBe('1.1.1');
    expect(retrieved.vector).toHaveLength(1536);
  });

  it('should perform k-NN search', async () => {
    // Insert test vectors
    await store.bulkInsert([
      { vector: Array(1536).fill(0.1), metadata: { id: 1 } },
      { vector: Array(1536).fill(0.5), metadata: { id: 2 } },
      { vector: Array(1536).fill(0.9), metadata: { id: 3 } }
    ]);

    const query = Array(1536).fill(0.15);
    const results = await store.search(query, { k: 2 });

    expect(results).toHaveLength(2);
    expect(results[0].metadata.id).toBe(1); // Closest match
    expect(results[0].distance).toBeLessThan(results[1].distance);
  });

  it('should filter by metadata', async () => {
    await store.bulkInsert([
      { vector: Array(1536).fill(0.1), metadata: { category: 'network' } },
      { vector: Array(1536).fill(0.2), metadata: { category: 'data' } }
    ]);

    const results = await store.search(
      Array(1536).fill(0.15),
      { k: 10, filter: { category: 'network' } }
    );

    expect(results.every(r => r.metadata.category === 'network')).toBe(true);
  });
});
```

#### D. Response Generation Tests

```typescript
// tests/unit/generation/response.test.ts
import { describe, it, expect, vi } from 'vitest';
import { ResponseGenerator } from '@/generation/response';

describe('ResponseGenerator', () => {
  it('should generate response with citations', async () => {
    const generator = new ResponseGenerator();
    const context = [
      { text: 'PCI-DSS 3.2.1 states...', section: '3.2.1' },
      { text: 'Requirement 4.1 requires...', section: '4.1' }
    ];

    const response = await generator.generate(
      'What are encryption requirements?',
      context
    );

    expect(response.answer).toContain('encryption');
    expect(response.citations).toHaveLength(2);
    expect(response.citations[0].section).toBe('3.2.1');
  });

  it('should handle insufficient context', async () => {
    const generator = new ResponseGenerator();
    const context = [];

    const response = await generator.generate('Random question?', context);

    expect(response.confidence).toBeLessThan(0.5);
    expect(response.answer).toContain('insufficient information');
  });

  it('should cache responses for identical queries', async () => {
    const generator = new ResponseGenerator({ cacheEnabled: true });
    const spy = vi.spyOn(generator, 'generateInternal');

    const query = 'What is PCI-DSS?';
    const context = [{ text: 'PCI-DSS is...', section: '1.0' }];

    await generator.generate(query, context);
    await generator.generate(query, context); // Should hit cache

    expect(spy).toHaveBeenCalledTimes(1); // Only called once
  });
});
```

### 2.3 Mocking Strategy

#### Mock AgentDB for Fast Tests

```typescript
// tests/mocks/agentdb.mock.ts
import { vi } from 'vitest';

export class MockAgentDB {
  private store = new Map<string, { vector: number[], metadata: any }>();

  async insert(vector: number[], metadata: any): Promise<string> {
    const id = `mock-${Date.now()}-${Math.random()}`;
    this.store.set(id, { vector, metadata });
    return id;
  }

  async search(query: number[], options: any): Promise<any[]> {
    // Simple mock: return all items sorted by random distance
    return Array.from(this.store.values())
      .map(item => ({
        ...item,
        distance: Math.random()
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, options.k || 5);
  }

  async close(): Promise<void> {
    this.store.clear();
  }
}

// Usage in tests
vi.mock('@/storage/agentdb', () => ({
  AgentDBStore: MockAgentDB
}));
```

#### Mock Agentic-Flow Swarm

```typescript
// tests/mocks/swarm.mock.ts
import { vi } from 'vitest';

export class MockSwarm {
  async initialize(): Promise<void> {
    return Promise.resolve();
  }

  async dispatch(task: string): Promise<any> {
    // Simulate agent processing with deterministic results
    return {
      result: `Processed: ${task}`,
      agent: 'mock-agent',
      duration: 100
    };
  }

  async coordinate(agents: string[]): Promise<void> {
    return Promise.resolve();
  }
}
```

### 2.4 Coverage Tools: c8

**Configuration:**
```json
{
  "c8": {
    "reporter": ["text", "lcov", "html"],
    "exclude": [
      "coverage/**",
      "tests/**",
      "**/*.test.ts",
      "**/*.spec.ts",
      "**/node_modules/**"
    ],
    "all": true,
    "check-coverage": true,
    "lines": 90,
    "functions": 90,
    "branches": 85,
    "statements": 90
  }
}
```

**Run Coverage:**
```bash
# Generate coverage report
npm run test:coverage

# View HTML report
open coverage/index.html

# Check coverage thresholds
npm run test:coverage:check
```

---

## 3. Integration Testing

### 3.1 Component Integration Tests

#### End-to-End Query Flow Test

```typescript
// tests/integration/query-flow.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { RAGSystem } from '@/system';
import { setupTestDB, teardownTestDB } from './helpers';

describe('Query Flow Integration', () => {
  let system: RAGSystem;

  beforeAll(async () => {
    await setupTestDB();
    system = new RAGSystem({ environment: 'test' });
    await system.initialize();
  });

  afterAll(async () => {
    await system.shutdown();
    await teardownTestDB();
  });

  it('should handle complete query lifecycle', async () => {
    // Ingest test document
    await system.ingest({
      content: 'PCI-DSS Requirement 1.1.1: Establish firewall rules...',
      metadata: { source: 'test-doc', version: 'v4.0' }
    });

    // Query the system
    const response = await system.query('What are firewall requirements?');

    // Validate response
    expect(response.answer).toContain('firewall');
    expect(response.citations).toHaveLength.greaterThan(0);
    expect(response.citations[0].section).toBe('1.1.1');
    expect(response.confidence).toBeGreaterThan(0.8);
    expect(response.latency).toBeLessThan(500);
  });

  it('should learn from feedback via RL', async () => {
    // Initial query
    const query = 'CVV storage requirements';
    const response1 = await system.query(query);
    const initialAccuracy = response1.confidence;

    // Provide positive feedback
    await system.feedback(response1.id, {
      rating: 5,
      correctAnswer: true
    });

    // Re-query after learning
    const response2 = await system.query(query);

    expect(response2.confidence).toBeGreaterThanOrEqual(initialAccuracy);
  });
});
```

### 3.2 Third-Party Integration Tests

#### AgentDB Integration

```typescript
// tests/integration/agentdb.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { AgentDB } from 'agentdb';
import { VectorStore } from '@/storage';

describe('AgentDB Integration', () => {
  it('should store and retrieve embeddings', async () => {
    const db = new AgentDB({ path: ':memory:' });
    const store = new VectorStore(db);

    // Store embeddings
    const id = await store.store({
      vector: Array(1536).fill(0.5),
      metadata: { text: 'Test', section: '1.1' }
    });

    // Retrieve
    const result = await store.retrieve(id);
    expect(result.metadata.section).toBe('1.1');
  });

  it('should perform HNSW search efficiently', async () => {
    const db = new AgentDB({
      path: ':memory:',
      index_type: 'hnsw'
    });

    // Insert 10k vectors
    const vectors = Array(10000).fill(0).map(() =>
      Array(1536).fill(0).map(() => Math.random())
    );
    await db.bulkInsert(vectors);

    // Search should complete in <50ms
    const start = Date.now();
    await db.search(Array(1536).fill(0.5), { k: 10 });
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(50);
  });
});
```

#### Agentic-Flow Swarm Integration

```typescript
// tests/integration/swarm.test.ts
import { describe, it, expect } from 'vitest';
import { Swarm } from 'agentic-flow';
import { SwarmCoordinator } from '@/orchestration';

describe('Agentic-Flow Swarm Integration', () => {
  it('should coordinate retrieval with multiple agents', async () => {
    const swarm = new Swarm({
      topology: 'hierarchical',
      agents: ['retriever', 'ranker', 'generator']
    });

    const coordinator = new SwarmCoordinator(swarm);
    await coordinator.initialize();

    const result = await coordinator.process({
      query: 'Encryption requirements',
      strategy: 'multi-agent'
    });

    expect(result.agents).toHaveLength(3);
    expect(result.consensus).toBeGreaterThan(0.8);
  });
});
```

### 3.3 Database Integration Tests

```typescript
// tests/integration/persistence.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { SessionStore, MemoryStore } from '@/storage';

describe('Persistence Integration', () => {
  let sessionStore: SessionStore;
  let memoryStore: MemoryStore;

  beforeEach(async () => {
    sessionStore = new SessionStore({ db: 'test.db' });
    memoryStore = new MemoryStore({ db: 'test.db' });
    await sessionStore.clear();
    await memoryStore.clear();
  });

  it('should persist query history across sessions', async () => {
    // Session 1: Execute query
    await sessionStore.saveQuery({
      sessionId: 'test-session',
      query: 'What is PCI-DSS?',
      response: 'PCI-DSS is...',
      timestamp: Date.now()
    });

    // Session 2: Retrieve history
    const history = await sessionStore.getHistory('test-session');

    expect(history).toHaveLength(1);
    expect(history[0].query).toBe('What is PCI-DSS?');
  });

  it('should store RL learning trajectories', async () => {
    // Store trajectory
    await memoryStore.storeTrajectory({
      query: 'Test query',
      attempts: [
        { action: 'retrieve', reward: 0.5 },
        { action: 'rerank', reward: 0.8 }
      ]
    });

    // Retrieve for RL training
    const trajectories = await memoryStore.getTrajectories({ limit: 10 });

    expect(trajectories).toHaveLength(1);
    expect(trajectories[0].attempts[1].reward).toBe(0.8);
  });
});
```

---

## 4. End-to-End Testing

### 4.1 Framework: Playwright

**Why Playwright:**
- 🌐 Real browser testing
- 🚀 Fast, reliable execution
- 📸 Screenshot & video capture
- 🔄 Auto-wait for elements
- 📊 Built-in tracing

**Configuration:**
```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60000,
  fullyParallel: true,
  retries: 2,
  workers: 4,
  reporter: [['html'], ['json', { outputFile: 'test-results.json' }]],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
    { name: 'firefox', use: { browserName: 'firefox' } },
    { name: 'webkit', use: { browserName: 'webkit' } }
  ]
});
```

### 4.2 User Journey Tests

```typescript
// tests/e2e/user-journeys.test.ts
import { test, expect } from '@playwright/test';

test.describe('PCI-DSS Query User Journey', () => {
  test('should complete successful query flow', async ({ page }) => {
    // Navigate to application
    await page.goto('/');

    // Enter query
    await page.fill('[data-testid="query-input"]',
      'What are the requirements for storing credit card data?'
    );
    await page.click('[data-testid="submit-query"]');

    // Wait for response
    await page.waitForSelector('[data-testid="response"]', {
      timeout: 5000
    });

    // Validate response
    const response = await page.textContent('[data-testid="response"]');
    expect(response).toContain('Requirement 3');

    // Check citations
    const citations = await page.locator('[data-testid="citation"]');
    await expect(citations).toHaveCount.greaterThan(0);

    // Validate latency indicator
    const latency = await page.textContent('[data-testid="latency"]');
    expect(parseInt(latency!)).toBeLessThan(500);
  });

  test('should handle learning from feedback', async ({ page }) => {
    await page.goto('/');

    // Execute query
    await page.fill('[data-testid="query-input"]', 'CVV storage rules');
    await page.click('[data-testid="submit-query"]');
    await page.waitForSelector('[data-testid="response"]');

    // Provide positive feedback
    await page.click('[data-testid="feedback-helpful"]');

    // Verify feedback recorded
    await expect(page.locator('[data-testid="feedback-success"]'))
      .toBeVisible();
  });
});
```

### 4.3 Performance Under Load

```typescript
// tests/e2e/load-test.ts
import { test, expect } from '@playwright/test';

test.describe('Load Testing', () => {
  test('should handle 50 concurrent queries', async ({ browser }) => {
    const queries = Array(50).fill(0).map((_, i) =>
      `Test query ${i}: What is PCI-DSS requirement ${i % 12 + 1}?`
    );

    const startTime = Date.now();

    const results = await Promise.all(
      queries.map(async (query) => {
        const page = await browser.newPage();
        await page.goto('/');
        await page.fill('[data-testid="query-input"]', query);
        await page.click('[data-testid="submit-query"]');
        await page.waitForSelector('[data-testid="response"]', {
          timeout: 10000
        });
        const latency = await page.textContent('[data-testid="latency"]');
        await page.close();
        return parseInt(latency!);
      })
    );

    const totalTime = Date.now() - startTime;
    const avgLatency = results.reduce((a, b) => a + b, 0) / results.length;
    const p95Latency = results.sort()[Math.floor(results.length * 0.95)];

    expect(avgLatency).toBeLessThan(500);
    expect(p95Latency).toBeLessThan(1000);
    expect(totalTime).toBeLessThan(30000); // Complete in 30s
  });
});
```

---

## 5. Accuracy Testing

### 5.1 Test Dataset Composition

| Quality Tier | Quantity | Purpose | Accuracy Target |
|--------------|----------|---------|----------------|
| 💎 Platinum (Expert) | 80-150 | Final validation | >99% |
| 🟢 Gold (Official) | 380-570 | Core accuracy | >97% |
| 🟡 Silver (Curated) | 370-570 | Training validation | >90% |
| 🔵 Bronze (Synthetic) | 150-200 | RL training | >85% |
| **Total** | **980-1,490** | **Complete test suite** | **>97%** |

### 5.2 Baseline Accuracy Testing (>85%)

```typescript
// tests/accuracy/baseline.test.ts
import { describe, it, expect } from 'vitest';
import { RAGSystem } from '@/system';
import { loadTestSet } from './helpers';

describe('Baseline Accuracy', () => {
  it('should achieve >85% accuracy on gold dataset', async () => {
    const system = new RAGSystem({ mode: 'baseline' });
    await system.initialize();

    const goldQuestions = await loadTestSet('gold');
    let correct = 0;

    for (const question of goldQuestions) {
      const response = await system.query(question.text);
      const isCorrect = evaluateResponse(response, question.groundTruth);
      if (isCorrect) correct++;
    }

    const accuracy = correct / goldQuestions.length;
    console.log(`Baseline accuracy: ${(accuracy * 100).toFixed(2)}%`);

    expect(accuracy).toBeGreaterThan(0.85);
  });
});

function evaluateResponse(response: any, groundTruth: any): boolean {
  // Semantic similarity check
  const similarity = cosineSimilarity(
    response.answerEmbedding,
    groundTruth.answerEmbedding
  );

  // Citation accuracy check
  const citationMatch = response.citations.some(c =>
    groundTruth.citations.includes(c.section)
  );

  return similarity > 0.85 && citationMatch;
}
```

### 5.3 RL Learning Target (>97%)

```typescript
// tests/accuracy/rl-learning.test.ts
import { describe, it, expect } from 'vitest';
import { RAGSystem } from '@/system';
import { RLTrainer } from '@/training';

describe('RL Learning Accuracy', () => {
  it('should improve accuracy to >97% after training', async () => {
    const system = new RAGSystem({ rlEnabled: true });
    await system.initialize();

    const trainer = new RLTrainer(system);
    const trainingSet = await loadTestSet('silver');
    const testSet = await loadTestSet('gold');

    // Train RL agent
    await trainer.train(trainingSet, {
      episodes: 1000,
      learningRate: 0.001
    });

    // Evaluate on test set
    let correct = 0;
    for (const question of testSet) {
      const response = await system.query(question.text);
      if (evaluateResponse(response, question.groundTruth)) correct++;
    }

    const accuracy = correct / testSet.length;
    console.log(`Post-RL accuracy: ${(accuracy * 100).toFixed(2)}%`);

    expect(accuracy).toBeGreaterThan(0.97);
  });
});
```

### 5.4 Adversarial Testing

```typescript
// tests/accuracy/adversarial.test.ts
import { describe, it, expect } from 'vitest';

describe('Adversarial Robustness', () => {
  const adversarialCases = [
    {
      name: 'Typos',
      query: 'Whut r the reqirements for encrypshun?',
      expectedSection: '3.4'
    },
    {
      name: 'Ambiguous query',
      query: 'Tell me about storage',
      expectedConfidence: '<0.6' // Should indicate uncertainty
    },
    {
      name: 'Out-of-scope',
      query: 'How to make pizza?',
      expectedAnswer: 'out-of-scope'
    },
    {
      name: 'Misleading context',
      query: 'Is CVV storage allowed if encrypted?',
      expectedAnswer: 'no' // Should resist misleading premise
    }
  ];

  for (const testCase of adversarialCases) {
    it(`should handle: ${testCase.name}`, async () => {
      const response = await system.query(testCase.query);
      // Validate robust handling
      expect(response).toMatchExpectation(testCase);
    });
  }
});
```

---

## 6. Performance Testing

### 6.1 Load Testing: k6

**Configuration:**
```javascript
// tests/performance/load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const queryLatency = new Trend('query_latency');

export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500'], // 95% under 500ms
    'errors': ['rate<0.05'],            // Error rate < 5%
  },
};

export default function () {
  const payload = JSON.stringify({
    query: 'What are the encryption requirements?',
    sessionId: `session-${__VU}`
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post('http://localhost:3000/api/query', payload, params);

  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 500ms': (r) => r.timings.duration < 500,
    'has citations': (r) => JSON.parse(r.body).citations.length > 0,
  });

  errorRate.add(!success);
  queryLatency.add(res.timings.duration);

  sleep(1);
}
```

**Run Load Test:**
```bash
k6 run tests/performance/load-test.js
```

### 6.2 Latency Profiling

```typescript
// tests/performance/latency-profile.test.ts
import { describe, it, expect } from 'vitest';
import { RAGSystem } from '@/system';
import { LatencyProfiler } from './profiler';

describe('Latency Profiling', () => {
  it('should meet latency budgets per component', async () => {
    const system = new RAGSystem();
    const profiler = new LatencyProfiler();

    await profiler.profile(async () => {
      await system.query('Test query');
    });

    const breakdown = profiler.getBreakdown();

    // Latency budget allocation
    expect(breakdown.queryProcessing).toBeLessThan(50);  // 50ms
    expect(breakdown.retrieval).toBeLessThan(150);       // 150ms
    expect(breakdown.reranking).toBeLessThan(100);       // 100ms
    expect(breakdown.generation).toBeLessThan(150);      // 150ms
    expect(breakdown.total).toBeLessThan(500);           // 500ms P95
  });
});
```

### 6.3 Memory Profiling

```typescript
// tests/performance/memory-profile.test.ts
import { describe, it, expect } from 'vitest';
import { measureMemory } from './helpers';

describe('Memory Usage', () => {
  it('should not leak memory during sustained load', async () => {
    const system = new RAGSystem();

    const initialMemory = process.memoryUsage().heapUsed;

    // Execute 1000 queries
    for (let i = 0; i < 1000; i++) {
      await system.query(`Test query ${i}`);
      if (i % 100 === 0) global.gc?.(); // Force GC periodically
    }

    const finalMemory = process.memoryUsage().heapUsed;
    const growth = (finalMemory - initialMemory) / 1024 / 1024; // MB

    console.log(`Memory growth: ${growth.toFixed(2)} MB`);
    expect(growth).toBeLessThan(100); // Max 100MB growth
  });

  it('should efficiently use AgentDB memory', async () => {
    const store = new AgentDBStore({ path: ':memory:' });

    // Insert 100k vectors
    const vectors = Array(100000).fill(0).map(() =>
      Array(1536).fill(0).map(() => Math.random())
    );

    const beforeInsert = process.memoryUsage().heapUsed;
    await store.bulkInsert(vectors);
    const afterInsert = process.memoryUsage().heapUsed;

    const memoryPerVector = (afterInsert - beforeInsert) / vectors.length;

    console.log(`Memory per vector: ${memoryPerVector.toFixed(2)} bytes`);
    expect(memoryPerVector).toBeLessThan(10000); // ~10KB per vector
  });
});
```

---

## 7. Iterative Refinement Process

### 7.1 Sprint Structure (2-Week Cycles)

```
┌─────────────────────────────────────────────────────────────┐
│  Sprint Planning (Day 1)                                    │
│  • Review backlog                                           │
│  • Select stories for sprint                               │
│  • Define acceptance criteria                              │
│  • Estimate story points                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Development (Days 2-9)                                     │
│  • TDD: Red → Green → Refactor                             │
│  • Daily standups                                          │
│  • Continuous integration                                  │
│  • Pair programming for complex features                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Testing & QA (Days 8-10)                                   │
│  • Run full test suite                                     │
│  • Manual exploratory testing                              │
│  • Performance validation                                  │
│  • Security review                                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Retrospective & Demo (Day 10)                              │
│  • Demo completed features                                 │
│  • Discuss what went well/poorly                          │
│  • Action items for next sprint                           │
│  • Update metrics dashboard                               │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Code Review Checklist

#### Pre-Merge Requirements

**Automated Checks:**
- ✅ All tests pass (unit, integration, e2e)
- ✅ Code coverage ≥90%
- ✅ No linting errors
- ✅ Type checking passes
- ✅ Security scan passes
- ✅ Build succeeds

**Manual Review:**
- [ ] Code follows TDD principles
- [ ] Functions are pure/testable
- [ ] Proper error handling
- [ ] Logging is appropriate
- [ ] Documentation is updated
- [ ] No hardcoded secrets
- [ ] Performance considerations addressed
- [ ] Accessibility standards met

**Testing:**
- [ ] Tests cover edge cases
- [ ] Tests are deterministic
- [ ] Mocks are appropriate
- [ ] Integration tests for new APIs
- [ ] Performance tests for critical paths

**Architecture:**
- [ ] Follows clean architecture principles
- [ ] Dependencies point inward
- [ ] Single responsibility principle
- [ ] Open/closed principle
- [ ] Code is modular and reusable

### 7.3 Refactoring Guidelines

#### When to Refactor

**Triggers:**
1. Code smell detected (duplication, long functions)
2. Test coverage falls below threshold
3. Performance degrades
4. New requirement requires significant changes
5. Technical debt accumulates

#### Refactoring Techniques

```typescript
// Example: Extract Method Refactoring

// BEFORE: Long function with multiple responsibilities
async function processQuery(query: string): Promise<Response> {
  // Parse query
  const tokens = query.split(' ');
  const intent = classifyIntent(tokens);

  // Retrieve documents
  const embedding = await embedQuery(query);
  const results = await vectorStore.search(embedding, 10);

  // Rerank
  const scored = results.map(r => ({
    ...r,
    score: calculateRelevance(r, query)
  }));
  const topResults = scored.sort((a, b) => b.score - a.score).slice(0, 5);

  // Generate response
  const context = topResults.map(r => r.text).join('\n');
  const response = await llm.generate(query, context);

  return { answer: response, citations: topResults };
}

// AFTER: Extracted to focused functions
async function processQuery(query: string): Promise<Response> {
  const intent = await analyzeQuery(query);
  const documents = await retrieveRelevantDocuments(query);
  const rankedDocs = await rerankDocuments(documents, query);
  const response = await generateResponse(query, rankedDocs);
  return response;
}

async function analyzeQuery(query: string): Promise<Intent> {
  const tokens = tokenize(query);
  return classifyIntent(tokens);
}

async function retrieveRelevantDocuments(query: string): Promise<Document[]> {
  const embedding = await embedQuery(query);
  return vectorStore.search(embedding, 10);
}

async function rerankDocuments(docs: Document[], query: string): Promise<Document[]> {
  const scored = docs.map(d => ({
    ...d,
    score: calculateRelevance(d, query)
  }));
  return scored.sort((a, b) => b.score - a.score).slice(0, 5);
}
```

#### Refactoring Safety

**Always:**
1. Ensure all tests pass before refactoring
2. Refactor one step at a time
3. Run tests after each change
4. Commit working code frequently
5. Use IDE refactoring tools

### 7.4 Quality Gates

#### Phase Gates

| Phase | Entry Criteria | Exit Criteria |
|-------|---------------|---------------|
| **Development** | Story accepted, tests written | All tests green, code reviewed |
| **Testing** | Code merged to main | >90% coverage, no critical bugs |
| **Staging** | All tests pass | Performance validated, security cleared |
| **Production** | Staging approval | Monitoring active, rollback plan ready |

#### Continuous Monitoring

```typescript
// Quality metrics tracking
interface QualityMetrics {
  testCoverage: number;        // Target: >90%
  accuracy: number;            // Target: >97%
  p95Latency: number;          // Target: <500ms
  errorRate: number;           // Target: <1%
  technicalDebt: number;       // Target: <10%
  securityVulnerabilities: number; // Target: 0 high/critical
}

// Automated quality checks
async function validateQualityGates(): Promise<boolean> {
  const metrics = await collectMetrics();

  return (
    metrics.testCoverage >= 0.90 &&
    metrics.accuracy >= 0.97 &&
    metrics.p95Latency <= 500 &&
    metrics.errorRate <= 0.01 &&
    metrics.technicalDebt <= 0.10 &&
    metrics.securityVulnerabilities === 0
  );
}
```

---

## 8. Continuous Refinement

### 8.1 Automated Testing Pipeline

```yaml
# .github/workflows/test.yml
name: Test Pipeline

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: npm ci
      - name: Run unit tests
        run: npm run test:unit
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      - name: Setup test DB
        run: ./scripts/setup-test-db.sh
      - name: Run integration tests
        run: npm run test:integration

  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3
      - name: Install Playwright
        run: npx playwright install
      - name: Run E2E tests
        run: npm run test:e2e
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/

  quality-gates:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, e2e-tests]
    steps:
      - name: Check coverage threshold
        run: npm run coverage:check
      - name: Run security scan
        run: npm audit --audit-level=high
      - name: Check performance budgets
        run: npm run performance:check
```

### 8.2 Metrics Dashboard

**Track Key Metrics:**
- Test coverage trends
- Accuracy over time
- Performance metrics (P50, P95, P99)
- Error rates
- Technical debt ratio

```typescript
// Dashboard data collection
interface DashboardMetrics {
  timestamp: Date;
  sprint: number;
  metrics: {
    coverage: { unit: number; integration: number; e2e: number };
    accuracy: { baseline: number; rl: number };
    performance: { p50: number; p95: number; p99: number };
    quality: { bugs: number; technicalDebt: number };
  };
}

// Collect and store metrics after each sprint
async function recordSprintMetrics(): Promise<void> {
  const metrics = await collectAllMetrics();
  await database.insert('sprint_metrics', metrics);
  await generateDashboard(metrics);
}
```

---

## Summary: Refinement Success Criteria

### Exit Criteria for Phase 4

✅ **Test Coverage:** >90% overall
✅ **Accuracy:** >97% on gold dataset
✅ **Performance:** <500ms P95 latency
✅ **Quality Gates:** All automated checks passing
✅ **Documentation:** Complete and up-to-date
✅ **Security:** No high/critical vulnerabilities
✅ **Code Review:** All PRs approved
✅ **Technical Debt:** <10% ratio

### Key Deliverables

1. ✅ Comprehensive test suite (980-1,490 tests)
2. ✅ >90% code coverage with c8
3. ✅ Automated CI/CD pipeline
4. ✅ Performance benchmarks with k6
5. ✅ Quality metrics dashboard
6. ✅ Refactored, production-ready code
7. ✅ Complete test documentation
8. ✅ Sprint retrospective reports

---

**Next Phase:** [05-COMPLETION.md](./05-COMPLETION.md) - Integration, deployment, and production readiness
