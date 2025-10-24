# Repository Cleanup & Migration Plan
## Epic 004: TypeScript Pivot - Execution Guide

**Version**: 1.0
**Date**: 2025-10-24
**Status**: Ready for Execution

---

## 🎯 Executive Summary

This plan orchestrates the clean migration from Rust-based neurosymbolic RAG (v3.0) to TypeScript-based AgentDB + ReasoningBank architecture. The migration preserves git history, archives working code, and establishes a clean foundation for Epic 004.

**Key Metrics**:
- **225 Rust files** to archive
- **13 Cargo workspace members** to preserve
- **5.5MB source code** to migrate
- **Zero downtime** for epic planning docs

---

## 📋 Phase 1: Pre-Migration Assessment & Safety

### 1.1 Create Safety Tag
```bash
# Tag current working state
git tag -a v3.0-neurosymbolic-end -m "Archive: End of Rust neurosymbolic RAG implementation"
git tag -a v3.0-phase2-complete -m "Archive: Phase 2 MRAP + Byzantine consensus complete"

# Push tags to origin
git push origin v3.0-neurosymbolic-end
git push origin v3.0-phase2-complete
```

**Validation**: Verify tags exist with `git tag -l`

### 1.2 Create Archive Branch
```bash
# Create archive branch from current main
git checkout main
git checkout -b archive/rust-neurosymbolic-v3

# Push archive branch to origin
git push -u origin archive/rust-neurosymbolic-v3

# Return to main
git checkout main
```

**Validation**: Confirm branch exists with `git branch -a | grep archive`

### 1.3 Backup Critical Files
```bash
# Create backup directory
mkdir -p /workspaces/doc-rag/archive/backups/pre-migration

# Backup configuration
cp -r /workspaces/doc-rag/config /workspaces/doc-rag/archive/backups/pre-migration/
cp /workspaces/doc-rag/Cargo.toml /workspaces/doc-rag/archive/backups/pre-migration/
cp /workspaces/doc-rag/Cargo.lock /workspaces/doc-rag/archive/backups/pre-migration/

# Backup test data
cp -r /workspaces/doc-rag/data /workspaces/doc-rag/archive/backups/pre-migration/

# Backup documentation
cp -r /workspaces/doc-rag/docs /workspaces/doc-rag/archive/backups/pre-migration/

# Create backup manifest
cat > /workspaces/doc-rag/archive/backups/pre-migration/MANIFEST.md << 'EOF'
# Pre-Migration Backup Manifest
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Git Commit: $(git rev-parse HEAD)
Branch: $(git branch --show-current)

## Backed Up Files
- config/ - System configuration files
- Cargo.toml - Rust workspace configuration
- Cargo.lock - Dependency lock file
- data/ - Test data and samples
- docs/ - Documentation files

## Recovery Instructions
1. Checkout archive branch: `git checkout archive/rust-neurosymbolic-v3`
2. Or restore from this backup: `cp -r * /workspaces/doc-rag/`
EOF
```

---

## 📋 Phase 2: Create Migration Branch

### 2.1 Create Epic Branch
```bash
# Create new feature branch for Epic 004
git checkout main
git checkout -b epic-004-typescript-pivot

# Set upstream
git push -u origin epic-004-typescript-pivot
```

### 2.2 Branch Strategy

**Branch Structure**:
```
main (protected)
├── epic-004-typescript-pivot (active development)
│   ├── feature/ingestion-pipeline
│   ├── feature/query-processing
│   ├── feature/agentdb-storage
│   └── feature/reasoningbank-rl
└── archive/rust-neurosymbolic-v3 (read-only archive)
```

**Branch Rules**:
- `main`: Protected, requires PR + review
- `epic-004-typescript-pivot`: Active development branch
- `archive/*`: Read-only, preserved for reference
- Feature branches: Created from `epic-004-typescript-pivot`

---

## 📋 Phase 3: Directory Restructure

### 3.1 New Directory Structure
```bash
# Create new TypeScript directory structure
mkdir -p /workspaces/doc-rag/src-ts/{api,ingestion,query,agents,storage,neural,learning,utils}
mkdir -p /workspaces/doc-rag/tests-ts/{unit,integration,e2e}
mkdir -p /workspaces/doc-rag/docs-ts/{api,architecture,guides}
mkdir -p /workspaces/doc-rag/config-ts/{dev,staging,prod}
mkdir -p /workspaces/doc-rag/scripts-ts/{build,deploy,migration}
mkdir -p /workspaces/doc-rag/data-ts/{test,fixtures,samples}
```

### 3.2 Target Structure
```
/workspaces/doc-rag/
├── src-ts/                          # NEW TypeScript source
│   ├── api/                         # REST API layer
│   │   ├── routes/                  # API routes
│   │   ├── middleware/              # Express middleware
│   │   ├── controllers/             # Request handlers
│   │   └── validators/              # Input validation
│   ├── ingestion/                   # Document processing
│   │   ├── chunking/                # Document chunking
│   │   ├── embedding/               # Vector embeddings
│   │   ├── parsers/                 # Format parsers (PDF, DOCX, etc)
│   │   └── pipeline/                # Ingestion orchestration
│   ├── query/                       # Query processing
│   │   ├── processor/               # Query analysis
│   │   ├── router/                  # Query routing
│   │   ├── retrieval/               # Document retrieval
│   │   └── response/                # Response generation
│   ├── agents/                      # Agent definitions
│   │   ├── swarm/                   # Swarm coordination
│   │   ├── specialized/             # Specialized agents
│   │   └── coordination/            # Agent communication
│   ├── storage/                     # AgentDB integration
│   │   ├── agentdb/                 # AgentDB client
│   │   ├── vector/                  # Vector operations
│   │   ├── cache/                   # Caching layer
│   │   └── persistence/             # Data persistence
│   ├── neural/                      # ruv-FANN WASM
│   │   ├── models/                  # Neural models
│   │   ├── training/                # Training utilities
│   │   ├── inference/               # Inference engine
│   │   └── optimization/            # Model optimization
│   ├── learning/                    # ReasoningBank RL
│   │   ├── trajectories/            # Trajectory tracking
│   │   ├── verdicts/                # Verdict judgment
│   │   ├── distillation/            # Memory distillation
│   │   └── algorithms/              # RL algorithms
│   └── utils/                       # Shared utilities
│       ├── logging/                 # Logging utilities
│       ├── metrics/                 # Metrics collection
│       ├── config/                  # Config management
│       └── errors/                  # Error handling
├── tests-ts/                        # NEW Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   └── e2e/                         # End-to-end tests
├── docs-ts/                         # NEW Documentation
│   ├── api/                         # API documentation
│   ├── architecture/                # Architecture docs
│   └── guides/                      # User guides
├── config-ts/                       # NEW Configuration
│   ├── dev/                         # Development config
│   ├── staging/                     # Staging config
│   └── prod/                        # Production config
├── scripts-ts/                      # NEW Utility scripts
│   ├── build/                       # Build scripts
│   ├── deploy/                      # Deployment scripts
│   └── migration/                   # Migration utilities
├── data-ts/                         # NEW Test data
│   ├── test/                        # Test datasets
│   ├── fixtures/                    # Test fixtures
│   └── samples/                     # Sample documents
├── epics/                           # KEEP: Epic planning
├── archive/                         # NEW: Archived Rust code
│   ├── backups/                     # Pre-migration backups
│   ├── src/                         # Archived Rust source
│   ├── tests/                       # Archived Rust tests
│   └── docs/                        # Archived Rust docs
├── .claude/                         # KEEP: Claude configuration
├── .claude-flow/                    # KEEP: Claude Flow metrics
├── memory/                          # KEEP: Agent memory
└── coordination/                    # KEEP: Swarm coordination
```

---

## 📋 Phase 4: File-by-File Cleanup Checklist

### 4.1 Archive Rust Source Code

**Files to Archive** (Move to `/workspaces/doc-rag/archive/src/`):

```bash
# Create archive structure
mkdir -p /workspaces/doc-rag/archive/src

# Archive all Rust source code
mv /workspaces/doc-rag/src/api /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/chunker /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/embedder /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/storage /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/query-processor /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/response-generator /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/integration /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/fact /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/symbolic /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/graph /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/mcp-adapter /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/ingestion /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/observability /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/performance /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/retriever /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/security /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/shared /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/tests /workspaces/doc-rag/archive/src/
mv /workspaces/doc-rag/src/pdf_extractor.rs /workspaces/doc-rag/archive/src/

# Remove empty src directory
rmdir /workspaces/doc-rag/src
```

**Verification**: `ls /workspaces/doc-rag/archive/src/ | wc -l` should show 19 items

### 4.2 Archive Rust Tests

```bash
# Create test archive
mkdir -p /workspaces/doc-rag/archive/tests

# Archive test files
mv /workspaces/doc-rag/tests/* /workspaces/doc-rag/archive/tests/

# Keep tests directory for new TypeScript tests
# Note: tests/ will be repopulated with TypeScript tests
```

### 4.3 Archive Build Artifacts

```bash
# Create build artifacts archive
mkdir -p /workspaces/doc-rag/archive/build

# Archive Cargo files
mv /workspaces/doc-rag/Cargo.toml /workspaces/doc-rag/archive/build/
mv /workspaces/doc-rag/Cargo.lock /workspaces/doc-rag/archive/build/

# Archive build directory if exists
if [ -d /workspaces/doc-rag/target ]; then
    mv /workspaces/doc-rag/target /workspaces/doc-rag/archive/build/
fi

# Archive TDD client
mv /workspaces/doc-rag/tdd-storage-client /workspaces/doc-rag/archive/build/

# Archive temporary directories
mv /workspaces/doc-rag/temp-fact-analysis /workspaces/doc-rag/archive/build/ 2>/dev/null || true
```

### 4.4 Clean Build Artifacts

```bash
# Remove profiling data
rm -f /workspaces/doc-rag/build_rs_cov.profraw
rm -f /workspaces/doc-rag/*.profraw

# Remove test logs
rm -f /workspaces/doc-rag/test_results*.log
rm -f /workspaces/doc-rag/test_results*.txt
rm -f /workspaces/doc-rag/test_fact

# Archive validation scripts
mkdir -p /workspaces/doc-rag/archive/scripts
mv /workspaces/doc-rag/validate_neo4j.py /workspaces/doc-rag/archive/scripts/ 2>/dev/null || true
```

### 4.5 Files to KEEP (No Action Required)

**Preserve These Directories**:
- ✅ `/workspaces/doc-rag/epics/` - Epic planning documents
- ✅ `/workspaces/doc-rag/.claude/` - Claude configuration
- ✅ `/workspaces/doc-rag/.claude-flow/` - Claude Flow metrics
- ✅ `/workspaces/doc-rag/memory/` - Agent memory
- ✅ `/workspaces/doc-rag/coordination/` - Swarm coordination
- ✅ `/workspaces/doc-rag/.git/` - Git repository
- ✅ `/workspaces/doc-rag/.github/` - GitHub configuration
- ✅ `/workspaces/doc-rag/.hive-mind/` - Hive mind coordination
- ✅ `/workspaces/doc-rag/.swarm/` - Swarm state
- ✅ `/workspaces/doc-rag/.roo/` - Roo configuration

**Preserve These Files**:
- ✅ `README.md` - Project README (will update)
- ✅ `CLAUDE.md` - Claude instructions (will update)
- ✅ `.gitignore` - Git ignore rules (will update)
- ✅ `LICENSE` - License file

### 4.6 Update Documentation

```bash
# Archive old docs, keep directory
mkdir -p /workspaces/doc-rag/archive/docs
mv /workspaces/doc-rag/docs/* /workspaces/doc-rag/archive/docs/

# Keep docs directory for new TypeScript documentation
# Will be repopulated with TypeScript-specific docs
```

### 4.7 Preserve Data & Configuration

**Data Directory** (KEEP, contains test data):
```bash
# Keep data directory intact
# No action required - will be used for TypeScript tests
ls -la /workspaces/doc-rag/data/
```

**Configuration** (PRESERVE, will migrate):
```bash
# Archive old config
mkdir -p /workspaces/doc-rag/archive/config
cp -r /workspaces/doc-rag/config/* /workspaces/doc-rag/archive/config/

# Keep config directory for migration
# Will be updated with TypeScript-compatible configurations
```

**Scripts** (PRESERVE, will migrate):
```bash
# Archive old scripts
mkdir -p /workspaces/doc-rag/archive/scripts-old
mv /workspaces/doc-rag/scripts/* /workspaces/doc-rag/archive/scripts-old/

# Keep scripts directory for new TypeScript scripts
```

---

## 📋 Phase 5: Dependency Migration

### 5.1 Remove Rust Dependencies

**Action**: Rust dependencies are archived with `Cargo.toml` in Phase 4.2

**Files Removed**:
- ✅ `Cargo.toml` (workspace and root)
- ✅ `Cargo.lock` (dependency lock)
- ✅ All `src/*/Cargo.toml` files (13 workspace members)

### 5.2 Create TypeScript Dependencies

**Initialize Node.js Project**:
```bash
# Create package.json
cat > /workspaces/doc-rag/package.json << 'EOF'
{
  "name": "doc-rag-typescript",
  "version": "4.0.0",
  "description": "TypeScript-based RAG with AgentDB + ReasoningBank",
  "main": "dist/index.js",
  "type": "module",
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=10.0.0"
  },
  "scripts": {
    "dev": "tsx watch src-ts/api/index.ts",
    "build": "tsc && tsc-alias",
    "start": "node dist/index.js",
    "test": "vitest",
    "test:unit": "vitest run tests-ts/unit",
    "test:integration": "vitest run tests-ts/integration",
    "test:e2e": "vitest run tests-ts/e2e",
    "test:coverage": "vitest run --coverage",
    "lint": "eslint src-ts --ext .ts",
    "lint:fix": "eslint src-ts --ext .ts --fix",
    "typecheck": "tsc --noEmit",
    "format": "prettier --write 'src-ts/**/*.ts' 'tests-ts/**/*.ts'",
    "format:check": "prettier --check 'src-ts/**/*.ts' 'tests-ts/**/*.ts'"
  },
  "keywords": [
    "rag",
    "retrieval",
    "agentdb",
    "reasoningbank",
    "typescript",
    "ai",
    "agents"
  ],
  "author": "Doc-RAG Team",
  "license": "MIT",
  "dependencies": {
    "express": "^4.18.2",
    "agentdb": "^1.0.0",
    "ruv-fann": "^0.1.6",
    "zod": "^3.22.4",
    "dotenv": "^16.3.1",
    "pino": "^8.16.2",
    "pino-pretty": "^10.2.3"
  },
  "devDependencies": {
    "@types/node": "^20.10.0",
    "@types/express": "^4.17.21",
    "typescript": "^5.3.3",
    "tsx": "^4.6.2",
    "vitest": "^1.0.4",
    "@vitest/coverage-v8": "^1.0.4",
    "eslint": "^8.55.0",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "prettier": "^3.1.0",
    "tsc-alias": "^1.8.8"
  }
}
EOF
```

**Create TypeScript Configuration**:
```bash
# Create tsconfig.json
cat > /workspaces/doc-rag/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "lib": ["ES2022"],
    "moduleResolution": "node",
    "outDir": "./dist",
    "rootDir": "./src-ts",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "paths": {
      "@api/*": ["./src-ts/api/*"],
      "@ingestion/*": ["./src-ts/ingestion/*"],
      "@query/*": ["./src-ts/query/*"],
      "@agents/*": ["./src-ts/agents/*"],
      "@storage/*": ["./src-ts/storage/*"],
      "@neural/*": ["./src-ts/neural/*"],
      "@learning/*": ["./src-ts/learning/*"],
      "@utils/*": ["./src-ts/utils/*"]
    }
  },
  "include": ["src-ts/**/*"],
  "exclude": ["node_modules", "dist", "archive"]
}
EOF
```

**Create ESLint Configuration**:
```bash
# Create .eslintrc.json
cat > /workspaces/doc-rag/.eslintrc.json << 'EOF'
{
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "ecmaVersion": 2022,
    "sourceType": "module",
    "project": "./tsconfig.json"
  },
  "plugins": ["@typescript-eslint"],
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/recommended-requiring-type-checking"
  ],
  "rules": {
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/explicit-function-return-type": "warn",
    "@typescript-eslint/no-unused-vars": "error"
  },
  "env": {
    "node": true,
    "es2022": true
  }
}
EOF
```

**Create Prettier Configuration**:
```bash
# Create .prettierrc.json
cat > /workspaces/doc-rag/.prettierrc.json << 'EOF'
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "arrowParens": "always"
}
EOF
```

### 5.3 NPM Packages Required

**Core Dependencies**:
- `express` - REST API framework
- `agentdb` - Vector database client
- `ruv-fann` - Neural network WASM
- `zod` - Schema validation
- `dotenv` - Environment variables
- `pino` - Logging

**Development Dependencies**:
- `typescript` - TypeScript compiler
- `tsx` - TypeScript execution
- `vitest` - Test framework
- `eslint` - Linting
- `prettier` - Code formatting

**Future Dependencies** (Add as needed):
- `@anthropic-ai/sdk` - Anthropic API client
- `openai` - OpenAI API client
- `langchain` - LangChain integration
- `pdf-parse` - PDF parsing
- `mammoth` - DOCX parsing
- `cheerio` - HTML parsing
- `axios` - HTTP client
- `ioredis` - Redis client (if needed)
- `pg` - PostgreSQL client (if needed)

---

## 📋 Phase 6: Configuration Migration

### 6.1 Environment Variables

**Create `.env.example`**:
```bash
cat > /workspaces/doc-rag/.env.example << 'EOF'
# Server Configuration
NODE_ENV=development
PORT=3000
HOST=0.0.0.0

# AgentDB Configuration
AGENTDB_URL=http://localhost:8080
AGENTDB_API_KEY=your_api_key_here

# Neural Network Configuration
NEURAL_MODEL_PATH=./models
NEURAL_CACHE_SIZE=1000

# ReasoningBank Configuration
REASONINGBANK_ENABLED=true
REASONINGBANK_ALGORITHM=decision_transformer

# API Keys (DO NOT COMMIT)
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...

# Logging
LOG_LEVEL=info
LOG_PRETTY=true

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT_MS=30000
EOF
```

**Update `.gitignore`**:
```bash
cat >> /workspaces/doc-rag/.gitignore << 'EOF'

# TypeScript
dist/
*.tsbuildinfo

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*

# Environment
.env
.env.local
.env.production

# Testing
coverage/
.nyc_output/

# Build artifacts
*.log
*.pid
*.seed
*.pid.lock

# Archive (already tracked)
# archive/
EOF
```

### 6.2 Configuration Files Migration

**Strategy**: Port existing Rust configs to TypeScript-compatible formats

**Files to Migrate**:
1. `config/database.yaml` → `config-ts/dev/database.json`
2. `config/neural.yaml` → `config-ts/dev/neural.json`
3. `config/api.yaml` → `config-ts/dev/api.json`

**Migration Script** (to be created):
```bash
# Create configuration migration utility
mkdir -p /workspaces/doc-rag/scripts-ts/migration
cat > /workspaces/doc-rag/scripts-ts/migration/config-migrator.ts << 'EOF'
// Utility to convert YAML configs to JSON
import { readFileSync, writeFileSync } from 'fs';
import { parse } from 'yaml'; // Will add yaml package

// TODO: Implement config migration logic
EOF
```

### 6.3 Secrets Management

**Strategy**: Use environment variables, never commit secrets

**Checklist**:
- ✅ `.env` added to `.gitignore`
- ✅ `.env.example` created with placeholders
- ✅ All secrets referenced via `process.env.*`
- ✅ Validation for required secrets at startup

---

## 📋 Phase 7: Git Operations

### 7.1 Commit Archive Operations

```bash
# Stage archive operations
git add archive/
git add -u  # Stage deletions

# Commit archive
git commit -m "archive: preserve Rust neurosymbolic RAG v3.0

- Archive 225 Rust source files to archive/src/
- Archive 13 Cargo workspace members
- Preserve test data and documentation
- Create pre-migration backups

This commit preserves the complete Rust implementation
before pivoting to TypeScript + AgentDB + ReasoningBank.

Archived components:
- API gateway (Axum)
- Document ingestion (chunking, embedding)
- FACT cache (<50ms performance)
- MRAP integration layer
- Byzantine consensus
- Symbolic reasoning (Datalog + Prolog)
- Neo4j graph database integration

Reference: Epic 004 - TypeScript Pivot
See: archive/rust-neurosymbolic-v3 branch for full history
"
```

### 7.2 Commit New Structure

```bash
# Stage new TypeScript structure
git add src-ts/
git add tests-ts/
git add docs-ts/
git add config-ts/
git add scripts-ts/
git add package.json
git add tsconfig.json
git add .eslintrc.json
git add .prettierrc.json
git add .env.example
git add .gitignore

# Commit new structure
git commit -m "feat: initialize TypeScript project structure

- Create modular TypeScript source structure
- Add package.json with core dependencies
- Configure TypeScript, ESLint, Prettier
- Set up test infrastructure with Vitest
- Create environment configuration templates

New architecture:
- src-ts/api/ - REST API layer
- src-ts/ingestion/ - Document processing
- src-ts/query/ - Query processing
- src-ts/agents/ - Agent definitions
- src-ts/storage/ - AgentDB integration
- src-ts/neural/ - ruv-FANN WASM
- src-ts/learning/ - ReasoningBank RL
- src-ts/utils/ - Shared utilities

Dependencies:
- Express (API framework)
- AgentDB (vector storage)
- ruv-FANN (neural networks)
- Vitest (testing)

Reference: Epic 004 Phase 1 - Foundation Setup
"
```

### 7.3 Push Changes

```bash
# Push epic branch
git push origin epic-004-typescript-pivot

# Verify push
git log --oneline -5
```

---

## 📋 Phase 8: Validation & Rollback

### 8.1 Validation Checklist

**Git Repository**:
- [ ] Archive branch exists: `git branch -a | grep archive/rust-neurosymbolic-v3`
- [ ] Tags created: `git tag -l | grep v3.0`
- [ ] Epic branch exists: `git branch -a | grep epic-004-typescript-pivot`
- [ ] All changes committed: `git status` is clean

**Directory Structure**:
- [ ] Archive directory created: `ls -la /workspaces/doc-rag/archive/`
- [ ] Rust code archived: `ls /workspaces/doc-rag/archive/src/ | wc -l` = 19
- [ ] TypeScript structure created: `ls /workspaces/doc-rag/src-ts/`
- [ ] Old `src/` removed: `! -d /workspaces/doc-rag/src`

**Configuration**:
- [ ] `package.json` exists: `cat /workspaces/doc-rag/package.json`
- [ ] `tsconfig.json` exists: `cat /workspaces/doc-rag/tsconfig.json`
- [ ] `.env.example` exists: `cat /workspaces/doc-rag/.env.example`
- [ ] `.gitignore` updated: `grep node_modules /workspaces/doc-rag/.gitignore`

**Preserved Files**:
- [ ] Epic docs intact: `ls /workspaces/doc-rag/epics/`
- [ ] Claude config intact: `ls /workspaces/doc-rag/.claude/`
- [ ] Memory intact: `ls /workspaces/doc-rag/memory/`
- [ ] Data intact: `ls /workspaces/doc-rag/data/`

**Backups**:
- [ ] Pre-migration backup: `ls /workspaces/doc-rag/archive/backups/pre-migration/`
- [ ] Backup manifest: `cat /workspaces/doc-rag/archive/backups/pre-migration/MANIFEST.md`

### 8.2 Rollback Strategy

**If Migration Fails**:

**Option 1: Rollback to Archive Branch**
```bash
# Abort current work
git checkout main
git branch -D epic-004-typescript-pivot

# Restore from archive branch
git checkout archive/rust-neurosymbolic-v3
git checkout -b main-restored

# Verify restoration
ls src/
cargo build
```

**Option 2: Restore from Backup**
```bash
# Restore from pre-migration backup
cd /workspaces/doc-rag
cp -r archive/backups/pre-migration/src ./
cp -r archive/backups/pre-migration/tests ./
cp archive/backups/pre-migration/Cargo.toml ./
cp archive/backups/pre-migration/Cargo.lock ./

# Verify restoration
cargo build
```

**Option 3: Git Reset**
```bash
# Reset to before migration
git checkout main
git reset --hard v3.0-neurosymbolic-end

# Verify reset
git log --oneline -5
ls src/
```

### 8.3 Validation Script

```bash
# Create validation script
cat > /workspaces/doc-rag/scripts-ts/migration/validate-migration.sh << 'EOF'
#!/bin/bash
set -e

echo "🔍 Validating Migration..."

# Check git repository
echo "📦 Checking Git Repository..."
git branch -a | grep -q "archive/rust-neurosymbolic-v3" && echo "✅ Archive branch exists" || echo "❌ Archive branch missing"
git tag -l | grep -q "v3.0-neurosymbolic-end" && echo "✅ Tag exists" || echo "❌ Tag missing"
git branch -a | grep -q "epic-004-typescript-pivot" && echo "✅ Epic branch exists" || echo "❌ Epic branch missing"

# Check directory structure
echo "📁 Checking Directory Structure..."
[ -d "/workspaces/doc-rag/archive/src" ] && echo "✅ Archive directory exists" || echo "❌ Archive directory missing"
[ ! -d "/workspaces/doc-rag/src" ] && echo "✅ Old src directory removed" || echo "❌ Old src directory still exists"
[ -d "/workspaces/doc-rag/src-ts" ] && echo "✅ New src-ts directory exists" || echo "❌ New src-ts directory missing"

# Check configuration
echo "⚙️  Checking Configuration..."
[ -f "/workspaces/doc-rag/package.json" ] && echo "✅ package.json exists" || echo "❌ package.json missing"
[ -f "/workspaces/doc-rag/tsconfig.json" ] && echo "✅ tsconfig.json exists" || echo "❌ tsconfig.json missing"
[ -f "/workspaces/doc-rag/.env.example" ] && echo "✅ .env.example exists" || echo "❌ .env.example missing"

# Check preserved files
echo "🗂️  Checking Preserved Files..."
[ -d "/workspaces/doc-rag/epics" ] && echo "✅ Epic docs preserved" || echo "❌ Epic docs missing"
[ -d "/workspaces/doc-rag/.claude" ] && echo "✅ Claude config preserved" || echo "❌ Claude config missing"
[ -d "/workspaces/doc-rag/memory" ] && echo "✅ Memory preserved" || echo "❌ Memory missing"
[ -d "/workspaces/doc-rag/data" ] && echo "✅ Data preserved" || echo "❌ Data missing"

echo "✨ Migration validation complete!"
EOF

chmod +x /workspaces/doc-rag/scripts-ts/migration/validate-migration.sh
```

---

## 📋 Phase 9: Post-Migration Setup

### 9.1 Install Dependencies

```bash
# Install NPM packages
cd /workspaces/doc-rag
npm install

# Verify installation
npm list --depth=0
```

### 9.2 Initialize Development Environment

```bash
# Create .env from example
cp .env.example .env

# Edit .env with actual values (DO NOT COMMIT)
# nano .env

# Run typecheck
npm run typecheck

# Run linting
npm run lint

# Run tests (will be empty initially)
npm run test
```

### 9.3 Update Documentation

```bash
# Update README.md with TypeScript instructions
# Update CLAUDE.md with new project structure
# Create migration notes in docs-ts/

# Commit documentation updates
git add README.md CLAUDE.md docs-ts/
git commit -m "docs: update for TypeScript migration"
```

---

## 📋 Phase 10: Communication & Handoff

### 10.1 Create Migration Report

```bash
cat > /workspaces/doc-rag/epics/004-ap/repository-analysis/migration-report.md << 'EOF'
# Migration Report: Rust → TypeScript

**Date**: 2025-10-24
**Branch**: epic-004-typescript-pivot
**Status**: ✅ Complete

## Summary
Successfully migrated from Rust neurosymbolic RAG to TypeScript foundation.

## Actions Taken
1. ✅ Created safety tags (v3.0-neurosymbolic-end)
2. ✅ Created archive branch (archive/rust-neurosymbolic-v3)
3. ✅ Archived 225 Rust source files
4. ✅ Preserved test data and documentation
5. ✅ Created new TypeScript structure
6. ✅ Initialized Node.js project
7. ✅ Configured TypeScript, ESLint, Prettier
8. ✅ Set up test infrastructure

## Preserved Components
- Epic planning (epics/)
- Claude configuration (.claude/)
- Agent memory (memory/)
- Test data (data/)
- Coordination state (.claude-flow/, .swarm/, .hive-mind/)

## Next Steps
1. Implement AgentDB storage layer
2. Build document ingestion pipeline
3. Integrate ruv-FANN neural networks
4. Implement ReasoningBank RL
5. Create REST API endpoints

## Rollback
If needed, use: `git checkout archive/rust-neurosymbolic-v3`

## References
- Archive Branch: archive/rust-neurosymbolic-v3
- Epic Branch: epic-004-typescript-pivot
- Tags: v3.0-neurosymbolic-end, v3.0-phase2-complete
EOF
```

### 10.2 Notify Swarm Agents

```bash
# Store migration status in memory
npx claude-flow@alpha hooks notify --message "Repository migration complete: Rust archived, TypeScript initialized"

# Update coordination memory
npx claude-flow@alpha memory store \
  --key "swarm/planner/migration-complete" \
  --namespace "epic-004" \
  --value '{"status":"complete","branch":"epic-004-typescript-pivot","archived":"archive/rust-neurosymbolic-v3","date":"2025-10-24"}'
```

---

## 🎯 Execution Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Pre-Migration Assessment | 15 min | None |
| 2 | Create Migration Branch | 5 min | Phase 1 |
| 3 | Directory Restructure | 10 min | Phase 2 |
| 4 | File-by-File Cleanup | 30 min | Phase 3 |
| 5 | Dependency Migration | 20 min | Phase 4 |
| 6 | Configuration Migration | 15 min | Phase 5 |
| 7 | Git Operations | 10 min | Phases 4-6 |
| 8 | Validation | 10 min | Phase 7 |
| 9 | Post-Migration Setup | 15 min | Phase 8 |
| 10 | Communication & Handoff | 10 min | Phase 9 |
| **TOTAL** | **Full Migration** | **~2.5 hours** | Sequential |

---

## 🚨 Risk Management

### High-Risk Operations

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| Data loss during archive | Pre-migration backup + git tags | Restore from backup |
| Git history corruption | Work on branch, not main | Reset to tag |
| Configuration loss | Backup config before deletion | Copy from archive |
| Accidental secret commit | Add .env to .gitignore first | git filter-branch |
| Failed dependency installation | Document all versions | Use archive dependencies |

### Critical Success Factors

1. ✅ **Complete backup before any destructive operations**
2. ✅ **Work on branch, never directly on main**
3. ✅ **Validate after each phase**
4. ✅ **Commit incrementally, not all at once**
5. ✅ **Test rollback procedure before starting**

---

## 📚 References

### Internal Documents
- [Epic 004 Master Plan](../epic-004-master-plan.md)
- [Technology Stack](../technology-stack.md)
- [Repository Analysis](./current-state-assessment.md)

### Git References
- Archive Branch: `archive/rust-neurosymbolic-v3`
- Epic Branch: `epic-004-typescript-pivot`
- Safety Tags: `v3.0-neurosymbolic-end`, `v3.0-phase2-complete`

### External Documentation
- [AgentDB Documentation](https://github.com/ruvnet/agentdb)
- [ruv-FANN Documentation](https://github.com/ruvnet/fann.js)
- [TypeScript Best Practices](https://www.typescriptlang.org/docs/handbook/intro.html)

---

## ✅ Completion Criteria

Migration is considered complete when:

1. ✅ All Rust code archived in `archive/` directory
2. ✅ Archive branch `archive/rust-neurosymbolic-v3` created and pushed
3. ✅ Safety tags created: `v3.0-neurosymbolic-end`, `v3.0-phase2-complete`
4. ✅ New TypeScript structure created in `src-ts/`
5. ✅ `package.json` and TypeScript configs created
6. ✅ All changes committed to `epic-004-typescript-pivot` branch
7. ✅ Validation script passes all checks
8. ✅ `npm install` succeeds
9. ✅ `npm run typecheck` succeeds
10. ✅ Epic planning docs preserved and accessible

**Final Verification Command**:
```bash
bash /workspaces/doc-rag/scripts-ts/migration/validate-migration.sh
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-24
**Status**: ✅ Ready for Execution
**Next Action**: Begin Phase 1 - Pre-Migration Assessment
