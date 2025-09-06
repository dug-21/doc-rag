# Archmemory Usage Examples

## Viewing All Architecture Data

The cohesive namespace design makes it easy to see everything:

```bash
# View ALL architecture data with one command
npx claude-flow@alpha memory list --pattern "architecture/*"

# Example output:
architecture/meta/initialized: true
architecture/overview/design-principles: 1. REST API design 2. Component composition 3. Service layer pattern
architecture/overview/architectural-decisions: Microservices architecture, Event-driven communication
architecture/overview/tech-stack: Frontend: React, Backend: Node.js, Database: PostgreSQL
architecture/directories/src_api: Directory: src/api | Purpose: API endpoints and business logic | Last updated: [date]
architecture/directories/src_components: Directory: src/components | Purpose: UI components and presentation | Last updated: [date]
architecture/directories/src_utils: Directory: src/utils | Purpose: Shared utilities and helpers | Last updated: [date]
```

## Complete Setup Example

```bash
# 1. Initialize architecture memory
npx claude-flow@alpha memory store --key "architecture/meta/initialized" --value "true"

# 2. Set project-wide principles
npx claude-flow@alpha memory store --key "architecture/overview/design-principles" \
  --value "1. Clean Architecture 2. SOLID principles 3. Test-driven development"

npx claude-flow@alpha memory store --key "architecture/overview/architectural-decisions" \
  --value "1. Hexagonal architecture for business logic isolation 2. Repository pattern for data access 3. Event sourcing for audit trails"

npx claude-flow@alpha memory store --key "architecture/overview/tech-stack" \
  --value "TypeScript, React 18, Node.js 20, PostgreSQL 15, Redis, Docker"

# 3. Mark directories
touch src/domain/.archtrack      # Business logic
touch src/infrastructure/.archtrack  # External integrations
touch src/presentation/.archtrack    # UI layer

# 4. View everything
npx claude-flow@alpha memory list --pattern "architecture/*"
```

## Querying Specific Areas

```bash
# View only overview information
npx claude-flow@alpha memory list --pattern "architecture/overview/*"

# View only directory information
npx claude-flow@alpha memory list --pattern "architecture/directories/*"

# Get specific value
npx claude-flow@alpha memory retrieve --key "architecture/overview/design-principles"
```

## Benefits of Cohesive Namespace

1. **Single Query**: `--pattern "architecture/*"` shows everything
2. **Organized**: Clear hierarchy (meta, overview, directories)
3. **No Namespace Confusion**: Everything uses keys, not mixed namespaces
4. **Easy Filtering**: Use patterns to query subsets
5. **Consistent**: All commands follow the same pattern