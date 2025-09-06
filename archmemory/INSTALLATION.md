# Archmemory Installation Guidelines

These are step-by-step instructions for Claude Code to install archmemory in your project.

## Prerequisites
- Claude Flow must be installed (check: `npx claude-flow@alpha --version`)
- `.claude/settings.json` must exist

## Installation Steps

### Step 1: Initialize Architecture Memory in Claude Flow

```bash
# Initialize the architecture tracking for this project
npx claude-flow@alpha memory store --key "architecture/meta/initialized" --value "true"
npx claude-flow@alpha memory store --key "architecture/overview/design-principles" --value "Project design principles will be stored here"
npx claude-flow@alpha memory store --key "architecture/overview/architectural-decisions" --value "Key architectural decisions will be stored here"
```

### Step 2: Add Hooks to .claude/settings.json

Use the Edit tool to add these two hooks to your `.claude/settings.json` file.

1. First, backup the current settings:
```bash
cp .claude/settings.json .claude/settings.json.backup-$(date +%Y%m%d-%H%M%S)
```

2. Read the current `.claude/settings.json` file to understand its structure.

3. Add this hook to the `tools.edit.hooks.pre` array (create the array if it doesn't exist):
```json
{
  "command": "dir=$(dirname \"$FILE_PATH\"); tracked_dir=\"\"; while [ \"$dir\" != \".\" ] && [ \"$dir\" != \"/\" ]; do if [ -f \"$dir/.archtrack\" ]; then tracked_dir=\"$dir\"; break; fi; dir=$(dirname \"$dir\"); done; if [ -n \"$tracked_dir\" ]; then echo \"Loading architecture context for $tracked_dir\"; npx claude-flow@alpha memory retrieve --key \"architecture/directories/$(echo $tracked_dir | sed 's/[^a-zA-Z0-9]/_/g')\" 2>/dev/null || true; npx claude-flow@alpha memory retrieve --key \"architecture/overview/design-principles\" 2>/dev/null || true; fi",
  "description": "Load architecture context and design principles if in tracked directory"
}
```

4. Add this hook to the `tools.edit.hooks.post` array (create the array if it doesn't exist):
```json
{
  "command": "dir=$(dirname \"$FILE_PATH\"); tracked_dir=\"\"; while [ \"$dir\" != \".\" ] && [ \"$dir\" != \"/\" ]; do if [ -f \"$dir/.archtrack\" ]; then tracked_dir=\"$dir\"; break; fi; dir=$(dirname \"$dir\"); done; if [ -n \"$tracked_dir\" ]; then key=\"architecture/directories/$(echo $tracked_dir | sed 's/[^a-zA-Z0-9]/_/g')\"; existing=$(npx claude-flow@alpha memory retrieve --key \"$key\" 2>/dev/null || echo ''); purpose=\"\"; if echo \"$tracked_dir\" | grep -q \"api\"; then purpose=\"API endpoints and business logic\"; elif echo \"$tracked_dir\" | grep -q \"component\"; then purpose=\"UI components and presentation\"; elif echo \"$tracked_dir\" | grep -q \"util\"; then purpose=\"Shared utilities and helpers\"; elif echo \"$tracked_dir\" | grep -q \"model\"; then purpose=\"Data models and schemas\"; elif echo \"$tracked_dir\" | grep -q \"service\"; then purpose=\"Business services and logic\"; else purpose=\"Application module\"; fi; summary=\"Directory: $tracked_dir | Purpose: $purpose | Last updated: $(date)\"; npx claude-flow@alpha memory store --key \"$key\" --value \"$summary\" 2>/dev/null || true; fi",
  "description": "Update architecture knowledge if in tracked directory"
}
```

Note: Make sure to maintain proper JSON syntax when adding these hooks. If the hooks arrays don't exist, create them within the tools.edit object structure.

### Step 3: Add Documentation to CLAUDE.md

Append this section to your CLAUDE.md file:

```markdown

## Archmemory: Directory Intelligence

Archmemory preserves architectural knowledge about key directories across all Claude sessions using Claude Flow's persistent memory.

### Quick Start:
```bash
# Mark important directories for tracking
touch src/api/.archtrack          # API endpoints and business logic
touch src/components/.archtrack    # UI components and patterns
touch src/utils/.archtrack         # Shared utilities and helpers

# Store project-wide design principles
npx claude-flow@alpha memory store --key "architecture/overview/design-principles" --value "Your design principles here"

# Store architectural decisions
npx claude-flow@alpha memory store --key "architecture/overview/architectural-decisions" --value "Key decisions: 1) REST API design, 2) Component-based UI, 3) Service layer pattern"
```

### How it works:
- **Directory Tracking**: Touch `.archtrack` in any directory to enable tracking
- **Automatic Learning**: When editing files in tracked directories, Claude:
  - Loads architecture context from the `architecture` namespace
  - Loads project-wide design principles
  - Updates directory knowledge after edits
- **Persistent Memory**: All knowledge stored in Claude Flow's project-level memory
- **Agent Access**: All agents and future sessions can access this knowledge

### Memory Structure:
All architecture data is stored under the `architecture/` key prefix:
- `architecture/meta/initialized`: Initialization flag
- `architecture/overview/design-principles`: Project-wide design principles
- `architecture/overview/architectural-decisions`: Key architectural decisions
- `architecture/overview/tech-stack`: Technology choices
- `architecture/overview/patterns`: Architectural patterns
- `architecture/directories/[directory_key]`: Specific directory knowledge

View all architecture data with:
```bash
npx claude-flow@alpha memory list --pattern "architecture/*"
```

### Best practices:
- Mark 3-7 truly important directories per project
- Store high-level design principles in overview keys
- Let Claude learn directory specifics through natural editing
- All agents working on tracked directories automatically get context
```

### Step 4: Create Helper Script (Optional)

Create `mark-directory.sh` for easier directory marking:

```bash
#!/bin/bash
# Helper to mark directories for architecture tracking

if [ "$#" -ne 1 ]; then
    echo "Usage: ./mark-directory.sh <directory-path>"
    exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "❌ Error: Directory '$DIR' does not exist"
    exit 1
fi

touch "$DIR/.archtrack"
echo "✅ Marked $DIR for architecture tracking"
echo "💡 Claude will now learn and remember this directory's purpose"

# Optionally prompt for immediate context
echo "Would you like to add initial context for this directory? (y/n)"
read -r response
if [ "$response" = "y" ]; then
    echo "Enter the purpose of $DIR:"
    read -r purpose
    key="architecture/directories/$(echo "$DIR" | sed 's/[^a-zA-Z0-9]/_/g')"
    npx claude-flow@alpha memory store --key "$key" --value "Directory: $DIR | Purpose: $purpose | Marked: $(date)"
    echo "✅ Initial context stored"
fi
```

Make it executable:
```bash
chmod +x mark-directory.sh
```

## Verification Steps

After installation, verify everything works:

1. Check initialization:
```bash
npx claude-flow@alpha memory retrieve --key "architecture/meta/initialized"
```

2. Test directory marking:
```bash
touch src/test/.archtrack
# Edit a file in src/test/ and check if hooks trigger
```

3. View all architecture memory:
```bash
npx claude-flow@alpha memory list --pattern "architecture/*"
```

## How Agents Access This Information

When agents work on tracked directories:
1. Pre-edit hook automatically loads:
   - Directory-specific architecture context
   - Project-wide design principles
2. This information is available in the agent's context
3. Post-edit hook updates the directory knowledge
4. All changes persist in Claude Flow's project-level memory

## Storing High-Level Design Information

Store project-wide architectural information:

```bash
# Design principles
npx claude-flow@alpha memory store --key "architecture/overview/design-principles" \
  --value "1. Separation of concerns 2. DRY principle 3. SOLID principles 4. API-first design"

# Tech stack decisions
npx claude-flow@alpha memory store --key "architecture/overview/tech-stack" \
  --value "Frontend: React, Backend: Node.js, Database: PostgreSQL, Cache: Redis"

# Architectural patterns
npx claude-flow@alpha memory store --key "architecture/overview/patterns" \
  --value "Repository pattern for data access, Service layer for business logic, MVC for API structure"
```

These overview keys are automatically loaded when working in any tracked directory.