# Archmemory Simplified

Directory intelligence for Claude Code using Claude Flow's persistent memory system.

## What it does

Archmemory helps Claude remember:
- The purpose and patterns of important directories
- Project-wide design principles and architectural decisions
- Directory-specific knowledge that persists across all sessions

All data is stored in Claude Flow's memory under the `architecture/` key prefix, accessible to all agents and sessions.

## Installation

Ask Claude Code to install archmemory:

```
Claude, please install archmemory by following the steps in archmemory-simplified/INSTALLATION.md
```

## How it works

1. **Cohesive Memory Structure**: All architecture data under `architecture/` prefix
2. **Directory Tracking**: Mark directories with `.archtrack` files
3. **Automatic Context**: Hooks load relevant architecture info when editing tracked directories
4. **Knowledge Sharing**: All agents and sessions share the same architectural knowledge

## Usage

### Mark directories for tracking
```bash
touch src/api/.archtrack
touch src/components/.archtrack
touch src/utils/.archtrack
```

### Store high-level design principles
```bash
npx claude-flow@alpha memory store --key "architecture/overview/design-principles" \
  --value "1. REST API design 2. Component composition 3. Service layer pattern"
```

### View ALL architecture knowledge with one command
```bash
npx claude-flow@alpha memory list --pattern "architecture/*"
```

## Memory Structure

All architecture data uses the `architecture/` prefix for easy querying:
- `architecture/meta/initialized`: Initialization flag
- `architecture/overview/design-principles`: Project-wide design principles
- `architecture/overview/architectural-decisions`: Key architectural decisions  
- `architecture/overview/tech-stack`: Technology choices
- `architecture/overview/patterns`: Architectural patterns
- `architecture/directories/[directory_key]`: Specific directory knowledge

Query everything with: `npx claude-flow@alpha memory list --pattern "architecture/*"`

## Key Features

- **Zero Configuration**: Just touch `.archtrack` files
- **Cohesive Namespace**: All data under `architecture/` for easy querying
- **Agent Integration**: All agents automatically get context when working in tracked directories
- **Progressive Learning**: Claude learns more about directories as you edit files
- **High-Level Storage**: Store design principles above the component level

## Best Practices

1. **Be Selective**: Only track 3-7 truly important directories
2. **Store Principles Early**: Add design principles and decisions when starting
3. **Let it Learn**: Claude will understand directories better over time
4. **Review Periodically**: Check stored knowledge with memory list command

## How Agents Use This

When any agent (spawned via Task tool) works on files in tracked directories:
1. Pre-edit hook loads all relevant context
2. Agent has access to directory purpose and project principles
3. Post-edit hook updates knowledge based on changes
4. All agents share the same architectural understanding

## Requirements

- Claude Flow (already installed if using Claude Code)
- Project must have `.claude/settings.json` file

## Minimal Footprint

Archmemory adds:
- 2 lightweight hooks to `.claude/settings.json`
- ~40 lines to CLAUDE.md
- No configuration files (just `.archtrack` markers)
- All data stored in Claude Flow's existing memory system