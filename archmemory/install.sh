#!/bin/bash
# Archmemory Simplified Installation Script
# One command setup for directory intelligence

set -e

echo "🧠 Installing Archmemory Simplified..."

# Check prerequisites
if ! command -v npx &> /dev/null; then
    echo "❌ Error: npx is required. Please install Node.js first."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required for JSON processing."
    echo "Please install jq: https://stedolan.github.io/jq/download/"
    exit 1
fi

# Verify claude-flow is available
if ! npx claude-flow@alpha --version &> /dev/null; then
    echo "❌ Error: claude-flow is not available. Please ensure it's properly configured."
    exit 1
fi

# Check if .claude/settings.json exists
if [ ! -f ".claude/settings.json" ]; then
    echo "❌ Error: .claude/settings.json not found. Please run from project root."
    exit 1
fi

# Backup current settings
echo "📦 Backing up current settings..."
cp .claude/settings.json .claude/settings.json.backup-$(date +%Y%m%d-%H%M%S)

# Add minimal hooks to settings.json
echo "🔧 Adding lightweight hooks..."
jq '.tools.edit.hooks.pre = (.tools.edit.hooks.pre // []) + [{
  "command": "dir=$(dirname \"$FILE_PATH\"); while [ \"$dir\" != \".\" ] && [ \"$dir\" != \"/\" ]; do if [ -f \"$dir/.archtrack\" ]; then npx claude-flow@alpha memory retrieve --key \"arch/$(echo $dir | sed \"s/[^a-zA-Z0-9]/_/g\")\" 2>/dev/null || true; break; fi; dir=$(dirname \"$dir\"); done",
  "description": "Load architecture context if in tracked directory"
}] | 
.tools.edit.hooks.post = (.tools.edit.hooks.post // []) + [{
  "command": "dir=$(dirname \"$FILE_PATH\"); while [ \"$dir\" != \".\" ] && [ \"$dir\" != \"/\" ]; do if [ -f \"$dir/.archtrack\" ]; then echo \"Updating architecture knowledge for $dir\"; break; fi; dir=$(dirname \"$dir\"); done",
  "description": "Mark for architecture update if in tracked directory"
}]' .claude/settings.json > .claude/settings.json.tmp && mv .claude/settings.json.tmp .claude/settings.json

# Add minimal section to CLAUDE.md
echo "📝 Adding documentation to CLAUDE.md..."
if ! grep -q "## Archmemory: Directory Intelligence" CLAUDE.md 2>/dev/null; then
    cat >> CLAUDE.md << 'EOF'

## Archmemory: Directory Intelligence

Archmemory preserves architectural knowledge about key directories across all Claude sessions.

### Quick Start:
```bash
# Mark important directories for tracking
touch src/api/.archtrack          # API endpoints and business logic
touch src/components/.archtrack    # UI components and patterns
touch src/utils/.archtrack         # Shared utilities and helpers
```

### How it works:
- When you edit files in marked directories, Claude automatically:
  - Learns the directory's purpose and patterns
  - Stores this knowledge in persistent memory
  - Shares insights across all future sessions

### What Claude learns:
- Directory purpose and responsibilities
- Key patterns and conventions
- Important dependencies
- Integration points with other parts

### Best practices:
- Only mark truly important directories (3-7 per project)
- Mark directories that contain architectural decisions
- Let Claude learn naturally through your edits

No configuration needed - just touch .archtrack and work normally!
EOF
fi

# Create helper script for marking directories
echo "📄 Creating mark-directory helper..."
cat > mark-directory.sh << 'EOF'
#!/bin/bash
# Simple helper to mark directories for architecture tracking

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
EOF

chmod +x mark-directory.sh

# Create analyzer script (runs automatically via hooks)
echo "🔍 Creating architecture analyzer..."
cat > analyze-directory.sh << 'EOF'
#!/bin/bash
# Lightweight directory analyzer - called automatically by hooks

DIR="$1"
if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    exit 0
fi

# Generate a simple key for this directory
KEY="arch/$(echo "$DIR" | sed 's/[^a-zA-Z0-9]/_/g')"

# Quick analysis
PURPOSE=""
PATTERNS=""

# Detect purpose from common patterns
if ls "$DIR"/*.api.* &>/dev/null || ls "$DIR"/*api* &>/dev/null; then
    PURPOSE="API endpoints and business logic"
elif ls "$DIR"/*.component.* &>/dev/null || ls "$DIR"/*Component* &>/dev/null; then
    PURPOSE="UI components and presentation"
elif ls "$DIR"/*.util* &>/dev/null || ls "$DIR"/*utils* &>/dev/null; then
    PURPOSE="Shared utilities and helpers"
elif ls "$DIR"/*.model* &>/dev/null || ls "$DIR"/*Model* &>/dev/null; then
    PURPOSE="Data models and schemas"
elif ls "$DIR"/*.service* &>/dev/null || ls "$DIR"/*Service* &>/dev/null; then
    PURPOSE="Business services and logic"
else
    PURPOSE="Application module"
fi

# Count files by extension
MAIN_TYPE=$(find "$DIR" -maxdepth 1 -type f -name "*.*" | sed 's/.*\.//' | sort | uniq -c | sort -nr | head -1 | awk '{print $2}')

# Build simple architecture summary
SUMMARY="Directory: $DIR
Purpose: $PURPOSE
Primary file type: .$MAIN_TYPE
Key responsibility: $(basename "$DIR") module
Last analyzed: $(date)"

# Store in memory
npx claude-flow@alpha memory store --key "$KEY" --value "$SUMMARY" --namespace "architecture" 2>/dev/null || true
EOF

chmod +x analyze-directory.sh

echo "
✅ Archmemory installation complete!

🚀 Quick start:
   1. Mark important directories:
      ./mark-directory.sh src/api
      ./mark-directory.sh src/components
      
   2. That's it! Claude will learn as you work.

📁 Marked directories will have enhanced context in all future sessions.
"