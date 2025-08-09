#\!/bin/bash
# Helper to mark directories for architecture tracking

if [ "$#" -ne 1 ]; then
    echo "Usage: ./mark-directory.sh <directory-path>"
    exit 1
fi

DIR="$1"

if [ \! -d "$DIR" ]; then
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
    npx claude-flow@alpha memory store "$key" "Directory: $DIR | Purpose: $purpose | Marked: $(date)"
    echo "✅ Initial context stored"
fi
EOF < /dev/null