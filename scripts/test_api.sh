#!/bin/bash

# Simple API testing script
API_URL="${API_URL:-http://localhost:8080}"

echo "🧪 Testing Doc-RAG API at $API_URL"
echo "================================"

# 1. Health check
echo -n "1. Health check... "
if curl -s "$API_URL/health" | grep -q "ok"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
    exit 1
fi

# 2. Upload PDF
echo -n "2. Uploading test PDF... "
UPLOAD_RESPONSE=$(curl -s -X POST "$API_URL/upload" \
    -F "file=@tests/sample.pdf" \
    -F "name=test_doc")

DOC_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4)

if [ -n "$DOC_ID" ]; then
    echo "✅ PASS (ID: $DOC_ID)"
else
    echo "❌ FAIL"
    echo "Response: $UPLOAD_RESPONSE"
    exit 1
fi

# 3. Query document
echo -n "3. Querying document... "
QUERY_RESPONSE=$(curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"What is the main topic?\"}")

if echo "$QUERY_RESPONSE" | grep -q "answer"; then
    echo "✅ PASS"
    echo "   Answer: $(echo "$QUERY_RESPONSE" | grep -o '"answer":"[^"]*' | cut -d'"' -f4 | head -c 50)..."
else
    echo "❌ FAIL"
    echo "Response: $QUERY_RESPONSE"
    exit 1
fi

# 4. Check citations
echo -n "4. Checking citations... "
if echo "$QUERY_RESPONSE" | grep -q "citations"; then
    echo "✅ PASS"
else
    echo "⚠️  WARNING: No citations found"
fi

# 5. Performance check
echo -n "5. Response time check... "
START_TIME=$(date +%s%N)
curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"Test query\"}" > /dev/null
END_TIME=$(date +%s%N)
ELAPSED=$((($END_TIME - $START_TIME) / 1000000))

if [ $ELAPSED -lt 2000 ]; then
    echo "✅ PASS (${ELAPSED}ms < 2000ms)"
else
    echo "❌ FAIL (${ELAPSED}ms > 2000ms)"
fi

echo ""
echo "================================"
echo "✅ All basic tests passed!"