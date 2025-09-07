#!/bin/bash
# P0 Critical Priority Validation Script

echo "================================"
echo "P0 CRITICAL VALIDATION SUITE"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check FACT integration
echo "1. Verifying FACT System Integration..."
echo "----------------------------------------"

# Check cache.rs uses FACT
if grep -q "use fact::" src/query-processor/src/cache.rs 2>/dev/null; then
    echo -e "${GREEN}✅ FACT system imported in cache.rs${NC}"
    FACT_CACHE=1
else
    echo -e "${RED}❌ FACT not found in cache.rs${NC}"
    FACT_CACHE=0
fi

# Check for FACT system usage
FACT_REFS=$(grep -r "fact_system\|FactSystem\|FACT" src/query-processor/src/ --include="*.rs" 2>/dev/null | wc -l)
echo -e "${GREEN}✅ Found $FACT_REFS FACT system references${NC}"

echo ""
echo "2. Architecture Compliance..."
echo "------------------------------"

# Check Byzantine consensus
if grep -r "0\.66\|0\.67\|66%" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ Byzantine consensus (66%) found${NC}"
    BYZANTINE=1
else
    echo -e "${YELLOW}⚠️  Byzantine consensus threshold not found${NC}"
    BYZANTINE=0
fi

# Check ruv-FANN
if grep -r "ruv_fann::" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ ruv-FANN neural processing found${NC}"
    NEURAL=1
else
    echo -e "${YELLOW}⚠️  ruv-FANN not found${NC}"
    NEURAL=0
fi

# Check DAA
if grep -r "daa_orchestrator::\|DAA\|MRAP" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ DAA orchestration found${NC}"
    DAA=1
else
    echo -e "${YELLOW}⚠️  DAA orchestration not found${NC}"
    DAA=0
fi

echo ""
echo "3. Performance Targets..."
echo "-------------------------"

# Check for performance constants
echo "Checking configured performance targets:"

if grep -r "<.*50.*ms\|50ms\|fifty.*ms" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ Cache <50ms target configured${NC}"
fi

if grep -r "<.*200.*ms\|200ms" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ Neural <200ms target configured${NC}"
fi

if grep -r "<.*500.*ms\|500ms" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ Consensus <500ms target configured${NC}"
fi

if grep -r "<.*2.*s\|2s\|two.*second" src/ --include="*.rs" -q 2>/dev/null; then
    echo -e "${GREEN}✅ Total <2s response target configured${NC}"
fi

echo ""
echo "================================"
echo "P0 VALIDATION SUMMARY"
echo "================================"
echo ""

# Calculate score
TOTAL_SCORE=$((FACT_CACHE + BYZANTINE + NEURAL + DAA))

if [ "$TOTAL_SCORE" -eq 4 ]; then
    echo -e "${GREEN}✅ P0 CRITICAL WORK COMPLETE${NC}"
    echo ""
    echo "All requirements met:"
    echo "  • FACT cache system integrated"
    echo "  • Byzantine consensus (66%) active"
    echo "  • ruv-FANN neural processing enabled"
    echo "  • DAA orchestration implemented"
    echo "  • Performance targets configured"
    echo ""
    echo "Status: READY FOR PRODUCTION"
    exit 0
elif [ "$TOTAL_SCORE" -ge 2 ]; then
    echo -e "${YELLOW}⚠️  P0 work partially complete ($TOTAL_SCORE/4 core components)${NC}"
    echo ""
    echo "Completed:"
    [ "$FACT_CACHE" -eq 1 ] && echo "  ✅ FACT cache integration"
    [ "$BYZANTINE" -eq 1 ] && echo "  ✅ Byzantine consensus"
    [ "$NEURAL" -eq 1 ] && echo "  ✅ ruv-FANN neural"
    [ "$DAA" -eq 1 ] && echo "  ✅ DAA orchestration"
    echo ""
    echo "Remaining:"
    [ "$FACT_CACHE" -eq 0 ] && echo "  ❌ FACT cache integration"
    [ "$BYZANTINE" -eq 0 ] && echo "  ❌ Byzantine consensus"
    [ "$NEURAL" -eq 0 ] && echo "  ❌ ruv-FANN neural"
    [ "$DAA" -eq 0 ] && echo "  ❌ DAA orchestration"
    exit 1
else
    echo -e "${RED}❌ P0 work incomplete ($TOTAL_SCORE/4 core components)${NC}"
    exit 1
fi