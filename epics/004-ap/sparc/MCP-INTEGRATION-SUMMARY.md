# MCP Integration Updates to SPARC Specification

**Document:** `/workspaces/doc-rag/epics/004-ap/sparc/01-SPECIFICATION.md`
**Date:** October 25, 2025
**Status:** ✅ COMPLETE

---

## Summary of Changes

The SPARC specification has been comprehensively updated to reflect **Model Context Protocol (MCP)** integration as the primary interface for the RAG system, replacing the traditional REST API approach.

---

## Key Updates

### 1. **FR2: Query Processing** → **FR2: Query Processing via MCP Tools**

**Changes:**
- Reframed as MCP tool-based integration
- All capabilities now exposed via MCP protocol
- Added MCP-specific acceptance criteria with tool call scenarios
- Updated performance targets to include MCP overhead breakdown
- Added zero network overhead benefit (stdio transport)

**Impact:**
- **Latency improvement**: -50ms to -100ms vs REST API
- **Cost reduction**: Zero API overhead for local queries

---

### 2. **NEW: FR8: MCP Server Implementation** (CRITICAL)

**Added comprehensive specification for:**

#### MCP Tools (5 total):
1. `query_pci_dss` - Simple requirement lookup (<500ms)
2. `compare_standards` - Cross-standard analysis (<800ms)
3. `search_requirements` - Semantic search (<300ms)
4. `explain_requirement` - Detailed explanation (<400ms)
5. `validate_compliance` - Compliance checking (<1000ms)

#### MCP Resources (4 URI schemes):
1. `pci-dss://requirement/{id}` - Direct requirement access
2. `hipaa://section/{id}` - HIPAA section access
3. `compliance://graph/{node}` - Knowledge graph access
4. `document://{standard}/{page}` - Page-level access

#### MCP Prompts (4 templates):
1. `requirement-lookup` - Standard requirement query
2. `compliance-check` - Validation workflow
3. `gap-analysis` - Identify missing controls
4. `implementation-guide` - Step-by-step implementation

**Performance Requirements:**
- Tool registration: <100ms
- Tool call overhead: <10ms (MCP protocol)
- Resource lookup: <100ms
- Streaming first chunk: <50ms
- Session state access: <5ms
- Concurrent sessions: 100+

**Security Requirements:**
- JWT token validation
- RBAC at tool level
- Rate limiting (100 calls/min per user)
- Input sanitization via JSON Schema
- Audit trail for all tool calls

---

### 3. **NFR1: Performance** → **NFR1: Performance (MCP-Enhanced)**

**Changes:**
- Added MCP-specific metrics with comparison to REST API
- Broke down MCP overhead (~25ms total vs 50-100ms REST)
- Updated cost per query: <$0.0005 (50% reduction)
- Added streaming and session state metrics

**MCP Advantages:**
- **Protocol overhead**: Only ~10ms (JSON-RPC parsing)
- **Authentication**: ~5ms (JWT, cached)
- **Tool dispatch**: ~5ms
- **Session lookup**: ~5ms
- **Total**: ~25ms vs 50-100ms for REST API

---

### 4. **NFR4: Security** → **NFR4: Security (MCP-Specific)**

**Added:**
- MCP authentication & authorization flows
- Tool-level permissions (RBAC)
- Session-based authorization (30-min TTL)
- MCP message signing (HMAC-SHA256)
- Tool discovery based on user roles
- URI-based resource permissions

---

### 5. **NFR5: Observability** → **NFR5: Observability (MCP-Enhanced)**

**Added MCP-specific metrics:**
- MCP tool call latency per tool
- MCP resource access latency per URI scheme
- MCP session duration and tool call count
- MCP streaming performance
- MCP error rates by error code
- Tool usage distribution
- Session state cache hit rate

---

### 6. **System Context: Claude as Primary Client**

**Major Update:**
- **Section 4.1**: New primary client section highlighting Claude Code/Desktop
- **Integration**: Model Context Protocol (stdio and HTTP transports)
- **Capabilities**: Direct tool calls, resource access, streaming, session preservation
- **Zero network overhead**: Local stdio transport

**Advantages of MCP over REST API:**
- ✅ Zero API overhead (~50-100ms saved)
- ✅ Native Claude integration
- ✅ Streaming built-in
- ✅ Automatic session management
- ✅ Type safety via JSON Schema
- ✅ Self-describing tools/resources

**Updated User Personas:**
- Compliance Officer: Uses MCP tools via Claude Code
- Security Auditor: Multi-standard analysis via Claude Desktop
- Developer: Implementation guidance via Claude API with MCP

---

### 7. **Use Cases: MCP-Based Workflows**

**Updated all use cases to MCP:**

- **UC1**: Simple lookup via `query_pci_dss` tool (<300ms, 0ms network)
- **UC2**: Comparative analysis via `compare_standards` with streaming (<800ms)
- **UC3**: Contextual follow-up with MCP session state (<400ms)
- **UC4**: Resource access via URI `pci-dss://requirement/3.2` (<100ms)
- **UC5**: Batch query via `gap-analysis` prompt template (<2000ms)

---

### 8. **Testing: MCP Protocol Compliance Phase**

**Added Phase 3: MCP Protocol Compliance (Week 9)**
- MCP protocol validator (stdio and HTTP)
- JSON Schema validation for all tools
- Error handling per MCP spec
- Authentication/authorization flows
- Rate limiting enforcement

**Updated Integration Tests (Phase 2):**
- All 5 MCP tools with various parameters
- All 4 resource URI schemes
- All 4 prompt templates
- Session state preservation
- Streaming response handling

**Updated Production Validation (Phase 5):**
- Full test bank via MCP tools
- MCP tool call success rate >99%
- Session state accuracy >99%
- 100 concurrent MCP sessions

---

### 9. **Budget: MCP Cost Analysis**

**Updated Budget Table:**
| Category | Estimate | MCP Impact |
|----------|----------|------------|
| Infrastructure Setup | $20K | -$5K (no API gateway) |
| MCP Server Development | +$10K | New component |
| **Total Budget** | **$244K-$316K** | +$5K (minimal) |
| **Annual Operating** | **$12K/year** | -$3K/year |

**Cost Savings:**
- API Gateway eliminated: -$2K setup, -$1.5K/year
- No API management tools: -$1K setup, -$0.5K/year
- Reduced networking: -$1K setup, -$1K/year bandwidth
- Simplified auth: -$1K setup
- **3-year savings**: ~$15K infrastructure

---

### 10. **Success Metrics: MCP-Specific Targets**

**Updated Metrics Table:**
| Metric | Target | MCP Advantage |
|--------|--------|---------------|
| P95 MCP Tool Latency | <500ms | -100ms vs REST |
| Cost per Query | <$0.0005 | 50% reduction |
| MCP Tool Success Rate | >99.5% | Built-in error handling |
| Session State Accuracy | >99% | MCP session management |

---

### 11. **Production Readiness Checklist: MCP Items**

**Added MCP-Specific Checklist (11 items):**
- [ ] All 5 MCP tools registered and functional
- [ ] All 4 resource URI schemes working
- [ ] All 4 prompt templates tested
- [ ] stdio transport validated (default)
- [ ] HTTP transport tested (optional)
- [ ] Session state persistence verified
- [ ] Streaming responses working
- [ ] MCP protocol compliance validated
- [ ] Tool discovery for all user roles
- [ ] Rate limiting enforced
- [ ] MCP error handling complete

---

### 12. **Glossary: MCP Terms**

**Added definitions:**
- **MCP**: Model Context Protocol - Standard for AI-native tool integration
- **MCP Server**: TypeScript server exposing tools, resources, prompts
- **MCP Tool**: Callable function (e.g., `query_pci_dss`)
- **MCP Resource**: URI-accessible content (e.g., `pci-dss://requirement/3.2`)
- **MCP Prompt**: Pre-configured prompt template with variables
- **stdio Transport**: Local communication (zero network overhead)
- **HTTP Transport**: Remote access via HTTP

---

## Quantified Benefits of MCP Integration

### Performance
- **Latency reduction**: -50ms to -150ms depending on percentile
- **MCP overhead**: Only 25ms (vs 50-100ms for REST API)
- **Streaming**: <50ms first chunk (progressive results)

### Cost
- **Per-query cost**: 50% reduction (<$0.0005 vs <$0.001)
- **Infrastructure savings**: -$5K setup, -$3K/year operating
- **3-year TCO**: ~$15K savings

### Developer Experience
- **Native Claude integration**: First-class tool support
- **Type safety**: JSON Schema validation at protocol level
- **Session management**: Automatic context preservation
- **Discovery**: Self-describing tools and resources

### Reliability
- **Tool success rate**: >99.5% (vs >99% for REST)
- **Session state accuracy**: >99%
- **Uptime**: Improved (local stdio resilient to network issues)

---

## Implementation Priorities

### Critical Path (Week 1-4):
1. ✅ Specification updated (COMPLETE)
2. MCP server scaffold (TypeScript + @modelcontextprotocol/sdk)
3. Implement 5 core MCP tools
4. AgentDB integration for tools

### High Priority (Week 5-8):
5. Resource URI handlers (4 schemes)
6. Prompt template engine
7. Session state management
8. Streaming response implementation

### Medium Priority (Week 9-10):
9. MCP protocol compliance testing
10. Authentication/authorization
11. Rate limiting
12. Error handling

### Pre-Production (Week 11-12):
13. Performance optimization
14. Load testing (100 concurrent MCP sessions)
15. Security audit
16. Production deployment

---

## Next Steps

1. **Architecture phase**: Design MCP server architecture
2. **Pseudocode phase**: Detail tool handlers and resource resolvers
3. **Implementation**: Build MCP server with AgentDB integration
4. **Testing**: MCP protocol compliance and integration tests
5. **Deployment**: Claude Code/Desktop integration

---

## Files Modified

- `/workspaces/doc-rag/epics/004-ap/sparc/01-SPECIFICATION.md` - Comprehensive MCP integration

---

## Memory Storage

**Keys stored in AgentDB:**
- `sparc/specification/mcp-updates` - File edit metadata
- Task completion metrics in `.swarm/memory.db`

---

**Status:** ✅ SPECIFICATION COMPLETE
**Confidence:** 95% (aligned with Epic 003 recommendation)
**Recommendation:** Proceed to SPARC Phase 2 (Pseudocode) with MCP architecture
