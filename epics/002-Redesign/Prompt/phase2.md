claude-flow hive-mind spawn "PHASE 2: QUERY PROCESSING ENHANCEMENT - SPARC/London 
  TDD Planning Phase

  QUEEN BEE MISSION: Execute Phase 2 using proven SPARC + London TDD methodology.  Create all planning documentation 1st.  THEN and only then, proceed directly to implementation.  You are to ensure there are no stubs, mocks, or todo's left in the code.

  === MANDATORY EXECUTION SEQUENCE ===
  STEP 1: PLANNING PHASE (NO CODE EXECUTION)
  • Create epics/002-Redesign/phase2/ directory structure
  • Generate complete SPARC planning documents BEFORE any implementation

  STEP 2: SPARC METHODOLOGY EXECUTION
  Sparc planning documents

  === PHASE 2 SCOPE DEFINITION ===
  TARGET: Implement query processing enhancement (Weeks 5-6 from 
  REVISED-IMPLEMENTATION-ROADMAP.md)
  CONSTRAINTS: epics/002-Redesign/architecture/CONSTRAINTS.md
  ARCHITECTURE: epics/002-Redesign/architecture/MASTER-ARCHITECTURE-v3.md Query 
  Router section

  === LONDON TDD PLANNING REQUIREMENTS ===
  • TEST-ARCHITECTURE.md: Define test-first approach before any code
  • Test scenarios for symbolic query routing (80%+ accuracy target)
  • Test scenarios for template response generation (<1s response time)
  • Integration test strategy with existing Phase 1 components
  • Mock strategies for external dependencies (Neo4j, ruv-fann)

  === CRITICAL PLANNING DELIVERABLES ===
  1. SPECIFICATION.md - Functional requirements for:
     • Query classification system routing logic
     • Natural language to logic parsing requirements  
     • Template engine with variable substitution
     • Citation formatting and audit trail generation

  2. PSEUDOCODE.md - Algorithmic design for:
     • Query classifier confidence scoring algorithm
     • Symbolic reasoning integration workflow
     • Template variable extraction from proof chains
     • Response formatting and validation logic

  3. ARCHITECTURE.md - Integration design for:
     • Extension points in existing src/query-processor
     • Interface design for symbolic/graph/vector routing
     • Template engine integration with src/response-generator
     • Performance optimization strategies for <1s target

  4. TEST-ARCHITECTURE.md - London TDD strategy:
     • Unit test design for each component
     • Integration test scenarios with Phase 1 infrastructure  
     • End-to-end test validation for constraint compliance
     • Performance test harness for response time validation

  === EXECUTION RESTRICTIONS ===
  • NO implementation until ALL planning documents complete
  • NO code generation during planning phase  
  • NO deviation from epics/002-Redesign/architecture/CONSTRAINTS.md

  === SUCCESS CRITERIA FOR PLANNING PHASE ===
  □ Complete SPARC document set in epics/002-Redesign/phase2/
  □ London TDD test strategy defined for all components
  □ Integration strategy with existing 308 Rust files documented
  □ Performance validation approach for <1s response time
  □ Zero implementation - pure planning phase completion

  === SUCCESS CRITERIA FOR EXECUTION PHASE ===
   - All requirements in Phase 2 successfully coded/implemented in the codebase
   - London TDD approach implemented
   - All CONSTRAINTS adhered to
   - All tests passing
   - As built architecture aligns with epics/002-Redesign/architecture/MASTER-ARCHITECTURE-v3.md Query 
  Router section

  Bring in highly specialized agents

  BEGIN WITH PLANNING ONLY. No code execution until complete SPARC documentation 
  approved.  Then swarm the build/implementation phase in london TDD