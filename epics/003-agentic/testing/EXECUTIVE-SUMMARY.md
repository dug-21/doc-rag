# Test Data Generation Research: Executive Summary

**Research Question**: How to generate high-quality test data for PCI-DSS RAG system to achieve >97% accuracy?

**Research Date**: October 24, 2025
**Researcher**: Research Agent
**Status**: ✅ Complete

---

## TL;DR

**Recommended Approach**: Hybrid methodology combining synthetic generation (RAGAS), active learning, and adversarial testing.

**Cost**: $200-650 for 400+ test questions
**Time**: 8-12 weeks to production-ready evaluation system
**Expected Accuracy**: >97% achievable

---

## Key Findings

### 1. Best Method: Hybrid Approach

Combining three complementary methods provides optimal ROI:

| Method | Purpose | Questions | Cost | Time |
|--------|---------|-----------|------|------|
| **RAGAS Synthetic** | Baseline coverage | 100 | $20 | 2 weeks |
| **Active Learning** | Target weak areas | 200 | $100 | 4 weeks |
| **Adversarial Testing** | Robustness validation | 50 | $50 | 2 weeks |
| **Multi-Hop Questions** | Complex reasoning | 50 | $30 | 2 weeks |
| **Total** | **Comprehensive** | **400** | **$200** | **10 weeks** |

### 2. RAGAS Framework is Production-Ready

- **Most mature** open-source RAG evaluation framework
- **Excellent documentation** and active community
- **Built-in cost tracking** and metrics
- **Reference-free evaluation** (no gold standard required)
- **Wide integrations**: LangChain, LlamaIndex, Haystack

### 3. Question Taxonomy Matters

Four question types with different difficulty:

| Type | Description | Retrieval Success | PCI-DSS Examples |
|------|-------------|-------------------|------------------|
| **fact_single** | One explicit fact | Highest (95%+) | "What is requirement 3.4?" |
| **summary** | Multiple facts aggregated | Good (85-90%) | "What are network segmentation requirements?" |
| **reasoning** | Inference required | Lowest (60-85%) | "Which requirements apply to cloud storage?" |
| **unanswerable** | Cannot be answered | Critical | "What will PCI-DSS 5.0 require?" |

**Implication**: Test set must include all types in realistic proportions.

### 4. Semantic Validation Essential

**Exact Match fails for compliance domains**:
- "PAN must be encrypted during transmission" ✓
- "Encrypt PAN when transmitting data" ✓ (same meaning)
- Exact Match: ❌ False (different strings)
- Semantic Similarity: ✅ 0.92 (captures equivalence)

**Citation validation equally important**:
- Correct answer with wrong citation = ❌ Failed
- Must validate both content AND sources

### 5. Adversarial Testing Prevents Hallucinations

RAG systems vulnerable to:
- **Knowledge poisoning**: 98.2% attack success with 0.04% corpus contamination
- **Misleading context**: Plausible but incorrect information
- **Ambiguous queries**: Vague questions without clarification
- **Edge cases**: Exceptions and boundary conditions

**Mitigation**: Dedicated adversarial test suite (50 cases, $50, 2 weeks)

### 6. Active Learning Most Cost-Effective

Instead of generating 500 random questions:
1. Start with 100 questions ($20)
2. Identify weak areas (free)
3. Generate 50 targeted questions per weak area ($10 each)
4. Iterate 4 times
5. **Result**: 300 high-quality questions for $100 vs $250 for random

**ROI**: 2.5x better

---

## State-of-the-Art Tools (2024-2025)

### Recently Released

1. **DataMorgana** (Jan 2025)
   - Customizable diversity (lexical, syntactic, semantic)
   - JSON configuration for persona/question types
   - Beta for SIGIR'2025 LiveRAG challenge
   - **PCI-DSS Fit**: ⭐⭐⭐⭐⭐ Excellent

2. **BenchmarkQED** (Microsoft, 2025)
   - Enterprise-scale automated benchmarking
   - AutoQ query generation (local to global)
   - **PCI-DSS Fit**: ⭐⭐⭐⭐ Very Good

3. **KAQG** (2025)
   - Difficulty calibration with IRT + Bloom's Taxonomy
   - Psychometric soundness
   - **PCI-DSS Fit**: ⭐⭐⭐⭐ Very Good

### Production-Ready

4. **RAGAS** (Most Popular)
   - 6+ evaluation metrics
   - Synthetic test generation
   - Cost tracking built-in
   - **PCI-DSS Fit**: ⭐⭐⭐⭐⭐ Excellent

5. **FlashRAG** (WWW2025)
   - 36 benchmark datasets
   - 23 RAG algorithms
   - Easy comparison
   - **PCI-DSS Fit**: ⭐⭐⭐⭐ Very Good

---

## Cost Analysis

### LLM Pricing (Oct 2024)

| Model | Input (1M tokens) | Output (1M tokens) | 100 Q&A Gen |
|-------|-------------------|-------------------|-------------|
| GPT-4o | $5 | $15 | $2-5 |
| Claude 3.5 Sonnet | $3 | $15 | $2-6* |
| GPT-4o Mini | $0.15 | $0.60 | $0.10-0.30 |

*Claude uses 20-30% more tokens due to tokenizer, offsetting lower per-token cost

### Method Comparison (500 Questions)

| Method | Cost | Time | Quality | ROI |
|--------|------|------|---------|-----|
| Pure Manual | $5,000 | 4 weeks | Excellent | ⭐⭐ Low |
| Pure Synthetic | $100 | 1 week | Good | ⭐⭐⭐⭐ High |
| **Hybrid (Recommended)** | **$650** | **12 weeks** | **Excellent** | **⭐⭐⭐⭐⭐ Very High** |
| **Automated Hybrid** | **$200** | **8 weeks** | **Very Good** | **⭐⭐⭐⭐⭐ Very High** |

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
```bash
pip install ragas langchain openai sentence-transformers
```
- Generate 100 bootstrap questions with RAGAS
- Run baseline evaluation
- Establish metrics: faithfulness, relevancy, citation accuracy
- **Deliverable**: Baseline metrics (likely 85-90% accuracy)

### Phase 2: Active Learning (Weeks 3-6)
- Identify weak areas from Phase 1
- Generate 50 targeted questions per iteration × 4 iterations
- **Deliverable**: 300 total questions, 92-94% accuracy

### Phase 3: Adversarial Testing (Weeks 7-8)
- Generate 50 adversarial cases (LLM-assisted)
- Test: ambiguity, negation, edge cases, misleading context
- **Deliverable**: Robustness validation (may expose gaps)

### Phase 4: Multi-Hop Questions (Weeks 9-10)
- Generate 50 multi-hop reasoning questions
- Test cross-referential understanding
- **Deliverable**: Complex reasoning evaluation

### Phase 5: Production Pipeline (Weeks 11-12)
- Implement semantic similarity validation
- Set up citation checking
- Create automated evaluation pipeline
- **Deliverable**: Production-ready evaluation system achieving >97%

---

## Recommended Immediate Actions

### This Week
1. ✅ Install RAGAS: `pip install ragas`
2. ✅ Prepare PCI-DSS documents (chunking, embedding)
3. ✅ Generate first 50 questions
4. ✅ Run initial evaluation

### Next Week
1. Complete 100 question bootstrap
2. Analyze failure patterns
3. Begin first active learning iteration
4. Set up metrics dashboard (faithfulness, relevancy, citations)

---

## Success Criteria

### Test Data Quality
- ✅ 400+ questions across all types
- ✅ Balanced difficulty distribution (30% easy, 40% medium, 30% hard)
- ✅ 100% coverage of PCI-DSS requirements
- ✅ Diverse question formulations (lexical, syntactic, semantic)

### RAG System Performance
- ✅ >97% overall accuracy
- ✅ >95% faithfulness (factual consistency)
- ✅ >90% answer relevancy
- ✅ >90% citation F1 score
- ✅ <3% hallucination rate

### Production Readiness
- ✅ Automated evaluation pipeline
- ✅ Documented methodology
- ✅ Reproducible results
- ✅ Ongoing monitoring system

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| LLM costs exceed budget | Medium | Low | Use GPT-4o Mini for generation |
| Questions low quality | Low | High | Multi-stage filtering |
| Coverage gaps | Medium | Medium | Structured generation by requirement |
| Accuracy target not met | Low | High | Extend active learning iterations |

---

## Bottom Line

**Question**: Is it possible to generate high-quality test data for >97% accuracy?

**Answer**: ✅ **Yes, with hybrid approach for $200-650 over 8-12 weeks**

**Confidence**: High (based on 2024-2025 research and production frameworks)

**Recommendation**: Proceed with RAGAS-based hybrid approach starting immediately.

---

## References

**Full Research Document**: `test-generation-methods.md` (comprehensive 78-page survey)

**Key Papers**:
- DataMorgana (Jan 2025) - Diverse Q&A generation
- Know Your RAG (COLING 2025) - Question taxonomy
- KAQG (2025) - Difficulty calibration with IRT
- MultiHop-RAG (2024) - Multi-hop reasoning benchmark

**Production Tools**:
- RAGAS: https://docs.ragas.io/
- FlashRAG: https://github.com/RUC-NLPIR/FlashRAG
- Evidently: https://www.evidentlyai.com/llm-guide/rag-evaluation

---

**Next Document to Read**: `test-generation-methods.md` for implementation details
