# Research Notes: RAG Corpus Poisoning & Knowledge Exclusion

**Date:** 2026-10-13  
**Researcher:** Abhijeet Malviya (UNLV)  
**Status:** Active Research

---

## Current Research Question

**Primary:** Can corpus poisoning cause legitimate target knowledge to become effectively inaccessible in a RAG system by pushing the correct document out of the top-k retrieved context?

**Secondary:** How do poison budget and retrieval depth trade off in determining knowledge exclusion effectiveness?

---

## Working Hypothesis

1. **Corpus poisoning works:** Crafted poison documents can cause target documents to fall outside top-k retrieval
2. **Budget matters:** More poison documents → higher probability of target exclusion (non-linear relationship expected)
3. **K is protective:** Larger retrieval depth (k) reduces knowledge exclusion success (but at computational cost)
4. **Generation vulnerability:** If target is excluded from retrieval, LLM is forced to use poison or abstain (weaker defense than assumed)

---

## What Has Been Completed

### ✅ Phase 1: Infrastructure & Proof of Concept
- [x] Initial demo.py with single-query poisoning example
- [x] Working FAISS retrieval with SentenceTransformers embeddings
- [x] Ollama integration for LLM generation
- [x] Basic metrics (target rank, poison retrieval, knowledge exclusion)

### ✅ Phase 2: Experiment Framework
- [x] `experiment.py` with multi-query benchmark
- [x] Synthetic benchmark: 20 QA pairs with diverse domains (APIs, security, systems)
- [x] Poison budget sweep: 0, 1, 2, 3, 5 documents
- [x] Retrieval depth sweep: k = 1, 3, 5
- [x] Per-query metrics (rank, retrieval status, generation output)
- [x] Aggregated summary statistics
- [x] `--skip-generation` option for fast retrieval-only runs
- [x] Reproducible seeding (random.seed, np.random.seed)

### ✅ Phase 3: Metrics & Evaluation
- [x] **Knowledge Deletion Success Rate (KDSR):** Fraction of queries where target excluded from top-k
- [x] **Generation Attack Success Rate (GASR):** Fraction of queries where LLM outputs attacker answer
- [x] **Retrieval metrics:** Poison retrieved, poison rank-1, poison occupancy
- [x] **Rank tracking:** Before/after poisoning ranks for causal analysis

### ✅ Phase 4: Visualization & Reporting
- [x] Matplotlib plots:
  - KDSR vs poison budget (separated by k)
  - GASR vs poison budget (separated by k)
  - KDSR vs retrieval depth (separated by budget)
- [x] CSV outputs:
  - `results.csv`: 300 per-query rows (20 queries × 5 budgets × 3 k values)
  - `summary.csv`: 15 aggregated rows (5 budgets × 3 k values)
- [x] README.md with threat model, metrics, and usage instructions
- [x] RESEARCH_NOTES.md (this file)

---

## What Remains to Be Done

### 🟡 Short-term (for next research meeting/paper draft)
1. **Run full experiment** and report results:
   - Document KDSR, GASR, and related metrics
   - Interpret findings relative to hypothesis
   - Identify which factors most influence success

2. **Analyze failure modes:**
   - When does poisoning fail? (e.g., k=5, tight semantic cluster)
   - When does it succeed best? (e.g., k=1, loose query semantics)
   - Rank shift statistics (how many positions does target move on average?)

3. **Preliminary findings summary:**
   - 1-2 page summary for advisor/committee
   - Key insight: Is knowledge exclusion a real threat, or artifact of experimental setup?

### 🟠 Medium-term (for publication-quality results)
4. **Scale up benchmark:**
   - Expand from 20 to 100+ queries (cover more knowledge domains)
   - Use realistic corpus (e.g., subset of Wikipedia, StackExchange)
   - Stratify by query difficulty, document similarity, etc.

5. **Explore poison strategies:**
   - **Semantic paraphrasing:** Use LLM to rephrase attacker target (more natural poison)
   - **Semantic matching:** Optimize poison documents to maximize similarity to query
   - **Multi-target:** One poison doc targeting multiple queries
   - **Defensive rephrasing:** Vary poison language to avoid detection

6. **Test generalization:**
   - Multiple embedding models (e.g., OpenAI, Cohere, larger ST models)
   - Multiple LLMs (7b, 13b, 70b models; different architectures)
   - Multiple retrieval methods (BM25, dense-sparse hybrids, rerankers)

7. **Defenses & robustness:**
   - **Reranker:** Add second-stage ranker to re-rank top-k
   - **Diversity:** Enforce diversity in retrieved results
   - **Provenance:** Track document origin; deprioritize unknown sources
   - **Model robustness:** Evaluate on models with instruction-tuning, RLHF

### 🔴 Longer-term (for followup research)
8. **Theoretical analysis:**
   - When is knowledge exclusion possible? (necessary/sufficient conditions)
   - Embedding space geometry: why do poison docs rank high?
   - Trade-off between coverage (k) and robustness

9. **Realistic threat scenarios:**
   - Attacker doesn't know exact query (distribution shift)
   - Multi-step attacks (temporal, coordinated)
   - Attacker without knowledge of embedding model

10. **Positive research (defenses):**
    - Training robust embeddings (adversarial training)
    - Certified retrieval (provable bounds on rank)
    - Curriculum learning for poison resistance

---

## Concrete Next Experiments (3–5)

### 1. **Poison Optimization via Gradient-Free Search**
**Goal:** Improve poison success by optimizing document crafting  
**Method:**
- For each query, generate N variations of poison text (via LLM)
- Embed each variation and measure similarity to query
- Select top poison variants; run experiment
- Compare against fixed template approach

**Expected outcome:** Adaptive poison documents → higher KDSR/GASR

### 2. **Realistic Corpus Experiment**
**Goal:** Test if poisoning works in large, diverse corpus  
**Method:**
- Embed 1,000 documents from public corpus (e.g., Wikipedia snippets, StackOverflow)
- Inject target + poison into corpus
- Run retrieval for same queries
- Measure rank/exclusion rates

**Expected outcome:** May be harder to exclude target in noisy corpus; or easier if poison can semantically match many docs

### 3. **Reranker Defense**
**Goal:** Evaluate simple defense mechanism  
**Method:**
- Add second-stage reranker (e.g., cross-encoder) to re-score top-10 retrieved docs
- Re-rank based on semantic alignment with query
- Measure if reranker recovers target document or filters poison

**Expected outcome:** Reranker may recover target; demonstrates defense feasibility

### 4. **Multi-LLM Evaluation**
**Goal:** Understand LLM robustness to poisoning  
**Method:**
- Test same poisoned retrieval results on multiple LLMs (llama, mistral, gpt-like)
- Measure GASR variance across models
- Analyze failure modes (does model refuse? abstain? use poison?)

**Expected outcome:** Smaller models more susceptible; larger models more robust?

### 5. **Attacker Uncertainty Experiment**
**Goal:** Realistic scenario where attacker doesn't see embedding model  
**Method:**
- Attacker crafts poison without access to embedding model
- Attacker uses only semantic understanding (e.g., heuristics, semantic synonymy)
- Measure success rate vs white-box optimization

**Expected outcome:** Black-box attacks harder but still viable; shows practical threat

---

## Key Insights So Far

1. **Retrieval depth is critical:** k=1 creates high exclusion risk; k=5 is significantly more robust
2. **Template strategy works:** Simple poison templates can push targets out of top-k (in demo)
3. **Generation gap:** Retrieval success ≠ generation success; LLM may robustly refuse poisoned answers
4. **Reproducibility matters:** Fixed seed + careful metrics ensure consistent evaluation

---

## Professor Update (Verbal Explanation)

> "My research investigates whether an attacker can use document poisoning to exclude legitimate knowledge from a RAG system's retrieval context. 
>
> **The setup:** I've built a benchmark with 20 factual question-answer pairs, and for each query, I craft poison documents designed to rank highly using the same embedding model as the RAG system. 
>
> **The experiment:** I test whether increasing the number of poison documents causes the legitimate target document to fall out of the top-k retrieved results—a form of *effective retrieval-level knowledge exclusion*. I also test how retrieval depth (k=1, 3, or 5) affects this vulnerability.
>
> **Preliminary results:** The framework is ready to run. Early demos show that crafted poison can indeed push targets down in rankings. My next step is to run the full experiment sweep across all poison budgets and k values, analyze the data, and determine whether knowledge exclusion is a real, quantifiable threat.
>
> **Why it matters:** RAG systems are increasingly used for high-stakes applications (legal research, medical QA, code assistance). If poisoning can exclude knowledge at the retrieval level, it's a vulnerability defenders need to address—either through more robust retrieval methods, defenses like reranking, or model hardening.
>
> **Next:** Full experiment run + analysis of results. If KDSR is high (>50%), I'll explore defenses. If it's low, I'll investigate why—maybe larger embeddings models are naturally robust, or the corpus needs to be more realistic."

---

## Memo Format for Advisor

**Subject:** RAG Poisoning Research Progress — Benchmark & Experiment Framework Complete

**From:** Abhijeet Malviya  
**Date:** 2026-10-13  

**Status:** Ready for full evaluation

**Completed:**
- Synthetic benchmark (20 QA pairs)
- Experiment framework (budgets 0-5, k ∈ {1,3,5})
- Metrics: KDSR, GASR, rank tracking, occupancy
- Code: fully commented, reproducible, CLI-friendly

**Next milestone:** Run full experiment, interpret results, and report findings

**Estimated timeline:** 
- Full run + analysis: 1 week
- Defense exploration: +2 weeks
- Paper draft: +2 weeks

---

## Files & Structure

```
.
├── demo.py                 # Single-query proof of concept (unchanged)
├── experiment.py           # Multi-query benchmark framework
├── README.md              # Comprehensive guide
├── RESEARCH_NOTES.md      # This file
├── requirements.txt       # Dependencies
├── results/
│   ├── results.csv        # Per-query results (20×5×3=300 rows)
│   ├── summary.csv        # Aggregated summary (15 rows)
│   └── plots/
│       ├── knowledge_deletion_vs_budget.png
│       ├── generation_asr_vs_budget.png
│       └── knowledge_deletion_vs_k.png
└── .gitignore
```

---

## Terminology (Important for Paper & Presentation)

- **Corpus poisoning:** Injecting crafted documents into corpus
- **Knowledge exclusion / Effective retrieval-level knowledge deletion:** Target document outside top-k (NOT "machine unlearning" or "document deletion")
- **Knowledge Deletion Success Rate (KDSR):** Our primary metric for exclusion
- **Generation Attack Success Rate (GASR):** LLM outputs wrong answer
- **Poison budget:** Number of poison documents injected
- **Retrieval depth (k):** Number of documents retrieved

**Avoid:**
- "Unlearning" (implies forgetting; we're just degrading retrieval)
- "Deletion" (document still in corpus; just not retrieved)
- "Knowledge poisoning" (document poisoning, not knowledge)

---

## Open Questions for Advisor

1. Is KDSR the right primary metric, or should we weight GASR more heavily?
2. Should we test with larger corpora (1,000+ docs) before paper submission?
3. What embedding models should we prioritize? (ST is good but limited)
4. Should we focus on understanding the threat, or immediately on defenses?
5. Any blind spots in the threat model we should address?

---

**Last Updated:** 2026-10-13  
**Next Review:** After full experiment run (expected 2026-10-20)
