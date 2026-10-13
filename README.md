# RAG Corpus Poisoning: Research on Effective Knowledge Exclusion

## Research Problem

Retrieval-Augmented Generation (RAG) systems combine document retrieval with language models to provide grounded, knowledge-rich answers. However, their security properties are understudied. **Can an attacker cause legitimate target knowledge to become effectively inaccessible in a RAG system by injecting poisoned documents that dominate retrieval rankings?**

This repository investigates whether corpus poisoning—strategically crafting and injecting documents—can push legitimate documents out of the top-k retrieval context, effectively excluding that knowledge from the language model's view.

## Research Hypothesis

**Primary Hypothesis:** An attacker can craft poisoned documents that, when added to a corpus, cause the legitimate target document to fall outside the top-k retrieved results, thereby preventing the language model from accessing that knowledge through retrieval-based context injection.

**Corollary:** As the poison budget (number of injected documents) increases, the probability that the target document is excluded from top-k retrieval increases, and retrieval depth (k) acts as a moderating factor.

## Threat Model

### Attacker Capabilities:
- **Corpus injection:** Attacker can add new documents to the corpus (but cannot modify or delete existing ones)
- **Knowledge of target:** Attacker knows which document/knowledge is the target
- **Document crafting:** Attacker can craft poison documents to optimize semantic similarity to the query

### Attacker Goals (ranked by severity):
1. **Knowledge Exclusion (strongest):** Target document is absent from top-k retrieval
2. **Generation Success (stronger):** LLM outputs the attacker's target answer based on poisoned documents
3. **Ranking Success (weaker):** Poison documents rank highly (e.g., #1) in retrieval

### Attacker Constraints:
- Poison documents must be retrievable for the same query
- Attacker cannot control the embedding model or retrieval mechanism
- Attacker cannot modify the language model's generation behavior

### Legitimate Scenarios:
This research assumes a realistic scenario where:
- The document corpus is collaborative or semi-open (e.g., Wikipedia, Q&A forums, documentation)
- An attacker can inject documents as a legitimate user (e.g., add a poorly-sourced article)
- The system does not have strong content verification/moderation

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│  User Query                                             │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Embedding Model: sentence-transformers/all-MiniLM-L6  │
│  (Encodes query and all documents to 384-dim vectors)  │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  FAISS Index                                            │
│  (Inner Product search over embeddings)                │
│  ├─ Legitimate Documents (clean corpus)                │
│  └─ Poison Documents (attacker-injected)               │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Top-k Retrieved Documents (k=1, 3, or 5)              │
│  Ranked by semantic similarity score                   │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  LLM (Ollama llama3.2:1b)                              │
│  Generates answer from only retrieved context         │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Generated Answer                                       │
└────────────────────────────────────────────────────────┘
```

## Metrics Definition

### Retrieval-Level Metrics

1. **Target Document Rank (before poisoning):** Position of legitimate document in top-k clean retrieval (1-indexed, or None if outside top-k)

2. **Target Document Rank (after poisoning):** Position of legitimate document in top-k poisoned retrieval (1-indexed, or None if outside top-k)

3. **Target in Top-k (clean):** Boolean indicating whether target document appears in clean retrieval (expected: True)

4. **Target in Top-k (poisoned):** Boolean indicating whether target document appears in poisoned retrieval

5. **Poison Retrieved:** Boolean indicating whether any poison document appears in top-k poisoned retrieval

6. **Poison Rank One:** Boolean indicating whether a poison document ranks #1

7. **Poison Occupancy in Top-k:** Fraction of top-k positions occupied by poison documents (e.g., 2/5 if 2 of 5 are poison)

### Attack-Level Metrics

8. **Knowledge Deletion Success Rate (KDSR):** 
   - *Definition:* Fraction of queries where the legitimate target document is **absent** from top-k retrieval
   - *Interpretation:* Effective retrieval-level exclusion of target knowledge
   - *Range:* 0.0 (never excluded) to 1.0 (always excluded)
   - *Target:* Higher is more successful attack

9. **Retrieval Attack Success Rate (RASR):**
   - *Definition:* Fraction of queries where at least one poison document is retrieved in top-k
   - *Interpretation:* Poison documents are retrievable alongside legitimate corpus
   - *Range:* 0.0 to 1.0
   - *Target:* Prerequisite for poisoning to work

10. **Generation Attack Success Rate (GASR):**
    - *Definition:* Fraction of queries where the language model outputs the attacker-target answer
    - *Interpretation:* LLM was influenced by poisoned documents to generate wrong answer
    - *Range:* 0.0 to 1.0
    - *Target:* Higher is more successful attack

### Experiment Variables

- **Poison Budget:** Number of poison documents injected (0, 1, 2, 3, 5)
- **Retrieval Depth (k):** Number of documents retrieved (1, 3, 5)
- **Query:** Information request targeting specific knowledge

## How to Run

### Prerequisites

1. **Python 3.9+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama running locally** with `llama3.2:1b` model:
   ```bash
   ollama pull llama3.2:1b
   ollama serve  # in another terminal
   ```

### Simple Proof of Concept

Run the basic single-query demonstration:

```bash
python demo.py
```

**Output:**
- Shows clean RAG results (target document retrieved)
- Shows poisoned RAG results (target document may be pushed out)
- Prints retrieval ranks, poisoned status, and generated answers
- Displays attack success metrics

**Expected output:** Target document rank should increase or disappear in poisoned results.

### Full Multi-Query Experiment

Run the comprehensive experiment over all benchmark queries, poison budgets, and retrieval depths:

```bash
python experiment.py
```

**Output files:**
- `results/results.csv` — Per-query results (300 rows: 20 queries × 5 budgets × 3 k values)
- `results/summary.csv` — Aggregated metrics by (budget, k) pair (15 rows)
- `results/plots/knowledge_deletion_vs_budget.png` — KDSR vs poison budget
- `results/plots/generation_asr_vs_budget.png` — GASR vs poison budget
- `results/plots/knowledge_deletion_vs_k.png` — KDSR vs retrieval depth

### Retrieval-Only Experiment (Faster)

To skip the LLM generation step and run retrieval experiments only:

```bash
python experiment.py --skip-generation
```

This runs ~10x faster and still computes all retrieval metrics and KDSR. Useful for iterating on poison document strategies.

### Explore Results

```bash
# View per-query results (CSV)
cat results/results.csv

# View aggregated summary (CSV)
cat results/summary.csv

# View plots
open results/plots/
```

## What Counts as a Successful Attack?

An attack is **successful** if:

1. **Retrieval-level success (necessary):** Target document rank increases AND/OR falls outside top-k
2. **Strong success (knowledge exclusion):** KDSR > 0.5 (target excluded from top-k in >50% of queries)
3. **Strongest success (generation-level):** GASR > 0.3 (LLM outputs wrong answer in >30% of queries)

The progression from weak to strong:
- ✓ Poison documents are retrievable (RASR > 0)
- → Target rank decreases (weaker)
- → Target absent from top-k (strong) — **KDSR metric**
- → LLM generates wrong answer (strongest) — **GASR metric**

## Expected Outputs and Interpretation

### Scenario A: Attack Succeeds
```
KDSR (k=1):       0.85  (target excluded 85% of the time)
KDSR (k=5):       0.60  (target excluded 60% of the time)
GASR (k=1):       0.70  (LLM outputs wrong answer 70% of the time)
```
→ **Interpretation:** Poisoning effectively excludes target knowledge, especially with small k.

### Scenario B: Attack Partially Succeeds
```
KDSR (k=1):       0.50  (marginal exclusion)
KDSR (k=5):       0.15  (minimal exclusion with large k)
GASR:             0.10  (LLM rarely outputs wrong answer)
```
→ **Interpretation:** Poisoning can push target slightly, but large k mitigates; LLM robust to poison.

### Scenario C: Attack Fails
```
KDSR (all k):     0.00  (target always in top-k)
GASR:             0.00  (LLM never outputs wrong answer)
```
→ **Interpretation:** Target document too semantically salient; poison documents cannot compete.

## Code Structure

- **`demo.py`** — Single-query proof of concept (unchanged)
- **`experiment.py`** — Multi-query benchmark experiment with metrics and plotting
  - `create_benchmark()` — 20 synthetic QA pairs with attacker-target answers and poison templates
  - `retrieve()` — FAISS retrieval using SentenceTransformers embeddings
  - `generate_answer()` — LLM generation via Ollama
  - `run_experiment_query()` — Single query experiment (one budget, one k)
  - `run_full_experiment()` — Full factorial experiment
  - `aggregate_results()` — Compute summary statistics
  - `plot_*()` — Matplotlib visualization functions
- **`README.md`** — This file
- **`RESEARCH_NOTES.md`** — Research progress and next steps
- **`requirements.txt`** — Python dependencies

## Current Limitations

1. **Small scale:** 20 benchmark queries (realistic but limited); could expand to 100+ for publication-quality evaluation

2. **Single embedding model:** Only sentence-transformers/all-MiniLM-L6-v2; results may not generalize to other models (e.g., OpenAI embeddings)

3. **Simple poison strategy:** Poison documents use template-filling approach; more sophisticated attacks (e.g., semantic paraphrasing, LLM-generated poison) not explored

4. **Simple corpus:** Only target + poison documents; no large realistic corpus with many distractors

5. **Small LLM:** Using llama3.2:1b for speed; larger models (7b, 13b) may be more/less robust to poisoning

6. **Uniform k across experiments:** Could test mixed retrieval strategies (e.g., retrieve k=10 but vary downstream usage)

7. **No user study:** Metrics are automatic; human evaluation of answer quality/incorrectness would strengthen claims

8. **Reproducibility:** Fixed random seed helps, but Ollama model behavior may vary across runs

## Future Work

See `RESEARCH_NOTES.md` for detailed next steps, including:
- Adaptive poison optimization
- Multi-document target scenarios
- Larger-scale evaluation
- Other embedding models and LLMs
- Defense mechanisms

## References & Related Work

This research is motivated by:
- **Data poisoning** in machine learning (Biggio et al., 2012)
- **Adversarial examples** for NLP (Belinkov & Bisk, 2019)
- **RAG security** (emerging area; see conference workshops)
- **Knowledge suppression** in QA systems

## License

Research code. Use and modify as needed for academic purposes.

## Contact

**Researcher:** Abhijeet Malviya (malvia1@unlv.nevada.edu)  
**Institution:** UNLV  
**Last Updated:** 2026-10-13
