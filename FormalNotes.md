# Formal Notes: RAG Strategies and Intelligent Routing

## 1. Retrieval-Augmented Generation: common foundation

**Retrieval-Augmented Generation (RAG)** is an application architecture in which a system retrieves external information relevant to a request and supplies that information as context to a language model when generating a response. Retrieval can access documents, databases, search indexes, knowledge graphs, or other permitted sources.

RAG ordinarily changes the information available **at inference time**. It does not, by itself, update the language model's learned parameters. Fine-tuning and RAG therefore address different mechanisms: fine-tuning changes model behavior/parameters, while RAG supplies external context. They can be combined.

### Indexing and query-time processing

**Indexing** commonly consists of loading documents, extracting text, dividing it into chunks, attaching metadata, generating embeddings, and storing searchable representations. **Query-time processing** identifies the information need, retrieves candidate evidence, optionally transforms or evaluates it, and generates a response.

A dense embedding maps input text to a numerical vector. Similarity search ranks stored vectors relative to a query vector under a chosen metric. An embedding is not a factuality score. A nearby passage can be related to the topic while failing to answer the question.

Chunking controls the unit of retrieval. Small chunks can lose context; large chunks can include distracting material and increase context usage. Overlap preserves some continuity across chunk boundaries but does not guarantee that every answer remains wholly inside one chunk. Metadata such as source, section, date, category, tenant, or version supports filtering and provenance.

### A general RAG pipeline

```text
Documents → extraction → chunking → indexing
Question → query representation → retrieval → evidence context → generation → response
```

The following strategies change how this pipeline retrieves, makes decisions, corrects failures, or represents knowledge. Their boundaries overlap: a single application may combine hybrid retrieval, agentic planning, and corrective validation.

## 2. Simple RAG / Normal RAG / Vanilla RAG

### Definition

**Simple RAG** is a straightforward retrieve-then-generate workflow in which the system performs one main retrieval operation for a question and generates an answer using the retrieved context. In its basic form, it does not decompose the question or repeatedly revise the retrieval query.

```text
Question → retrieve top-k passages → construct context → generate answer
```

Citation formatting, relevance checks, or an answer-validation guard can be added without changing the basic one-pass retrieval pattern. If the application automatically retries retrieval after failure, that correction is an extension beyond the plain one-pass strategy.

### Suitable questions

Simple RAG works well when the question is clear and one or a small number of passages contain the answer. Example: **“What is ChaiTech's annual learning budget?”** The learning policy supplies the amount, time period, and approval conditions.

### Strengths and limitations

- Relatively simple to implement, inspect, and benchmark.
- Often uses fewer retrieval/model operations than iterative strategies.
- Can fail when wording mismatches the source, the top results are irrelevant, or the question spans several independent facts.
- A fluent answer is possible even with weak context unless the application provides appropriate constraints and validation.

### Implementation in this repository

The `SimpleRAG` subgraph performs one retrieval, grades evidence, generates a draft, and checks grounding/completeness. It does not rewrite or retrieve again internally. If evidence or answer validation fails, the **parent router workflow escalates to the separate CRAG subgraph**. Responses report both the initial and final strategy so this extension is visible.

## 3. Hybrid RAG

### Definition

**Hybrid RAG** combines complementary retrieval methods—commonly dense semantic retrieval and sparse lexical retrieval—before generating an answer. Dense retrieval can match related meanings; lexical methods such as BM25 can emphasize exact terms, names, identifiers, and rare words.

```text
                    → dense retrieval →
Question                              fusion → optional reranking → generation
                    → lexical search →
```

Hybrid retrieval concerns how evidence is found. It does not inherently imply iterative planning, self-correction, or a particular generation model.

### Fusion and reranking

Dense and lexical scores generally have different scales. A naive average can be misleading without an appropriate normalization/calibration method. Rank-based fusion is one alternative.

For Reciprocal Rank Fusion:

```text
RRF(document) = sum over retrieval lists of 1 / (k + rank(document))
```

Ranks are one-based; a document absent from a list contributes zero for that list. `k` is a rank-smoothing constant, not a confidence percentage. Weighted fusion is another option, with weights and score normalization validated on representative queries.

A reranker then evaluates the relationship between a query and candidate passages more directly. Reranking is distinct from fusion: fusion combines retrieval outputs; reranking reorders candidates using another relevance assessment.

### Example and tradeoffs

“Upskilling allowance” may benefit from semantic retrieval when the document says “learning budget.” An exact policy code may benefit from lexical matching. Combining the two can improve coverage, but introduces additional infrastructure, tuning, and duplicate-management concerns. It does not guarantee a better result on every query.

### Repository status

Hybrid RAG is explained here for conceptual completeness. **This repository uses dense Pinecone retrieval and does not implement a BM25 branch or RRF.** The earlier capstone used those techniques; this assignment implements three different strategies described below without relabeling dense search as hybrid search.

## 4. Agentic RAG

### Definition

**Agentic RAG** is a family of architectures in which a model or decision policy chooses retrieval-related actions based on the question, context, and intermediate observations. These actions may include selecting a knowledge source, deciding whether to retrieve, planning subquestions, calling tools, evaluating results, and deciding whether further evidence is needed.

```text
Goal → decide action → call permitted tool → inspect observation
                    ↑                         ↓
                    └──── continue / revise ──┘
                          or produce response
```

Agentic behavior is a matter of actual decision-making and feedback. Merely putting a fixed chain inside a class named `Agent` does not create autonomous planning. Tool boundaries, allowed actions, stopping conditions, and resource limits remain application responsibilities.

### Multi-step RAG as an agentic pattern

In multi-step RAG, the system decomposes a request into distinct evidence needs. Each subquestion is investigated, then the evidence is synthesized into one answer. Some implementations also allow later subquestions to depend on earlier factual observations. Such dependent reasoning needs careful propagation of verified facts and provenance.

Example: **“Compare the carry-forward rules for annual leave and learning budget.”** One evidence need concerns leave carry-forward limits and expiry; another concerns learning-budget rollover. A final answer must compare both.

### Strengths and limitations

- Supports comparisons, multi-document questions, and conditional research workflows.
- Can select different tools or sources where the application permits them.
- Introduces planning errors, accumulated reasoning errors, more calls, greater latency, and more opportunities to mishandle untrusted retrieved text.
- Requires explicit tool permissions, bounded iterations, traceability, and evidence checks.

### Implementation in this repository

The root router makes a model-based strategy decision. The `MultiStepRAG` subgraph generates two to four complementary subquestions, retrieves and grades evidence for each, permits one corrective search per failed subquestion, deduplicates evidence, and synthesizes the result. It supports evidence collection across planned subquestions, **not unrestricted autonomous browsing or a knowledge-graph reasoning engine**. The initial plan is fixed during that execution; it is not repeatedly replanned from intermediate answers.

## 5. Corrective RAG (CRAG)

### Definition

**Corrective RAG** evaluates the adequacy of retrieved evidence and takes corrective action when the evidence is unreliable, irrelevant, or insufficient. The defining idea is a feedback loop between retrieval evaluation and corrective information gathering.

```text
Retrieve → evaluate documents → sufficient? → generate
                              ↓ no
                        corrective action → retrieve/evaluate again
```

Corrective actions can include query rewriting, broader or alternative retrieval, different sources, filtering/refining evidence, or external search where permitted. The original CRAG research discusses retrieval evaluation and corrective retrieval actions including broader information gathering; a course implementation based on bounded query rewriting is a practical subset, not a claim to reproduce every research component.

### Relevance versus sufficiency

**Relevance** asks whether a passage pertains to the information need. **Sufficiency** asks whether the selected evidence can support an answer to the requested question. A relevant leave document may not provide a parental-leave entitlement. Several related passages can still be insufficient.

### Example

The question “What is the skill-up money thing?” may require a rewrite toward “annual learning budget and reimbursement policy.” Another query may initially retrieve unrelated ChaiTech policies. The grader's observed failure then justifies correction. The router cannot know the actual quality of unseen search results merely from the question text.

### Retry and failure rules

CRAG must have stopping criteria, such as a maximum number of corrective retrieval attempts. Otherwise it can repeatedly ask for information absent from the corpus. Exhausted correction should result in a clear limitation, clarification request, or permitted fallback—not an invented answer.

### Implementation in this repository

`CorrectiveRAG` rewrites the query using the original goal and actual failure feedback, retrieves from the same permitted Pinecone corpus/source scope, grades the documents, and generates/validates an answer when possible. By default it allows **two corrective retrieval attempts**. If Simple RAG ran first, its initial retrieval is additional, so a Simple-to-CRAG request can make three searches by default.

The project does not broaden a user-selected source filter or add web search. An impossible filter can therefore lead to a correct abstention rather than recovery. A corrected query cannot create a missing policy.

## 6. Self-RAG

### Definition

**Self-RAG (Self-Reflective Retrieval-Augmented Generation)** is a research approach that teaches a language model to control retrieval and assess generated content using reflection-related decisions. The original method uses a specially trained model and reflection signals concerning retrieval need, relevance, support, and usefulness.

In practical application discussions, “self-reflective RAG” is also used more broadly for systems that inspect generated answers, check their support and completeness, and decide whether to regenerate or retrieve again. This broader usage should be distinguished from a faithful implementation of the original trained Self-RAG method.

### Questions a self-reflective workflow may ask

1. Is additional retrieval necessary?
2. Are retrieved passages relevant?
3. Is the answer supported by those passages?
4. Does the answer satisfy the user's request?
5. Should the answer be revised, another retrieval performed, or the system abstain?

### Grounding and completeness

**Grounding** measures whether claims are supported by the available evidence. **Completeness** measures whether the requested parts are addressed. “The budget is INR 20,000” can be grounded yet incomplete if the user also asked about approval and rollover. A complete-sounding response can be ungrounded if it invents one of those conditions.

Using the same LLM to generate and judge can produce correlated errors. Reflection prompts do not create an infallible verifier. Human review and external evaluation remain important.

### Repository status

All implemented strategies use a separate answer-check call for grounding and completeness. This is **application-level reflective validation**, not a separately implemented original Self-RAG model. The assignment's three implemented strategies remain Simple RAG, CRAG, and Multi-step RAG.

## 7. Multi-Query RAG

### Definition

**Multi-Query RAG** generates multiple alternative search formulations for the same underlying information need, retrieves for each formulation, and combines/deduplicates the results before generation. Its purpose is usually to reduce dependence on a single wording and increase evidence recall.

```text
Original question → alternative query formulations → multiple retrievals
                 → union/fusion + deduplication → optional reranking → answer
```

For one learning-budget question, alternatives might target “learning allowance,” “professional development budget,” and “course reimbursement policy.” These are alternate expressions of one goal.

### Difference from multi-step planning

Multi-query expansion usually searches the **same intent in different ways**. Multi-step planning breaks a request into **different information needs**. Asking separately about annual-leave carry-forward and learning-budget rollover is decomposition, not merely query paraphrasing.

Multi-query techniques can be used inside an agentic or corrective workflow. The categories are composable. A CRAG rewrite based on an observed failure is also not identical to generating several alternative queries in advance.

### Tradeoffs

More formulations can improve recall but also introduce topic drift, redundant passages, additional query cost, and increased context noise. Query count, fusion, deduplication, and final evidence selection need constraints.

### Repository status

The repository does not implement a separate parallel Multi-Query RAG strategy. Its multi-step planner produces complementary subquestions; its CRAG branch produces feedback-driven corrections. Naming those operations accurately is part of the design.

## 8. Graph RAG

### Definition

**Graph RAG** augments retrieval and generation with graph-structured knowledge. Nodes may represent entities, documents, concepts, or communities; edges represent relationships. Retrieval can exploit graph neighborhoods, paths, entity links, community summaries, and supporting source text.

There are multiple Graph RAG designs. Some emphasize entity-centric local retrieval; others create community summaries for broad questions across a corpus. Not every graph-based RAG system uses the same indexing or retrieval method.

```text
Documents → entities/relationships + provenance → graph/index
Question → relevant entities/communities/paths → supporting evidence → generation
```

### Example

For a corpus relating employees, teams, projects, vendors, and compliance obligations, a graph can help connect information distributed across documents. A question such as “Which teams depend on vendors affected by policy X?” may require relationship traversal beyond one similar text passage.

### Strengths and limitations

- Can support relationship-oriented and corpus-wide questions.
- Requires reliable entity/relationship extraction, identity resolution, provenance, and graph maintenance.
- Incorrect edges or entity merges can propagate errors.
- Graph summaries and traversed relationships still need source support; a graph is not automatically a truth database.

### Important distinction: LangGraph is not Graph RAG

**LangGraph models application control flow and state transitions.** It does not automatically construct a knowledge graph of document entities and relationships. A workflow diagram with retrieval nodes is therefore not evidence of Graph RAG.

This repository uses LangGraph to orchestrate the router and strategy workflows. It **does not implement knowledge-graph retrieval**, and Pinecone vectors alone are not presented as a knowledge graph.

## 9. Comparison summary

| Strategy | Main adaptation | Typical trigger | Main additional cost/complexity |
|---|---|---|---|
| Simple RAG | One retrieve/generate pass | Focused, well-supported question | Lowest workflow complexity |
| Hybrid RAG | Combine complementary retrievers | Exact terms plus semantic variation | Multiple indexes/searches and fusion |
| Agentic RAG | Choose actions/tools using observations | Complex or conditional information need | Planning, control, tool and stopping policies |
| CRAG | Correct poor retrieval using evaluation feedback | Irrelevant/insufficient retrieved evidence | Evaluation and repeated information gathering |
| Self-RAG | Reflection-driven retrieval/generation decisions | Need to assess support/usefulness | Specialized training in the original method; evaluator limits in application variants |
| Multi-Query RAG | Alternative formulations of one goal | Query wording limits recall | Several retrievals and result combination |
| Graph RAG | Retrieve through graph relationships/summaries | Relationship-centric or global questions | Graph construction, resolution, maintenance, provenance |

## 10. Intelligent routing policy in this application

The router receives the user's question and bounded recent history. It outputs a typed strategy, a resolved standalone question, a concise reason, and an optional clarification.

- Clear single-policy fact → Simple RAG.
- Noisy or confusing search terminology → CRAG may be selected initially.
- Comparison or several distinct policy requirements → Multi-step RAG.
- Observed poor evidence or failed answer validation on Simple RAG → automatic escalation to CRAG.
- Unresolved references or out-of-scope input → clarification without retrieval.

The initial route is an LLM decision, not a keyword switch. The post-retrieval escalation is grounded in an actual grader/check result. Neither is guaranteed correct; both are logged and tested. The UI distinguishes initial and final strategies so the recorded explanation matches execution.

## 11. Evaluation concepts

**Routing agreement** compares the chosen strategy with a labeled expectation or set of acceptable routes. **Retrieval relevance** concerns evidence usefulness. **Source recall** asks whether expected supporting sources were retrieved/cited. **Citation integrity** checks that source identifiers map to actual supplied evidence. **Grounding** and **completeness** are content-level properties; they are not equivalent to citation formatting.

Latency, retrieval count, model-call count, retries, token usage, abstention rate, and operational failures should be examined alongside answer quality. A strategy with more steps is not automatically better. A correct abstention can be preferable to an unsupported answer, while unnecessary abstention on answerable questions still indicates a quality problem.

The included demo runner checks expected routes/statuses, source filenames, citation markers, and required text patterns. These checks provide reproducible diagnostics; they are not a universal semantic-accuracy metric. Cloud execution and manual evidence review are still required before claiming real-world performance.

## 12. Common misconceptions to avoid

- Embeddings do not train the chat model on the documents.
- High vector similarity does not prove that a passage answers the question.
- Top-k is a retrieval count, not a confidence threshold.
- A citation marker does not prove that its accompanying sentence is supported.
- A query rewrite cannot supply a policy absent from the allowed corpus.
- More retrieval attempts can add cost and noise rather than improve the answer.
- Multi-query expansion and multi-step decomposition are different operations.
- LangGraph workflow orchestration is not knowledge-graph retrieval.
- Prompt-based answer checking is not a faithful implementation of the original trained Self-RAG model.
- A diagram or class name is not sufficient evidence that a strategy is implemented; inspect its distinct execution path.
