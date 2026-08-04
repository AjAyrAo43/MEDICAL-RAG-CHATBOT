# 🏥 Medical RAG Chatbot — Complete Interview Preparation Handbook
### Principal-Level · Staff-Level · Production-Level · Research-Level
> **Prepared as of July 2026 | Covers everything from Beginner → Research → Industry Trends**

---

# TABLE OF CONTENTS

1. [Section 1 — Resume Verification & Deep Challenge](#section-1)
2. [Section 2 — RAG Fundamentals](#section-2)
3. [Section 3 — Chunking (300+ Questions)](#section-3)
4. [Section 4 — Embeddings](#section-4)
5. [Section 5 — Vector Database & Pinecone](#section-5)
6. [Section 6 — Hybrid Search](#section-6)
7. [Section 7 — Cross Encoder Re-ranking (200+ Questions)](#section-7)
8. [Section 8 — Prompt Engineering](#section-8)
9. [Section 9 — Llama 3.3 70B Deep Dive](#section-9)
10. [Section 10 — FastAPI Deep Dive](#section-10)
11. [Section 11 — AWS Architecture](#section-11)
12. [Section 12 — Docker & CI/CD](#section-12)
13. [Section 13 — MLflow & MLOps](#section-13)
14. [Section 14 — RAG Evaluation (500+ Questions)](#section-14)
15. [Section 15 — Scenario-Based Questions (500+)](#section-15)
16. [Section 16 — Production Incidents](#section-16)
17. [Section 17 — Security & HIPAA](#section-17)
18. [Section 18 — Coding Interview Questions](#section-18)
19. [Section 19 — System Design Interview](#section-19)
20. [Section 20 — HR & Behavioral](#section-20)
21. [Section 21 — Live Industry Trends 2026](#section-21)

---

# SECTION 1 — RESUME VERIFICATION & DEEP CHALLENGE {#section-1}

## 1.1 "Production RAG" — Every Word Challenged

### Resume Verification Questions
1. What is your definition of "production"? How does your definition differ from a personal project or prototype?
2. Is your system currently serving real users or real medical professionals?
3. What is the SLA (Service Level Agreement) you committed to?
4. What is your p50, p95, p99 latency? How do you measure it?
5. What is your error rate? How do you define an error in a RAG context?
6. What is your uptime percentage? How do you achieve it?
7. How many concurrent users does the system support?
8. What is your daily request volume?
9. What is your monthly AWS bill? How do you optimize it?
10. What monitoring do you have in place?
11. What alerting do you have configured?
12. Who gets paged when the system goes down at 2 AM?
13. What is your mean time to recovery (MTTR)?
14. What is your mean time between failures (MTBF)?
15. Have you ever had a production incident? Walk me through it in detail.

### Architecture Questions
16. Draw the complete architecture of your production system on a whiteboard.
17. What are the critical paths in your system? Which component failure causes total outage?
18. Where are your single points of failure?
19. How do you handle partial failures gracefully?
20. What does your data pipeline look like end to end?
21. How does a request flow from a user query to the final response?
22. Where do you add observability at each step?
23. How do you version your deployed system?
24. What is your rollback procedure if a bad deployment goes out?
25. How do you handle schema migrations in Pinecone?

### Deep Implementation Questions
26. Show me the exact code structure of your RAG pipeline.
27. What framework did you use for orchestration — pure LangChain, custom pipeline, or hybrid?
28. How do you handle token limits across the retrieval + generation pipeline?
29. How do you handle concurrent requests in FastAPI?
30. What is your connection pooling strategy for external services?
31. How do you handle timeouts at each layer?
32. What retry logic do you implement?
33. Do you use circuit breakers? Where and why?
34. What caching strategy do you employ?
35. How do you validate user input before it enters the pipeline?

### Production Questions
36. What does your logging infrastructure look like?
37. What log aggregation tool do you use (CloudWatch, ELK, Datadog)?
38. How do you trace a single request end-to-end through your system?
39. What metrics do you push to CloudWatch or equivalent?
40. How do you handle spikes in traffic?
41. What is your warm-up strategy for cold starts?
42. How do you handle database connection limits?
43. What is your request queuing strategy under load?
44. How do you protect against runaway queries?
45. What is your data retention policy for logs and metrics?

### Research Questions
46. What papers influenced your production RAG design?
47. How does your production system differ from the original RAG paper (Lewis et al., 2020)?
48. Have you read "REALM" (Guu et al., 2020)? How does it compare?
49. What is the difference between open-domain QA RAG and your medical RAG?
50. What benchmarks (PubMedQA, MedQA, MIMIC-III) would you use to validate your system?

### Optimization Questions
51. What was your pre-optimization latency? Post-optimization?
52. What bottleneck did you identify first and how?
53. How do you profile an async FastAPI application?
54. What was your embedding throughput before and after optimization?
55. How do you batch embedding calls to reduce API costs?
56. Have you implemented request coalescing?
57. What is your cache hit rate and how did you improve it?
58. Have you tried quantized models for inference?
59. What is the cost per query, and how did you bring it down?
60. How do you measure and reduce cold start time?

### Cost Optimization Questions
61. What is your monthly Pinecone cost? How many vectors are stored?
62. What is your monthly Groq/inference API cost? How do you control it?
63. What is your AWS EC2 cost? What instance type? Why?
64. Have you considered spot instances? Why or why not?
65. What is your embedding model cost? How do you minimize re-embedding?
66. How do you decide when to scale up vs scale out?
67. What is your storage cost for raw documents?
68. Have you implemented any cost allocation tagging in AWS?
69. How do you forecast cost growth as user volume doubles?
70. What cost guardrails do you have in place to prevent runaway costs?

### Security Questions
71. How do you handle PHI (Protected Health Information) in your RAG pipeline?
72. Are you HIPAA compliant? Walk me through your BAA (Business Associate Agreement) situation with AWS.
73. How do you prevent prompt injection attacks?
74. How do you prevent users from extracting training data or vector store contents?
75. What authentication mechanism do you use?
76. How do you manage API secrets and keys?
77. What network security measures are in place (VPC, Security Groups)?
78. How do you encrypt data at rest and in transit?
79. Do you have any rate limiting? At what level?
80. How do you audit access to medical data?

### Failure Handling Questions
81. What happens when Pinecone is unavailable?
82. What happens when the LLM API rate limits you?
83. What happens when the embedding model times out?
84. What happens when FastAPI runs out of worker threads?
85. What happens when an EC2 instance terminates unexpectedly?
86. What happens when a Docker container crashes in production?
87. What happens when GitHub Actions pipeline fails mid-deploy?
88. How do you detect silent failures (where no exception is raised but output is wrong)?
89. What happens when a user uploads a corrupted PDF?
90. What happens when Pinecone returns zero results?

### Scaling Questions
91. How would you handle 10x the current load tomorrow?
92. What is your horizontal scaling strategy?
93. What is your vertical scaling strategy and when do you prefer it?
94. How does Pinecone handle scaling?
95. How do you scale the embedding computation layer?
96. What is the maximum Pinecone index size you tested with?
97. How does latency degrade as the vector count doubles?
98. How do you handle index rebuilds during scaling?
99. What is your database scaling strategy for metadata storage?
100. How do you scale CI/CD pipelines when multiple engineers commit simultaneously?

---

## 1.2 "Hybrid Search" — Every Word Challenged

101. Define hybrid search precisely. What exactly is hybrid in your system?
102. What are the two components of your hybrid search?
103. How do you combine dense and sparse scores? Walk me through the math.
104. What fusion strategy do you use — weighted sum, RRF, or something else?
105. How do you tune the alpha (balance) between dense and sparse?
106. How did you determine the optimal alpha value?
107. What happens if alpha = 0.0? Alpha = 1.0?
108. How does your hybrid search perform on medical terminology vs layman terms?
109. How do you evaluate whether hybrid is better than pure dense on your dataset?
110. What are the failure cases of hybrid search?

---

## 1.3 "Cross Encoder Re-ranking" — Every Word Challenged

111. Why did you add a cross encoder on top of hybrid search? Wasn't hybrid good enough?
112. How many candidates do you retrieve before re-ranking?
113. What is the latency added by re-ranking? Is it acceptable?
114. What cross encoder model did you use specifically?
115. How did you evaluate whether re-ranking improved your system?
116. What is the throughput of your re-ranker in queries per second?
117. Have you tried batch re-ranking to improve throughput?
118. What happens if the cross encoder re-ranks poorly?
119. How do you monitor re-ranking quality in production?
120. Is your cross encoder fine-tuned on medical data?

---

## 1.4 "End-to-End RAG Evaluation" — Every Word Challenged

121. What does "End-to-End" mean here? What does it NOT include?
122. What specific metrics do you measure?
123. Who labels the ground truth answers — humans or an LLM?
124. What is your evaluation dataset size?
125. How representative is your evaluation dataset of production traffic?
126. How often do you run evaluation? On every deploy?
127. What is your acceptance threshold for each metric?
128. What happens if a deploy fails evaluation — does the pipeline auto-reject it?
129. How do you handle distribution shift between your eval set and production?
130. Have you done A/B testing of different RAG configurations?

---

## 1.5 "Medical Question Answering" — Every Word Challenged

131. What types of medical questions does your system handle?
132. What is the source corpus of your medical knowledge base?
133. How do you ensure medical accuracy?
134. Do you add disclaimers to answers? How?
135. What happens when a user asks about drug dosages or treatment recommendations?
136. Is your system HIPAA compliant? Walk me through all HIPAA safeguards.
137. How do you handle out-of-distribution medical questions?
138. What is your hallucination rate on medical questions?
139. How do you handle conflicting information in your medical corpus?
140. What is the difference between clinical NLP challenges and general NLP challenges?

---

# SECTION 2 — RAG FUNDAMENTALS {#section-2}

## 2.1 What Is RAG?

**Beginner**
1. What does RAG stand for?
2. Explain RAG to a non-technical person.
3. What problem does RAG solve?
4. What are the three core stages of RAG?
5. What is a retriever? What is a generator?

**Intermediate**
6. How does RAG differ from a standard LLM prompt?
7. What is "grounding" in the context of RAG?
8. How does RAG reduce hallucination?
9. What is "open-domain QA" and how does RAG enable it?
10. What is the difference between extractive QA and abstractive QA?

**Advanced**
11. What are the failure modes of RAG?
12. What is retrieval failure vs. generation failure?
13. How do you diagnose whether a wrong answer is a retrieval problem or a generation problem?
14. What is the "lost in the middle" problem in RAG?
15. How does chunk ordering in the prompt affect LLM response quality?

**Staff-Level**
16. How do you design a RAG system for a domain where ground truth labels are unavailable?
17. How do you handle distributional shift in retrieval (when new document types are added)?
18. What is the tension between precision and recall in RAG and how do you balance it?
19. How do you design RAG for real-time document updates?
20. What organizational infrastructure do you need to run RAG at scale?

**Research-Level**
21. Explain the original RAG paper (Lewis et al., 2020) — what did it introduce?
22. What is REALM (Retrieval-Enhanced Language Model Pre-training) and how does it differ?
23. What is FiD (Fusion in Decoder) and how does it improve multi-document RAG?
24. What is kNN-LM and how does it relate to RAG?
25. What is "Atlas" (Izacard et al., 2022) and what did it contribute to RAG research?

## 2.2 Why RAG vs. Alternatives?

**Why Not Fine-Tuning?**
26. What is fine-tuning and why is it NOT sufficient for knowledge-intensive tasks?
27. What is catastrophic forgetting and how does it affect fine-tuned models?
28. How do you decide between RAG and fine-tuning?
29. What is the cost of fine-tuning a 70B model vs. building a RAG system?
30. Can you use RAG and fine-tuning together? When?
31. What is knowledge memorization in fine-tuned models? How does it compare to RAG?
32. Fine-tuned models can't update knowledge without retraining — why is this a production problem?
33. What is the "frozen knowledge" problem?
34. When would you choose fine-tuning over RAG despite its limitations?
35. How do LoRA/QLoRA change the fine-tuning vs RAG trade-off?

**Why Not Full-Context Prompting?**
36. What is context stuffing? Why doesn't it work at scale?
37. What is the context window limit of Llama 3.3 70B?
38. What happens at the edges of the context window (attention degradation)?
39. What is the "lost in the middle" phenomenon?
40. What is the cost per request if you use 128K tokens every time vs. targeted retrieval?
41. How does latency scale with context length for transformer models?
42. Why do long contexts increase hallucination risk?
43. What is position bias in transformer attention?
44. What is the "needle in a haystack" test and what does it reveal about long-context models?
45. When would you prefer full-context prompting over RAG?

**Why Not SQL Database?**
46. What is the fundamental difference between semantic search and exact lookup?
47. Can you do fuzzy search in SQL? How does it compare to vector search?
48. What is the complexity of a full-text search in PostgreSQL vs. ANN in Pinecone?
49. What type of queries does a vector database excel at that SQL cannot?
50. Can you combine SQL and vector search? How?

**Why Not Knowledge Graph?**
51. What is a knowledge graph and what types of queries does it excel at?
52. When would a medical knowledge graph (like SNOMED CT or ICD-10 graph) outperform RAG?
53. What is GraphRAG and how does it combine both?
54. What is the construction cost of a medical knowledge graph vs. a vector index?
55. What are the limitations of knowledge graphs for free-text medical literature?

**Why Vector Search?**
56. What is semantic similarity and why does vector search capture it?
57. What is an embedding space and why does it encode meaning?
58. Why is cosine similarity the preferred metric for text embeddings?
59. What is approximate nearest neighbor (ANN) search and why is it necessary at scale?
60. What is the curse of dimensionality and how does it affect vector search?

---

# SECTION 3 — CHUNKING (300+ QUESTIONS) {#section-3}

## 3.1 Foundational Chunking Questions

1. What is chunking in the context of RAG?
2. Why do we chunk documents instead of indexing the whole document?
3. What happens if your chunk size is too small?
4. What happens if your chunk size is too large?
5. What is the fundamental trade-off in chunk size selection?
6. How does chunk size affect retrieval precision?
7. How does chunk size affect retrieval recall?
8. How does chunk size affect the LLM's context utilization?
9. How does chunk size affect embedding quality?
10. How does chunk size affect hallucination rates?
11. What is chunk overlap? Why is it used?
12. What happens if overlap is 0?
13. What happens if overlap equals chunk size?
14. How does overlap affect storage cost in Pinecone?
15. How does overlap affect recall for boundary-spanning concepts?
16. Explain why 512 tokens is a common chunk size choice.
17. Why not 256 tokens?
18. Why not 1024 tokens?
19. Why not 2048 tokens?
20. What is the relationship between chunk size and embedding model token limit?
21. What is the typical BERT-based model token limit and why does it matter for chunking?
22. How does chunk size interact with the LLM context window?
23. What is a "semantic unit" and why should chunks respect it?
24. What is the difference between a token and a word in chunking?
25. Why do we measure chunk size in tokens rather than characters or words?

## 3.2 Fixed Chunking

26. Define fixed chunking. What is the algorithm?
27. What are the advantages of fixed chunking?
28. What are the disadvantages of fixed chunking?
29. When does fixed chunking fail?
30. What happens when a sentence is split mid-way by fixed chunking?
31. What is the performance overhead of fixed chunking?
32. How do you implement fixed chunking in Python? Write the code.
33. Does fixed chunking preserve document structure? Why or why not?
34. How does fixed chunking handle different languages?
35. What is the recommended overlap for fixed chunking and why?

## 3.3 Recursive Chunking

36. What is recursive chunking?
37. What are the recursive separators in LangChain's RecursiveCharacterTextSplitter?
38. Why does LangChain try `\n\n`, then `\n`, then ` `, then `""`?
39. What does "recursive" mean in this context?
40. How does recursive chunking handle markdown differently from plain text?
41. What is the advantage of recursive chunking over fixed chunking?
42. How does recursive chunking affect chunk size consistency?
43. What happens when none of the separators are found in the text?
44. What is the computational complexity of recursive chunking?
45. When would you prefer recursive chunking for medical documents?

## 3.4 Semantic Chunking

46. What is semantic chunking?
47. How does semantic chunking differ from syntactic chunking?
48. What algorithm does semantic chunking use to find split points?
49. What embedding model is used internally in semantic chunking?
50. What is the cosine distance threshold for splitting in semantic chunking?
51. How do you tune the similarity threshold for semantic chunking?
52. What is the computational cost of semantic chunking vs. fixed chunking?
53. Can you use semantic chunking at ingestion time at scale? What are the bottlenecks?
54. How do you evaluate whether semantic chunking produces better chunks?
55. What is the latency of semantic chunking for a 100-page medical PDF?
56. What happens if two semantically similar but factually different sections are merged?
57. How does semantic chunking handle medical texts with dense technical terminology?
58. What research paper introduced semantic chunking?
59. What is the difference between NLTK sentence tokenizer and semantic chunker?
60. How does semantic chunking affect downstream retrieval NDCG?

## 3.5 Sentence Chunking

61. What is sentence chunking?
62. What tokenizer do you use for sentence splitting in medical text?
63. Why is sentence chunking challenging for medical text? Give specific examples.
64. How do abbreviations (Dr., mg., i.v.) break sentence tokenizers?
65. What is the difference between NLTK `sent_tokenize` and spaCy sentence segmentation?
66. Why might spaCy outperform NLTK for medical sentence chunking?
67. What is `scispaCy` and when would you use it over standard spaCy?
68. How do you handle very long sentences in medical text?
69. How do you handle very short sentences (single words, headers)?
70. What is sentence-window retrieval and how does it combine sentence chunking with context?

## 3.6 Paragraph & Sliding Window Chunking

71. What is paragraph chunking?
72. How do you detect paragraph boundaries in a plain text medical document?
73. How do you detect paragraph boundaries in a PDF?
74. What is the problem with paragraph chunking for very long paragraphs?
75. What is sliding window chunking?
76. What is the mathematical relationship between chunks and windows in sliding window?
77. How does sliding window compare to fixed chunking with overlap?
78. What is the computational cost of sliding window chunking?
79. When does sliding window chunking help most?
80. What is the storage overhead of sliding window chunking?

## 3.7 Hierarchical Chunking

81. What is hierarchical chunking?
82. What are "parent" and "child" chunks in hierarchical chunking?
83. How does parent-child retrieval work in RAG?
84. What is the typical child chunk size and parent chunk size?
85. Why is hierarchical chunking beneficial for medical documents?
86. How do you implement parent-child retrieval in LangChain?
87. What is the `ParentDocumentRetriever` in LangChain?
88. What is the document store used to store parent documents?
89. How does hierarchical chunking affect Pinecone storage?
90. What are the failure cases of hierarchical chunking?
91. How does RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) extend hierarchical chunking?
92. What is the RAPTOR algorithm step by step?
93. What clustering algorithm does RAPTOR use?
94. How does RAPTOR handle multi-level document hierarchies?
95. What is the cost of building a RAPTOR index?

## 3.8 Markdown & Structured Document Chunking

96. How do you chunk a markdown document while respecting headers?
97. What is `MarkdownHeaderTextSplitter` in LangChain?
98. How do you preserve header metadata in each chunk?
99. Why is metadata-enriched chunking important for retrieval?
100. How do you handle markdown tables during chunking?
101. What is the best strategy for chunking numbered lists in medical guidelines?
102. How do you handle markdown code blocks in medical documentation?
103. What happens if a header spans multiple pages?
104. How do you represent document hierarchy in chunk metadata?
105. Why does metadata filtering during retrieval depend on chunking quality?

## 3.9 PDF Chunking

106. What are the different methods to extract text from a PDF?
107. What is the difference between a native PDF and a scanned PDF?
108. What libraries do you use for PDF text extraction? (PyPDF2, pdfplumber, pypdf, PyMuPDF)
109. Why is PyMuPDF (fitz) considered superior to PyPDF2 for complex PDFs?
110. How do you handle PDF column layouts in medical journals?
111. How do you handle page numbers, headers, and footers in PDF chunking?
112. How do you handle footnotes in medical PDFs?
113. How do you handle citations and references sections?
114. What is the difference between text-based PDF extraction and OCR?
115. How do you detect whether a PDF requires OCR?
116. How do you handle mixed-content PDFs (some pages text, some scanned)?
117. What is the role of bounding boxes in PDF text extraction?
118. How do you handle multi-column academic medical papers?
119. How do you preserve reading order in multi-column PDFs?
120. What tools do you use for layout analysis of medical PDFs?

## 3.10 OCR & Scanned PDF Chunking

121. What is OCR? How does it work internally?
122. What is Tesseract and how does it perform on medical text?
123. What is AWS Textract and how does it compare to Tesseract?
124. What is the accuracy of modern OCR on medical handwriting?
125. How do you handle OCR errors in the retrieval pipeline?
126. What is the impact of OCR errors on embedding quality?
127. How do you post-process OCR output to fix common medical terminology errors?
128. What is the pipeline from scanned PDF → text → chunks → embeddings?
129. What is deskewing and why is it important for OCR preprocessing?
130. What is binarization in OCR preprocessing?
131. How do you handle low-quality scans with noise?
132. What is confidence scoring in OCR and how do you use it?
133. How do you handle mixed-language OCR (English + Latin medical terms)?
134. What is the cost of AWS Textract per page vs. open-source OCR?
135. How do you parallelize OCR for a large batch of scanned PDFs?

## 3.11 Table Chunking

136. What is the challenge of chunking tables in medical documents?
137. How does table structure get lost during naive text extraction?
138. What is the best strategy for representing a table as text for embedding?
139. How does serializing a table (row-by-row) compare to embedding the raw HTML?
140. What is the challenge of tables with merged cells?
141. How do you handle tables that span multiple pages?
142. What is the role of table captions and headers in chunking?
143. Should tables be chunked at all, or indexed as complete units?
144. What is the difference between indexing a table and indexing a summary of a table?
145. How do you handle drug interaction tables in medical literature?

## 3.12 Image Chunking (Multimodal)

146. What is image chunking in the context of RAG?
147. What is a multimodal RAG pipeline?
148. How do you embed medical images (X-rays, MRI scans)?
149. What model do you use for medical image embeddings? CLIP? BioViL?
150. How do you chunk a document that contains figures with captions?
151. What is the difference between embedding the caption vs. embedding the image itself?
152. How does ColPali (column-level patch interaction) change image retrieval?
153. What is GPT-4o's role in multimodal RAG?
154. How do you handle diagrams and flowcharts in medical guidelines?
155. What is the future of multimodal RAG in medical AI?

## 3.13 Metadata-Aware Chunking

156. What metadata should you attach to each chunk in a medical RAG system?
157. List 15 metadata fields for medical document chunks.
158. How does metadata enable filtered retrieval?
159. What is the trade-off between rich metadata and storage cost in Pinecone?
160. How do you extract metadata automatically from a PDF?
161. How do you handle metadata that is missing or ambiguous?
162. How do you use publication date metadata to filter outdated medical guidelines?
163. How do you filter by medical specialty in retrieval?
164. How does Pinecone metadata filtering work under the hood?
165. What is the performance impact of metadata filters on ANN search?

## 3.14 Advanced & Experimental Chunking

166. What is "late chunking"?
167. How does late chunking leverage full-document embeddings before splitting?
168. What is contextual retrieval (Anthropic, 2024)?
169. How does contextual retrieval prepend chunk-level context using an LLM?
170. What is the cost of contextual retrieval at indexing time?
171. Is contextual retrieval worth the cost? Under what conditions?
172. What is proposition chunking?
173. How does proposition chunking convert text into atomic factual statements?
174. What is the advantage of proposition chunks for precise factual retrieval?
175. What is the disadvantage of proposition chunking?
176. How does document-level vs. chunk-level embedding affect retrieval?
177. What is multi-vector retrieval?
178. How does ColBERT's multi-vector approach differ from single-vector approaches?
179. What is the storage cost of ColBERT embeddings vs. single-vector embeddings?
180. What is the query-time latency of ColBERT vs. single-vector?

## 3.15 Chunking Benchmarking & Production

181. How do you benchmark different chunking strategies?
182. What metric do you use to evaluate chunking quality?
183. What is "chunk coherence" as a metric?
184. What is "retrieval recall at K" and how does chunking affect it?
185. How do you create a chunking evaluation dataset?
186. How do you A/B test chunking strategies in production?
187. How do you monitor chunking quality over time as new documents are added?
188. What is chunk drift and how do you detect it?
189. How do you handle chunking pipeline failures for corrupt documents?
190. What is your chunking throughput (documents per second) at production scale?
191. How do you scale chunking horizontally?
192. What message queue do you use for distributed chunking?
193. How do you deduplicate chunks?
194. What is the impact of duplicate chunks on retrieval quality?
195. How do you detect near-duplicate chunks?

## 3.16 Mathematical Deep Dive on Chunking

196. If a document has N tokens and chunk size is C with overlap O, how many chunks are produced? Write the formula.
197. **Answer**: `num_chunks = ceil((N - O) / (C - O))` — explain why.
198. How does overlap affect the total tokens stored in Pinecone?
199. **Answer**: `total_tokens = num_chunks × C` — derive the cost function.
200. What is the relationship between chunk size and retrieval latency?
201. How does chunk size affect the embedding similarity score distribution?
202. What is information entropy of a chunk and how does it relate to embedding quality?
203. Prove mathematically why smaller chunks have higher variance in cosine similarity scores.
204. What is the Zipf distribution and how does it apply to medical document token distributions?
205. What is TF-IDF per chunk and how does it relate to BM25 chunk scoring?

## 3.17 Dynamic & Adaptive Chunking

206. What is adaptive chunking?
207. How do you dynamically choose chunk size based on document type?
208. How do you detect document type automatically?
209. What ML model would you use to classify document layout?
210. What is the benefit of using different chunk sizes for different document sections?
211. How do you handle a pipeline that processes 50+ different document formats?
212. What is a chunking configuration registry?
213. How do you update chunking configs without reindexing everything?
214. What is incremental re-chunking and when is it necessary?
215. How do you handle chunking for streaming document ingestion?

## 3.18 Multilingual Chunking

216. How do you chunk medical documents in multiple languages?
217. What is the difference between subword tokenization across languages?
218. How does BPE tokenization handle Arabic or Chinese medical text?
219. What is the challenge of Chinese medical text chunking (no spaces)?
220. What is `jieba` and how does it help?
221. How do multilingual embedding models handle cross-lingual chunks?
222. What is the LaBSE model and how does it compare to mBERT for multilingual chunking?
223. How do you handle medical abbreviations that differ across languages?
224. How does romanized medical terminology affect chunking in non-Latin scripts?
225. What is the production strategy for a global medical RAG system serving 50+ languages?

## 3.19 LangChain Chunking Implementation

226. What are all the text splitters available in LangChain?
227. What is `CharacterTextSplitter` and when do you use it?
228. What is `RecursiveCharacterTextSplitter` and when do you use it?
229. What is `TokenTextSplitter` and why is it different from character splitters?
230. Which tokenizer does `TokenTextSplitter` use by default?
231. How does `length_function` parameter work in LangChain splitters?
232. What is `keep_separator` parameter in LangChain splitters?
233. How do you implement a custom text splitter in LangChain?
234. What is `HTMLHeaderTextSplitter` and when do you use it for medical content?
235. How do you chain multiple splitters in LangChain?
236. How do you implement a pipeline that chunks 10,000 medical PDFs overnight?
237. What logging do you add to your chunking pipeline?
238. How do you checkpoint progress in a long-running chunking job?
239. What happens to in-progress chunking if the process dies?
240. How do you resume a failed chunking job?

## 3.20 Chunking Anti-Patterns

241. What is "chunking too small" anti-pattern? What does it cause?
242. What is "chunking too large" anti-pattern?
243. What is "no overlap" anti-pattern?
244. What is "ignoring document structure" anti-pattern?
245. What is "not cleaning text before chunking" anti-pattern?
246. What is "not removing boilerplate" anti-pattern?
247. What is "using character count instead of token count" anti-pattern?
248. What is "not versioning your chunking strategy" anti-pattern?
249. What is "chunking without metadata" anti-pattern?
250. What is "indexing duplicate chunks" anti-pattern?

## 3.21 Chunking Edge Cases

251. What happens when a medical document is a single continuous paragraph?
252. What happens when a document has a 5000-token table?
253. What happens when a section has only one sentence?
254. What happens when a document has no punctuation?
255. What happens with legal medical disclaimers (repeated boilerplate)?
256. How do you handle case reports that mix narrative and clinical data?
257. How do you handle clinical trial protocols with numbered sections?
258. How do you handle a PDF where text extraction produces garbled text?
259. How do you handle documents with embedded watermarks?
260. How do you handle encrypted PDFs?
261. What happens when two documents have identical section titles?
262. How do you chunk a 500-page cardiology textbook?
263. How do you handle a document with 90% figures and 10% text?
264. What happens when chunk boundaries split drug dosage information?
265. How do you detect when a critical piece of information is split across two chunks?

## 3.22 Production Chunking Architecture

266. Describe your complete production chunking pipeline.
267. What tools do you use for document ingestion?
268. How do you parallelize document processing across multiple workers?
269. What message queue (SQS, Kafka, Celery) do you use for chunking jobs?
270. How do you handle document updates — do you re-chunk the whole document?
271. How do you implement incremental indexing (only index new/changed documents)?
272. How do you detect document changes?
273. What is your strategy for document versioning in the vector store?
274. How do you delete old chunks when a document is updated?
275. How do you track which chunks came from which documents?
276. What is your chunking pipeline SLA? What is the maximum latency for a new document to appear in search?
277. How do you monitor chunking pipeline health?
278. What alerts do you configure for chunking pipeline failures?
279. How do you audit chunking quality over time?
280. What is your disaster recovery plan for chunking pipeline failure?

## 3.23 Research-Level Chunking Questions

281. What does the paper "Dense Passage Retrieval for Open-Domain Question Answering" (DPR) say about optimal chunk size?
282. What does the BEIR benchmark reveal about chunking strategies?
283. What is the relationship between chunk size and retrieval performance across different domains?
284. What is "contextual compression" in LangChain and how does it relate to chunking?
285. What is LLMLingua and how does it compress context windows?
286. How does LLMLingua affect RAG if applied after retrieval?
287. What is RECOMP (Abstractive Compression for RAG) and how does it work?
288. What is selective context and how does it reduce context window usage?
289. How does chunking interact with chain-of-thought reasoning in the generator?
290. What is the optimal chunking strategy for a LLM with a 1M token context window?
291. If context windows become unlimited, does chunking become irrelevant? Why or why not?
292. What is "chunk-aware fine-tuning" and when is it useful?
293. What is the difference between "chunk retrieval" and "token retrieval"?
294. What is GNN-based document chunking?
295. How does document graph structure inform chunking decisions?
296. What is "entailment-based chunking"?
297. How do you evaluate chunking quality using Natural Language Inference?
298. What is the paper "Lost in the Middle" and what does it reveal about chunk ordering?
299. What is "positional attention bias" and how does it affect which chunks the LLM uses?
300. How would you design a chunking system that learns the optimal strategy from retrieval feedback?

---

# SECTION 4 — EMBEDDINGS {#section-4}

## 4.1 Embedding Fundamentals

1. What is an embedding?
2. What is an embedding space?
3. What does it mean for two embeddings to be similar?
4. What is a dense embedding vs. a sparse embedding?
5. What is the difference between word embeddings, sentence embeddings, and document embeddings?
6. What is the difference between encoder-only, decoder-only, and encoder-decoder models for embeddings?
7. Why are BERT-style encoder models preferred for embeddings?
8. What is mean pooling vs. CLS pooling?
9. Which pooling strategy is better for retrieval? Why?
10. What is the role of the `[CLS]` token in BERT?

## 4.2 Embedding Models

11. What embedding model did you use in your project? Why?
12. How did you benchmark your embedding model choice?
13. What is BGE (BAAI General Embeddings)? Who created it?
14. What is E5 (Embeddings from Bidirectional Encoders)? How does it differ from BGE?
15. What is the MTEB (Massive Text Embedding Benchmark)? What does it measure?
16. What is the current MTEB leaderboard top-performer as of 2026?
17. What is the difference between `bge-small`, `bge-base`, and `bge-large`?
18. What is `bge-m3`? What is unique about it?
19. What is the `instructor` embedding model? Why is it unique?
20. What is the key innovation of `instructor` — instruction-following embeddings?
21. What is OpenAI `text-embedding-3-small` vs. `text-embedding-3-large`?
22. What is Cohere Embed v3? What are its key features?
23. What is Voyage AI's embedding offering? When would you use it?
24. What is Jina AI's embedding model? What is the maximum sequence length?
25. What is GTE (General Text Embeddings)?
26. What is the difference between sentence-transformers library and direct HuggingFace usage?
27. How do you load a sentence-transformer model?
28. What is the `all-MiniLM-L6-v2` model and where would you use it?
29. What is `multi-qa-mpnet-base-dot-v1` and when would you use it over cosine models?
30. What is the difference between bi-encoder and cross-encoder? (Revisited for embeddings)

## 4.3 Medical Embeddings

31. What is PubMedBERT and how does it differ from general BERT?
32. What is BioBERT? How was it trained?
33. What is ClinicalBERT? What corpus was it trained on?
34. What is BioLinkBERT?
35. What is GatorTron?
36. What is DRAGON (Domain-Robust Alignment for Generalizable Open-domain Retrieval)?
37. Why would a general-purpose embedding model underperform on medical text?
38. What is domain gap in embeddings?
39. How would you fine-tune an embedding model on medical Q&A pairs?
40. What training data would you use for medical embedding fine-tuning?
41. What is contrastive learning and how does it apply to embedding training?
42. What is the NT-Xent loss (InfoNCE) for contrastive embedding training?
43. What is the mathematical formulation of NT-Xent loss?
44. What are positive and negative pairs in contrastive training?
45. What is hard negative mining and why is it critical?

## 4.4 Embedding Mathematics

46. What is cosine similarity? Write the formula.
47. `cos(θ) = (A · B) / (||A|| × ||B||)` — explain every term.
48. Why is cosine similarity preferred over Euclidean distance for text?
49. When would Euclidean distance be preferred?
50. What is dot product similarity and how does it differ from cosine?
51. When is dot product > cosine similarity in retrieval quality?
52. What is normalized dot product? How does it relate to cosine?
53. What is the Pearson correlation coefficient and how does it relate to cosine similarity?
54. What is the triangle inequality in metric spaces and does cosine satisfy it?
55. Is cosine similarity a proper metric? Why or why not?
56. What is Manhattan distance and when might you use it for embeddings?
57. What is Hamming distance and how is it used for binary embeddings?
58. What is Jaccard similarity and how does it compare to cosine for sparse vectors?
59. What is the mathematical proof that cosine similarity is scale-invariant?
60. Why does normalizing vectors before dot product give cosine similarity?

## 4.5 Approximate Nearest Neighbor (ANN)

61. What is exact nearest neighbor search? Why is it not scalable?
62. What is the time complexity of brute-force nearest neighbor search?
63. What is ANN and what is the trade-off it makes?
64. What is the ANN recall@k metric?
65. What is the time complexity of HNSW search?
66. What is the space complexity of HNSW?
67. What is IVF (Inverted File Index)?
68. What is PQ (Product Quantization)?
69. What is HNSW + PQ and when is it used?
70. What is ScaNN (Scalable Nearest Neighbors, Google)?
71. What is FAISS and who made it?
72. What algorithms does FAISS implement?
73. What is DiskANN and when is it useful?
74. What is the recall vs. latency trade-off in ANN?
75. How do you tune HNSW parameters (M, ef_construction, ef_search)?

## 4.6 Embedding Dimensions, Storage & Latency

76. What is the embedding dimension of your chosen model?
77. What is the relationship between embedding dimension and ANN recall?
78. What is the relationship between embedding dimension and storage cost?
79. How much memory does one embedding of dimension 1536 take at float32?
80. **Answer**: `1536 × 4 bytes = 6144 bytes ≈ 6 KB` — calculate for 1M vectors.
81. What is Matryoshka Representation Learning (MRL)?
82. How does MRL allow you to truncate embedding dimensions without losing quality?
83. What is the OpenAI embedding model's support for MRL?
84. What is the trade-off of using lower-dimension embeddings?
85. How do you benchmark embedding latency? What tool do you use?
86. What is the p99 embedding latency of your chosen model on CPU vs. GPU?
87. What is the throughput (embeddings/second) for batch vs. single inference?
88. What is the optimal batch size for embedding inference?
89. How do you implement async batched embedding inference in Python?
90. How do you handle embedding API rate limits?

## 4.7 Embedding Normalization & Pooling

91. What is L2 normalization and why is it applied to embeddings before indexing?
92. How do you normalize a vector in numpy?
93. What is the effect of normalization on dot product search?
94. What is mean pooling mathematically?
95. What is max pooling and when is it used?
96. What is weighted mean pooling (attention-weighted)?
97. What is the CLS pooling approach and when does it work better?
98. What is last-token pooling (used in decoder-based LLMs for embeddings)?
99. How do decoder-based embedding models (LLM2Vec, NV-Embed) work?
100. What is the LLM2Vec paper? What is its key contribution?

## 4.8 Embedding Drift & Re-indexing

101. What is embedding drift?
102. When does embedding drift occur?
103. How do you detect embedding drift in production?
104. What happens if you upgrade your embedding model without re-indexing?
105. How do you implement a zero-downtime re-indexing strategy?
106. How long does re-indexing take for 1M documents?
107. What is the cost of re-indexing 1M documents with OpenAI embeddings?
108. How do you version your embedding model in Pinecone?
109. How do you A/B test a new embedding model against the current one in production?
110. What is your monitoring strategy for embedding quality over time?

## 4.9 Fine-Tuned vs. General Embeddings

111. When should you fine-tune an embedding model?
112. What data do you need to fine-tune an embedding model?
113. What is the minimum dataset size for embedding fine-tuning to be effective?
114. What is hard negative mining and how do you implement it?
115. What is the MNRL (MultipleNegativesRankingLoss) loss function?
116. How do you generate training pairs for medical embedding fine-tuning?
117. What is synthetic data generation for embedding training?
118. How do you evaluate a fine-tuned embedding model on medical retrieval?
119. What is the typical improvement (NDCG delta) from domain-specific fine-tuning?
120. What is the compute cost of fine-tuning a BGE-base model?

## 4.10 Sparse Embeddings & SPLADE

121. What are sparse embeddings?
122. What is SPLADE?
123. How does SPLADE differ from BM25?
124. What is the SPLADE loss function?
125. Why does SPLADE outperform BM25 on out-of-vocabulary queries?
126. What is the sparsity regularization in SPLADE?
127. How do you index SPLADE vectors in Pinecone?
128. What is Pinecone's native sparse-dense hybrid support?
129. What is the storage overhead of SPLADE vectors?
130. When would you use SPLADE over BM25 in a medical RAG system?

---

# SECTION 5 — VECTOR DATABASE & PINECONE {#section-5}

## 5.1 Pinecone Architecture & Fundamentals

1. What is Pinecone and who created it?
2. What is the core data structure Pinecone uses internally?
3. What ANN algorithm does Pinecone use?
4. What is the difference between Pinecone's serverless and pod-based infrastructure?
5. What is a Pinecone index?
6. What is a Pinecone namespace?
7. What is a Pinecone collection?
8. What is the difference between a namespace and a collection?
9. What is the maximum number of vectors per Pinecone index?
10. What is the maximum metadata size per vector?
11. What is the maximum number of metadata fields per vector?
12. What is the default embedding dimension limit in Pinecone?
13. What is the Pinecone API rate limit?
14. How does Pinecone handle concurrent writes?
15. What consistency model does Pinecone use (eventual vs. strong)?

## 5.2 Pinecone Operations

16. What is the `upsert` operation in Pinecone?
17. What happens if you upsert a vector with an existing ID?
18. What is the maximum batch size for upsert operations?
19. How do you implement batched upserts efficiently?
20. How do you delete vectors by ID?
21. How do you delete vectors by metadata filter?
22. How do you implement incremental indexing with Pinecone?
23. What is Pinecone's `fetch` operation?
24. What is Pinecone's `describe_index_stats` operation?
25. What information does `describe_index_stats` return?
26. How do you check if a specific vector exists in the index?
27. How do you retrieve a specific vector by ID?
28. What is Pinecone's query operation and what parameters does it take?
29. What is `top_k` in Pinecone queries?
30. What is `include_metadata` and why should you use it carefully?

## 5.3 Pinecone Metadata Filtering

31. What is metadata filtering in Pinecone?
32. What operators does Pinecone support in metadata filters?
33. What is the `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte` operator?
34. What is the `$in` and `$nin` operator in Pinecone?
35. How do you filter by multiple conditions simultaneously?
36. What is the `$and` and `$or` operator in Pinecone?
37. How does metadata filtering affect ANN search performance?
38. What is the "selective filtering" challenge in Pinecone?
39. What happens when a metadata filter eliminates 99% of vectors?
40. How do you optimize queries with highly selective metadata filters?
41. What is the difference between pre-filtering and post-filtering?
42. Why does Pinecone use pre-filtering (integrated into ANN)?
43. What is the performance cost of metadata filtering on large indexes?
44. How do you index metadata fields in Pinecone?
45. What metadata types does Pinecone support?

## 5.4 HNSW Deep Dive

46. What is HNSW? What does the acronym stand for?
47. Explain HNSW graph structure at the intuition level.
48. What is a "navigable small world" graph?
49. What is the hierarchical layer structure in HNSW?
50. How does HNSW index construction work step by step?
51. What is the M parameter in HNSW?
52. What is the ef_construction parameter in HNSW?
53. What is the ef_search parameter in HNSW?
54. What is the space complexity of HNSW?
55. What is the time complexity of HNSW search?
56. How does HNSW compare to IVF in recall vs. latency?
57. What is the theoretical basis of "small world" networks?
58. What is the Watts-Strogatz small world model?
59. How does randomness in HNSW layer assignment affect graph quality?
60. What is the deletion problem in HNSW and how is it handled?

## 5.5 IVF & PQ

61. What is IVF (Inverted File Index)?
62. What is the Voronoi diagram and how does it relate to IVF?
63. How does k-means clustering create IVF centroids?
64. What is the `nlist` parameter in IVF?
65. What is the `nprobe` parameter in IVF?
66. How does `nprobe` affect recall vs. latency?
67. What is Product Quantization?
68. How does PQ compress vectors?
69. What is the quantization error in PQ?
70. What is the reconstruction error bound in PQ?
71. What is the relationship between PQ subvectors and recall?
72. What is IVFPQ? When is it preferred over HNSW?
73. What is Scalar Quantization?
74. What is Binary Quantization and when is it useful?
75. What is the trade-off of binary quantization for medical text embeddings?

## 5.6 Why Pinecone? Why Not Alternatives?

**Why Not Weaviate?**
76. What is Weaviate and what are its main features?
77. What is the HNSW implementation in Weaviate?
78. What is the difference between Weaviate's vectorizer modules and Pinecone's external embeddings?
79. What is Weaviate's GraphQL API?
80. What is Weaviate's multi-tenancy support?
81. When would you choose Weaviate over Pinecone?
82. What is Weaviate's performance on large-scale indexes?

**Why Not Milvus?**
83. What is Milvus and who maintains it?
84. What is Zilliz (the managed cloud version of Milvus)?
85. What index types does Milvus support?
86. What is Milvus's scalability model vs. Pinecone?
87. When would Milvus be preferred for a medical RAG system?
88. What is PyMilvus?

**Why Not Qdrant?**
89. What is Qdrant and what language is it written in?
90. What is the key technical innovation in Qdrant?
91. What is Qdrant's payload filtering?
92. What is sparse vector support in Qdrant?
93. What is Qdrant's multi-vector support?
94. What is Qdrant's binary quantization support?
95. Why has Qdrant become popular in 2025-2026?

**Why Not pgvector?**
96. What is pgvector?
97. What is the performance limitation of pgvector for large-scale retrieval?
98. What is `pg_embedding` and how does it differ from pgvector?
99. What is HNSW support in pgvector?
100. When is pgvector the right choice over Pinecone?
101. What is the advantage of keeping vectors and relational data in the same database?
102. What is the pgvector `ivfflat` index?
103. What are the limitations of pgvector at 10M+ vectors?
104. What is pgai?
105. What is the Neon vector database and how does it extend pgvector?

**Why Not Elasticsearch?**
106. What is Elasticsearch's approximate kNN search?
107. What is the HNSW implementation in Elasticsearch?
108. What is the advantage of Elasticsearch for hybrid search vs. Pinecone?
109. What is `text_expansion` query in Elasticsearch for SPLADE?
110. When would you choose Elasticsearch as your vector database?

## 5.7 Pinecone Production Operations

111. How do you back up a Pinecone index?
112. What is a Pinecone collection backup strategy?
113. How do you restore a Pinecone index from a collection?
114. What is the disaster recovery procedure for Pinecone failure?
115. How do you migrate between Pinecone regions?
116. How do you monitor Pinecone query latency in production?
117. What CloudWatch metrics do you track for Pinecone integration?
118. What is the Pinecone SDK version you used and what breaking changes have occurred?
119. How do you handle Pinecone API downtime?
120. What is your fallback strategy if Pinecone is unavailable?

## 5.8 Pinecone Cost Optimization

121. What is the Pinecone pricing model?
122. How do you calculate the cost of storing 1M vectors at dimension 1536?
123. What is the difference between serverless and pod-based Pinecone pricing?
124. How do you reduce storage cost while maintaining retrieval quality?
125. When should you use a pod-based index over serverless?
126. How does namespace usage affect billing?
127. What is the cost of 1M queries on Pinecone serverless?
128. How do you optimize query costs by reducing `top_k`?
129. What is the cost impact of including metadata in query results?
130. How do you budget Pinecone costs for a growing medical corpus?

---

# SECTION 6 — HYBRID SEARCH {#section-6}

## 6.1 BM25 Deep Dive

1. What is BM25? What does the acronym stand for?
2. Write the BM25 scoring formula.
3. `BM25(D, Q) = Σ IDF(qi) × [f(qi,D)(k1+1)] / [f(qi,D) + k1(1-b+b×|D|/avgdl)]`
4. Explain every term in the BM25 formula.
5. What is IDF in BM25? Write the formula.
6. What is TF saturation in BM25? Why is it important?
7. What is the k1 parameter in BM25? Typical values?
8. What is the b parameter in BM25? Typical values?
9. What does b=0 mean? What does b=1 mean?
10. What is the difference between BM25 and TF-IDF?
11. What is the mathematical limitation of BM25 for synonyms?
12. How does BM25 handle out-of-vocabulary terms?
13. Why does BM25 fail for medical acronyms?
14. What is BM25+ and how does it fix the zero-floor TF issue?
15. What is BM25L?

## 6.2 TF-IDF

16. Write the TF-IDF formula.
17. What is the raw TF formulation?
18. What is log-normalized TF?
19. What is augmented TF?
20. What is the standard IDF formula?
21. What is the smooth IDF formula?
22. Why is BM25 considered superior to TF-IDF for retrieval?
23. When might TF-IDF still be preferred?
24. How do you implement BM25 in Python using `rank_bm25`?
25. How do you implement BM25 in Elasticsearch?

## 6.3 Dense Retrieval

26. What is dense retrieval?
27. How does dense retrieval differ from sparse retrieval (BM25)?
28. What is DPR (Dense Passage Retrieval)? How was it trained?
29. What is the bi-encoder architecture for dense retrieval?
30. How do you train a DPR model?
31. What are positive and negative passages in DPR training?
32. What is in-batch negatives training for DPR?
33. What is the ANN search process in dense retrieval?
34. What is the latency of dense retrieval vs. BM25?
35. What is the recall difference between dense and sparse retrieval?

## 6.4 Hybrid Search Fusion

36. What is the motivation for hybrid search?
37. What is the complementarity between dense and sparse retrieval?
38. What is linear interpolation (weighted sum) for score fusion?
39. `hybrid_score = α × dense_score + (1-α) × sparse_score` — explain.
40. What is the challenge of different score scales between dense and sparse?
41. What is score normalization? What methods do you use?
42. What is min-max normalization for scores?
43. What is z-score normalization for scores?
44. What is rank-based normalization (RRF)?
45. What is Reciprocal Rank Fusion (RRF)?
46. Write the RRF formula: `RRF(d) = Σ 1/(k + rank_i(d))` — explain.
47. What is the k constant in RRF? Typical value?
48. Why is k=60 the commonly used default?
49. What is the advantage of RRF over score-based fusion?
50. What are the disadvantages of RRF?
51. How do you determine the optimal alpha for weighted hybrid search?
52. How do you tune alpha for medical text specifically?
53. What is the grid search approach for alpha tuning?
54. What is the Bayesian optimization approach for alpha tuning?
55. Can alpha be different for different query types?

## 6.5 Hybrid Search Production

56. How do you implement hybrid search in Pinecone?
57. What is Pinecone's native hybrid search support?
58. How do you implement BM25 indexing separately from Pinecone?
59. What is the architecture of a system with separate dense and sparse indexes?
60. How do you synchronize updates across dense and sparse indexes?
61. What is the latency overhead of hybrid search vs. pure dense?
62. How do you benchmark hybrid vs. dense vs. sparse retrieval on your corpus?
63. What metric do you use to evaluate retrieval quality?
64. How do you diagnose when hybrid is not better than pure dense?
65. What is the failure mode of hybrid search for medical long-tail queries?
66. How do you scale hybrid search to 100M+ documents?
67. What is the role of Elasticsearch in a hybrid search architecture?
68. What is Vespa and why is it used for hybrid search?
69. What is the future of hybrid search with SPLADE and ColBERT?
70. How does SPLADE improve over BM25 in hybrid search?

## 6.6 Mathematical Analysis of Hybrid Search

71. Prove that RRF is score-invariant (rankings don't depend on absolute scores).
72. What is the retrieval recall improvement from hybrid over dense-only on BEIR?
73. What is the theoretical justification for combining dense and sparse retrieval?
74. What is the Cranfield paradigm for IR evaluation?
75. What is the relationship between MAP (Mean Average Precision) and retrieval quality?
76. Derive NDCG formula: `NDCG@k = DCG@k / IDCG@k`.
77. Derive DCG: `DCG@k = Σ rel_i / log2(i+1)` — explain each term.
78. What is the difference between binary relevance and graded relevance in NDCG?
79. What is MRR (Mean Reciprocal Rank)? Write the formula.
80. When is MRR preferred over NDCG?

---

# SECTION 7 — CROSS ENCODER RE-RANKING (200+ QUESTIONS) {#section-7}

## 7.1 Bi-Encoder vs. Cross Encoder Fundamentals

1. What is a bi-encoder? Describe the architecture.
2. What is a cross encoder? Describe the architecture.
3. What is the fundamental difference in how query and document are processed?
4. In a bi-encoder, are query and document embeddings computed independently?
5. In a cross encoder, are they computed together?
6. Why is a cross encoder more accurate than a bi-encoder?
7. Why is a cross encoder slower than a bi-encoder?
8. What is the time complexity of bi-encoder retrieval at inference time?
9. What is the time complexity of cross encoder scoring?
10. Why can't a cross encoder be used for first-stage retrieval?
11. What is the ANN bottleneck in bi-encoder retrieval?
12. What is the interaction mechanism in a cross encoder?
13. How does attention across query+document tokens enable better relevance scoring?
14. What is the standard cross encoder input format?
15. `[CLS] query [SEP] document [SEP]` — explain each token.

## 7.2 Cross Encoder Architecture

16. What base model is typically used for cross encoders?
17. Why is BERT the standard choice for cross encoder base?
18. What is the cross encoder fine-tuning process?
19. What dataset is used to fine-tune MS MARCO cross encoders?
20. What is the MS MARCO dataset and why is it the standard for re-ranking training?
21. What is the training loss for cross encoder fine-tuning?
22. What is Binary Cross Entropy loss for cross encoder training?
23. What is pointwise vs. pairwise vs. listwise training for re-rankers?
24. What is the difference between a classification head and a regression head in cross encoders?
25. How do you implement cross encoder inference with batch processing?

## 7.3 Cross Encoder Models

26. What is `cross-encoder/ms-marco-MiniLM-L-6-v2`?
27. What is `cross-encoder/ms-marco-MiniLM-L-12-v2`?
28. What is the trade-off between L-6 and L-12 cross encoder?
29. What is `cross-encoder/ms-marco-electra-base`?
30. What is the BGE Reranker family of models?
31. What is `BAAI/bge-reranker-base`?
32. What is `BAAI/bge-reranker-large`?
33. What is `BAAI/bge-reranker-v2-m3`?
34. What is the improvement of bge-reranker-v2 over v1?
35. What is MonoT5 and how does it work?
36. What is the T5 architecture and why is it used for re-ranking?
37. How does MonoT5 frame re-ranking as text generation?
38. What is the input/output format of MonoT5?
39. What is DuoT5 and how does it compare to MonoT5?
40. What is RankT5?

## 7.4 Cohere Rerank

41. What is Cohere Rerank?
42. What is the Cohere Rerank 3.5 API?
43. What is the latency of Cohere Rerank for 100 candidates?
44. What is the cost of Cohere Rerank per 1000 queries?
45. How do you integrate Cohere Rerank with LangChain?
46. What is the maximum document length for Cohere Rerank?
47. How does Cohere Rerank handle multilingual re-ranking?
48. What is the accuracy difference between Cohere Rerank and a local cross encoder?
49. When would you prefer Cohere Rerank over a local model?
50. What is the vendor lock-in risk with Cohere Rerank?

## 7.5 RankGPT

51. What is RankGPT?
52. How does RankGPT use GPT-4 for re-ranking?
53. What is the input prompt format for RankGPT?
54. What is sliding window strategy in RankGPT?
55. What is the latency of RankGPT for 20 candidates?
56. What is the cost of RankGPT at scale?
57. When is RankGPT worth the cost vs. a local cross encoder?
58. What is the accuracy of RankGPT vs. traditional cross encoders?
59. What is the paper "Is ChatGPT Good at Search?" and what did it find?
60. What is the limitation of LLM-based re-rankers?

## 7.6 ColBERT & Late Interaction

61. What is ColBERT?
62. What is late interaction in ColBERT?
63. How does ColBERT differ from a bi-encoder?
64. How does ColBERT differ from a cross encoder?
65. What is the MaxSim operation in ColBERT?
66. Write the MaxSim formula for ColBERT scoring.
67. `Score(q, d) = Σ_qi max_dj(E_q[qi] · E_d[dj]^T)` — explain.
68. What is the storage cost of ColBERT embeddings?
69. Why does ColBERT require more storage than a bi-encoder?
70. How does ColBERT enable efficient retrieval with late interaction?
71. What is PLAID (ColBERT v2's index)?
72. What is the PLAID candidate filtering mechanism?
73. What is ColBERT v2 and how does it reduce storage?
74. What is residual compression in ColBERT v2?
75. What is RAGatouille and how does it simplify ColBERT deployment?

## 7.7 Re-ranking Strategy & Pipeline Design

76. What is the standard retrieval → re-ranking pipeline?
77. How many candidates do you retrieve before re-ranking?
78. What is the recall@100 vs. recall@10 trade-off?
79. How do you determine the optimal N for candidates to re-rank?
80. How do you validate the optimal N experimentally?
81. What is the latency budget for re-ranking in a RAG system?
82. How do you batch cross encoder inference to improve throughput?
83. What is the maximum batch size for cross encoder inference on GPU?
84. What is the GPU memory requirement for running cross encoder inference?
85. How do you implement async cross encoder re-ranking?
86. What is the concurrency model for re-ranking in FastAPI?
87. How do you run re-ranking as a background task?
88. What is the latency penalty of re-ranking for a medical query?
89. How do you implement caching for re-ranking results?
90. What is the cache key strategy for re-ranking?

## 7.8 Re-ranking Evaluation

91. How do you measure the improvement from re-ranking?
92. What is NDCG@10 improvement from bi-encoder to cross encoder?
93. What is the typical NDCG improvement from re-ranking on BEIR benchmarks?
94. How do you create a labeled evaluation set for re-ranking quality?
95. What is the difference between offline and online evaluation of re-ranking?
96. How do you measure user click-through rate as a signal for re-ranking quality?
97. What is implicit feedback for re-ranking evaluation?
98. How do you handle position bias in click-through feedback?
99. What is counterfactual evaluation for re-ranking?
100. How do you A/B test re-ranking in production?

## 7.9 Medical Domain Re-ranking

101. Should your cross encoder be fine-tuned on medical data?
102. What medical re-ranking datasets exist?
103. What is BioASQ and how would you use it for re-ranking fine-tuning?
104. What is the TREC-COVID dataset?
105. What is MedQA and how does it inform re-ranking training?
106. How do you generate synthetic re-ranking training data for medical Q&A?
107. What is UDAPDR (Unsupervised Domain Adaptation via LLM Prompting)?
108. How does UDAPDR work for adapting re-rankers to new domains?
109. What is the benefit of domain-specific re-ranking for rare disease queries?
110. What is the performance difference between general and medical-fine-tuned re-rankers?

## 7.10 Re-ranking Failure Cases & Optimization

111. What is re-ranking failure mode #1: low recall before re-ranking?
112. What happens if you pass poor-quality candidates to the cross encoder?
113. What is the "garbage in, garbage out" problem in re-ranking?
114. How do you handle the case where all top-K candidates are irrelevant?
115. What is re-ranking failure mode #2: cross encoder overfitting to training distribution?
116. How do you detect distribution mismatch in re-ranking?
117. What is re-ranking failure mode #3: latency spikes under load?
118. How do you implement re-ranking timeout with fallback to bi-encoder ordering?
119. What is re-ranking failure mode #4: scoring inconsistency across batches?
120. How do you normalize cross encoder scores across batches?

## 7.11 Production Re-ranking Operations

121. How do you monitor re-ranking quality in production?
122. What metrics do you log per request for re-ranking?
123. How do you detect when re-ranking is making things worse?
124. What is your re-ranking model versioning strategy?
125. How do you deploy a new re-ranking model with zero downtime?
126. What is the model loading strategy for cross encoders in FastAPI?
127. How do you preload the cross encoder model at server startup?
128. How do you handle multiple cross encoder models for A/B testing?
129. What is the memory footprint of a cross encoder in inference mode?
130. How do you serve cross encoder inference at 1000 QPS?

## 7.12 Advanced Re-ranking Research

131. What is learned sparse retrieval vs. re-ranking?
132. What is FIRST (Full-Interaction Retrieval with Sparse Transformers)?
133. What is SPANN and how does it relate to re-ranking?
134. What is the paper "Multi-Stage Document Ranking with BERT"?
135. What is the Passage Re-Ranking with BERT paper?
136. What is the difference between pointwise, pairwise, and listwise learning to rank?
137. What is RankNet?
138. What is LambdaMART?
139. What is XGBoost-based learning to rank?
140. How does gradient boosted re-ranking compare to neural re-ranking?

## 7.13 Re-ranking Code Questions

141. Write Python code to implement cross encoder re-ranking with `sentence-transformers`.
142. How do you implement batched cross encoder inference in PyTorch?
143. Write code to integrate cross encoder re-ranking into a LangChain retriever.
144. How do you implement async cross encoder re-ranking in FastAPI?
145. How do you implement caching of re-ranking results with Redis?
146. How do you implement a fallback from cross encoder to BM25 on timeout?
147. Write code to evaluate NDCG@10 improvement from re-ranking.
148. How do you parallelize cross encoder inference across multiple GPUs?
149. Write code to fine-tune a cross encoder on medical Q&A pairs.
150. How do you implement cross encoder re-ranking as a microservice?

## 7.14 Re-ranking System Design

151. Design a re-ranking microservice that handles 1000 QPS.
152. How do you handle re-ranking for different medical specialties?
153. What is the load balancing strategy for a re-ranking service?
154. How do you implement re-ranking with model warm-up?
155. What is the auto-scaling trigger for a re-ranking service?
156. How do you implement graceful degradation when the re-ranker is overloaded?
157. What is the circuit breaker pattern for re-ranking?
158. How do you implement rate limiting for re-ranking requests?
159. How do you log re-ranking inputs and outputs for audit?
160. What is the data retention policy for re-ranking logs?

## 7.15 Re-ranking Latency Optimization

161. How do you reduce cross encoder latency?
162. What is ONNX Runtime and how does it accelerate cross encoder inference?
163. What is TensorRT and when would you use it for cross encoder?
164. What is quantization of cross encoder models?
165. What is INT8 quantization for BERT-based cross encoders?
166. What is the accuracy degradation from INT8 quantization?
167. What is knowledge distillation for cross encoder compression?
168. What is `cross-encoder/ms-marco-TinyBERT-L-2-v2` and when would you use it?
169. How do you implement early exit in cross encoder inference?
170. What is adaptive computation time for re-ranking?

## 7.16 Re-ranking Security

171. What are the security risks of a public re-ranking API?
172. Can a malicious user manipulate re-ranking scores?
173. What is re-ranking adversarial attacks?
174. How do you prevent prompt injection through document content in re-ranking?
175. What is content-based re-ranking manipulation?
176. How do you rate limit re-ranking requests per user?
177. What is the risk of exposing re-ranking model internals?
178. How do you protect PHI in documents during re-ranking?
179. What is the auditability requirement for re-ranking in medical AI?
180. How do you ensure deterministic re-ranking for audit trails?

## 7.17 Future of Re-ranking

181. What is the trend toward using LLMs directly for re-ranking?
182. What is the future of ColBERT in production RAG systems?
183. What is the impact of sub-second LLM inference on re-ranking strategies?
184. Will dense + re-ranking pipelines be replaced by end-to-end learned retrievers?
185. What is a learned retriever (joint training of retrieval + re-ranking)?
186. What is SPANN-style re-ranking?
187. What is the role of MTEB in measuring re-ranking progress?
188. How does Cohere Rerank 3.5 compare to open-source alternatives?
189. What is voyage-rerank and how does it position against Cohere?
190. What is the role of flash attention in cross encoder acceleration?

## 7.18 Re-ranking Mathematical Analysis

191. Prove that cross encoder can model query-document term interactions.
192. What is the attention pattern difference between bi-encoder and cross encoder?
193. What is cross-attention and how does it differ from self-attention?
194. Mathematically, why does token-level interaction improve relevance estimation?
195. What is the Bayes optimal ranking and how close do cross encoders get?
196. What is the theoretical bound on re-ranking improvement given oracle first-stage recall@k?
197. Derive the relationship between recall@k and maximum achievable NDCG@10 after re-ranking.
198. What is the expected gain formula from re-ranking with perfect cross encoder?
199. What is the statistical test you would use to prove re-ranking improves quality?
200. What is paired t-test vs. Wilcoxon signed-rank test for re-ranking evaluation?

---

# SECTION 8 — PROMPT ENGINEERING {#section-8}

## 8.1 Prompt Template Design

1. What is a prompt template?
2. What are the components of a RAG prompt template?
3. What is the system prompt vs. user prompt vs. assistant prompt?
4. How do you structure a medical QA system prompt?
5. Write a complete system prompt for your medical RAG chatbot.
6. What is the difference between instructive prompts and role-based prompts?
7. How do you inject retrieved context into a prompt template?
8. What is the `{context}` vs. `{question}` variable in a RAG prompt?
9. How do you handle the case where retrieved context is empty?
10. What is the "no relevant information found" fallback strategy?

## 8.2 Few-Shot Prompting

11. What is few-shot prompting?
12. What is zero-shot prompting?
13. What is one-shot prompting?
14. When do you add examples to a medical RAG prompt?
15. How do you select the best few-shot examples?
16. What is dynamic few-shot selection?
17. How do you implement dynamic few-shot selection using semantic similarity?
18. What is the risk of few-shot examples in medical prompts?
19. How do you prevent few-shot examples from biasing the model?
20. What is the computational cost of few-shot prompting?

## 8.3 Chain of Thought (CoT)

21. What is chain of thought prompting?
22. What paper introduced chain of thought prompting?
23. When does CoT improve medical question answering?
24. What is zero-shot chain of thought ("Let's think step by step")?
25. What is auto CoT?
26. What is tree of thought (ToT)?
27. What is the difference between CoT and CoT-SC (Self-Consistency)?
28. What is the cost of CoT vs. direct answer?
29. When does CoT fail for medical questions?
30. What is the relationship between CoT and hallucination?

## 8.4 Grounding & Citations

31. What is grounding in RAG?
32. How do you instruct the LLM to cite its sources?
33. What is the prompt structure for citation-aware responses?
34. How do you verify that citations are accurate (hallucinated citations)?
35. What is source attribution validation?
36. How do you format citations for medical answers?
37. What is the "strictly grounded" vs. "partially grounded" response classification?
38. How do you detect when the LLM generates information not in the retrieved context?
39. What is the role of the instruction "Answer ONLY based on the provided context"?
40. What happens when the LLM ignores this instruction?

## 8.5 Prompt Injection & Security

41. What is prompt injection?
42. What is direct prompt injection?
43. What is indirect prompt injection?
44. How can a malicious medical document inject instructions into your prompt?
45. What is the Gandalf challenge and what does it demonstrate?
46. How do you sanitize user queries before prompt construction?
47. How do you sanitize retrieved documents before injecting them into prompts?
48. What is the risk of embedding malicious instructions in vector store documents?
49. What is the "ignore previous instructions" attack?
50. What is the "jailbreak" attack and how do you prevent it?
51. What are LLM guardrails? Give examples.
52. What is Nemo Guardrails (NVIDIA)?
53. What is LlamaGuard and how does it work?
54. What is the Llama Guard taxonomy of harmful categories?
55. What is the role of input/output content moderation in medical AI?

## 8.6 Structured Output & JSON Mode

56. What is structured output from an LLM?
57. What is JSON mode in OpenAI API?
58. What is the Groq JSON mode?
59. How do you validate LLM JSON output?
60. What library do you use for structured output validation?
61. What is Pydantic and how does it work with LLM output?
62. What is Outlines library for structured generation?
63. What is the grammar-constrained generation approach?
64. How do you handle malformed JSON from LLM responses?
65. What is retry logic for structured output failures?

## 8.7 Hallucination Reduction

66. What prompt techniques reduce hallucination?
67. What is the "Retrieve then Read" instruction strategy?
68. What is the "Don't know" instruction for reducing confabulation?
69. What is the role of temperature in hallucination?
70. Why does temperature=0 not completely prevent hallucination?
71. What is the "self-consistency" technique for reducing hallucination?
72. What is "chain of verification" (CoVe)?
73. What is "self-RAG" and how does it reduce hallucination?
74. What is the factual consistency scoring prompt?
75. How do you implement a factual checker as a separate LLM call?

## 8.8 Prompt Optimization

76. What is DSPy (Demonstrate-Search-Predict) and how does it automate prompt optimization?
77. What is the OPRO (Optimization by PROmpting) approach?
78. What is automated prompt engineering?
79. What is LLM-as-optimizer for prompt tuning?
80. What is the APE (Automatic Prompt Engineer) paper?
81. What is prefix tuning vs. prompt tuning vs. in-context learning?
82. What is soft prompt tuning?
83. What is the LORA-equivalent for prompts?
84. How do you A/B test different prompts in production?
85. What is the metric you use to compare prompts?

---

# SECTION 9 — LLAMA 3.3 70B DEEP DIVE {#section-9}

## 9.1 Llama Architecture

1. What is Llama 3.3 70B? Who created it?
2. What is the transformer architecture used in Llama 3?
3. What is the difference between encoder-only, decoder-only, and encoder-decoder architectures?
4. Why is Llama a decoder-only architecture?
5. How many transformer layers does Llama 3.3 70B have?
6. What is the hidden dimension of Llama 3.3 70B?
7. What is the number of attention heads?
8. What is the number of KV heads? (GQA)
9. What is the feedforward network dimension?
10. What is the activation function in Llama 3's FFN?
11. What is SwiGLU and how does it differ from ReLU?
12. What is the SwiGLU formula?
13. What is RMSNorm and why is it used instead of LayerNorm?
14. What is the RMSNorm formula?
15. What is the computational advantage of RMSNorm?

## 9.2 Tokenizer

16. What tokenizer does Llama 3 use?
17. What is BPE (Byte Pair Encoding)?
18. How does BPE tokenization work? Explain the algorithm step by step.
19. What is the vocabulary size of Llama 3's tokenizer?
20. How does the Llama 3 tokenizer handle medical terminology?
21. What is the difference between character-level, word-level, and subword tokenization?
22. What is the advantage of subword tokenization over word-level?
23. What is the sentencepiece library?
24. How does Llama 3 tokenizer handle whitespace?
25. What is the token count for a typical medical question vs. a general question?

## 9.3 Context Window

26. What is the context window of Llama 3.3 70B?
27. What is RoPE (Rotary Position Embedding)?
28. Why did Llama 3 switch to RoPE from absolute positional encoding?
29. What is the mathematical formulation of RoPE?
30. How does RoPE enable context window extension?
31. What is YaRN (Yet another RoPE extensioN method)?
32. What is the quality degradation at long context lengths?
33. What is the "needle in a haystack" evaluation for Llama 3.3?
34. What is the practical vs. theoretical context window?
35. How does context length affect inference latency and memory?

## 9.4 Attention Mechanism

36. What is self-attention? Explain Q, K, V.
37. Write the scaled dot-product attention formula.
38. `Attention(Q,K,V) = softmax(QK^T / √dk) × V` — explain every term.
39. Why do we divide by √dk?
40. What is multi-head attention?
41. How are multiple attention heads combined?
42. What is the purpose of multiple attention heads?
43. What is Grouped Query Attention (GQA)?
44. How does GQA reduce KV cache memory?
45. What is the memory saving from GQA vs. MHA?
46. What is Multi-Query Attention (MQA)?
47. What is the difference between MQA and GQA?
48. What is flash attention?
49. How does flash attention reduce memory from O(n²) to O(n)?
50. What is the computational complexity of standard vs. flash attention?
51. What is flash attention v2?
52. What is flash attention v3?
53. What is sliding window attention?
54. What is ring attention?
55. What is the "attention sink" phenomenon?

## 9.5 Inference & KV Cache

56. What is autoregressive generation?
57. What is the KV cache?
58. Why does KV cache prevent redundant computation?
59. What is the memory cost of the KV cache for Llama 3.3 70B?
60. Calculate the KV cache size for batch=1, seq=4096, Llama 3.3 70B.
61. How does KV cache memory scale with sequence length?
62. How does KV cache memory scale with batch size?
63. What is PagedAttention (vLLM)?
64. How does PagedAttention solve KV cache fragmentation?
65. What is the memory efficiency improvement from PagedAttention?
66. What is continuous batching in vLLM?
67. What is the prefill vs. decode phase in LLM inference?
68. What is TTFT (Time to First Token)?
69. What is TPOT (Time Per Output Token)?
70. What is the relationship between prefill latency and context length?

## 9.6 Quantization

71. What is model quantization?
72. What is FP32 vs. FP16 vs. BF16?
73. What is INT8 quantization?
74. What is INT4 quantization?
75. What is the quality degradation from FP16 to INT4 for Llama 3.3 70B?
76. What is GPTQ quantization?
77. What is AWQ (Activation-aware Weight Quantization)?
78. What is GGUF format and why is it used for local deployment?
79. What is the memory requirement for Llama 3.3 70B in FP16?
80. `70B × 2 bytes ≈ 140 GB` — what GPUs can serve it?
81. What is the memory requirement in INT4?
82. `70B × 0.5 bytes ≈ 35 GB` — what GPUs can serve it now?
83. What is model sharding for multi-GPU inference?
84. What is tensor parallelism?
85. What is pipeline parallelism?

## 9.7 Sampling & Generation Parameters

86. What is temperature in LLM sampling?
87. What is the mathematical effect of temperature on the softmax distribution?
88. What does temperature = 0 do?
89. What does temperature = 1.0 do?
90. What does temperature > 1.0 do?
91. What is top-k sampling?
92. What is top-p (nucleus) sampling?
93. How does top-p select the token set?
94. What is min-p sampling?
95. What is repetition penalty?
96. What is presence penalty vs. frequency penalty (OpenAI)?
97. What is greedy decoding?
98. What is beam search?
99. What is the difference between beam search and nucleus sampling?
100. What is the best temperature for medical question answering?

## 9.8 Why Groq for Inference?

101. What is Groq? What is the Language Processing Unit (LPU)?
102. How does Groq LPU differ from NVIDIA GPU architecture?
103. What is the Groq inference latency for Llama 3.3 70B?
104. What is the Groq throughput (tokens/second) for Llama 3.3 70B?
105. What is the cost of Groq inference per 1M tokens?
106. What is the rate limit on Groq's API?
107. How do you handle Groq rate limits in production?
108. What is the fallback strategy when Groq is rate limited?
109. What is the Groq context window for Llama 3.3 70B?
110. How does Groq compare to Together AI, Fireworks AI, and Cerebras?

## 9.9 Why Llama? Why Not Alternatives?

111. Why choose Llama 3.3 70B over GPT-4o for a medical RAG system?
112. What is the quality difference between Llama 3.3 70B and GPT-4o?
113. What are the privacy advantages of Llama over OpenAI?
114. What is the cost comparison between Groq Llama and OpenAI GPT-4o per token?
115. Why would you choose Claude 3.7 Sonnet for medical RAG?
116. What is Claude's "extended thinking" feature and when is it useful for medical QA?
117. What is Gemini 2.5 Pro and what is its multimodal advantage?
118. What is Gemini 2.5's context window and how does it affect RAG design?
119. What is Llama 4 and how does it differ from Llama 3.3?
120. What is DeepSeek V3 and why is it significant?
121. What is the reasoning advantage of DeepSeek R2?
122. What is Qwen 3 235B and how does it compete with frontier models?
123. What is Mistral Large 2 and when would you use it?
124. What is Command-R+ from Cohere and why is it specifically designed for RAG?
125. What is Phi-4 and what is the small model advantage?

## 9.10 Llama 3.3 Production Optimization

126. What is speculative decoding?
127. How does speculative decoding reduce inference latency?
128. What is the draft model in speculative decoding?
129. What is vLLM and why is it the standard for Llama serving?
130. What is SGLang and how does it compare to vLLM?
131. What is TensorRT-LLM and when would you use it?
132. What is DeepSpeed Inference?
133. What is ORCA batching?
134. What is the optimal batch size for Llama 3.3 70B inference?
135. What is the throughput-latency trade-off in LLM serving?

---

# SECTION 10 — FASTAPI DEEP DIVE {#section-10}

## 10.1 FastAPI Fundamentals

1. What is FastAPI?
2. Who created FastAPI and what is its core philosophy?
3. What is Starlette and how does FastAPI build on it?
4. What is ASGI vs. WSGI?
5. Why is ASGI necessary for modern Python web APIs?
6. What is Uvicorn?
7. What is Gunicorn and how does it differ from Uvicorn?
8. What is the typical Gunicorn + Uvicorn production setup?
9. What is the worker model in Uvicorn?
10. How many workers should you configure for a RAG API server?

## 10.2 Async in FastAPI

11. What is Python asyncio?
12. What is a coroutine?
13. What is the `async def` vs. `def` in FastAPI route handlers?
14. What is the event loop in Python?
15. What happens when you block the event loop?
16. What is the difference between `await` and blocking call?
17. How do you run blocking code in an async FastAPI handler?
18. What is `run_in_executor` and when do you use it?
19. What is `asyncio.gather` and how do you use it for parallel retrieval?
20. What is the GIL and how does it affect FastAPI?
21. How does `asyncio` bypass the GIL for I/O operations?
22. What is `httpx` vs. `requests` for async HTTP calls in FastAPI?
23. How do you implement concurrent embedding API calls in FastAPI?
24. What is connection pooling with `httpx.AsyncClient`?
25. How do you implement proper lifecycle management for async clients?

## 10.3 Dependency Injection

26. What is FastAPI's dependency injection system?
27. How do you define a dependency in FastAPI?
28. What is `Depends()`?
29. How do you implement singleton pattern with FastAPI dependencies?
30. How do you inject the Pinecone client as a dependency?
31. How do you inject the LLM client as a dependency?
32. How do you implement authentication as a dependency?
33. What is yield-based dependency for resource management?
34. How do you implement database connection management with yield dependencies?
35. What is dependency override for testing?

## 10.4 Streaming

36. What is streaming in FastAPI?
37. What is `StreamingResponse`?
38. What is Server-Sent Events (SSE)?
39. How do you implement SSE for LLM streaming output?
40. What is the difference between SSE and WebSockets?
41. How do you implement streaming LLM response in FastAPI?
42. What is the generator pattern for streaming?
43. How do you handle client disconnection during streaming?
44. What is the `asyncio.CancelledError` in streaming handlers?
45. How do you implement streaming with token counting?

## 10.5 Authentication & Security

46. What is JWT (JSON Web Token)?
47. What are the three parts of a JWT?
48. What is the JWT signing algorithm HS256 vs. RS256?
49. How do you implement JWT authentication in FastAPI?
50. What is OAuth2 password flow in FastAPI?
51. What is the `OAuth2PasswordBearer` in FastAPI?
52. How do you implement API key authentication?
53. What is HTTPS and why is it mandatory for medical APIs?
54. How do you configure CORS in FastAPI?
55. What are the security risks of a permissive CORS policy?

## 10.6 Performance & Scaling

56. What is the request lifecycle in FastAPI?
57. What is middleware in FastAPI?
58. How do you implement request logging middleware?
59. How do you implement rate limiting middleware?
60. What is `slowapi` for FastAPI rate limiting?
61. How do you implement request timeout middleware?
62. What is the performance impact of middleware?
63. How do you implement response caching with `fastapi-cache`?
64. What is the role of Redis for FastAPI caching?
65. How do you implement semantic caching for medical queries?
66. What is semantic caching and how is it different from exact key caching?
67. How do you implement request validation with Pydantic?
68. What is the performance of Pydantic v2 vs. v1?
69. How do you benchmark FastAPI endpoint latency?
70. What is the `locust` tool for load testing FastAPI?

## 10.7 FastAPI in Production

71. How do you deploy FastAPI with Docker?
72. What is the recommended Docker setup for FastAPI + Uvicorn?
73. How do you configure health check endpoints?
74. What is a readiness probe vs. liveness probe?
75. How do you implement graceful shutdown in FastAPI?
76. What is the `lifespan` context manager in FastAPI?
77. How do you handle model loading in the `startup` event?
78. What is the `on_event("startup")` pattern?
79. How do you implement background tasks in FastAPI?
80. What is `BackgroundTasks` in FastAPI?

## 10.8 WebSockets in FastAPI

81. What is the WebSocket protocol?
82. How do you implement WebSocket endpoint in FastAPI?
83. What is the use case for WebSockets in a medical chatbot?
84. How does WebSocket compare to SSE for streaming?
85. How do you handle WebSocket connection lifecycle?
86. How do you implement WebSocket authentication?
87. How do you broadcast to multiple WebSocket clients?
88. What is the memory cost of maintaining WebSocket connections?
89. How do you scale WebSocket connections across multiple servers?
90. What is sticky session and why is it needed for WebSockets?

---

# SECTION 11 — AWS ARCHITECTURE {#section-11}

## 11.1 EC2 Deep Dive

1. What is AWS EC2?
2. What EC2 instance type did you use? Why?
3. What is the difference between general-purpose (m-series) and compute-optimized (c-series) instances?
4. What is an AMI (Amazon Machine Image)?
5. What is the difference between on-demand, reserved, and spot instances?
6. Why would you use spot instances for a medical RAG system (or not)?
7. What is an EC2 placement group?
8. What is the difference between cluster, spread, and partition placement groups?
9. How do you automate EC2 instance configuration with user data?
10. What is an EC2 instance profile and IAM role?

## 11.2 Load Balancing

11. What is the AWS Elastic Load Balancer (ELB)?
12. What is the difference between ALB, NLB, and CLB?
13. Why would you use ALB for a FastAPI application?
14. What is the ALB target group?
15. What is the health check configuration for ALB?
16. What is connection draining (deregistration delay)?
17. What is sticky sessions in ALB?
18. How does ALB handle SSL termination?
19. What is the ALB access log and what information does it contain?
20. What is WAF (Web Application Firewall) and how do you integrate it with ALB?

## 11.3 Auto Scaling

21. What is AWS Auto Scaling?
22. What is an Auto Scaling Group (ASG)?
23. What is a launch template in ASG?
24. What is a scaling policy?
25. What is target tracking scaling policy?
26. What is step scaling policy?
27. What is scheduled scaling?
28. What metric do you use to trigger auto scaling for a RAG system?
29. What is the warm-up period in Auto Scaling?
30. How do you prevent scale thrashing?
31. What is cooldown period?
32. What is predictive scaling?
33. How do you handle stateful components during auto scaling?
34. What is the minimum, maximum, and desired capacity in ASG?
35. How do you test your auto scaling configuration?

## 11.4 IAM & Security

36. What is IAM (Identity and Access Management)?
37. What is the principle of least privilege?
38. What is an IAM role vs. IAM user?
39. What is an IAM policy?
40. What is an instance profile?
41. How does EC2 assume an IAM role?
42. What is AWS Secrets Manager?
43. How do you store Pinecone API key in Secrets Manager?
44. How does your FastAPI application retrieve secrets at runtime?
45. What is AWS KMS and how do you use it for encryption?
46. What is the KMS key policy?
47. How do you encrypt Pinecone API keys at rest?
48. What is VPC (Virtual Private Cloud)?
49. What is a security group?
50. What is the difference between inbound and outbound rules?

## 11.5 CloudWatch & Monitoring

51. What is AWS CloudWatch?
52. What is the difference between CloudWatch metrics, logs, and alarms?
53. What custom metrics do you push to CloudWatch for your RAG system?
54. What is a CloudWatch dashboard?
55. What is a CloudWatch alarm?
56. How do you configure CloudWatch alarm notification to SNS?
57. What is CloudWatch Log Insights?
58. How do you query logs with CloudWatch Log Insights?
59. What is CloudWatch Container Insights for Docker?
60. What is CloudWatch Synthetics for uptime monitoring?
61. What is X-Ray and how does it enable distributed tracing?
62. How do you integrate X-Ray with FastAPI?
63. What is OpenTelemetry and how does it compare to X-Ray?
64. How do you instrument LangChain calls with OpenTelemetry?
65. What is a trace, span, and baggage in distributed tracing?

## 11.6 Networking & Security Groups

66. What is a VPC?
67. What is a subnet?
68. What is the difference between public and private subnet?
69. What is an Internet Gateway?
70. What is a NAT Gateway?
71. Why would your EC2 instance be in a private subnet?
72. How does traffic flow from the internet to an EC2 in a private subnet?
73. What is a security group and how is it stateful?
74. What is a NACL (Network Access Control List) and how is it stateless?
75. What is the difference between security groups and NACLs?

## 11.7 Deployment & Infrastructure as Code

76. What is CloudFormation?
77. What is the difference between CloudFormation and Terraform?
78. Have you written any IaC for your AWS deployment?
79. What is the benefit of IaC over manual console configuration?
80. What is AWS CDK (Cloud Development Kit)?
81. What is the difference between CDK and CloudFormation?
82. What is Elastic Beanstalk and why might you use it?
83. What is ECS (Elastic Container Service)?
84. What is EKS (Elastic Kubernetes Service)?
85. When would you migrate from EC2 to ECS for your Docker-based RAG system?

## 11.8 Cost Optimization

86. What is the AWS Cost Explorer?
87. How do you set up billing alerts?
88. What is AWS Savings Plans?
89. What is the right-sizing recommendation?
90. How do you use Compute Optimizer for EC2 right-sizing?
91. What is the cost of EC2 t3.medium vs. m5.xlarge for your workload?
92. What is data transfer cost in AWS?
93. How do you minimize cross-AZ data transfer?
94. What is S3 intelligent tiering and when do you use it?
95. How do you estimate your total monthly AWS bill?

## 11.9 Disaster Recovery

96. What is RPO (Recovery Point Objective)?
97. What is RTO (Recovery Time Objective)?
98. What are the four DR strategies (backup-restore, pilot light, warm standby, active-active)?
99. Which DR strategy did you implement for your medical RAG system?
100. How do you back up your Pinecone index to S3?
101. How do you restore from backup if Pinecone is corrupted?
102. What is your RTO for a total system failure?
103. How do you handle AWS region failure?
104. What is Route 53 failover routing?
105. How do you test your disaster recovery procedure?

---

# SECTION 12 — DOCKER & CI/CD {#section-12}

## 12.1 Docker Fundamentals

1. What is Docker?
2. What is a Docker container vs. a Docker image?
3. What is the difference between a container and a VM?
4. What is the Docker daemon?
5. What is a Dockerfile?
6. What is the Docker build process?
7. What are Docker layers?
8. What is layer caching in Docker builds?
9. How do you optimize layer caching in your Dockerfile?
10. What is the impact of layer order on build time?

## 12.2 Dockerfile Deep Dive

11. What is the `FROM` instruction?
12. What is the difference between `CMD` and `ENTRYPOINT`?
13. What is the difference between `RUN`, `COPY`, and `ADD`?
14. Why should you use `COPY` instead of `ADD` in most cases?
15. What is the `.dockerignore` file?
16. What should you include in `.dockerignore` for a Python FastAPI project?
17. What is a multi-stage Docker build?
18. How does multi-stage build reduce image size?
19. What is the difference between a build stage and a runtime stage?
20. What is the base image choice for a FastAPI Python application?
21. Why should you use `python:3.11-slim` over `python:3.11`?
22. What is Alpine Linux and when is it used for base images?
23. What is the security risk of running containers as root?
24. How do you create a non-root user in a Dockerfile?
25. What is the principle of least privilege in Docker?

## 12.3 Docker Networking

26. What is Docker networking?
27. What are the Docker network types (bridge, host, overlay, none)?
28. What is the bridge network?
29. How do containers on the same bridge network communicate?
30. What is Docker DNS resolution for service discovery?
31. What is host networking and when is it used?
32. What is port mapping (`-p`)?
33. What is `EXPOSE` in Dockerfile?
34. How do you implement Docker networking for FastAPI + Redis?
35. What is Docker Compose networking?

## 12.4 Docker Volumes & Storage

36. What is a Docker volume?
37. What is the difference between a volume and a bind mount?
38. When do you use volumes for a FastAPI application?
39. How do you persist MLflow data with Docker volumes?
40. What is the tmpfs mount?
41. How do you manage secrets in Docker? (Not environment variables)
42. What is Docker Secrets?
43. How do you use AWS Secrets Manager with Docker?
44. What is the risk of storing secrets in environment variables?
45. What is the risk of storing secrets in Docker images?

## 12.5 Docker Optimization

46. How do you minimize Docker image size?
47. What is the `pip install --no-cache-dir` flag?
48. What is the benefit of combining `RUN` commands with `&&`?
49. How do you use BuildKit for faster Docker builds?
50. What is `DOCKER_BUILDKIT=1`?
51. What is Docker BuildKit caching?
52. What is the difference between `--mount=type=cache` and layer caching?
53. How do you scan Docker images for vulnerabilities?
54. What is `docker scan` or `trivy`?
55. What is the role of SBOM (Software Bill of Materials) in Docker security?

## 12.6 GitHub Actions

56. What is GitHub Actions?
57. What is a GitHub Actions workflow?
58. What is a GitHub Actions job?
59. What is a GitHub Actions step?
60. What is a GitHub Actions runner?
61. What is the difference between `ubuntu-latest` and self-hosted runners?
62. What is a GitHub Actions secret?
63. How do you store AWS credentials in GitHub Actions secrets?
64. What is the principle of least privilege for GitHub Actions secrets?
65. What is OIDC (OpenID Connect) for GitHub Actions and AWS authentication?

## 12.7 CI/CD Pipeline

66. What does your CI/CD pipeline look like step by step?
67. What triggers your CI/CD pipeline?
68. What tests do you run in CI?
69. What is linting in CI? What tools do you use?
70. What is `ruff` and how does it compare to `flake8` + `black`?
71. What is type checking in CI? What tool do you use?
72. What is `mypy` for static type checking?
73. What is unit testing in CI?
74. What is integration testing in CI for a RAG system?
75. How do you test RAG quality in CI without calling expensive APIs?

## 12.8 Deployment Strategies

76. What is blue-green deployment?
77. Explain how you would implement blue-green deployment for your RAG API.
78. What is canary deployment?
79. How does canary deployment reduce risk?
80. What percentage of traffic do you send to canary initially?
81. What metrics trigger a canary rollback?
82. What is rolling deployment?
83. What is the difference between rolling and blue-green?
84. What is feature flag deployment?
85. How do you implement feature flags in a FastAPI application?

## 12.9 Rollback & Recovery

86. What is your rollback procedure when a deploy fails?
87. How do you detect a failed deployment?
88. What is an automated rollback trigger?
89. How do you implement automated rollback in GitHub Actions?
90. What is the smoke test after deployment?
91. How long do you maintain the previous version for rollback?
92. What is the rollback procedure for Pinecone index changes?
93. How do you handle database migration rollbacks?
94. What is a feature toggle for safe rollback?
95. What is the Git strategy (GitFlow vs. trunk-based) for your deployment?

## 12.10 Secrets Management

96. What is the risk of hardcoding secrets?
97. What tools do you use for secrets management?
98. What is HashiCorp Vault?
99. What is AWS Secrets Manager vs. Parameter Store?
100. How do you inject secrets into a Docker container at runtime?
101. What is the `docker run --env-file` approach?
102. What is the risk of printing secrets in logs?
103. How do you audit secret access?
104. What is secret rotation and how do you implement it without downtime?
105. What is the SOPS (Secrets OPerationS) tool?

---

# SECTION 13 — MLFLOW & MLOPS {#section-13}

## 13.1 MLflow Fundamentals

1. What is MLflow?
2. What are the four components of MLflow?
3. What is the MLflow Tracking component?
4. What is an MLflow experiment?
5. What is an MLflow run?
6. What is an MLflow metric?
7. What is an MLflow parameter?
8. What is an MLflow artifact?
9. What is the MLflow tracking server?
10. What is the default MLflow backend store?
11. What is the MLflow artifact store?
12. What is the difference between a local and remote MLflow server?
13. How do you start the MLflow UI?
14. What is the MLflow API for logging metrics?
15. `mlflow.log_metric("faithfulness", 0.87, step=1)` — explain.

## 13.2 MLflow Tracking in RAG

16. What RAG-specific metrics do you track in MLflow?
17. How do you log retrieval quality metrics to MLflow?
18. How do you log generation quality metrics to MLflow?
19. How do you log chunking configuration as parameters?
20. How do you log embedding model name as a parameter?
21. How do you log the entire RAG pipeline configuration as parameters?
22. What artifacts do you log to MLflow for RAG experiments?
23. How do you log evaluation datasets as artifacts?
24. How do you log RAGAS evaluation results to MLflow?
25. How do you log retrieval examples as artifacts?

## 13.3 MLflow Model Registry

26. What is the MLflow Model Registry?
27. What is a registered model?
28. What is a model version?
29. What are the model stages in MLflow Registry?
30. What is the difference between "Staging" and "Production" stages?
31. How do you transition a model from Staging to Production?
32. How do you add annotations and tags to model versions?
33. How do you implement model approval workflow with MLflow Registry?
34. How do you roll back to a previous model version?
35. What is the MLflow model schema?

## 13.4 MLflow Experiments for RAG

36. How do you structure MLflow experiments for RAG optimization?
37. What experiment name structure do you use?
38. How do you compare different chunking strategies in MLflow?
39. How do you compare different embedding models in MLflow?
40. How do you compare different retrieval strategies in MLflow?
41. How do you compare different prompt templates in MLflow?
42. What is the parent-child run structure in MLflow?
43. How do you implement hyperparameter search with MLflow?
44. What is the `mlflow.start_run(nested=True)` pattern?
45. How do you reproduce an MLflow run exactly?

## 13.5 MLflow Artifact Logging

46. What types of artifacts do you log for a RAG experiment?
47. How do you log a pandas DataFrame as an artifact?
48. How do you log a matplotlib figure as an artifact?
49. How do you log a JSON configuration file as an artifact?
50. How do you log a RAGAS evaluation report as an artifact?
51. How do you log retrieval examples with ground truth?
52. How do you log model weights as artifacts?
53. What is the MLflow artifact URI format?
54. How do you download artifacts programmatically?
55. How do you version datasets with MLflow?

## 13.6 MLflow Production Operations

56. How do you deploy MLflow server in production?
57. What database backends does MLflow support?
58. Why use PostgreSQL as MLflow backend vs. SQLite?
59. What is the S3 backend for MLflow artifacts?
60. How do you secure the MLflow server with authentication?
61. How do you scale the MLflow server?
62. What is the MLflow REST API?
63. How do you query MLflow runs programmatically?
64. What is `mlflow.search_runs()`?
65. How do you implement automated model comparison in CI/CD?

## 13.7 MLflow Reproducibility

66. What makes a RAG experiment reproducible?
67. What is the role of random seeds in RAG experiments?
68. How do you pin all dependencies for reproducibility?
69. What is the `mlflow.log_artifact("requirements.txt")` pattern?
70. How do you version your evaluation dataset?
71. How do you ensure the embedding model version is logged?
72. How do you ensure the LLM model version is logged?
73. How do you ensure Pinecone index configuration is logged?
74. What is environment reproducibility in MLflow?
75. What is `mlflow.log_dict(config, "config.json")`?

## 13.8 MLflow vs. Alternatives

76. What is Weights & Biases (W&B) and how does it compare to MLflow?
77. What is Neptune.ai?
78. What is Comet ML?
79. What is DVC (Data Version Control) and how does it complement MLflow?
80. What is LangSmith and why might it be preferred over MLflow for RAG?
81. What is the advantage of LangSmith for LLM-specific tracking?
82. What is Phoenix (Arize) for LLM observability?
83. What is Helicone for LLM API logging?
84. What is TruLens for RAG evaluation tracking?
85. How do you choose between MLflow and LangSmith for a RAG project?

---

# SECTION 14 — RAG EVALUATION (500+ QUESTIONS) {#section-14}

## 14.1 RAG Evaluation Framework Overview

1. What is RAG evaluation?
2. Why is RAG evaluation harder than classification evaluation?
3. What are the three levels of RAG evaluation?
4. What is retrieval evaluation?
5. What is generation evaluation?
6. What is end-to-end RAG evaluation?
7. What is RAGAS?
8. What are the core RAGAS metrics?
9. Who created RAGAS?
10. What is the paper "RAGAS: Automated Evaluation of Retrieval Augmented Generation"?

## 14.2 Faithfulness (100 Questions)

11. What is faithfulness in RAG?
12. What is the difference between faithfulness and factual correctness?
13. How does RAGAS compute faithfulness?
14. What is the faithfulness score formula?
15. What is claim extraction in faithfulness evaluation?
16. How does RAGAS extract claims from the generated answer?
17. How does RAGAS check if each claim is supported by the context?
18. What is NLI (Natural Language Inference) in faithfulness evaluation?
19. What is the role of an LLM as judge in faithfulness?
20. What is the RAGAS faithfulness prompt?
21. What faithfulness score is acceptable for a medical RAG system?
22. What faithfulness score is unacceptable?
23. What is a faithfulness regression?
24. How do you detect faithfulness regression in CI/CD?
25. What causes low faithfulness scores?
26. What is the relationship between chunk quality and faithfulness?
27. What is the relationship between reranking and faithfulness?
28. What is the relationship between prompt design and faithfulness?
29. How do you improve faithfulness for medical questions?
30. What is the "strictly grounded" instruction and does it always help?
31. What is faithfulness vs. hallucination rate?
32. What is the difference between intrinsic and extrinsic hallucination?
33. What is sentence-level vs. token-level faithfulness?
34. What is the SAFE (Search-Augmented Faithfulness Evaluation) approach?
35. What is FACTSCORE and how does it measure faithfulness?
36. What is FactBench?
37. How does faithfulness differ for open-ended vs. factoid questions?
38. What is the faithfulness challenge for multi-hop reasoning?
39. How do you evaluate faithfulness for medical answers that require clinical inference?
40. What is the limit of automated faithfulness evaluation?
41. When does LLM-as-judge fail for faithfulness?
42. What is the meta-evaluation of faithfulness metrics?
43. How do human faithfulness ratings compare to automated ones?
44. What is the inter-annotator agreement for medical faithfulness?
45. What is Cohen's kappa for faithfulness annotation agreement?
46. How do you build a faithfulness labeled dataset for medical QA?
47. What is the role of medical expert review in faithfulness evaluation?
48. What is the difference between faithfulness and answer quality?
49. How do you monitor faithfulness in production?
50. What is the production faithfulness alert threshold?
51. How do you triage faithfulness failures?
52. What is faithfulness attribution?
53. How do you identify which retrieval failure caused unfaithful output?
54. What is the causal chain from retrieval to faithfulness failure?
55. What is the impact of context length on faithfulness?
56. What is "context utilization" and how does it relate to faithfulness?
57. What is the faithfulness improvement from reranking?
58. Quantify: by how much does adding re-ranking improve faithfulness?
59. What is the faithfulness score with and without re-ranking in your system?
60. What is the faithfulness score variation across medical specialties?
61. Why might cardiology questions have different faithfulness than oncology questions?
62. How do you create specialty-specific faithfulness benchmarks?
63. What is the cost of running RAGAS faithfulness evaluation at scale?
64. How do you optimize RAGAS evaluation for cost efficiency?
65. What is batch evaluation in RAGAS?
66. How do you parallelize RAGAS evaluation?
67. What is the evaluation latency for 100 Q&A pairs in RAGAS?
68. How do you reduce RAGAS evaluation latency?
69. What is the faithfulness score for your current production system?
70. How has faithfulness changed over time in your system?
71. What changes caused the most faithfulness improvement?
72. What changes caused faithfulness regression?
73. How do you version faithfulness evaluation results in MLflow?
74. What dashboard do you use to monitor faithfulness trends?
75. What is the confidence interval for faithfulness scores?
76. What is statistical significance in faithfulness comparison?
77. When is a faithfulness improvement statistically significant?
78. What is the minimum evaluation dataset size for reliable faithfulness comparison?
79. What is the sampling strategy for faithfulness evaluation dataset?
80. How do you ensure evaluation dataset is representative of production traffic?
81. What is the difference between faithfulness and answer coverage?
82. What is answer coverage in RAG?
83. What is the relationship between faithfulness and context precision?
84. What is the relationship between faithfulness and answer relevancy?
85. Can you have high faithfulness but low answer quality?
86. Give an example where faithfulness is high but the answer is wrong.
87. What is a factually grounded but clinically wrong answer?
88. How does your system handle clinical knowledge not in the corpus?
89. What is the "don't know" response and when should it be triggered?
90. How does a "don't know" response affect faithfulness score?
91. What is the relationship between faithfulness and user satisfaction?
92. What user study design would you use to correlate faithfulness with satisfaction?
93. What is the A/B test design for faithfulness improvement validation?
94. What is the MDE (minimum detectable effect) for faithfulness A/B testing?
95. What is the required sample size for a faithfulness A/B test?
96. What is the Bonferroni correction for multiple faithfulness metric testing?
97. What is the false discovery rate (FDR) in faithfulness evaluation?
98. What is Type I and Type II error in faithfulness evaluation?
99. How do you report faithfulness results to medical stakeholders?
100. What is the regulatory implication of low faithfulness in medical AI?

## 14.3 Answer Relevancy

101. What is answer relevancy in RAGAS?
102. How does RAGAS compute answer relevancy?
103. What is the reverse question generation approach in answer relevancy?
104. How does embedding similarity between original and generated questions measure relevancy?
105. What is the answer relevancy formula in RAGAS?
106. What is the difference between answer relevancy and faithfulness?
107. Can you have high faithfulness but low answer relevancy?
108. Give an example of a faithful but irrelevant answer.
109. What causes low answer relevancy in medical QA?
110. How do you improve answer relevancy?
111. What prompt changes improve answer relevancy?
112. What is the relationship between question type and answer relevancy?
113. How do you measure answer relevancy for factoid vs. explanatory questions?
114. What is the non-committal penalty in RAGAS answer relevancy?
115. What is a non-committal answer?
116. How does temperature affect answer relevancy?
117. What is the answer relevancy score for your production system?
118. How do you monitor answer relevancy in production?
119. What is the answer relevancy alert threshold?
120. How do you improve answer relevancy for multi-part medical questions?

## 14.4 Context Precision

121. What is context precision in RAGAS?
122. How does RAGAS compute context precision?
123. What is the precision@k formula for context?
124. `Context Precision@k = (Σ Precision@k × rel_k) / total relevant` — explain.
125. What is the difference between context precision and retrieval precision?
126. What causes low context precision?
127. How does re-ranking improve context precision?
128. What is the relationship between top-k retrieval count and context precision?
129. What is the trade-off between context precision and context recall?
130. How do you optimize context precision?

## 14.5 Context Recall

131. What is context recall in RAGAS?
132. How does RAGAS compute context recall?
133. What is the ground truth answer's role in context recall computation?
134. What is the relationship between context recall and retrieval recall@k?
135. What causes low context recall?
136. How does hybrid search improve context recall?
137. How does chunk overlap affect context recall?
138. How does chunk size affect context recall?
139. What is the trade-off between context recall and context precision?
140. What is the optimal retrieval top-k for maximizing context recall?

## 14.6 Retrieval Metrics (Precision@K, Recall@K, MRR, MAP, NDCG)

141. What is Precision@K?
142. Write the formula for Precision@K.
143. What is Recall@K?
144. Write the formula for Recall@K.
145. What is the difference between Precision@K and Recall@K?
146. What is MRR (Mean Reciprocal Rank)?
147. Write the MRR formula.
148. When is MRR preferred over Precision@K?
149. What is MAP (Mean Average Precision)?
150. Write the MAP formula.
151. What is the difference between AP and MAP?
152. What is NDCG (Normalized Discounted Cumulative Gain)?
153. Write the NDCG formula.
154. What is DCG (Discounted Cumulative Gain)?
155. Write the DCG formula with graded relevance.
156. What is IDCG (Ideal DCG)?
157. Why do we normalize by IDCG?
158. What is the difference between binary and graded relevance in NDCG?
159. What is the discount function in DCG and why is it log2?
160. What is Hit Rate (HR@K)?
161. What is the difference between Hit Rate and Recall@K?
162. What is R-Precision?
163. What is Precision@R?
164. What is the F1 measure for retrieval?
165. What is the relationship between F1, precision, and recall?
166. What is Success@K?
167. What is MRR@K vs. MRR?
168. How do you compute NDCG with non-binary relevance for medical Q&A?
169. How do you assign relevance grades to retrieved medical passages?
170. What is the Cranfield evaluation framework?
171. What is TREC evaluation methodology?
172. What is the BEIR benchmark and what does it measure?
173. What medical datasets are in BEIR?
174. What is the MIRAGE benchmark for medical RAG?
175. What is PubMedQA?
176. What is MedQA (USMLE-style questions)?
177. What is MedMCQA?
178. What is BioASQ?
179. How do you create your own medical retrieval evaluation benchmark?
180. What is the minimum annotation requirements for a valid retrieval benchmark?

## 14.7 LLM-as-a-Judge

181. What is LLM-as-a-Judge?
182. What is the paper "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"?
183. What is the position bias in LLM-as-a-Judge?
184. What is verbosity bias in LLM-as-a-Judge?
185. What is self-enhancement bias?
186. What is the inter-model agreement problem in LLM-as-a-Judge?
187. What is calibration of LLM judges?
188. How do you calibrate an LLM judge with human annotations?
189. What is the difference between pairwise and pointwise LLM judging?
190. What is G-Eval?
191. What is the G-Eval framework from OpenAI?
192. How does G-Eval use CoT for evaluation?
193. What is the judge model choice for medical QA evaluation?
194. Should you use the same LLM for generation and judging? Why not?
195. What is the self-evaluation bias?
196. How do you mitigate self-evaluation bias?
197. What is the jury-of-judges approach?
198. What is the correlation between LLM judge and human judgment for medical QA?
199. What is Pearson correlation for judge-human alignment?
200. What is Spearman rank correlation for judge-human alignment?

## 14.8 Human Evaluation

201. What is human evaluation in RAG?
202. When is human evaluation necessary vs. automated evaluation?
203. What is the standard human evaluation protocol for medical QA?
204. What dimensions do you ask humans to evaluate?
205. What is a Likert scale and how do you use it for RAG evaluation?
206. What is pairwise human evaluation (A/B preference)?
207. What is absolute human evaluation (individual scoring)?
208. What is inter-annotator agreement?
209. What is Cohen's kappa?
210. What is Fleiss' kappa for multi-annotator agreement?
211. What is Krippendorff's alpha?
212. What is an acceptable inter-annotator agreement for medical annotation?
213. How do you train medical annotators?
214. What is the cost of human evaluation for a medical RAG system?
215. How do you scale human evaluation to thousands of questions?

## 14.9 A/B Testing

216. What is A/B testing in RAG?
217. What is the null hypothesis in a RAG A/B test?
218. What is statistical significance?
219. What is p-value?
220. What is the significance threshold (α)?
221. What is statistical power (1-β)?
222. What is Type I error in A/B testing?
223. What is Type II error in A/B testing?
224. What is the minimum detectable effect (MDE)?
225. How do you calculate the required sample size for a RAG A/B test?
226. What is the t-test for A/B testing continuous metrics?
227. What is the chi-square test for A/B testing proportions?
228. What is the bootstrap confidence interval?
229. What is the Bonferroni correction for multiple comparisons?
230. How do you implement A/B testing for retrieval strategy changes?

## 14.10 RAGAS Deep Dive

231. What is the RAGAS framework architecture?
232. What LLM does RAGAS use internally?
233. How do you configure RAGAS to use a specific LLM?
234. What is the RAGAS `evaluate()` function?
235. What is the RAGAS `Dataset` format?
236. What is the RAGAS `EvaluationDataset`?
237. How do you prepare RAGAS input data?
238. What is the `question`, `answer`, `contexts`, `ground_truth` schema?
239. What is the cost of running RAGAS on 1000 examples?
240. How do you optimize RAGAS cost?
241. What is the RAGAS batch size parameter?
242. How do you parallelize RAGAS evaluation?
243. What is RAGAS async mode?
244. How do you integrate RAGAS with MLflow?
245. How do you log RAGAS results to MLflow?
246. What is the RAGAS TestsetGenerator?
247. How does RAGAS generate synthetic test data?
248. What is the evolution graph in RAGAS test generation?
249. What is simple, reasoning, and multi-context question evolution?
250. How do you generate medical-specific test questions with RAGAS?

## 14.11 DeepEval

251. What is DeepEval?
252. How does DeepEval compare to RAGAS?
253. What metrics does DeepEval provide?
254. What is the `GEval` metric in DeepEval?
255. What is the `ContextualPrecisionMetric` in DeepEval?
256. What is the `FaithfulnessMetric` in DeepEval?
257. What is the `HallucinationMetric` in DeepEval?
258. How do you write a custom metric in DeepEval?
259. What is the DeepEval `assert_test` function?
260. How do you integrate DeepEval into CI/CD?

## 14.12 LangSmith

261. What is LangSmith?
262. What is a LangSmith trace?
263. What is a LangSmith run?
264. What is a LangSmith dataset?
265. What is a LangSmith evaluation?
266. What is the LangSmith feedback mechanism?
267. How do you collect user feedback in LangSmith?
268. What is LangSmith's online evaluation feature?
269. How do you set up automated evaluation in LangSmith?
270. How do you compare LangSmith experiments?

## 14.13 Online Evaluation

271. What is online evaluation vs. offline evaluation?
272. What user signals can be used for online evaluation in medical RAG?
273. What is implicit feedback in online evaluation?
274. What is explicit feedback in online evaluation?
275. What is the thumbs up/down feedback system?
276. What is the follow-up question signal for relevance?
277. What is session abandonment as a negative signal?
278. What is the copy-paste behavior as a positive signal?
279. How do you design an online evaluation system that doesn't annoy users?
280. What is the sample rate for online evaluation?
281. What is the logging infrastructure for online evaluation?
282. What is the real-time vs. batch processing of online feedback?
283. How do you prevent gaming of feedback mechanisms?
284. What is selection bias in online evaluation?
285. How do you correct for position bias in online evaluation?
286. What is inverse propensity scoring?
287. How do you implement real-time quality dashboards?
288. What is the alert threshold for online metric degradation?
289. How do you handle the cold start problem for online evaluation?
290. What is the shadow mode evaluation strategy?

## 14.14 Production Monitoring

291. What metrics do you monitor in production for RAG quality?
292. What is latency monitoring for RAG?
293. What is p99 latency and why is it important?
294. What is throughput monitoring?
295. What is error rate monitoring?
296. What is cost per query monitoring?
297. What is query volume monitoring?
298. What is query type distribution monitoring?
299. What is retrieval quality drift monitoring?
300. What is answer quality drift monitoring?
301. What is embedding drift monitoring in production?
302. How do you detect when retrieval quality degrades silently?
303. What is a canary query set?
304. How do you run a canary query set on every deployment?
305. What is the SLA for medical RAG query response time?
306. How do you implement SLA breach alerts?
307. What is the on-call rotation for production RAG monitoring?
308. What is the incident response procedure for quality degradation?
309. How do you conduct a post-mortem for a quality regression?
310. What is the RCA (Root Cause Analysis) process for RAG quality issues?

## 14.15 Golden Dataset & Regression Testing

311. What is a golden dataset?
312. How do you build a golden dataset for medical RAG?
313. What is the minimum size of a golden dataset?
314. How do you ensure golden dataset quality?
315. What is the medical expert involvement in golden dataset creation?
316. How do you version the golden dataset?
317. How do you update the golden dataset over time?
318. What is regression testing in RAG?
319. How do you implement regression testing in CI/CD?
320. What is the acceptance threshold for regression tests?
321. What happens when a deploy causes a regression test failure?
322. What is the historical performance comparison in regression testing?
323. How do you handle changing golden dataset answers over time?
324. What is the difference between a golden dataset and a test set?
325. How do you ensure the golden dataset covers medical specialty distribution?

## 14.16 Evaluation Metrics Summary Table

326. Define each metric: Faithfulness, Answer Relevancy, Context Precision, Context Recall, MRR, MAP, NDCG, Hit Rate.
327. Which metric is most important for a medical RAG system? Why?
328. Which metric do you sacrifice when optimizing for faithfulness?
329. What is the trade-off between context precision and context recall?
330. What is the Pareto frontier of RAG evaluation metrics?
331. How do you create a composite RAG quality score?
332. What weights would you assign to each metric for medical QA?
333. What is the weighted harmonic mean of RAG metrics?
334. How do you visualize RAG evaluation results?
335. What is a radar chart for multi-metric RAG evaluation?
336. How do you compare two RAG systems across multiple metrics?
337. What is the dominance criterion for multi-objective RAG comparison?
338. What is the Pareto dominance criterion?
339. What is the hypervolume indicator for multi-objective evaluation?
340. How do you implement multi-objective RAG optimization?

## 14.17 Hallucination Detection

341. What is hallucination in LLMs?
342. What is intrinsic hallucination?
343. What is extrinsic hallucination?
344. What is factual hallucination vs. faithful hallucination?
345. What is the hallucination rate metric?
346. How do you compute hallucination rate?
347. What is the HaluEval benchmark?
348. What is the TruthfulQA benchmark?
349. What is the FActScore metric?
350. How does FActScore work?
351. What is the SAFE (Search-Augmented Faithfulness Evaluation) method?
352. What is chain-of-verification (CoVe) for hallucination detection?
353. What is SelfCheckGPT?
354. How does SelfCheckGPT work?
355. What is the consistency-based hallucination detection approach?
356. How do you detect hallucination in medical answers?
357. What is the clinical NLP hallucination challenge?
358. How do you distinguish clinical inference from hallucination?
359. What is the risk of false positives in hallucination detection?
360. How do you reduce false positives in medical hallucination detection?

## 14.18 Continuous Evaluation

361. What is continuous evaluation?
362. What is the difference between batch and continuous evaluation?
363. How do you implement continuous evaluation as a pipeline?
364. What triggers continuous evaluation runs?
365. What is the evaluation frequency for production RAG?
366. How do you handle evaluation data staleness?
367. How do you implement online continuous evaluation with production traffic sampling?
368. What is the infrastructure for continuous evaluation?
369. How do you store continuous evaluation results?
370. What is the continuous evaluation dashboard?
371. How do you alert on continuous evaluation metric drift?
372. What is the rolling window evaluation approach?
373. What is the exponential moving average for evaluation metrics?
374. How do you detect sudden vs. gradual quality degradation?
375. What is anomaly detection for evaluation metrics?

## 14.19 Benchmark Datasets

376. What is MedQA?
377. What is MedMCQA?
378. What is PubMedQA?
379. What is BioASQ?
380. What is MedBench?
381. What is MMLU Medical subset?
382. What is USMLE Steps 1, 2, 3 benchmark performance?
383. What is the MIRAGE benchmark for medical RAG?
384. What is HealthBench (OpenAI)?
385. What is the NEJM Case Record benchmark?
386. What is the MedRAG toolkit?
387. What medical corpus does MedRAG use?
388. How do you evaluate your system on MedQA?
389. What is your system's accuracy on MedQA-style questions?
390. How does your system compare to GPT-4 on medical benchmarks?

## 14.20 Evaluation Cost Optimization

391. What is the cost of running RAGAS on 1000 examples with GPT-4?
392. How do you reduce evaluation cost using a cheaper judge model?
393. What is the accuracy difference between GPT-4-judge and GPT-3.5-judge?
394. How do you validate that a cheaper judge gives consistent results?
395. What is the minimum viable judge model for medical faithfulness evaluation?
396. How do you use local models (Llama, Mistral) as judges to reduce cost?
397. What is the quality trade-off of using Llama-3 as a judge vs. GPT-4?
398. How do you implement caching of evaluation calls to reduce cost?
399. What is the batch API discount for evaluation calls?
400. How do you estimate total evaluation cost for a production system?

## 14.21 Evaluation Anti-Patterns

401. What is the "evaluation on training data" anti-pattern?
402. What is data leakage in RAG evaluation?
403. What is the "fixed golden dataset" anti-pattern?
404. What is the "evaluating only happy paths" anti-pattern?
405. What is the "single metric" anti-pattern?
406. What is the "ignoring latency" anti-pattern?
407. What is the "no statistical testing" anti-pattern?
408. What is the "evaluating without production representative data" anti-pattern?
409. What is the "ignoring evaluation drift" anti-pattern?
410. What is the "not evaluating edge cases" anti-pattern?

## 14.22 Evaluation Research Papers

411. What is the RAGAS paper? Key contributions?
412. What is the ARES paper (Automated RAG Evaluation System)?
413. What is the CRUD-RAG paper?
414. What is the RGB (Retrieval-Generation Benchmark) paper?
415. What is the TruLens paper?
416. What is the DeepEval methodology paper?
417. What is the BLEU score and why is it insufficient for RAG evaluation?
418. What is ROUGE and why is it insufficient for RAG evaluation?
419. What is BERTScore?
420. When is BERTScore useful for RAG evaluation?
421. What is the limitation of BERTScore for medical QA?
422. What is the BLEURT metric?
423. What is the MoverScore metric?
424. What is UniEval?
425. What is the G-Eval paper?

## 14.23 Evaluation System Architecture

426. Design an evaluation system for a production medical RAG chatbot.
427. What is the evaluation pipeline architecture?
428. What database do you use to store evaluation results?
429. What is the ETL pipeline for evaluation data?
430. How do you sample production traffic for evaluation?
431. What is the evaluation micro-service architecture?
432. How do you ensure evaluation does not impact production latency?
433. What is the evaluation SLA?
434. How do you version evaluation results?
435. What is the evaluation result schema?

## 14.24 Evaluation Security

436. How do you protect PHI in evaluation datasets?
437. How do you anonymize medical evaluation data?
438. What is k-anonymity for medical evaluation data?
439. How do you ensure evaluation data is not used for LLM training?
440. What is the data lineage requirement for evaluation in medical AI?
441. How do you audit evaluation data access?
442. What is the HIPAA requirement for evaluation data handling?
443. How do you deidentify patient data for evaluation?
444. What is differential privacy for evaluation data?
445. How do you comply with GDPR for evaluation data in Europe?

## 14.25 Future of RAG Evaluation

446. What is the future of automated RAG evaluation?
447. Will LLM judges replace human evaluation entirely?
448. What is the role of multi-agent evaluation systems?
449. What is Constitutional AI and how does it apply to RAG evaluation?
450. What is the role of preference learning in RAG evaluation?
451. What is RLHF-based evaluation?
452. What is the DPO (Direct Preference Optimization) approach for evaluation?
453. What is reward modeling for RAG quality?
454. What is the future of domain-specific evaluation for medical AI?
455. What is the role of regulatory bodies (FDA, EMA) in medical AI evaluation?
456. What is FDA's stance on AI/ML-based medical devices?
457. What is the SaMD (Software as a Medical Device) classification?
458. What is the FDA guidance on AI transparency and auditability?
459. What is the EU AI Act's impact on medical RAG systems?
460. What is GDPR Article 22 and how does it affect automated medical decisions?
461. What is the role of clinical validation studies for medical RAG?
462. What is IRB (Institutional Review Board) approval for medical AI evaluation?
463. What is the difference between retrospective and prospective clinical evaluation?
464. What is the CONSORT AI extension for reporting medical AI trials?
465. What is the SPIRIT-AI extension for clinical AI protocol reporting?

## 14.26 Cost vs. Quality Trade-off Evaluation

466. How do you measure the cost-quality Pareto frontier in RAG?
467. What is the cost of high faithfulness vs. lower faithfulness?
468. What is the quality impact of using a smaller LLM for generation?
469. What is the quality impact of reducing top-k retrieval?
470. What is the quality impact of removing re-ranking?
471. What is the quality impact of cheaper embedding models?
472. How do you optimize the cost-quality trade-off for your budget?
473. What is the minimum quality acceptable for a medical RAG system?
474. Who defines the minimum quality threshold for medical AI?
475. What is the accountability structure for quality decisions?

## 14.27 Evaluation Data Curation

476. How do you curate a high-quality medical evaluation dataset?
477. What is stratified sampling for evaluation dataset curation?
478. How do you balance question difficulty in the evaluation set?
479. How do you balance medical specialty coverage?
480. What is adversarial example curation for evaluation?
481. What is edge case curation for evaluation?
482. How do you handle ambiguous medical questions in evaluation?
483. What is the annotation guideline for medical QA evaluation?
484. How do you handle disagreements among medical annotators?
485. What is the adjudication process for annotation disagreements?
486. How do you ensure annotation quality?
487. What is annotation quality control (QC)?
488. How do you use inter-annotator agreement as a QC metric?
489. What is pilot annotation and why is it important?
490. What is the training phase for medical annotators?
491. How do you prevent annotator fatigue?
492. What is the annotation session length recommendation?
493. What is the compensation model for medical annotation?
494. What are the regulatory requirements for annotation records?
495. How do you archive evaluation datasets for regulatory compliance?

## 14.28 Evaluation at Scale

496. How do you run RAG evaluation on 100,000 examples?
497. What infrastructure do you need for large-scale evaluation?
498. How do you parallelize evaluation across multiple machines?
499. What is the evaluation throughput bottleneck?
500. How do you reduce evaluation time from 1 week to 1 hour?

---

# SECTION 15 — SCENARIO-BASED QUESTIONS (500+) {#section-15}

## 15.1 System Failure Scenarios

1. **Pinecone is completely unavailable.** Walk me through your exact response process.
2. **Pinecone is responding but with 10x latency.** How do you detect and mitigate this?
3. **Pinecone returns 0 results for all queries.** Root cause analysis?
4. **Pinecone returns corrupted metadata.** How do you detect and handle it?
5. **Your Pinecone index is accidentally deleted.** What is your recovery procedure?
6. **Pinecone index becomes inconsistent after a partial upsert failure.** How do you reconcile?
7. **The LLM API (Groq) is completely down.** What is your fallback?
8. **The LLM API returns garbled output for 10% of requests.** How do you detect this?
9. **The LLM API rate limits you at 3 AM.** How does your system handle this?
10. **The LLM API returns a 500 error.** What is your retry logic?
11. **The embedding API is unavailable.** How does this affect query time?
12. **The embedding model returns inconsistent dimensions.** How do you detect this?
13. **AWS EC2 instance is terminated by a spot reclamation.** What happens to in-flight requests?
14. **The Docker container crashes mid-response.** What happens to the user?
15. **GitHub Actions pipeline fails during deployment.** How do you detect and rollback?
16. **FastAPI runs out of worker threads.** What does the user experience?
17. **The server disk is full.** How does this affect logging?
18. **A memory leak causes OOM after 24 hours.** How do you detect and fix it?
19. **The network connection between EC2 and Pinecone has 500ms added latency.** Impact?
20. **The cross encoder model file is corrupted.** How does your system fail gracefully?

## 15.2 Data Pipeline Scenarios

21. **A new batch of 1,000 medical PDFs arrives.** Walk me through the ingestion pipeline.
22. **A medical PDF contains 50% scanned pages.** How does your pipeline handle this?
23. **A PDF is corrupted and cannot be parsed.** What happens?
24. **A PDF is password-protected.** How do you handle it?
25. **A PDF contains only images with no text layer.** What is your OCR pipeline?
26. **OCR fails on a low-quality scan.** What is your fallback?
27. **A document is uploaded in a language other than English.** How do you handle it?
28. **A duplicate document is uploaded.** How do you detect and handle duplicates?
29. **A document that was previously indexed is updated.** How do you re-index?
30. **A document needs to be deleted from the knowledge base.** Walk me through the deletion process.
31. **New documents arrive at 1 per second continuously.** How does your pipeline scale?
32. **A 1,000-page medical textbook is uploaded.** How does chunking perform?
33. **A document contains a 100-row drug interaction table.** How do you chunk it?
34. **A document has inconsistent formatting across pages.** How does chunking handle it?
35. **Chunking produces 0 chunks for a document.** What went wrong?

## 15.3 Quality Degradation Scenarios

36. **Hallucination rate increases from 2% to 15% after a deploy.** Root cause?
37. **Faithfulness score drops from 0.85 to 0.60 after adding new documents.** Why?
38. **Context precision drops for cardiology questions only.** What do you investigate?
39. **Answer relevancy is low for multi-part questions.** How do you fix it?
40. **The re-ranker is consistently putting the worst chunk first.** Root cause?
41. **Users report wrong drug dosages in answers.** Immediate response?
42. **The LLM is generating answers not grounded in context.** How do you detect and fix?
43. **Retrieval quality is good but generation quality is low.** What changed?
44. **Retrieval quality is low but generation quality is good.** What changed?
45. **Quality is good for common questions but poor for rare disease queries.** Why?

## 15.4 Scaling Scenarios

46. **User volume doubles overnight from 1,000 to 2,000/day.** What breaks first?
47. **User volume grows to 100,000/day gradually.** What is your scaling roadmap?
48. **You receive 10,000 concurrent requests during peak hours.** How does the system behave?
49. **A viral social media post sends 100,000 users to your system in 1 hour.** Response?
50. **A medical conference sends 50,000 professionals to your system simultaneously.** Plan?
51. **The vector index grows from 1M to 100M documents.** What changes?
52. **Embedding model inference becomes the bottleneck at 10,000 QPS.** Solution?
53. **Pinecone query latency grows as index size increases.** How do you mitigate?
54. **Re-ranking latency grows under high load.** How do you auto-scale re-ranking?
55. **FastAPI worker pool is exhausted.** What is your auto-scaling trigger?

## 15.5 Security Attack Scenarios

56. **A user submits a prompt injection attack.** Detect and response?
57. **A user uploads a malicious PDF with embedded prompt injection.** How is it handled?
58. **A user tries to extract the contents of your vector store.** Prevention?
59. **A user performs a jailbreak attack to get harmful medical advice.** Response?
60. **A user probes your system to extract the system prompt.** Prevention?
61. **A competitor attempts to steal your medical knowledge base via API scraping.** Detection?
62. **A user uploads a document poisoned with false medical information.** Detection?
63. **An attacker submits 10,000 requests/second (DDoS).** Response?
64. **An attacker gains access to your API key.** Incident response?
65. **A data breach exposes user queries containing PHI.** HIPAA incident response?

## 15.6 Medical Domain Scenarios

66. **A user asks about maximum safe dosage of acetaminophen.** How does your system respond?
67. **A user asks about drug interactions for 5 simultaneous medications.** Quality?
68. **A user asks a question about a rare disease with no documents in the knowledge base.** Response?
69. **A user asks a question that requires clinical reasoning beyond the retrieved documents.** Response?
70. **Two documents in the knowledge base contradict each other.** How does the LLM handle this?
71. **A user asks for personalized treatment recommendations.** How do you handle this ethically?
72. **A user is in an emergency and needs immediate medical help.** System response?
73. **A user asks a question in medical jargon vs. layman terms.** Quality difference?
74. **A medical guideline is updated — old version is in the knowledge base.** How do you handle it?
75. **A user asks about experimental treatments not in standard guidelines.** System behavior?

## 15.7 Data Quality Scenarios

76. **Your medical corpus has 30% OCR errors.** Impact on retrieval quality?
77. **Your corpus has inconsistent terminology (Brand vs. Generic drug names).** How do you handle?
78. **Your corpus mixes old and new clinical guidelines.** How do you filter by date?
79. **Your corpus has duplicate information across multiple documents.** Impact?
80. **Your corpus has missing sections (OCR failure on some pages).** Detection?
81. **Your corpus grows 10x overnight due to a data import error.** How do you detect and rollback?
82. **Your corpus contains documents in 10 different formats.** Chunking strategy?
83. **Your corpus has inconsistent metadata (some documents missing specialty tags).** Impact?
84. **Your corpus includes patient data that should be excluded.** Discovery and remediation?
85. **Your corpus has a 5-year-old document that contradicts current evidence.** Priority?

## 15.8 Performance Scenarios

86. **Query latency spikes to 30 seconds for 1% of queries.** Investigation?
87. **Embedding computation takes 5 seconds per query.** What is wrong?
88. **Re-ranking adds 8 seconds to query latency.** How do you fix it?
89. **Pinecone query takes 3 seconds when it usually takes 100ms.** Root cause?
90. **LLM generation takes 45 seconds for a long response.** Acceptable? Solution?
91. **Cold start after auto-scaling adds 60 seconds to first query.** How do you reduce?
92. **Memory usage grows to 16GB on a t3.medium instance.** What is happening?
93. **CPU usage is 100% during peak hours.** Investigation and solution?
94. **Database connection pool is exhausted.** Symptoms and solution?
95. **Network bandwidth is saturated.** Investigation?

## 15.9 Integration Failure Scenarios

96. **LangChain version upgrade breaks your RAG pipeline.** Response?
97. **Pinecone SDK upgrade changes the API interface.** How do you handle?
98. **The embedding model deprecates the API endpoint you use.** Migration?
99. **Groq changes its pricing model and becomes too expensive.** Migration strategy?
100. **A third-party library has a critical security vulnerability.** Response?
101. **Python version upgrade causes dependency conflicts.** Debugging?
102. **Docker base image update introduces a breaking change.** Detection?
103. **GitHub Actions runner update changes behavior.** Investigation?
104. **AWS region pricing changes.** Cost optimization response?
105. **Pinecone introduces breaking changes to metadata filtering.** Migration?

## 15.10 Compliance Scenarios

106. **A HIPAA audit reveals potential PHI in your vector store.** Remediation?
107. **A user requests deletion of all their data (GDPR Right to be Forgotten).** Process?
108. **A medical professional asks for sources of every claim.** Can your system provide this?
109. **A regulator asks for audit logs of all queries and responses.** Are they available?
110. **A legal team needs to understand the decision-making process for a specific answer.** Process?
111. **A patient claims your system gave incorrect medical advice.** Incident response?
112. **A malpractice lawsuit cites your system's advice.** Documentation you need?
113. **A GDPR data transfer request from EU requires data export.** Process?
114. **A security audit requires penetration testing of your AI system.** Process?
115. **An FDA inquiry about your system's classification as SaMD.** Your response?

## 15.11 Infrastructure Scenarios

116. **AWS goes down in us-east-1.** Failover procedure?
117. **Your EC2 instance type is deprecated by AWS.** Migration?
118. **Docker Hub rate limits your image pulls in CI/CD.** Solution?
119. **GitHub Actions minutes run out mid-month.** What happens to CI/CD?
120. **Your AWS budget is exceeded by 3x in one day.** Investigation?
121. **Auto scaling creates 100 instances due to a scaling bug.** Emergency response?
122. **A zombie process consumes all CPU on the server.** Detection and response?
123. **Your S3 bucket for MLflow artifacts becomes full.** Impact?
124. **A network partition separates your application from Pinecone.** Behavior?
125. **The ALB health check fails for all instances.** Impact on users?

## 15.12 Concurrent User Scenarios

126. **100 users submit the same query simultaneously.** Can you serve all?
127. **1000 users upload PDFs simultaneously.** Ingestion pipeline behavior?
128. **Multiple users share the same account.** Data isolation?
129. **A user submits 1000 queries per second from a script.** Rate limiting?
130. **Multiple CI/CD pipelines run evaluation simultaneously.** Conflict?
131. **Two engineers deploy simultaneously.** Race condition?
132. **Concurrent Pinecone upserts from multiple workers.** Consistency?
133. **Multiple re-ranking requests compete for GPU resources.** Queuing?
134. **A user starts a very long conversation that fills the context window.** Truncation?
135. **Multiple monitoring agents write to CloudWatch simultaneously.** Throttling?

## 15.13 Model & AI Scenarios

136. **The LLM starts generating toxic medical content.** Detection and prevention?
137. **The LLM confidently generates wrong drug names.** Safeguards?
138. **Embedding drift is detected in production.** What do you do?
139. **A new embedding model outperforms your current one by 10%.** Migration plan?
140. **The cross encoder starts re-ranking adversarially (worst first).** Detection?
141. **The LLM response is truncated mid-sentence.** Handling?
142. **Streaming response breaks after 50 tokens.** Debugging?
143. **The LLM generates a very long response (10,000 tokens).** Impact?
144. **The LLM temperature is accidentally set to 2.0 in production.** Symptoms?
145. **The cross encoder returns NaN scores.** What happened?

## 15.14 Cost Explosion Scenarios

146. **Monthly cost triples unexpectedly.** Investigation and response?
147. **A single user makes 100,000 API calls in one day.** Detection?
148. **Pinecone costs exceed budget due to unexpected growth.** Optimization?
149. **LLM API costs spike due to very long prompts.** Detection?
150. **Embedding costs explode due to re-indexing bug.** Prevention?

## 15.15 Agentic & Multi-Step Scenarios

151. **A medical question requires information from 5 different documents.** Multi-hop RAG?
152. **A question requires step-by-step medical calculation.** CoT integration?
153. **A question requires comparing two treatments.** Comparative RAG?
154. **A question is ambiguous.** Clarification flow?
155. **A question has multiple valid answers.** Handling?
156. **A question is outside your medical domain entirely.** Response?
157. **A user continues a multi-turn conversation.** Context management?
158. **A user asks a follow-up that requires the previous answer.** Context inclusion?
159. **A user asks for a summary of 50 documents.** Scalable summarization?
160. **A user asks to compare your system to GPT-4.** Response?

## 15.16 Data Governance Scenarios

161. **A data source for your knowledge base is taken down.** Impact?
162. **Copyright claim is made on a document in your knowledge base.** Response?
163. **A medical journal updates its citation policy.** Compliance?
164. **A government regulation changes medical treatment guidelines.** Knowledge base update?
165. **A pandemic causes outdated treatment protocols to be urgent.** Priority update?

## 15.17 Evaluation Failure Scenarios

166. **Faithfulness drops by 20% after a new document batch.** Debugging?
167. **RAGAS evaluation fails to complete due to rate limits.** Handling?
168. **Evaluation results are inconsistent between runs.** Root cause?
169. **The evaluation LLM gives inconsistent judgments.** Mitigation?
170. **The golden dataset becomes stale.** Update process?

## 15.18 Multi-tenant Scenarios

171. **Two hospitals use your system with different medical corpora.** Architecture?
172. **A hospital wants to add proprietary documents not shared with others.** Isolation?
173. **A tenant's documents are exposed to another tenant.** Security incident?
174. **Different tenants have different latency SLAs.** Implementation?
175. **A tenant wants to delete all their data.** Process?

## 15.19 Edge Case Query Scenarios

176. **User submits an empty query.** System behavior?
177. **User submits a 10,000-word query.** Handling?
178. **User submits a query in emoji only.** Behavior?
179. **User submits a query with SQL injection.** Prevention?
180. **User submits a query with XSS script.** Prevention?
181. **User submits a query in binary.** Handling?
182. **User submits a query in ancient Greek.** Behavior?
183. **User submits a query that is completely gibberish.** Behavior?
184. **User submits the same query 10,000 times in a loop.** Rate limiting?
185. **User submits a query designed to maximize token usage.** Protection?

## 15.20 Production Readiness Scenarios

186. **Your system is asked to handle 10x traffic in 2 hours notice.** Plan?
187. **The CTO asks for a runbook for on-call engineers.** What's in it?
188. **A new engineer joins and needs to understand the system in 1 hour.** Documentation?
189. **An external security audit finds 3 critical vulnerabilities.** Priority and response?
190. **A major competitor launches a similar system.** How do you differentiate?

---

# SECTION 16 — PRODUCTION INCIDENTS {#section-16}

## 16.1 Memory Leak

1. What is a memory leak in a Python application?
2. How do you detect a memory leak in a running FastAPI server?
3. What tools do you use to profile Python memory? (tracemalloc, memray, memory_profiler)
4. What is `tracemalloc`?
5. What is `memray` and how does it visualize memory usage?
6. What is the common cause of memory leaks in LangChain applications?
7. What is a reference cycle in Python and how does garbage collection handle it?
8. How does the Python GC handle circular references?
9. What is `gc.collect()` and when should you call it?
10. What is a common memory leak cause with Pinecone connections?
11. How do you detect memory leak in a Docker container?
12. What is `docker stats` and what metrics does it show?
13. How do you set memory limits on Docker containers?
14. What happens when a Docker container exceeds its memory limit?
15. How do you implement memory monitoring and alerting?

## 16.2 GPU OOM

16. What is GPU OOM (Out of Memory) error?
17. What causes GPU OOM in LLM inference?
18. What is the GPU memory requirement for Llama 3.3 70B in FP16?
19. How do you reduce GPU memory usage?
20. What is gradient checkpointing and does it help at inference?
21. What is the role of KV cache in GPU OOM?
22. How does batch size affect GPU memory?
23. How do you implement dynamic batch size for GPU memory management?
24. What is `torch.cuda.empty_cache()` and when do you call it?
25. How do you monitor GPU memory in production?

## 16.3 Slow Queries

26. How do you define a "slow query" in your RAG system?
27. What is the p99 latency threshold for alerts?
28. How do you identify the bottleneck in a slow query?
29. What is the breakdown of latency in your RAG pipeline?
30. How do you instrument each step for latency measurement?
31. What is the typical latency for Pinecone query vs. LLM generation?
32. How do you profile async FastAPI handlers?
33. What is the `time.perf_counter()` approach for latency measurement?
34. How do you use OpenTelemetry spans for latency tracing?
35. What is the first thing you check when queries suddenly become slow?

## 16.4 Cold Starts

36. What is a cold start in a RAG application?
37. What causes cold starts?
38. What is the cold start latency for your system?
39. How do you minimize cold start latency?
40. What is model pre-loading at server startup?
41. How do you keep the cross encoder model warm?
42. What is the role of connection pooling in reducing cold starts?
43. How do you warm up Pinecone connections?
44. What is a warm-up endpoint?
45. How do you implement health check warm-up?

## 16.5 Docker Crash & Recovery

46. What causes Docker container crashes?
47. How do you configure Docker restart policies?
48. What is the difference between `--restart=always` and `--restart=on-failure`?
49. How do you implement Docker health checks?
50. What is the Docker `HEALTHCHECK` instruction?
51. How does orchestration (ECS, Kubernetes) handle container crashes?
52. What is the backoff strategy for container restarts?
53. How do you prevent crash loops?
54. How do you analyze a Docker container crash post-mortem?
55. What logs do you collect from a crashed container?

## 16.6 Connection Pool Exhaustion

56. What is connection pool exhaustion?
57. What is the connection pool limit for SQLAlchemy?
58. What is the connection pool for an HTTP client (httpx)?
59. How do you configure connection pool size for Pinecone client?
60. How do you detect connection pool exhaustion?
61. What error messages indicate connection pool exhaustion?
62. How do you tune connection pool size?
63. What is the queue size for waiting connections?
64. What is the connection timeout vs. pool timeout?
65. How do you gracefully handle pool exhaustion in production?

## 16.7 Deadlock & Race Conditions

66. What is a deadlock in a multi-threaded Python application?
67. Can asyncio have deadlocks? Under what circumstances?
68. What is a race condition in FastAPI?
69. What is the global state problem in FastAPI applications?
70. How do you use locks in asyncio?
71. What is `asyncio.Lock()`?
72. What is `asyncio.Semaphore()` for controlling concurrency?
73. How do you prevent race conditions in Pinecone upserts?
74. How do you detect deadlocks in production?
75. What is the circuit breaker pattern for deadlock prevention?

## 16.8 Retry Storms & Circuit Breakers

76. What is a retry storm?
77. How do retries under failure create retry storms?
78. What is exponential backoff?
79. What is jitter and why is it added to exponential backoff?
80. What is the maximum retry count for LLM API calls?
81. What is a circuit breaker pattern?
82. What are the three states of a circuit breaker?
83. When does a circuit breaker open?
84. When does a circuit breaker close (recover)?
85. How do you implement a circuit breaker in Python?
86. What is `tenacity` library for retry logic?
87. What is the `pybreaker` library for circuit breakers?
88. How do you configure circuit breaker for Pinecone?
89. How do you configure circuit breaker for the LLM API?
90. What is the half-open state in circuit breakers?

## 16.9 Cache Stampede

91. What is a cache stampede?
92. What is thundering herd problem?
93. How does cache stampede occur in a RAG system?
94. What is the dog-pile effect?
95. How do you prevent cache stampede?
96. What is probabilistic early expiration?
97. What is the cache lock approach?
98. What is the background refresh approach?
99. How do you implement cache warm-up?
100. What is the cache warming strategy after deployment?

## 16.10 High Latency Investigation

101. Walk me through a high latency incident investigation step by step.
102. What is your first action when p99 latency exceeds SLA?
103. How do you use distributed tracing to find the latency source?
104. What is the runbook for high latency incidents?
105. How do you communicate high latency incidents to users?
106. What is your escalation path for high latency?
107. How do you implement an emergency latency mitigation (e.g., dropping re-ranking)?
108. What is the feature flag for disabling re-ranking under load?
109. How do you implement graceful degradation for high latency?
110. How do you validate that latency is restored after mitigation?

---

# SECTION 17 — SECURITY & HIPAA {#section-17}

## 17.1 Prompt Injection

1. What is prompt injection in LLM applications?
2. What is direct prompt injection?
3. What is indirect prompt injection?
4. Give an example of indirect prompt injection through a medical document.
5. What is the "ignore previous instructions" attack?
6. What is the hierarchy of instructions in LLM prompts?
7. How do you implement instruction isolation?
8. What is the XML/tag-based instruction separation approach?
9. What is the benefit of putting user input in clearly delimited blocks?
10. What is the role of fine-tuned instruction-following in injection resistance?
11. What is the OWASP LLM Top 10? List all 10.
12. What is LLM01: Prompt Injection in OWASP LLM Top 10?
13. What is LLM02: Insecure Output Handling?
14. What is LLM03: Training Data Poisoning?
15. What is LLM06: Sensitive Information Disclosure?
16. How do you implement input sanitization for medical RAG?
17. What regex patterns do you use for prompt injection detection?
18. What is semantic similarity-based injection detection?
19. What is the LlamaGuard injection detection approach?
20. How do you test your system for prompt injection?

## 17.2 Data Poisoning & Vector Poisoning

21. What is data poisoning in the context of RAG?
22. What is vector store poisoning?
23. How can an attacker poison a medical RAG knowledge base?
24. What is the impact of poisoned medical data?
25. How do you prevent unauthorized document uploads?
26. What is document provenance tracking?
27. How do you implement document approval workflow?
28. What is content scanning for document uploads?
29. How do you detect malicious content in PDFs?
30. What is the PDF parsing sandbox?

## 17.3 HIPAA Compliance

31. What is HIPAA?
32. What is PHI (Protected Health Information)?
33. What are the 18 HIPAA identifiers?
34. How do you ensure PHI is not stored in your vector database?
35. What is de-identification in HIPAA?
36. What is the safe harbor method for HIPAA de-identification?
37. What is the expert determination method?
38. What is a BAA (Business Associate Agreement) and with whom do you need one?
39. Do you have a BAA with AWS?
40. Do you have a BAA with Pinecone?
41. Do you have a BAA with OpenAI or Groq?
42. What HIPAA safeguards apply to cloud-hosted medical AI?
43. What is the minimum necessary standard in HIPAA?
44. What is the HIPAA Security Rule's encryption requirement?
45. What is AES-256 encryption and where do you use it?
46. What is TLS 1.3 and why is it important for medical APIs?
47. What is HIPAA audit log requirement?
48. How long must you retain HIPAA audit logs?
49. What is the HIPAA Breach Notification Rule?
50. What is the 60-day breach notification requirement?

## 17.4 Authentication & Authorization

51. What is the difference between authentication and authorization?
52. What is JWT and how does it work?
53. What is the JWT secret and how do you protect it?
54. What is RS256 JWT signing? Why is it more secure than HS256?
55. What is JWT expiration and how do you handle token refresh?
56. What is refresh token and access token pattern?
57. What is OAuth 2.0 and when do you use it for medical apps?
58. What is OpenID Connect (OIDC)?
59. What is RBAC (Role-Based Access Control)?
60. What roles would you define for a medical RAG system?
61. What is ABAC (Attribute-Based Access Control)?
62. How do you implement RBAC in FastAPI?
63. What is the principle of least privilege in API design?
64. How do you implement API key management?
65. What is API key rotation?

## 17.5 Encryption

66. What is encryption at rest?
67. What is encryption in transit?
68. What is AES-256?
69. What is TLS and what version should you use?
70. What is mTLS (mutual TLS) and when do you need it?
71. What is the difference between symmetric and asymmetric encryption?
72. What is the key management requirement for HIPAA?
73. What is AWS KMS?
74. How do you use KMS for database encryption?
75. What is envelope encryption?

## 17.6 Network Security

76. What is a VPC and how does it protect your system?
77. What is a security group and what rules do you configure?
78. What is the principle of deny by default for security groups?
79. What ports do you open for a medical RAG API?
80. What is a WAF (Web Application Firewall)?
81. What is AWS WAF?
82. What is OWASP API Top 10?
83. What is API01: Broken Object Level Authorization?
84. What is API02: Broken Authentication?
85. What is API03: Excessive Data Exposure?
86. What is DDoS protection for a medical API?
87. What is AWS Shield Standard vs. Advanced?
88. What is rate limiting at the API level?
89. What is IP-based rate limiting vs. user-based rate limiting?
90. What is the CAPTCHA approach for bot prevention?

## 17.7 Secrets Management

91. What are the types of secrets in your system?
92. What is the risk of secrets in environment variables?
93. What is the risk of secrets in Docker images?
94. What is the risk of secrets in git history?
95. How do you detect secrets in git history? (git-secrets, truffleHog)
96. What is AWS Secrets Manager?
97. What is the secret rotation strategy?
98. What is the zero-downtime secret rotation procedure?
99. What is HashiCorp Vault?
100. What is SOPS (Secrets OPerationS)?

## 17.8 AI-Specific Security

101. What is model theft via API?
102. How do you detect model extraction attacks?
103. What is adversarial examples for medical AI?
104. What is membership inference attack?
105. What is model inversion attack?
106. What is differential privacy for LLMs?
107. What is the risk of LLM memorization of training data?
108. What is the Copilot/GitHub memorization controversy?
109. How do you test for training data memorization?
110. What is the role of AI red teaming in medical AI?

## 17.9 Incident Response

111. What is your security incident response plan?
112. What is the incident response team (IRT)?
113. What is the incident classification (P0, P1, P2, P3)?
114. What is the P0 incident response SLA?
115. What is the escalation path for a security incident?
116. What is the role of a CISO in medical AI security?
117. What is a tabletop exercise for security incident response?
118. What is the post-incident review process?
119. What documentation is required for HIPAA breach reporting?
120. Who must be notified in a HIPAA breach?

---

# SECTION 18 — CODING INTERVIEW {#section-18}

## 18.1 Chunking Implementation

**Question 1**: Implement a recursive character text splitter.
```python
def recursive_split(text: str, chunk_size: int, chunk_overlap: int, 
                    separators: list[str]) -> list[str]:
    # Implement this function
    pass
```

**Question 2**: Implement a token-aware chunker using tiktoken.
```python
import tiktoken

def token_aware_chunk(text: str, max_tokens: int, overlap_tokens: int, 
                      model: str = "gpt-4") -> list[str]:
    # Count tokens correctly and split
    pass
```

**Question 3**: Implement a chunk deduplication function.
```python
def deduplicate_chunks(chunks: list[str], similarity_threshold: float = 0.95) -> list[str]:
    # Remove near-duplicate chunks using embeddings
    pass
```

## 18.2 Retriever Implementation

**Question 4**: Implement a basic vector retriever with Pinecone.
```python
from pinecone import Pinecone

def retrieve(query: str, top_k: int, pc: Pinecone, index_name: str, 
             embed_fn) -> list[dict]:
    pass
```

**Question 5**: Implement a hybrid search retriever.
```python
def hybrid_retrieve(query: str, top_k: int, alpha: float, 
                    dense_index, sparse_index) -> list[dict]:
    # Combine dense and sparse with alpha weighting
    pass
```

**Question 6**: Implement Reciprocal Rank Fusion.
```python
def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[str]:
    # Merge ranked lists using RRF
    pass
```

## 18.3 FastAPI Implementation

**Question 7**: Implement an async streaming RAG endpoint.
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(query: str):
    # Implement streaming RAG response
    pass
```

**Question 8**: Implement JWT authentication middleware.
```python
from fastapi import Depends, HTTPException
from jose import JWTError, jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    pass
```

**Question 9**: Implement rate limiting middleware.
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")
async def query_endpoint(request: Request, query: str):
    pass
```

## 18.4 Caching Implementation

**Question 10**: Implement semantic caching for RAG queries.
```python
import redis
import numpy as np

class SemanticCache:
    def __init__(self, redis_client: redis.Redis, embed_fn, threshold: float = 0.95):
        self.redis = redis_client
        self.embed_fn = embed_fn
        self.threshold = threshold
    
    def get(self, query: str) -> str | None:
        pass
    
    def set(self, query: str, answer: str, ttl: int = 3600):
        pass
```

**Question 11**: Implement an LRU cache for embedding results.
```python
from functools import lru_cache
import hashlib

class EmbeddingCache:
    def get_embedding(self, text: str) -> list[float]:
        pass
```

## 18.5 MLflow Integration

**Question 12**: Implement RAGAS evaluation with MLflow logging.
```python
import mlflow
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def evaluate_and_log(dataset, run_name: str):
    pass
```

**Question 13**: Implement experiment comparison in MLflow.
```python
import mlflow

def compare_experiments(exp_name: str, metric: str) -> pd.DataFrame:
    # Return top N runs sorted by metric
    pass
```

## 18.6 Evaluation Implementation

**Question 14**: Implement precision@k for retrieval evaluation.
```python
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    pass
```

**Question 15**: Implement NDCG@k.
```python
def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    pass
```

**Question 16**: Implement MRR.
```python
def mean_reciprocal_rank(queries: list[str], retrieved: list[list[str]], 
                          relevant: list[set[str]]) -> float:
    pass
```

## 18.7 Security Implementation

**Question 17**: Implement prompt injection detection.
```python
def detect_prompt_injection(query: str, patterns: list[str]) -> bool:
    pass
```

**Question 18**: Implement PII detection and redaction.
```python
import spacy

def redact_pii(text: str, nlp) -> str:
    # Detect and redact names, dates, locations
    pass
```

## 18.8 Docker Implementation

**Question 19**: Write an optimized Dockerfile for FastAPI RAG application.
```dockerfile
# Write your Dockerfile here
```

**Question 20**: Write a docker-compose.yml for local development.
```yaml
# Include FastAPI, Redis for caching, MLflow server
```

## 18.9 Cross Encoder Re-ranking

**Question 21**: Implement cross encoder re-ranking.
```python
from sentence_transformers import CrossEncoder

def rerank(query: str, candidates: list[str], model: CrossEncoder, 
           top_k: int) -> list[tuple[str, float]]:
    pass
```

**Question 22**: Implement async batched re-ranking.
```python
async def async_rerank(query: str, candidates: list[str], 
                       model: CrossEncoder, batch_size: int = 16) -> list[str]:
    pass
```

## 18.10 System Health Monitoring

**Question 23**: Implement a health check endpoint with component status.
```python
@app.get("/health")
async def health_check():
    # Check Pinecone, LLM API, embedding model
    pass
```

**Question 24**: Implement request latency logging middleware.
```python
@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    pass
```

---

# SECTION 19 — SYSTEM DESIGN INTERVIEW {#section-19}

## 19.1 Medical RAG at 100 Users

**Problem**: Design a Medical RAG platform for 100 users.

### Requirements
- 100 concurrent users
- Medical document corpus of 100,000 pages
- Sub-3-second response time
- HIPAA compliant
- 99.9% uptime

### Architecture
- Single EC2 instance (m5.large or m5.xlarge)
- FastAPI + Uvicorn with 4 workers
- Pinecone serverless index
- Groq for LLM inference
- Redis (ElastiCache) for caching
- S3 for document storage
- CloudWatch for monitoring
- ALB for load balancing (even at small scale for future)

### Key Questions at This Scale
1. What EC2 instance type would you use?
2. How many Uvicorn workers?
3. What is the Pinecone index size?
4. Do you need Redis at 100 users?
5. What is your caching hit rate expected to be?
6. Do you need auto-scaling at 100 users?
7. What is your monthly cost at this scale?
8. What is your deployment procedure?
9. What monitoring alerts do you set up?
10. What is your backup strategy?

## 19.2 Medical RAG at 10,000 Users

**Problem**: Scale the system to 10,000 concurrent users.

### New Challenges
- Load distribution across multiple instances
- Embedding inference becomes a bottleneck
- Connection pool limits on Pinecone
- LLM API rate limits
- Cache becomes critical

### Architecture Changes
- Move to Auto Scaling Group (min: 3, max: 20 instances)
- ALB with sticky sessions for WebSocket connections
- ElastiCache Redis cluster for distributed caching
- Separate embedding microservice
- Queue-based document ingestion (SQS)
- CloudFront CDN for static assets
- Enhanced monitoring with X-Ray

### Key Questions at This Scale
11. What is your horizontal scaling strategy for FastAPI?
12. How do you handle session state across multiple instances?
13. How do you implement distributed semantic caching?
14. How do you handle LLM API rate limits at 10K users?
15. What is the embedding service architecture?
16. How do you monitor per-instance health?
17. What is your deployment strategy (blue-green)?
18. What is the estimated monthly cost?
19. How do you test auto-scaling?
20. What is the inter-service communication pattern?

## 19.3 Medical RAG at 1 Million Users

**Problem**: Scale the platform to 1 million concurrent users.

### New Challenges
- Multi-region deployment
- Microservices architecture
- Message queue for async processing
- Dedicated inference cluster
- Advanced caching strategy
- Real-time monitoring at scale
- Compliance at scale

### Architecture Changes
- Multi-region active-active deployment
- Service mesh (AWS App Mesh or Istio)
- Dedicated retrieval service cluster
- Dedicated re-ranking service cluster
- Dedicated generation service cluster
- Apache Kafka for event streaming
- Elasticsearch for hybrid search at scale
- Redis Cluster for distributed caching
- Global Accelerator for traffic routing
- Kubernetes (EKS) for container orchestration

### Key Questions at This Scale
21. How do you design a microservices RAG architecture?
22. What is the service dependency graph?
23. How do you implement multi-region Pinecone?
24. What is the global data consistency strategy?
25. How do you implement cross-region failover?
26. What is the architecture for real-time document ingestion at this scale?
27. What is the Kafka topic structure for RAG events?
28. How do you implement distributed tracing across microservices?
29. What is the monitoring strategy at 1M users?
30. What is the estimated monthly cost and optimization strategy?

## 19.4 Medical RAG at 10 Million Users

**Problem**: Netflix-scale Medical RAG platform.

### New Challenges
- Multiple data centers
- Extreme cost optimization
- CDN for edge caching
- Custom inference infrastructure
- Real-time personalization
- Advanced ML infrastructure

### Architecture Changes
- Global multi-region active-active
- Custom LLM inference cluster (A100/H100 GPUs)
- Edge caching with personalization
- GraphRAG for complex medical knowledge
- Feature store for user personalization
- ML platform for continuous learning
- Advanced A/B testing infrastructure
- Chaos engineering for resilience

### Key Design Questions
31. How do you cache at the edge for RAG?
32. What is the role of personalization at this scale?
33. How do you build a custom LLM serving infrastructure?
34. What is the architecture for continuous learning from user feedback?
35. How do you implement chaos engineering?
36. What is the cost optimization strategy at 10M users?
37. How do you implement global compliance (HIPAA, GDPR, others)?
38. What is the governance structure for medical AI at scale?
39. How do you prevent system-wide failures?
40. What is the role of SRE (Site Reliability Engineering) at this scale?

## 19.5 System Design Deep Dive Questions

41. What is your data model for the complete system?
42. What is the API contract between services?
43. What is the message schema for Kafka events?
44. What is the caching strategy at each layer?
45. How do you implement distributed locking?
46. What is the consensus mechanism for distributed decisions?
47. How do you handle hot partitions in Kafka?
48. What is the back-pressure mechanism?
49. How do you implement graceful degradation at each service?
50. What is the complete disaster recovery plan?

---

# SECTION 20 — HR & BEHAVIORAL {#section-20}

## 20.1 Project Explanation

1. Tell me about your Medical RAG Chatbot project in 2 minutes.
2. What was the most difficult technical challenge you faced?
3. What would you do differently if you built it from scratch today?
4. What is the proudest achievement in this project?
5. What is the biggest mistake you made and how did you recover?

## 20.2 Deep Technical Challenges

6. Walk me through a specific production incident you resolved.
7. Describe the most complex debugging session in this project.
8. What was the hardest optimization you implemented?
9. Describe a time when you had to make a difficult technical trade-off.
10. What external resources (papers, blogs, people) most influenced your design decisions?

## 20.3 Architecture Decisions

11. Why did you choose LangChain over building a custom RAG pipeline?
12. What would you change about LangChain now that you've used it in production?
13. Why Pinecone over other vector databases?
14. Would you still choose Pinecone today? Why or why not?
15. Why FastAPI over Flask, Django, or other frameworks?
16. What drove the decision to use Docker?
17. Why AWS over GCP or Azure?
18. Why MLflow over W&B?
19. What drove the choice of Llama 3.3 70B over other models?
20. Why cross encoder re-ranking specifically?

## 20.4 Failure Stories

21. Describe a deployment that went wrong. What happened?
22. What was the most surprising failure you encountered?
23. What was the hardest bug to find in the system?
24. Describe a time when your evaluation metrics were misleading.
25. What was a performance optimization that backfired?

## 20.5 Leadership & Ownership

26. How did you prioritize features during development?
27. How did you handle scope creep?
28. How did you manage technical debt?
29. What documentation did you create for the system?
30. How would you onboard a new team member on this project?

## 20.6 Future Improvements

31. What is the #1 improvement you would make if you had 3 more months?
32. How would you implement GraphRAG for medical knowledge?
33. How would you implement agentic capabilities for multi-step medical reasoning?
34. How would you add multimodal support (X-rays, MRI images)?
35. How would you implement real-time learning from user feedback?
36. How would you add support for electronic health records (EHR)?
37. How would you implement personalization based on user specialty?
38. How would you add multilingual support?
39. How would you implement clinical decision support features?
40. How would you get FDA clearance for your system?

## 20.7 Behavioral Competency Questions

41. Tell me about a time you had to learn a new technology very quickly for this project.
42. Tell me about a time you disagreed with a technical decision and how you handled it.
43. Tell me about a time you had to balance quality vs. speed.
44. How do you stay current with rapidly evolving LLM and RAG technology?
45. What papers did you read that most influenced this project?
46. How do you evaluate when to adopt a new technology?
47. What is your process for technical decision-making?
48. How do you handle ambiguity in requirements?
49. How do you measure your own technical growth?
50. Where do you see AI-powered medical applications in 5 years?

---

# SECTION 21 — LIVE INDUSTRY TRENDS 2026 {#section-21}

## 21.1 Latest LLM Models (2025-2026)

### Current Frontier Models (as of July 2026)
1. What is GPT-5 and how does it differ from GPT-4o for RAG?
2. What is Claude 4 and what is its extended thinking advantage for medical reasoning?
3. What is Gemini 2.5 Pro and what is its 2M token context window advantage?
4. What is Llama 4 and how does it differ from Llama 3.3?
5. What is the Mixture of Experts architecture in Llama 4?
6. What is DeepSeek V3 and why is it significant?
7. What is DeepSeek R2 and its reasoning advantage?
8. What is Qwen 3 235B and how does it compare to frontier models?
9. What is Mistral Large 2 and when would you use it?
10. What is Command-R+ and why is it specifically RAG-optimized?
11. What is Phi-4 and what is the small model advantage?
12. What is Kimi 1.5 from Moonshot AI?
13. What is the difference between reasoning models (o3, R2, Gemini Thinking) and standard models?
14. When should you use a reasoning model for medical RAG?
15. What is the cost trade-off of using reasoning models for medical questions?
16. What is the latency trade-off of reasoning models?
17. What is "thinking budget" in reasoning models?
18. How do you implement streaming with reasoning models?
19. What is the role of tool use in modern LLMs for RAG?
20. What is structured output from modern LLMs and how has it improved?

### Model Selection for Medical RAG (2026)
21. Which model performs best on medical benchmarks as of 2026?
22. What is HealthBench (OpenAI) and what does it measure?
23. Which model has the highest faithfulness for medical RAG?
24. Which model has the lowest hallucination rate for medical questions?
25. What is the cost comparison across all frontier models per 1M tokens?
26. What is the latency comparison across models?
27. What is the context window comparison?
28. Which model has the best tool/function calling support?
29. Which model is best for structured output generation?
30. Which model is preferred by AI-native medical companies today?

## 21.2 Latest RAG Research & Techniques (2025-2026)

### GraphRAG
31. What is GraphRAG (Microsoft Research)?
32. How does GraphRAG build a knowledge graph from documents?
33. What is community detection in GraphRAG?
34. What are global queries vs. local queries in GraphRAG?
35. How does GraphRAG handle medical ontology relationships?
36. What is the cost of building a GraphRAG index?
37. When would GraphRAG outperform standard RAG for medical questions?
38. What is the GraphRAG paper (Edge et al., 2024)?
39. What is LightRAG and how does it improve GraphRAG?
40. What is HippoRAG and how does it model human memory for RAG?

### Agentic RAG
41. What is Agentic RAG?
42. How does Agentic RAG differ from standard RAG?
43. What is the ReAct (Reason+Act) pattern for Agentic RAG?
44. What is LangGraph and how does it enable agentic RAG?
45. What is a conditional edge in LangGraph?
46. What is state management in LangGraph?
47. What is Self-RAG?
48. What is the Self-RAG critique and generate mechanism?
49. What is Corrective RAG (CRAG)?
50. How does CRAG detect poor retrieval and trigger web search fallback?
51. What is Adaptive RAG?
52. How does Adaptive RAG route queries to different retrieval strategies?
53. What is the query complexity classifier in Adaptive RAG?
54. What is the paper "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"?
55. What is Iterative RAG?
56. What is multi-hop RAG and when is it necessary for medical questions?
57. What is the IRCOT (Interleaving Retrieval with Chain of Thought) approach?
58. What is FLARE (Forward Looking Active REtrieval)?
59. How does FLARE predict when to retrieve additional context?
60. What is the paper on FLARE?

### Contextual Retrieval
61. What is Anthropic's contextual retrieval (September 2024)?
62. How does contextual retrieval prepend context to chunks?
63. What is the cost of contextual retrieval at indexing time?
64. How does contextual retrieval improve BM25 recall?
65. What is the performance improvement reported by Anthropic?

### HyDE
66. What is HyDE (Hypothetical Document Embeddings)?
67. How does HyDE use an LLM to generate a hypothetical answer before retrieval?
68. How does HyDE work with medical questions?
69. What is the query expansion mechanism in HyDE?
70. What are the failure cases of HyDE?
71. What is the cost overhead of HyDE?
72. What is multi-query retrieval and how does it compare to HyDE?
73. What is RAG Fusion and how does it use multiple generated queries?
74. What is Step-Back Prompting?
75. How does Step-Back prompting improve medical question retrieval?

### RAPTOR
76. What is RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)?
77. How does RAPTOR build a hierarchical document tree?
78. What is the clustering step in RAPTOR?
79. What is Gaussian Mixture Model clustering in RAPTOR?
80. What is UMAP dimensionality reduction in RAPTOR?
81. What are the leaf nodes and summary nodes in RAPTOR?
82. How does tree traversal retrieval work in RAPTOR?
83. How does collapsed tree retrieval work in RAPTOR?
84. When does RAPTOR outperform standard RAG?
85. What is the paper "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"?

## 21.3 Latest Retrieval Techniques

### SPLADE
86. What is SPLADE v2?
87. How does SPLADE combine learned lexical and semantic retrieval?
88. What is the SPLADE training objective?
89. How does SPLADE outperform BM25 on out-of-vocabulary medical terms?
90. What is the inference cost of SPLADE vs. BM25?

### ColBERT
91. What is the latest ColBERT version (PLAID)?
92. What is the retrieval quality of ColBERT v2 on BEIR?
93. What is the RAGatouille library for ColBERT deployment?
94. How do you integrate ColBERT with LangChain?
95. What is the storage cost of ColBERT for 1M documents?

### Latest Embedding Models
96. What is the MTEB leaderboard state in 2026?
97. What embedding model tops MTEB as of 2026?
98. What is NV-Embed-v2 from NVIDIA?
99. What is the GTE-Qwen2 embedding model?
100. What is the E5-Mistral embedding model?
101. What is LLM2Vec?
102. How do decoder LLMs become embedding models?
103. What is the bidirectional training trick in LLM2Vec?
104. What is the performance gain of LLM2Vec over encoder-based models?
105. What is Cohere Embed v3 and what is its multilingual support?

## 21.4 Agentic AI & MCP

### AI Agents
106. What is an AI agent in 2026?
107. What is the difference between a reactive agent and a planning agent?
108. What is the ReAct framework?
109. What is AutoGen from Microsoft?
110. What is CrewAI?
111. What is LangGraph and how does it compare to LangChain?
112. What is the OpenAI Agents SDK?
113. What is MCP (Model Context Protocol) from Anthropic?
114. How does MCP standardize tool calling for LLMs?
115. What is the security risk of MCP in medical AI?

### When Agents vs. RAG
116. When should you use agents instead of RAG?
117. When should you use RAG instead of agents?
118. What is the "agentic RAG" combination?
119. What are the safety concerns of autonomous agents in medical AI?
120. What is the human-in-the-loop requirement for medical agents?

## 21.5 Latest Vector Databases (2026)

121. What is the current state of the vector database market?
122. How has Pinecone evolved since 2023?
123. What is Pinecone's inference feature?
124. What is Qdrant's latest performance benchmark?
125. What is Weaviate v1.25+ and its new features?
126. What is Milvus 2.4+ and its improvements?
127. What is pgvector 0.7+ and HNSW support?
128. What is the VectorDB benchmark (ANN Benchmarks 2025)?
129. Which vector database is preferred by AI startups in 2026?
130. What is the trend toward multi-vector and sparse-dense hybrid in all DBs?

## 21.6 Latest MLOps Tools

### LLM Serving
131. What is vLLM 0.5+ and its new features?
132. What is SGLang and how does it compare to vLLM?
133. What is TensorRT-LLM 0.13+ improvements?
134. What is Triton Inference Server from NVIDIA?
135. What is the role of speculative decoding in production 2026?

### LLM Observability
136. What is LangSmith 2026 features?
137. What is Phoenix (Arize) for LLM observability?
138. What is Helicone for LLM API cost tracking?
139. What is Braintrust for LLM evaluation?
140. What is Humanloop for LLM prompt management?

### Evaluation Frameworks
141. What is RAGAS v0.2 and what changed?
142. What is DeepEval 2025 features?
143. What is the role of Pytest-based evaluation frameworks?
144. What is the Inspect AI framework from UK AI Safety Institute?
145. What is OpenAI Evals?

## 21.7 Latest Deployment Options

### Cloud Inference
146. What is AWS Bedrock and when would you use it vs. Groq?
147. What is Azure AI Foundry (formerly Azure OpenAI)?
148. What is Google Vertex AI and Gemini deployment?
149. What is Amazon Bedrock knowledge bases?
150. What is Google's CCAI (Contact Center AI) for medical?

### Serverless GPU
151. What is Modal and when is it ideal for RAG inference?
152. What is RunPod for GPU inference?
153. What is Together AI inference?
154. What is Fireworks AI inference?
155. What is Groq 2026 and its new models?
156. What is Cerebras inference and its wafer-scale chip advantage?
157. What is the cost comparison of all inference providers?
158. What is the latency comparison?
159. What is the context window comparison?
160. Which provider is preferred for HIPAA-compliant medical inference?

## 21.8 What Is Outdated vs. State of the Art (2026)

### Outdated Approaches
161. Why is naive chunking (fixed 512 tokens) considered outdated?
162. Why is simple cosine similarity without hybrid considered outdated?
163. Why is single-stage retrieval without re-ranking outdated?
164. Why is FAISS without metadata filtering limiting for production?
165. Why is LangChain v0.1 pipeline considered outdated?
166. Why are absolute positional encodings outdated?
167. Why is BM25-only retrieval outdated for domain-specific RAG?
168. Why is BLEU/ROUGE for RAG evaluation considered outdated?
169. Why is single-vector retrieval outdated vs. multi-vector?
170. Why is on-premise GPU deployment outdated for most startups?

### Current Best Practices
171. What is the current best practice for chunking in production RAG?
172. What is the current best practice for hybrid search?
173. What is the current best practice for re-ranking?
174. What is the current best practice for RAG evaluation?
175. What is the current best practice for LLM serving?
176. What is the current best practice for RAG monitoring?
177. What is the current best practice for multi-turn conversation in RAG?
178. What is the current best practice for knowledge base updates?
179. What is the current best practice for RAG security?
180. What is the current best practice for medical AI compliance?

## 21.9 What FAANG Uses vs. What Startups Use

181. What RAG architecture does Google Search use?
182. What is Perplexity AI's retrieval architecture?
183. What does Cohere's production RAG look like?
184. What does Anthropic's internal RAG for Claude look like?
185. What does Microsoft Copilot's RAG architecture use?
186. What vector database does Uber use at scale?
187. What does Databricks' production RAG (DBRX) use?
188. What inference infrastructure does OpenAI use?
189. What does Meta AI use for RAG in their products?
190. What embedding model does Amazon use for product search?

## 21.10 Research Papers You Must Know (2024-2026)

191. "RAG vs. Fine-Tuning" — what did this paper find?
192. "Lost in the Middle" — key finding for RAG prompt design?
193. "RAGAS" paper — key contribution?
194. "Self-RAG" paper — key mechanism?
195. "CRAG" paper — when retrieval is wrong?
196. "GraphRAG" paper — community summary contribution?
197. "HyDE" paper — hypothetical document embedding?
198. "RAPTOR" paper — tree-based hierarchical RAG?
199. "Contextual Retrieval" blog — Anthropic's contribution?
200. "ColBERT v2" paper — efficient late interaction?
201. "SPLADE v2" paper — learned sparse retrieval?
202. "LLM2Vec" paper — decoder as encoder?
203. "BGE-M3" paper — multi-functionality embedding?
204. "Flash Attention 3" paper — attention acceleration?
205. "GQA" paper — grouped query attention?
206. "Matryoshka Representation Learning" paper?
207. "PagedAttention / vLLM" paper?
208. "MMLU" paper — medical AI benchmark?
209. "GPT-4 technical report" — medical benchmark results?
210. "MedPaLM 2" paper — Google's medical LLM?
211. "Health-LLM" paper — health domain challenges?
212. "HealthBench" paper — comprehensive medical LLM evaluation?
213. "DSPy" paper — declarative language model programming?
214. "OPRO" paper — optimization by prompting?
215. "ReAct" paper — reasoning and acting in LLMs?
216. "LangGraph" paper / documentation — stateful agent design?
217. "MCP" paper / documentation — model context protocol?
218. "Constitutional AI" paper — Anthropic's safety approach?
219. "Direct Preference Optimization" paper — DPO?
220. "Mixtral" paper — mixture of experts?

## 21.11 Future Directions

221. What is the future of RAG with 10M context window models?
222. Will RAG become obsolete with very large context windows?
223. What is the role of memory networks in future medical AI?
224. What is the role of continual learning for medical knowledge?
225. What is the role of knowledge graphs alongside RAG?
226. What is multimodal RAG for radiology AI?
227. What is the future of personalized medical AI?
228. What is the role of federated learning for private medical AI?
229. What is the future of AI agents in clinical decision support?
230. What will medical AI systems look like in 2030?

---

# APPENDIX A — MATHEMATICAL REFERENCE

## A.1 Key Formulas

| Formula | Name | Context |
|---------|------|---------|
| `cos(θ) = (A·B)/(‖A‖‖B‖)` | Cosine Similarity | Vector search |
| `NDCG@k = DCG@k / IDCG@k` | NDCG | Retrieval eval |
| `DCG@k = Σ rel_i / log₂(i+1)` | DCG | Retrieval eval |
| `MRR = (1/Q) Σ 1/rank_q` | Mean Reciprocal Rank | Retrieval eval |
| `RRF(d) = Σ 1/(k + rank_i(d))` | Reciprocal Rank Fusion | Hybrid search |
| `BM25 = Σ IDF(qi) × f(qi,D)(k1+1) / [f(qi,D) + k1(1-b+b|D|/avgdl)]` | BM25 | Sparse retrieval |
| `Attention(Q,K,V) = softmax(QK^T/√dk)V` | Scaled Dot-Product Attention | Transformer |
| `hybrid = α × dense + (1-α) × sparse` | Hybrid Score Fusion | Hybrid search |
| `num_chunks = ⌈(N-O)/(C-O)⌉` | Chunk Count | Chunking |
| `InfoNCE = -log[exp(sim(q,k+)/τ) / Σ exp(sim(q,ki)/τ)]` | Contrastive Loss | Embedding training |

## A.2 Complexity Reference

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Brute-force NN | O(n·d) | O(n·d) |
| HNSW search | O(log n) | O(n·M) |
| IVF search | O(nprobe·n/nlist + d) | O(n·d) |
| BM25 scoring | O(q·avg_posting) | O(V·avg_posting) |
| Cross encoder | O(n · L²) | O(n · L) |
| Bi-encoder | O(d) per query | O(n·d) |
| Transformer attention | O(L²·d) | O(L²) |
| Flash Attention | O(L²·d) compute, O(L) memory | O(L) |

---

# APPENDIX B — INTERVIEW TIPS

## B.1 How to Answer Every Question

For **every technical question**, structure your answer as:
1. **Define** the concept clearly
2. **Explain** the mechanism/algorithm
3. **Give a concrete example** from your project
4. **Discuss trade-offs** and alternatives
5. **Mention production considerations**
6. **Reference relevant research** if applicable

## B.2 Anti-Patterns to Avoid

❌ Saying "I used LangChain for that" without explaining what LangChain does internally  
❌ Confusing precision and recall  
❌ Not knowing the math behind cosine similarity  
❌ Not knowing HNSW internals when you use Pinecone  
❌ Not knowing what HIPAA requires for medical AI  
❌ Not knowing the difference between faithfulness and factual correctness  
❌ Saying "it works well" without metrics  
❌ Not knowing your system's actual latency numbers  
❌ Not knowing the cost of your production system  

## B.3 Questions to Always Ask the Interviewer

1. What scale does your current RAG system operate at?
2. What is the most challenging problem you're facing in your RAG system?
3. What evaluation methodology do you use?
4. What is your team's approach to RAG quality monitoring?
5. How do you handle knowledge base updates?

---

*This handbook covers 3,000+ interview questions across all domains of Medical RAG, LLM Engineering, Search, Backend, MLOps, System Design, and Production AI. Review each section systematically and ensure you can answer at all difficulty levels: Beginner → Research → Production.*

**Total Questions: ~3,200+**  
**Sections: 21 major sections + 2 appendices**  
**Coverage: Beginner to Research-Level, 2026 Industry Current**
