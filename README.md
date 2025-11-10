# Vibe Matcher - Fashion Semantic Retrieval Prototype

## Internship Assignment — Nexora AI

### Overview

Vibe Matcher is a lightweight semantic retrieval prototype designed to simulate a fashion recommendation system that matches a user's "vibe" query to fashion products using vector embeddings and cosine similarity. The project demonstrates how text embeddings, semantic search and cosine similarity can be combined to retrieve contextually relevant items, even when the words in the query and product descriptions don't exactly match.

---

### Objective

> Input a "vibe" query -> Embed it -> Compare it with product vectors -> Output top 3 matches by similarity

This prototype fulfills the core requirements of the task:
- A dataset of 10 mock fashion items with descriptive text and vibe tags.
- Use of a SentenceTransformer model (as a fallback to OpenAI embeddings due to API limitations).
- Computation of cosine similarity using sklearn.
- Handling of weak matches and empty inputs.
- Visualization of similarity scores and latency for each query.

---

### Modules Breakdown

#### 1. Data Preparation

A Pandas DataFrame of 10 sample fashion products was created with:

- `name`: Product title
- `desc`: Short natural-language description
- `tags`: Style identifiers (e.g., "boho", "cozy", "formal")

Rich descriptions are used instead of keywords because natural text yields better embeddings for semantic models.

#### 2. Embedding Generation

Used the open-source model:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

Each description is embedded into a 768-dimensional vector. Normalization ensures that the cosine similarity equals the dot product.

#### 3. Query Handling & Vector Similarity Search

- Each user query is embedded and compared against product vectors.
- The top 3 most similar items are retrieved using cosine similarity.
- Latency for each query is recorded.

#### 4. Fallback & Edge Case Handling

|Case|Handling Logic|
|---|---|
|Weak Match (<0.7)|Return feedback taht no strong match found|
|Empty Query|Return error Message|
|Duplicate Vibe|Use tags to break ties or display multiple top items|

#### 5. Testing & Visualization

- 10 test queries were run to measure similarity and latency.
- Visualization shows:
    - Best similarity score per query
    - Latency (seconds) per query

Despite no queries exceeding the 0.7 similarity threshold, all performed between 0.4–0.6, indicating semantic relevance but vocabulary mismatch due to small dataset size.

#### 6. Observations & Improvements

##### Observations

- Scores between 0.4–0.6 indicate moderate semantic alignment.
- Queries with adjectives like "bohemian" or "sporty" performed worse due to missing descriptors in product texts.
- Latency stayed consistently below 0.05s, indicating high efficiency.

##### Improvements

- Expand dataset (15–25 entries) to better represent diverse fashion styles.
- Add more expressive and contextually rich descriptions.
- Experiment with higher-quality transformer models (e.g., `paraphrase-mpnet-base-v2`).
- Integrate with a vector DB (e.g., Pinecone, FAISS) for scalable retrieval.

---

### What I Learned

This project gave me my first hands-on exposure to AI concepts in practice. Even though I only had theoretical understanding of tokenization, BPE, and the transformer architecture, I learned:

- How embeddings translate text into numerical meaning.
- How cosine similarity measures contextual alignment between vectors.
- How to use open-source models for semantic retrieval.
- The importance of good data design and rich descriptions for meaningful embeddings.

It also helped me appreciate the "why" behind the models — how representation learning powers real-world applications like recommendation systems

---

### Tech Stack

- Language: Python 3.10+
- Libraries:
    - pandas
    - sentence-transformers
    - sklearn
    - matplotlib
    - time

---

### License

MIT License

---

This assignment was my first attempt at building an AI prototype from scratch. While I initially relied heavily on guidance and examples, I came out understanding how real-world systems translate vague human "vibes" into numerical meaning.

---