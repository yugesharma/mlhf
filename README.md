---
title: CSDS553 Demo
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---



# NFL Chatbot — RAG‑powered NFL Bot Players & Teams (2025)

For the 2025 season, a lightweight Retrieval-Augmented Generation (RAG) chatbot will respond to inquiries on NFL players and teams. It uses compact, effective LLMs in conjunction with a dense-vector index (FAISS) constructed from SportsDataIO roster/news data to provide quick, source-grounded replies through a Gradio user interface.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Models](#models)
- [Data & Indexing](#data--indexing)
- [Project Structure](#project-structure)
- [Metrics & Monitoring](#metrics--monitoring)

---

## Overview
An LLM-powered chat interface focused on NFL players and teams is called **NFL Chatbot**. It caters to those who are interested in current events and quick details about the team, position, college, weight, and experience. A tiny language model is grounded in an FAISS index of carefully selected player bios and news to respond to queries.

- **UI:** Gradio web app
- **Retrieval:** FAISS (dense search)
- **Embeddings:** e5‑small‑v2
- **Generation:** Qwen3‑0.6B (local) or gpt‑oss‑20b (Hugging Face Inference API)

## Key Features
- Ask natural‑language questions about players/teams
- Uses dense retrieval over curated bios + latest news
- Choose **local** or **hosted** LLM path
- Basic Prometheus metrics: request counts & in‑flight gauges

## Architecture
```
User Query → e5-small-v2 → FAISS top‑k → Prompt w/ context → LLM (local Qwen3‑0.6B or HF gpt-oss-20b)   →Answer Text → Gradio UI
```

## Models
| Model        | Architecture                                           | Parameters   | Purpose                                                               |
|--------------|--------------------------------------------------------|--------------|-----------------------------------------------------------------------|
| e5-small-v2  | Small text encoder, 12 layers, embedding size 384      | 33.4 million | Convert text into embeddings for RAG                                  |
| Qwen3-0.6B   | Causal language model with ~28 layers                  | 0.6 billion  | Generate answers from retrieved context (hosted locally)              |
| gpt-oss-20b  | Autoregressive Mixture-of-Experts; ~24 layers          | 21.5 billion | Generate answers via Hugging Face Inference API (fallback/alternative)|

## Data & Indexing
- **Source:** SportsDataIO API (2024 season rosters) + curated player news articles
- **Preprocess:** Convert structured JSON → readable unstructured passages (bios + news)
- **Chunking:** Name / team / season‑anchored chunks
- **Embeddings:** e5‑small‑v2 → 384‑d normalized vectors
- **Index:** FAISS IVF/Flat (configurable) persisted to `database/players.index`
- **Metadata:** Per‑chunk JSON stored in `database/metadata.json` (includes `text` and fields like player_id, team, season)

## Project Structure
```
.
├─ app.py                      
├─ database/
│  ├─ players.index           
│  └─ metadata.json                 
├─ tests/
│  └─ test_smoke.py             
├─ requirements.txt
└─ README.md
```

## Metrics & Monitoring
The app instruments two Prometheus metrics:
- `chatbot_requests_total{model_type="local|api"}` — counter
- `chatbot_requests_in_progress` — counter




