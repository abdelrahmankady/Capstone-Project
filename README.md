# EpiWave

EpiWave is an AI-powered platform for analyzing EEG recordings and interacting with the results through a conversational agent. Upload `.edf` files, get seizure detection predictions from a trained deep learning model, and ask follow-up questions about the findings — or about neuroscience in general.

> **Disclaimer:** This project is for educational and informational purposes only. It does not provide medical diagnoses or clinical recommendations. Consult a qualified neurologist for any medical decisions.

## How It Works

The system is built around three components that work together through a unified Flask API:

**Signal Processing & ML Inference** — Raw `.edf` files are parsed with MNE, bandpass-filtered (0.5–40 Hz), and run through spike detection (3σ threshold) and RMS-based seizure candidate detection. A fine-tuned MobileNetV2 classifier then predicts the overall scan state (normal, preictal, or seizure).

**RAG Knowledge Base** — After inference, the analysis report is automatically chunked and embedded into a ChromaDB vector store using Google's `gemini-embedding-2`. When the user asks a question, relevant chunks are retrieved via cosine similarity and passed to the LLM as context.

**Conversational Agent** — Powered by Gemini for streaming responses. The agent answers scan-specific questions by citing the retrieved context, and handles general neuroscience questions using its training knowledge plus live DuckDuckGo web search for up-to-date information.

## Project Structure

```
├── api/                        Flask API layer
│   ├── main.py                 Entry point, serves frontend & registers blueprints
│   ├── predict_routes.py       POST /api/predict — upload, inference, auto-ingest
│   ├── chatbot_routes.py       POST /api/chat — RAG pipeline, streaming response
│   ├── eeg_routes.py           POST /api/eeg/analyze — detailed EEG analysis
│   └── shared/eeg_schema.py    Shared validation schemas
│
├── chatbot/                    Conversational AI
│   ├── config.py               Centralized settings from .env
│   ├── llm_wrapper.py          Google Gemini API client
│   ├── chat.py                 Standalone CLI interface
│   ├── agents/                 LLM execution pipeline
│   │   ├── retrieve.py         Query ChromaDB for relevant chunks
│   │   ├── verify.py           Filter by cosine similarity threshold (0.60)
│   │   └── respond.py          Build prompt, stream LLM response, web search
│   ├── rag/                    Vector store management
│   │   ├── ingest.py           Format analysis results into documents
│   │   └── vectorize.py        Chunk, embed, upsert, query ChromaDB
│   └── eeg/                    Signal processing pipeline
│       ├── parser.py           Read .edf via MNE
│       ├── analyzer.py         Filtering, spike detection, seizure detection
│       └── visualizer.py       Generate JSON chart data for frontend
│
├── ml/                         Machine learning
│   ├── epiwave_predict_api.py  Preprocessing + Keras model inference
│   └── epiwave_multiclass_model_tuned.py   Training script
│
├── frontend/                   Client UI
│   ├── index.html              Landing page
│   ├── upload.html             File upload + chat interface
│   └── brain.html              3D brain visualization (WebGL)
│
├── tests/                      Test suite
│   ├── test_eeg_pipeline.py    Parser, analyzer, visualizer
│   ├── test_rag_pipeline.py    Ingest, vectorize, ChromaDB
│   ├── test_agents.py          Retrieve, verify, respond
│   ├── test_security.py        Path traversal, prompt injection, guardrails
│   └── test_memory.py          RSS memory bounds
│
├── data/eeg_scans/             Place .edf files here
├── chroma_db/                  Persistent vector store (auto-created)
├── requirements.txt            All dependencies
└── .env.example                Environment variable template
```

## Setup

**Requirements:** Python 3.11+, a Google AI Studio API key ([get one here](https://aistudio.google.com/apikey)).

```bash
git clone <repo-url>
cd llm_api

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Configuration

Edit `.env` to configure:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. Your Gemini API key |
| `GOOGLE_LLM_MODEL` | `gemini-2.0-flash` | Model used for chat responses |
| `GOOGLE_EMBEDDING_MODEL` | `gemini-embedding-2` | Model used for text embeddings |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector store location |
| `EEG_SCANS_PATH` | `./data/eeg_scans` | Where to find/save .edf files |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MAX_HISTORY_TURNS` | `6` | Conversation history window |
| `FLASK_PORT` | `5000` | Server port |

## Running

```bash
python api/main.py
```

Open `http://127.0.0.1:5000` in your browser.

## Tests

```bash
pytest              # run all
pytest -v           # verbose
pytest tests/test_security.py -v   # specific module
```

## Design Notes

**Why DuckDuckGo for web search?** Google's Search Grounding API hits aggressive rate limits on the free tier (20 requests/day). DuckDuckGo search via `ddgs` is unlimited and free, so general questions always get answered.

**Why a 0.60 similarity threshold?** Lower thresholds caused the agent to pull scan data into unrelated conversations (e.g., saying "hello" would trigger a full medical report dump). 0.60 ensures only genuinely relevant chunks reach the LLM.

**Why three prompt branches?** The LLM needs to behave differently depending on context: (1) scan chunks found → cite findings, (2) scans exist but query is vague → respond naturally, (3) no scans at all → general education mode.

**Why server-side auto-ingestion?** Previously the frontend triggered RAG ingestion after ML prediction, which caused race conditions and caching bugs. Now `predict_routes.py` handles ingestion immediately after analysis completes.
