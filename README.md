# ⚽ FantasyEdge — Agentic FPL Analyst

An AI agent that autonomously analyzes Fantasy Premier League data to help you pick optimal teams. Unlike a fixed pipeline, the agent **decides its own actions** based on your question — choosing which tools to call, in what order, and when to stop.

Built with LangGraph's ReAct pattern: the LLM reasons, acts, observes, and loops until it has enough information to answer.

## 🤖 What Makes This Agentic (Not a Pipeline)

A pipeline runs the same steps every time. This agent adapts:

| Your question | Agent's path |
|--------------|-------------|
| "Pick my team" | get_gameweek → get_player_stats (×4 positions) → get_fixtures → check_availability → build_squad |
| "Is Salah worth it?" | get_player_stats (Salah) → get_fixtures (Liverpool) → respond |
| "Best midfielder under £8m" | get_player_stats (MID, max_price=8) → respond |
| "I have Palmer, build around him" | get_player_stats (Palmer) → build_squad (locked: Palmer) → respond |

Same tools, completely different execution paths. The LLM decides — not the developer.

## 🏗️ Architecture

```
User Question
      │
      ▼
┌──────────┐
│   LLM    │ ← Thinks: "What do I need?"
│  (Brain) │
└────┬─────┘
     │ decides to call a tool
     ▼
┌──────────┐
│  Tool    │ ← One of 6 tools (stats, fixtures, availability, squad, memory, gameweek)
└────┬─────┘
     │ returns data
     ▼
┌──────────┐
│   LLM    │ ← Thinks: "Do I need more info?"
│  (Brain) │
└────┬─────┘
     │
     ├── needs more → calls another tool (loop)
     └── has enough → writes final response
```

### Three-Layer Design

**Core Layer (no LLM needed):**
- `scoring.py` — Documented scoring formula with configurable weights and confidence levels
- `optimizer.py` — Constraint solver (budget, positions, max 3 per team) using greedy algorithm
- `fpl_data.py` — FPL API integration with caching and fallback for reliability

**Agent Layer:**
- `tools.py` — 6 tools the LLM can call autonomously
- `graph.py` — LangGraph ReAct agent with reasoning loop
- `memory.py` — Three memory types (conversation, preferences, squad state)

**Evaluation Layer:**
- `evaluator.py` — Backtesting against baselines with measurable metrics

### Scoring Engine

```
Player Score = (form × 3.0) + (xGI × 2.5) + (value × 1.5) 
             + (fixture_ease × 2.0) + (availability × 5.0) + (momentum × 0.5)
```

Each pick includes a confidence level (HIGH/MEDIUM/LOW) based on:
- Sample size (minutes played)
- Form-xG agreement (sustainable or lucky?)
- Fixture clarity
- Fitness certainty

### Three Memory Types

| Memory | Persists | Example |
|--------|---------|---------|
| Conversation | Current session | "You just recommended Salah as captain" |
| Preferences | Across sessions | "I'm a Liverpool fan, always include one LFC player" |
| Squad State | Across sessions | "My current team has Palmer, Haaland, and Saka" |

### Tool Reliability

- **API timeout handling** — falls back to cached data if FPL API is slow
- **Stale data warnings** — tells user when using cached data
- **Confidence scoring** — every pick explains why it's HIGH/MEDIUM/LOW confidence

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/fantasyedge.git
cd fantasyedge
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Groq API key
```

### Run — CLI (quickest test)

```bash
python -m agent.graph
```

Type questions like "Pick my team" or "Is Haaland worth the price?"

### Run — Streamlit UI

```bash
streamlit run ui/app.py
```

### Run — FastAPI

```bash
uvicorn api.api:app --reload
# Visit http://localhost:8000/docs
```

### Run — Evaluation

```bash
python -m eval.evaluator
```

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent Orchestration | **LangGraph (ReAct)** | Autonomous reasoning loop with tool calling |
| LLM | **Groq (Llama 3.3 70B)** | Free, fast — only used for reasoning + final report |
| Data Source | **FPL API** | Free, real-time, no auth needed |
| Scoring | **Custom Python** | Documented formula, not LLM guesswork |
| Optimization | **Greedy + constraints** | Algorithmic, guarantees valid squads |
| API | **FastAPI** | REST endpoint with session management |
| Frontend | **Streamlit** | Chat UI with reasoning trace toggle |
| Memory | **JSON persistence** | Preferences + squad state across sessions |

## 📁 Project Structure

```
fantasyedge/
├── core/
│   ├── fpl_data.py          # FPL API + caching + fallback
│   ├── scoring.py           # Documented scoring engine + confidence
│   └── optimizer.py         # Constraint solver (algorithmic)
├── agent/
│   ├── tools.py             # 6 tools the agent calls autonomously
│   ├── graph.py             # LangGraph ReAct agent
│   └── memory.py            # 3 memory types
├── api/
│   └── api.py               # FastAPI with session management
├── ui/
│   └── app.py               # Streamlit with reasoning trace
├── eval/
│   ├── evaluator.py         # Backtesting framework
│   └── results/             # Evaluation outputs
├── data/
│   └── preferences.json     # User preferences (persistent)
├── requirements.txt
└── README.md
```

## 💡 Key Design Decisions

**Only 1 out of 6 tools uses the LLM.** The Scout, Fixture, Availability, and Optimizer tools are pure Python — no LLM. Only the final presentation uses Groq. This is deliberate: LLMs reason well but can't do math or respect constraints reliably.

**Greedy optimizer over LLM selection.** Asking an LLM to "pick 15 players within £100m" fails because LLMs can't track cumulative budgets. Our optimizer guarantees valid squads every time.

**Caching for reliability.** If the FPL API is down, the agent uses cached data and warns the user. It never crashes — it degrades gracefully.

**Three memory layers for real agent behavior.** Without memory, every conversation starts from scratch. With squad state memory, the agent can suggest transfers instead of full rebuilds.

## 🔮 Potential Extensions

- [ ] Evaluate agent picks against actual gameweek results
- [ ] Transfer suggestions based on current vs recommended team
- [ ] Chip strategy advisor (Bench Boost, Triple Captain, Free Hit)
- [ ] Differential picks for head-to-head leagues
- [ ] Historical performance tracking with charts

## 📜 License

MIT
