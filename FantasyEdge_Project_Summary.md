# FantasyEdge v2 — Complete Project Summary

## PROJECT OVERVIEW

FantasyEdge is an agentic AI system for Fantasy Premier League team selection.
It uses LangGraph's ReAct pattern where the LLM autonomously decides which tools
to call based on the user's question — NOT a fixed pipeline.

The project exists at: fantasyedge-v2/ on Abhay's local machine (Windows, VS Code)
LLM Provider: Groq (free tier, model: llama-3.3-70b-versatile)
FPL API: https://fantasy.premierleague.com/api/ (free, no auth needed)

## PROJECT STRUCTURE

```
fantasyedge-v2/
├── core/                    # Pure Python — NO LLM calls
│   ├── __init__.py
│   ├── fpl_data.py          # FPL API data fetching + caching + fallback
│   ├── scoring.py           # Player scoring engine with documented formula
│   └── optimizer.py         # Constraint solver for valid squad selection
├── agent/                   # Agentic layer — LLM reasoning
│   ├── __init__.py
│   ├── tools.py             # 6 tools the agent can call
│   ├── graph.py             # LangGraph ReAct agent
│   └── memory.py            # 3 memory types
├── api/
│   └── api.py               # FastAPI REST endpoint
├── ui/
│   └── app.py               # Streamlit frontend with reasoning trace
├── eval/
│   ├── __init__.py
│   ├── evaluator.py         # Backtesting framework
│   └── results/             # Evaluation output files
├── data/
│   └── preferences.json     # User preferences (persistent)
├── .streamlit/
│   └── config.toml          # Dark theme config
├── .github/workflows/       # (empty, for future automation)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## FILE-BY-FILE SUMMARY

### core/fpl_data.py
- Fetches all data from FPL API (bootstrap-static endpoint = all players, teams, gameweeks)
- Fetches fixtures with difficulty ratings (FDR 1-5)
- Builds clean Pandas DataFrames for players and fixtures
- CACHING LAYER: saves fetched data to data/fpl_cache.json, falls back to cache if API fails
- Calculates average fixture difficulty per team over next 3 gameweeks
- Key constants: SQUAD_RULES (budget=1000, positions, max 3 per team)
- Main entry point: get_all_data() returns dict with players df, fixtures df, fixture_scores, current_gw, teams

### core/scoring.py
- Documented scoring formula:
  score = (form × 3.0) + (xGI × 2.5) + (value × 1.5) + (fixture_ease × 2.0) + (availability × 5.0) + (momentum × 0.5)
- Configurable WEIGHTS dict at top of file
- score_player() returns: score, components breakdown, confidence level, confidence reasons
- Confidence levels (HIGH/MEDIUM/LOW) based on: minutes played, form-xG agreement, fixture clarity, fitness
- get_top_players() filters by position, form, price, team with optional params
- explain_score() generates human-readable explanation of a player's score
- score_all_players() adds score/confidence columns to the DataFrame

### core/optimizer.py
- Greedy algorithm with constraint checking (NOT LLM-based)
- Constraints: £100m budget, 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per team
- Supports locked_players (must include) and exclude_players
- _select_starting_xi() picks best 11 respecting formation rules (min 3 DEF, 2 MID, 1 FWD)
- Picks captain (highest score) and vice captain (second highest)
- Returns: squad, starting_xi, bench, captain, vice_captain, total_cost, remaining_budget, formation, valid, issues
- format_squad_summary() generates readable text output

### agent/tools.py
- 6 tools decorated with @tool (LangChain tool decorator):
  1. get_player_stats — search/rank players with filters (position, form, price, team, name)
  2. get_fixtures — check upcoming fixture difficulty for a team
  3. check_availability — check if a player is fit/injured/doubtful
  4. build_squad — optimize valid 15-player squad (calls optimizer.py internally)
  5. manage_memory — view/update user preferences and squad state
  6. get_gameweek_info — get current gameweek number and deadline
- _data_cache dict stores loaded+scored data for reuse across tool calls in a session
- _ensure_data() loads and scores all players on first tool call
- ALL_TOOLS list exported for the agent

### agent/graph.py
- ReAct agent using LangGraph
- SYSTEM_PROMPT: detailed instructions for FPL analysis, references scoring formula, tells agent when to use which tools
- System prompt dynamically includes user preferences and squad state from memory
- AgentState: just messages (using LangGraph's add_messages annotation)
- build_agent(): creates StateGraph with 2 nodes:
  - "agent" node: LLM thinks and optionally calls tools
  - "tools" node: ToolNode executes whatever tool the LLM chose
- Conditional edge: if LLM called a tool → go to tools → back to agent (loop)
                     if LLM didn't call a tool → END (it's done thinking)
- chat() function: takes message + optional history, returns response + tool_calls + messages
- CLI mode at bottom for terminal testing: python -m agent.graph

### agent/memory.py
- Three memory types:
  1. Conversation memory — handled by LangGraph's message history (no custom code)
  2. User preferences — JSON file at data/preferences.json
     Keys: favorite_team, must_include_players, never_pick_players, prefer_budget_picks, captain_preference, risk_tolerance, notes
  3. Squad state — JSON file at data/squad_state.json
     Keys: current_squad, budget_remaining, free_transfers, chips_available, gameweek_history
- Functions: load/save preferences, load/save squad state, get summaries for display

### eval/evaluator.py
- Backtests agent's scoring engine against real gameweek results
- get_gameweek_points() fetches actual points from FPL live endpoint
- Three baselines:
  1. Form-only baseline (picks by form, no fixtures/injuries)
  2. Popularity baseline (picks most-selected players)
  3. Random valid team (random within constraints, averaged over 5 runs)
- Also calculates theoretical optimal (hindsight — best possible team after results known)
- Metrics: agent_points, efficiency (agent/optimal %), win rate vs each baseline
- evaluate_multiple_gameweeks() runs across a range and saves results to eval/results/
- print_eval_report() prints formatted comparison table

### ui/app.py
- Streamlit frontend with chat interface
- Adds project root to sys.path for imports
- Handles GROQ_API_KEY from Streamlit Cloud secrets or .env
- Sidebar shows: agent pipeline description, memory state, sample questions
- Toggle for "Show reasoning trace" — displays tool calls and intermediate results
- Extracts tool_calls and reasoning steps from LangGraph message history
- Error handling for rate limits
- Session state maintains: messages, agent_history (for conversation continuity), agent instance

### api/api.py
- FastAPI endpoint at POST /chat
- ChatRequest: message + session_id
- ChatResponse: response + tool_calls + processing_time + session_id
- Simple in-memory session storage (dict) for conversation continuity
- DELETE /session/{id} to clear history
- GET /health for health check

## KEY DESIGN DECISIONS

1. Only the Presenter/final response uses LLM. Scoring, optimization, and data processing are pure Python.
   Interview talking point: "I use AI where it adds value and algorithms where they're more reliable."

2. ReAct pattern (not pipeline): LLM decides which tools to call and when to stop.
   Different questions take different paths through the tools.

3. Three memory layers enable real conversational behavior:
   - "I already have Salah" → agent remembers from conversation
   - "I'm a Liverpool fan" → persists to preferences.json
   - "What transfers should I make?" → reads squad_state.json

4. Caching + fallback for reliability: if FPL API is down, uses cached data and warns user.

5. Confidence scoring on every pick: HIGH/MEDIUM/LOW with reasons. Shows engineering maturity.

6. FastAPI alongside Streamlit: same agent, two interfaces. Shows separation of concerns.

## WHAT'S BEEN BUILT VS WHAT'S REMAINING

### DONE:
- All 9 code files written and reviewed
- README.md with full documentation
- Project structure with proper __init__.py files
- .env.example, .gitignore, requirements.txt, .streamlit/config.toml

### NEEDS TESTING:
- Run python -m agent.graph in CLI and test with real FPL data
- Run streamlit run ui/app.py and test the UI
- Run uvicorn api.api:app --reload and test FastAPI
- Debug any import errors or API issues

### NEEDS DOING:
- Push to GitHub (github.com/abhaythomas/fantasyedge)
- Deploy to Streamlit Cloud (add GROQ_API_KEY as secret)
- Run eval/evaluator.py on past gameweeks to get benchmark numbers
- Take screenshots for README
- Add to CV

## ABHAY'S SETUP
- Machine: Windows PC
- Editor: VS Code
- Python: 3.14 (via venv)
- Terminal: PowerShell (use venv\Scripts\activate, not source)
- Groq API key: already has one (same as EarningsLens and FinPulse)
- GitHub: github.com/abhaythomas

## ABHAY'S OTHER PROJECTS (for context)
1. EarningsLens — Adaptive RAG for financial documents (deployed on Streamlit Cloud)
   - Live: https://earningslens-b4fjpwzptyqempqmzuhuff.streamlit.app/
   - GitHub: github.com/abhaythomas/earnings_lens
   - Recently added: Compare Mode, company sidebar, 5 new transcripts

2. FinPulse — Autonomous portfolio monitoring agent (daily email via GitHub Actions)
   - GitHub: github.com/abhaythomas/finpulse
   - Real portfolio: DEEPAKNTR, ETERNAL (Zomato), JUNIORBEES, NIFTYBEES
   - Running daily at 8 AM IST

3. LLaMA fine-tuning — ARC Challenge (university Kaggle project)
   - GitHub: github.com/abhaythomas/Team-Project-ARC-Competition

## ABHAY'S BACKGROUND
- M.Sc. Data Science at Universität Mannheim (graduating early 2027)
- B.Sc. Statistics from St. Xavier's College Mumbai
- Ex-BASF (Working Student, Digitalization, Mar 2024 - Mar 2026)
- Ex-PwC India (Business Analyst, Aug 2021 - Aug 2023)
- Thesis: Benchmarking Monocular Visual SLAM under Prof. Dr. Keuper
- German: A2 level, actively learning
- Currently applying to: BMW (7 roles), SAP, Aleph Alpha, Siemens, Gen Re, Briink, startups
- Primary target: Gen AI / AI Engineering internships in Germany (English-first)

## FEEDBACK TO INCORPORATE (from external review)
1. Must show architecture depth, engineering maturity, and evaluation — not just "LangChain + functions"
2. Scoring engine must be documented and testable (DONE — scoring.py has formula + weights)
3. Constraint solver must be algorithmic, not LLM (DONE — optimizer.py)
4. Three memory types needed (DONE — memory.py)
5. Evaluation framework comparing against baselines (DONE — evaluator.py)
6. Tool reliability logic: timeout handling, fallback, confidence scoring (DONE — fpl_data.py caching, scoring.py confidence)
7. UI must show: clean chat, visible tool calls, reasoning trace toggle (DONE — ui/app.py)
8. The system must be TRULY agentic (ReAct loop, not pipeline) — agent decides its own actions (DONE — graph.py)

## COMMON ISSUES TO WATCH FOR
- Windows: use venv\Scripts\activate (not source)
- Groq rate limits: 30 requests/min, 1000/day on free tier. This project uses ~2-6 calls per question.
- langchain imports: use langchain_text_splitters (not langchain.text_splitter) for newer versions
- FPL API: completely free, no auth, but can be slow during gameweek deadlines
- Python 3.14 compatibility warnings with Pydantic v1 — can be ignored
