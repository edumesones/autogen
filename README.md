# AutoGen Experiments — Multi-Agent QA Systems

Hands-on projects built with **Microsoft AutoGen v0.4** (async AgentChat API),
exploring multi-agent question answering, cost control, tool use (MCP), and
multimodal web automation. Each subfolder is a self-contained experiment with its
own README.

## Projects

### 🔎 `github/` — AutoGen QA with GitHub MCP integration
Multi-agent QA system extended with **Model Context Protocol (MCP)** tools for deep
GitHub repository analysis (code, issues, PRs, commit history). Specialized agents —
**Researcher, Analyst, Fact-Checker, Synthesizer, Critic** — collaborate in automatic
or human-in-the-loop mode, with **per-session / per-agent / per-message cost tracking**
and Markdown export of the full conversation for audit.

### 💰 `q_a_cost/` — Cost-aware multi-agent QA
Same five-role agent team focused on **token/cost accounting**: tracks OpenAI usage and
cost per session, agent and message, with an optional Docker-backed **Code Executor**
agent for data-analysis answers. Supports fully automatic or step-by-step human review.

### 🖼️ `screenshots/` — Magentic-One multimodal & screenshot agent
Demonstrates robust handling of **multimodal (text + image)** agent responses and web
**screenshot capture**. Implements a `FixedMultimodalWebSurfer` that safely processes
multimodal payloads, fixing the classic `'list' object has no attribute 'strip'` crash
in MultimodalWebSurfer, with a coordinating assistant agent for navigation/capture/analysis.

## Common themes

- **Role-specialized multi-agent design** (research → analyze → fact-check → synthesize → critique).
- **Human-in-the-loop vs automatic** execution modes.
- **Cost observability** as a first-class concern.
- **Tool/MCP extensibility** and **multimodal robustness**.

## Requirements

Python 3.9+ · AutoGen v0.4+ · Node.js + npx (MCP / Playwright) · Docker (optional,
code executor) · per-project `requirements.txt`. See each subfolder's README for setup.
