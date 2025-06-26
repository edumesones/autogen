# Advanced AutoGen QA System with GitHub MCP Integration

## Overview

This project implements an advanced, modular Question Answering (QA) system using Microsoft's AutoGen v4.0 AgentChat API, extended with Model Context Protocol (MCP) tools for deep GitHub repository analysis. It features a multi-agent architecture for reliable, cost-aware question answering, supporting both automatic and interactive workflows.

---

## Features

- **Multi-Agent System**: Specialized agents for research, analysis, fact-checking, synthesis, and critique.
- **GitHub MCP Integration**: Agents can use MCP tools to analyze repositories, issues, code, PRs, and commit history.
- **AutoGen v0.4**: Leverages the latest async AgentChat API for efficient, scalable conversations.
- **Cost Tracking**: Tracks and reports OpenAI API token usage and cost per session, per agent, and per message.
- **Interactive & Automatic Modes**: Choose between fully automatic agent collaboration or step-by-step human review and approval.
- **Markdown Export**: Saves detailed conversation flows and analysis to Markdown files for audit and review.
- **Extensible Tools**: Easily add custom tools (e.g., web search, calculator) for agent use.

---

## Requirements

- **Python 3.9+**
- **Node.js** and **npx** (for MCP GitHub integration)
- **Docker** (optional, for code execution agent)
- See `requirements.txt` for Python dependencies

---

## Setup

1. **Clone the repository** and navigate to this directory:
   ```bash
   cd autogen_pruebas/github
   ```
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Node.js** (if not already installed):
   - [Download Node.js](https://nodejs.org/)
   - Ensure `npx` is available in your PATH
4. **Set up environment variables** (create a `.env` file or set in your shell):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GITHUB_TOKEN=your_github_personal_access_token
   MODEL=gpt-4o-mini  # or another supported OpenAI model
   TEMPERATURE=0.1
   MAX_ROUNDS=10
   LOG_LEVEL=INFO
   QA_WORK_DIR=qa_workspace
   ENABLE_GITHUB_MCP=true
   ```
   - **GITHUB_TOKEN**: Required for MCP GitHub integration. [Create a GitHub PAT](https://github.com/settings/tokens)

---

## Usage

Run the main script:

```bash
python main.py
```

You will be prompted to choose a mode:
- **[1] Automatic**: Agents collaborate automatically to answer the question.
- **[2] Interactive**: You review and approve each agent's response step by step.

You can enter your own question and context. If MCP is enabled, agents will use GitHub tools for repository/code analysis.

### Example (Automatic Mode)
- The system answers your question using all available agents and GitHub MCP tools.
- Results, costs, and conversation are displayed and saved to a Markdown file.

### Example (Interactive Mode)
- Enter your own question.
- Review, approve, or request revisions for each agent's response.
- Final answer and full conversation are saved for audit.

---

## GitHub MCP Integration
- Requires a valid GitHub Personal Access Token (PAT) with repo access.
- MCP server is started automatically using `npx @modelcontextprotocol/server-github`.
- Agents can:
  - Explore repositories
  - Search and analyze code
  - Track issues and PRs
  - Analyze commit history and file contents

---

## Cost Tracking & Reporting
- The system tracks input/output tokens and cost for each agent and the overall session.
- Cost breakdowns are shown in the console and in the Markdown export.

---

## Custom Tools
You can add custom tools (e.g., web search, calculator) by passing them to the system. See the `web_search_tool` and `calculator_tool` examples in the code.

---

## Troubleshooting
- Ensure your OpenAI API key and GitHub token are valid and have sufficient permissions.
- Node.js and npx must be installed and available in your PATH for MCP to work.
- For code execution, Docker must be installed and running (if enabled).
- Review logs and Markdown exports for detailed error and cost information.

---

## License
MIT License. See repository for details. 