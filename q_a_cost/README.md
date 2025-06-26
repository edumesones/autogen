# Advanced AutoGen QA System (`main_humancost.py`)

## Overview

This project implements an advanced, modular Question Answering (QA) system using Microsoft's AutoGen v0.4 AgentChat API. It features a multi-agent architecture for deep, reliable, and cost-aware question answering, supporting both automatic and interactive human-in-the-loop workflows.

---

## Features

- **Multi-Agent System**: Specialized agents for research, analysis, fact-checking, synthesis, and critique.
- **AutoGen v0.4 Integration**: Leverages the latest async AgentChat API for efficient, scalable conversations.
- **Cost Tracking**: Tracks and reports OpenAI API token usage and cost per session, per agent, and per message.
- **Interactive & Automatic Modes**: Choose between fully automatic agent collaboration or step-by-step human review and approval.
- **Markdown Export**: Saves detailed conversation flows and analysis to Markdown files for audit and review.
- **Extensible Tools**: Easily add custom tools (e.g., web search, calculator) for agent use.

---

## Agent Roles

- **Researcher**: Gathers facts, statistics, and sources.
- **Analyst**: Analyzes data, trends, and relationships.
- **Fact Checker**: Verifies claims and checks for accuracy.
- **Synthesizer**: Integrates findings into coherent answers.
- **Critic**: Reviews and suggests improvements for quality.
- **Code Executor** (optional): Executes code for data analysis (requires Docker).

---

## Setup & Requirements

1. **Clone the repository** and ensure you have Python 3.9+.
2. **Install dependencies** (see your project requirements):
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables** (create a `.env` file or set in your shell):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL=gpt-4o-mini  # or another supported OpenAI model
   TEMPERATURE=0.1
   MAX_ROUNDS=10
   LOG_LEVEL=INFO
   QA_WORK_DIR=qa_workspace
   ```

---

## Usage

Run the main script:

```bash
python main_humancost.py
```

You will be prompted to choose a mode:
- **[1] Automatic**: Agents collaborate automatically to answer the question.
- **[2] Interactive**: You review and approve each agent's response step by step.

### Example (Automatic Mode)
- The system answers: "What are the main benefits of using Python for data science?"
- Results, costs, and conversation are displayed and saved to a Markdown file.

### Example (Interactive Mode)
- Enter your own question.
- Review, approve, or request revisions for each agent's response.
- Final answer and full conversation are saved for audit.

---

## Cost Tracking & Reporting
- The system tracks input/output tokens and cost for each agent and the overall session.
- Cost breakdowns are shown in the console and in the Markdown export.

---

## Custom Tools
You can add custom tools (e.g., web search, calculator) by passing them to the system. See the `web_search_tool` and `calculator_tool` examples in the code.

---

## Extending & Customizing
- **Agent Prompts**: Modify `SystemMessageTemplates` for custom agent behavior.
- **Team Structure**: Use round-robin or selector-based agent teams.
- **Cost Model**: Update `CostCalculator.MODEL_PRICING` for new models or pricing changes.

---

## Troubleshooting
- Ensure your OpenAI API key is valid and has sufficient quota.
- For code execution, Docker must be installed and running (if enabled).
- Review logs and Markdown exports for detailed error and cost information.

---

## License
MIT License. See repository for details. 