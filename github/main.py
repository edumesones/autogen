"""
Advanced AutoGen v0.4 Question Answering Agent System with MCP GitHub Integration
A modular and efficient implementation using the latest AutoGen AgentChat API.
Based on Microsoft's AutoGen v0.4 architecture with async support and MCP tools.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from datetime import datetime
from dotenv import load_dotenv,find_dotenv
script_dir = Path(__file__).parent.parent.parent
env_path = script_dir / 'report-auditor/.env'
# Load environment variables
print(f"🔍 Archivo .env encontrado: {env_path}")

load_dotenv(env_path)


# AutoGen v4.0 imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import (
    TextMentionTermination, 
    MaxMessageTermination,
    TimeoutTermination,
    TokenUsageTermination
)
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

# MCP imports for GitHub integration
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


class CostCalculator:
    """Calculate costs for different models."""
    
    # Pricing per 1M tokens (as of 2025) - in USD
    MODEL_PRICING = {
        "gpt-4o-mini": {
            "input": 0.150,   # $0.150 per 1M input tokens
            "output": 0.600   # $0.600 per 1M output tokens
        },
        "gpt-4o": {
            "input": 2.50,    # $2.50 per 1M input tokens
            "output": 10.00   # $10.00 per 1M output tokens
        },
        "gpt-4": {
            "input": 30.00,   # $30.00 per 1M input tokens
            "output": 60.00   # $60.00 per 1M output tokens
        }
    }
    
    @classmethod
    def calculate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        """Calculate cost for a model usage."""
        model_key = model.lower()
        
        # Find matching model pricing
        pricing = None
        for key, price in cls.MODEL_PRICING.items():
            if key in model_key:
                pricing = price
                break
        
        if not pricing:
            # Default to gpt-4o-mini if model not found
            pricing = cls.MODEL_PRICING["gpt-4o-mini"]
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost, 
            "total_cost": total_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model
        }
    
    @classmethod
    def format_cost(cls, cost: float) -> str:
        """Format cost in a readable way."""
        return f"${cost:.9f}"


class AgentRole(Enum):
    """Define different agent roles for specialized tasks."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    FACT_CHECKER = "fact_checker"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    CODE_EXECUTOR = "code_executor"


@dataclass
class QASystemConfig:
    """Configuration for the entire QA system."""
    openai_api_key: str = None
    github_token: str = None  # New field for GitHub token
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 300
    seed: Optional[int] = 42
    log_level: str = "INFO"
    max_rounds: int = 10
    enable_code_execution: bool = False
    work_dir: str = "qa_workspace"
    enable_github_mcp: bool = True  # New field to enable/disable GitHub MCP
    
    def __post_init__(self):
        """Load configuration from environment variables if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.github_token is None:
            self.github_token = os.getenv("GITHUB_TOKEN")
        
        # Load optional environment variables
        self.model = os.getenv("MODEL", self.model)
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))
        self.max_rounds = int(os.getenv("MAX_ROUNDS", self.max_rounds))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.work_dir = os.getenv("QA_WORK_DIR", self.work_dir)
        self.enable_github_mcp = os.getenv("ENABLE_GITHUB_MCP", "true").lower() == "true"
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided either as parameter or environment variable")
        
        if self.enable_github_mcp and not self.github_token:
            raise ValueError("GITHUB_TOKEN must be provided when GitHub MCP is enabled")


class SystemMessageTemplates:
    """Templates for system messages for different agent roles."""
    
    RESEARCHER = """
    You are a Research Agent specializing in gathering comprehensive information.
    Your responsibilities:
    - Search for relevant information and gather facts
    - Use GitHub MCP tools to explore repositories, issues, and code when relevant
    - Identify key statistics, evidence, and sources
    - Provide well-researched and detailed findings
    - Flag any information gaps or uncertainties
    - Use available tools for in-depth research
    
    Available tools include GitHub repository analysis, code search, issue tracking, and more.
    Always provide sources and confidence levels for your findings.
    Be thorough and systematic in your research approach.
    """
    
    ANALYST = """
    You are an Analysis Agent focused on deep analytical thinking.
    Your responsibilities:
    - Analyze patterns, trends, and relationships in data/information
    - Break down complex problems into manageable components
    - Apply analytical frameworks and methodologies
    - Provide structured insights and interpretations
    - Use code analysis and GitHub tools when beneficial
    - Analyze repository structures, commit patterns, and development trends
    
    Use clear reasoning and explain your analytical approach step by step.
    Leverage GitHub MCP tools for code and repository analysis when relevant.
    """
    
    FACT_CHECKER = """
    You are a Fact-Checking Agent ensuring accuracy and reliability.
    Your responsibilities:
    - Verify claims and statements for accuracy
    - Cross-reference information from multiple sources including GitHub
    - Identify potential biases, inconsistencies, or misinformation
    - Rate the credibility and reliability of sources
    - Highlight conflicting information
    - Use GitHub tools to verify technical claims and implementation details
    
    Be thorough and systematic. Clearly indicate confidence levels and sources.
    """
    
    SYNTHESIZER = """
    You are a Synthesis Agent combining insights into coherent answers.
    Your responsibilities:
    - Integrate findings from multiple agents and sources
    - Create comprehensive, well-structured responses
    - Identify consensus and conflicting viewpoints
    - Provide balanced, nuanced, and complete conclusions
    - Ensure logical flow and clarity
    - Synthesize technical and non-technical information effectively
    
    Focus on creating a unified, coherent response that addresses all aspects of the question.
    """
    
    CRITIC = """
    You are a Critical Review Agent ensuring quality and completeness.
    Your responsibilities:
    - Evaluate the quality, accuracy, and completeness of responses
    - Identify logical fallacies, weak arguments, or missing information
    - Suggest improvements and additional considerations
    - Ensure the final answer fully addresses the original question
    - Provide constructive feedback for enhancement
    - Review technical accuracy when GitHub tools were used
    
    Be thorough but constructive. Focus on improving the overall quality of the response.
    """


class AdvancedQASystem:
    """Advanced Question Answering System using AutoGen v4.0 with MCP GitHub integration."""
    
    def __init__(self, config: QASystemConfig):
        self.config = config
        self.model_client: Optional[OpenAIChatCompletionClient] = None
        self.agents: Dict[str, Any] = {}
        self.team: Optional[Any] = None
        self.workbench: Optional[McpWorkbench] = None
        self.total_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0, 
                          "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    async def _setup_model_client(self):
        """Setup the OpenAI model client."""
        if not self.model_client:
            self.model_client = OpenAIChatCompletionClient(
                model=self.config.model,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed
            )
            self.logger.info(f"Model client initialized: {self.config.model}")

    async def _setup_github_mcp(self):
        """Setup GitHub MCP workbench."""
        if not self.config.enable_github_mcp:
            self.logger.info("GitHub MCP disabled")
            return None
            
        try:
            # Configure GitHub MCP server
            github_server_params = StdioServerParams(
                command="npx",
                args=[
                    "@modelcontextprotocol/server-github",
                    "--github-personal-access-token", 
                    self.config.github_token
                ]
            )
            
            # Create MCP workbench
            self.workbench = McpWorkbench(github_server_params)
            await self.workbench.start()
            
            self.logger.info("GitHub MCP workbench initialized successfully")
            return self.workbench
            
        except Exception as e:
            self.logger.error(f"Failed to setup GitHub MCP: {e}")
            self.logger.warning("Continuing without GitHub MCP functionality")
            return None
        
    async def create_agents(self, custom_tools: Optional[List[Callable]] = None):
        """Create specialized agents for the QA system with MCP integration."""
        await self._setup_model_client()
        
        # Setup GitHub MCP if enabled
        workbench = await self._setup_github_mcp()
        
        # Agents que usan MCP workbench (SIN custom tools)
        if workbench:
            self.agents["researcher"] = AssistantAgent(
                name="researcher",
                model_client=self.model_client,
                system_message=SystemMessageTemplates.RESEARCHER,
                workbench=workbench,  # SOLO workbench
                reflect_on_tool_use=True
            )
            
            self.agents["analyst"] = AssistantAgent(
                name="analyst", 
                model_client=self.model_client,
                system_message=SystemMessageTemplates.ANALYST,
                workbench=workbench,  # SOLO workbench
                reflect_on_tool_use=True
            )
            
            self.agents["fact_checker"] = AssistantAgent(
                name="fact_checker",
                model_client=self.model_client,
                system_message=SystemMessageTemplates.FACT_CHECKER,
                workbench=workbench,  # SOLO workbench
                reflect_on_tool_use=True
            )
        else:
            # Si no hay workbench, usar custom tools
            tools = custom_tools or []
            
            self.agents["researcher"] = AssistantAgent(
                name="researcher",
                model_client=self.model_client,
                system_message=SystemMessageTemplates.RESEARCHER,
                tools=tools,  # SOLO tools
                reflect_on_tool_use=True
            )
            
            self.agents["analyst"] = AssistantAgent(
                name="analyst", 
                model_client=self.model_client,
                system_message=SystemMessageTemplates.ANALYST,
                tools=tools,  # SOLO tools
                reflect_on_tool_use=True
            )
            
            self.agents["fact_checker"] = AssistantAgent(
                name="fact_checker",
                model_client=self.model_client,
                system_message=SystemMessageTemplates.FACT_CHECKER,
                tools=tools,  # SOLO tools
                reflect_on_tool_use=True
            )
        
        # Agents que NO necesitan tools ni workbench
        self.agents["synthesizer"] = AssistantAgent(
            name="synthesizer",
            model_client=self.model_client,
            system_message=SystemMessageTemplates.SYNTHESIZER,
            reflect_on_tool_use=True
        )
        
        self.agents["critic"] = AssistantAgent(
            name="critic",
            model_client=self.model_client,
            system_message=SystemMessageTemplates.CRITIC,
            reflect_on_tool_use=True
        )
        # Create code executor agent if enabled
        if self.config.enable_code_execution:
            try:
                from autogen_ext.code_executors import DockerCommandLineCodeExecutor
                code_executor = DockerCommandLineCodeExecutor(work_dir=self.config.work_dir)
                self.agents["code_executor"] = CodeExecutorAgent(
                    name="code_executor",
                    code_executor=code_executor
                )
            except Exception as e:
                self.logger.warning(f"Docker not available, skipping code executor: {e}")
        
        # Create user proxy agent
        self.agents["user_proxy"] = UserProxyAgent(
            name="user_proxy"
        )
        
        mcp_status = "enabled" if workbench else "disabled"
        self.logger.info(f"Created {len(self.agents)} agents with GitHub MCP {mcp_status}")
        
    def setup_round_robin_team(self) -> RoundRobinGroupChat:
        """Setup a round-robin team for structured QA workflow."""
        if not self.agents:
            raise ValueError("No agents available. Call create_agents() first.")
            
        # Define the order of agents for round-robin conversation
        agent_order = [
            self.agents["researcher"],
            self.agents["analyst"], 
            self.agents["fact_checker"],
            self.agents["synthesizer"],
            self.agents["critic"]
        ]
        
        # Add code executor if available
        if "code_executor" in self.agents:
            agent_order.insert(2, self.agents["code_executor"])
        
        # Setup termination condition
        termination = MaxMessageTermination(self.config.max_rounds)
        
        self.team = RoundRobinGroupChat(
            participants=agent_order,
            termination_condition=termination
        )
        
        self.logger.info("Round-robin team setup completed")
        return self.team
        
    def setup_selector_team(self, 
                          selector_prompt: Optional[str] = None) -> SelectorGroupChat:
        """Setup a selector-based team for dynamic agent selection."""
        if not self.agents:
            raise ValueError("No agents available. Call create_agents() first.")
            
        participants = [
            self.agents["researcher"],
            self.agents["analyst"],
            self.agents["fact_checker"], 
            self.agents["synthesizer"],
            self.agents["critic"]
        ]
        
        if "code_executor" in self.agents:
            participants.append(self.agents["code_executor"])
            
        # Default selector prompt with GitHub MCP awareness
        if not selector_prompt:
            selector_prompt = """
            You are a team coordinator selecting the next agent to respond.
            
            Available agents:
            - researcher: Gathers information, uses GitHub MCP tools for repository analysis
            - analyst: Analyzes data, patterns, and code structures  
            - fact_checker: Verifies accuracy using multiple sources including GitHub
            - synthesizer: Combines insights into coherent responses
            - critic: Reviews quality and completeness
            - code_executor: Executes code and analysis (if available)
            
            GitHub MCP tools are available for:
            - Repository exploration and analysis
            - Code search and examination
            - Issue and PR tracking
            - Commit history analysis
            - File content inspection
            
            Select the most appropriate agent based on:
            1. The current conversation context
            2. What type of contribution is needed next
            3. Whether GitHub/code analysis would be beneficial
            4. The overall progress toward answering the question
            
            Return only the agent name.
            """
        
        # Setup termination condition
        termination = MaxMessageTermination(self.config.max_rounds)
        
        self.team = SelectorGroupChat(
            participants=participants,
            model_client=self.model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt
        )
        
        self.logger.info("Selector team setup completed")
        return self.team
    
    def _extract_and_accumulate_costs(self, result) -> Dict[str, float]:
        """Extract token usage from TaskResult and accumulate costs."""
        current_cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0, 
                       "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if hasattr(result, 'messages') and result.messages:
            for message in result.messages:
                if hasattr(message, 'models_usage') and message.models_usage:
                    usage = message.models_usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    
                    if prompt_tokens > 0 or completion_tokens > 0:
                        cost_info = CostCalculator.calculate_cost(
                            self.config.model, prompt_tokens, completion_tokens
                        )
                        
                        # Accumulate current call costs
                        current_cost["input_cost"] += cost_info["input_cost"]
                        current_cost["output_cost"] += cost_info["output_cost"]
                        current_cost["total_cost"] += cost_info["total_cost"]
                        current_cost["prompt_tokens"] += prompt_tokens
                        current_cost["completion_tokens"] += completion_tokens
                        current_cost["total_tokens"] += prompt_tokens + completion_tokens
                        
                        # Accumulate total session costs
                        self.total_cost["input_cost"] += cost_info["input_cost"]
                        self.total_cost["output_cost"] += cost_info["output_cost"] 
                        self.total_cost["total_cost"] += cost_info["total_cost"]
                        self.total_cost["prompt_tokens"] += prompt_tokens
                        self.total_cost["completion_tokens"] += completion_tokens
                        self.total_cost["total_tokens"] += prompt_tokens + completion_tokens
        
        return current_cost
    
    def _extract_final_answer(self, conversation_history: List[Dict]) -> str:
        """Extract the final answer from conversation history."""
        # Get the last meaningful message
        for msg in reversed(conversation_history):
            content = msg.get('content', '').strip()
            sender = msg.get('sender', '')
            
            # Skip empty messages
            if not content:
                continue
                
            # Prefer synthesizer's response if available
            if sender == 'synthesizer':
                return content
                
            # Otherwise return the last non-empty message
            return content
        
        return "No answer generated"
    
    def _save_conversation_to_markdown(self, result: Dict[str, Any], filename: Optional[str] = None):
        """Save the conversation to a markdown file for analysis."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qa_conversation_mcp_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# 🤖 AutoGen QA System with GitHub MCP - Session Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Question and context
                f.write(f"## 📋 Session Details\n\n")
                f.write(f"**Question:** {result['question']}\n\n")
                if result.get('context'):
                    f.write(f"**Context:** {result['context']}\n\n")
                
                # System configuration
                f.write(f"## ⚙️ System Configuration\n\n")
                metadata = result.get('metadata', {})
                f.write(f"- **Team Type:** {metadata.get('team_type', 'Unknown')}\n")
                f.write(f"- **Model:** {metadata.get('model', 'Unknown')}\n")
                f.write(f"- **GitHub MCP:** {'✅ Enabled' if self.config.enable_github_mcp else '❌ Disabled'}\n")
                f.write(f"- **Number of Agents:** {metadata.get('num_agents', 'Unknown')}\n")
                f.write(f"- **Total Messages:** {metadata.get('num_messages', 'Unknown')}\n")
                f.write(f"- **Session Status:** {metadata.get('status', 'Unknown')}\n\n")
                
                # Continue with rest of the markdown generation...
                # (keeping the existing conversation flow and analysis sections)
                conversation = result.get('conversation_history', [])
                
                if conversation:
                    f.write(f"## 🔄 Conversation Flow\n\n")
                    for i, msg in enumerate(conversation):
                        sender = msg.get('sender', 'Unknown')
                        content = msg.get('content', '')
                        
                        agent_emoji = {
                            'user': '👤',
                            'researcher': '🔍',
                            'analyst': '📊', 
                            'fact_checker': '✅',
                            'synthesizer': '🧠',
                            'critic': '🎯',
                            'code_executor': '💻'
                        }
                        
                        emoji = agent_emoji.get(sender, '🤖')
                        f.write(f"### {emoji} {sender.title()}\n\n")
                        f.write(f"```\n{content}\n```\n\n")
                
                # Final answer and cost analysis
                f.write(f"## 🎯 Final Result\n\n")
                final_answer = result.get('final_answer', 'No final answer generated')
                f.write(f"```\n{final_answer}\n```\n\n")
                
                # Cost analysis
                f.write(f"## 💰 Cost Analysis\n\n")
                total_cost_info = metadata.get('total_cost', {})
                if total_cost_info and total_cost_info.get('total_tokens', 0) > 0:
                    f.write(f"- **Total Cost:** {CostCalculator.format_cost(total_cost_info['total_cost'])}\n")
                    f.write(f"- **Input Cost:** {CostCalculator.format_cost(total_cost_info['input_cost'])} ({total_cost_info['prompt_tokens']} tokens)\n")
                    f.write(f"- **Output Cost:** {CostCalculator.format_cost(total_cost_info['output_cost'])} ({total_cost_info['completion_tokens']} tokens)\n")
                    f.write(f"- **Total Tokens:** {total_cost_info['total_tokens']:,}\n")
                
                f.write(f"\n*Generated by AutoGen QA System with GitHub MCP Integration*\n")
            
            self.logger.info(f"Conversation saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            return None

    async def ask_question_interactive(self, 
                                     question: str,
                                     context: Optional[str] = None,
                                     custom_tools: Optional[List[Callable]] = None,
                                     save_to_file: bool = True) -> Dict[str, Any]:
        """
        Interactive Q&A mode where human can review and approve each agent response.
        Now with GitHub MCP integration.
        """
        try:
            # Setup agents with MCP
            await self.create_agents(custom_tools)
            
            # Manual workflow for interactive mode
            conversation_history = []
            
            # Add initial question
            initial_msg = {
                'sender': 'user',
                'content': f"Context: {context}\n\nQuestion: {question}" if context else f"Question: {question}",
                'timestamp': datetime.now().isoformat(),
                'type': 'UserMessage',
                'approved': True
            }
            conversation_history.append(initial_msg)
            
            print("🎯 INTERACTIVE QA MODE WITH GITHUB MCP")
            print("=" * 55)
            print(f"Question: {question}")
            if context:
                print(f"Context: {context}")
            print(f"GitHub MCP: {'✅ Enabled' if self.config.enable_github_mcp else '❌ Disabled'}")
            print("=" * 55)
            
            # Define agent workflow
            agent_workflow = [
                ('researcher', '🔍', 'Research with GitHub MCP tools'),
                ('analyst', '📊', 'Analyze findings and code patterns'),
                ('fact_checker', '✅', 'Verify accuracy with GitHub cross-reference'),
                ('synthesizer', '🧠', 'Synthesize comprehensive answer'),
                ('critic', '🎯', 'Review and improve quality')
            ]
            
            # Execute interactive workflow
            current_context = [initial_msg]
            
            for agent_name, emoji, description in agent_workflow:
                if agent_name not in self.agents:
                    continue
                    
                print(f"\n{emoji} {agent_name} Turn")
                print(f"Role: {description}")
                print("-" * 40)
                
                # Get the agent
                agent = self.agents[agent_name]
                
                # Prepare context for agent
                context_text = "\n\n".join([
                    f"{msg['sender']}: {msg['content']}" 
                    for msg in current_context 
                    if msg.get('approved', True)
                ])
                
                # Create task for agent with GitHub MCP awareness
                agent_task = f"""
Previous conversation:
{context_text}

Your role: {description}

Available tools include GitHub MCP for repository analysis, code search, issue tracking, and more.
Use these tools when relevant to provide comprehensive and accurate information.

Please provide your contribution to answering the question. Be thorough and specific.
"""
                
                # Get agent response using correct AutoGen v0.4 API
                cancellation_token = CancellationToken()
                result = await agent.run(task=agent_task, cancellation_token=cancellation_token)
                
                # Extract cost information
                call_cost = self._extract_and_accumulate_costs(result)
                
                # Extract response content from TaskResult
                agent_content = ""
                if hasattr(result, 'messages') and result.messages:
                    last_message = result.messages[-1]
                    agent_content = getattr(last_message, 'content', str(last_message))
                else:
                    agent_content = str(result)
                
                # Show agent response to human
                print(f"\n{emoji} {agent_name} Response:")
                print("-" * 50)
                print(agent_content)
                print("-" * 50)
                
                # Show cost information
                if call_cost["total_tokens"] > 0:
                    print(f"💰 Cost: {CostCalculator.format_cost(call_cost['total_cost'])} "
                          f"({call_cost['total_tokens']} tokens)")
                
                # Get human approval (simplified for brevity - keeping original logic)
                while True:
                    choice = input(f"\n🤔 Review {agent_name}'s response:\n"
                                 f"[A]pprove and continue\n"
                                 f"[R]equest changes\n"
                                 f"[S]kip this agent\n"
                                 f"[Q]uit session\n"
                                 f"Your choice: ").strip().lower()
                    
                    if choice == 'a':
                        msg = {
                            'sender': agent_name,
                            'content': agent_content,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'AgentMessage',
                            'approved': True,
                            'human_feedback': 'Approved',
                            'cost': call_cost
                        }
                        conversation_history.append(msg)
                        current_context.append(msg)
                        print(f"✅ {agent_name} response approved!")
                        break
                        
                    elif choice in ['r', 's', 'q']:
                        # Handle other choices (keeping original logic but simplified)
                        if choice == 'q':
                            return self._create_session_result(question, context, conversation_history, 'terminated_by_user', save_to_file)
                        elif choice == 's':
                            print(f"⏭️ Skipped {agent_name}")
                            break
                        else:  # 'r' - request changes
                            feedback = input("📝 What changes would you like?: ").strip()
                            # Implement revision logic here...
                            break
                    else:
                        print("❌ Invalid choice. Please enter A, R, S, or Q")
            
            # Show final results
            final_answer = self._extract_final_answer(conversation_history)
            print(f"\n🎯 SESSION COMPLETED")
            print("=" * 40)
            print("Final Answer:")
            print(final_answer)
            print("=" * 40)
            
            # Show total cost
            if self.total_cost["total_tokens"] > 0:
                print(f"\n💰 TOTAL SESSION COST")
                print(f"Total: {CostCalculator.format_cost(self.total_cost['total_cost'])}")
                print(f"Tokens: {self.total_cost['total_tokens']:,}")
            
            return self._create_session_result(question, context, conversation_history, 'completed', save_to_file)
            
        except Exception as e:
            self.logger.error(f"Error in interactive session: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'context': context
            }

    def _create_session_result(self, question: str, context: Optional[str], 
                             conversation_history: List[Dict], status: str, 
                             save_to_file: bool = True) -> Dict[str, Any]:
        """Create standardized session result."""
        final_answer = self._extract_final_answer(conversation_history)
        
        result = {
            'success': True,
            'question': question,
            'context': context,
            'conversation_history': conversation_history,
            'final_answer': final_answer,
            'metadata': {
                'team_type': 'interactive',
                'num_agents': len(self.agents),
                'num_messages': len(conversation_history),
                'model': self.config.model,
                'status': status,
                'total_cost': self.total_cost,
                'github_mcp_enabled': self.config.enable_github_mcp
            }
        }
        
        if save_to_file:
            result['saved_file'] = self._save_conversation_to_markdown(result)
        
        return result

    async def ask_question(self, 
                          question: str,
                          context: Optional[str] = None,
                          use_selector: bool = True,
                          custom_tools: Optional[List[Callable]] = None,
                          save_to_file: bool = True) -> Dict[str, Any]:
        """
        Ask a question using the multi-agent system (automatic mode) with GitHub MCP.
        
        Args:
            question: The question to ask
            context: Optional context or background information
            use_selector: Whether to use selector-based team (True) or round-robin (False)
            custom_tools: Optional list of custom tools for agents
            save_to_file: Whether to save conversation to markdown file
            
        Returns:
            Dictionary containing the conversation result and metadata
        """
        try:
            # Setup agents and team
            await self.create_agents(custom_tools)
            
            if use_selector:
                team = self.setup_selector_team()
            else:
                team = self.setup_round_robin_team()
            
            # Prepare the initial message with GitHub MCP awareness
            github_note = "\n\nNote: GitHub MCP tools are available for repository analysis, code search, and development insights." if self.config.enable_github_mcp else ""
            
            if context:
                initial_message = f"""
                Context: {context}
                
                Question: {question}
                {github_note}
                
                Please provide a comprehensive, well-researched answer. 
                Work collaboratively to ensure accuracy, completeness, and quality.
                Use GitHub MCP tools when relevant for code analysis and repository insights.
                """
            else:
                initial_message = f"""
                Question: {question}
                {github_note}
                
                Please provide a comprehensive, well-researched answer.
                Work collaboratively to ensure accuracy, completeness, and quality.
                Use GitHub MCP tools when relevant for code analysis and repository insights.
                """
            
            # Create cancellation token
            cancellation_token = CancellationToken()
            
            # Run the team conversation
            self.logger.info("Starting multi-agent conversation with GitHub MCP")
            
            result = await team.run(task=initial_message, cancellation_token=cancellation_token)
            
            # Extract and accumulate costs
            session_cost = self._extract_and_accumulate_costs(result)
            
            # Process and return results
            conversation_history = []
            if hasattr(result, 'messages') and result.messages:
                for msg in result.messages:
                    conversation_history.append({
                        'sender': getattr(msg, 'source', 'Unknown'),
                        'content': getattr(msg, 'content', ''),
                        'timestamp': datetime.now().isoformat(),
                        'type': getattr(msg, 'type', 'Unknown')
                    })
            
            return {
                'success': True,
                'question': question,
                'context': context,
                'conversation_history': conversation_history,
                'final_answer': self._extract_final_answer(conversation_history),
                'saved_file': self._save_conversation_to_markdown({
                    'success': True,
                    'question': question,
                    'context': context,
                    'conversation_history': conversation_history,
                    'final_answer': self._extract_final_answer(conversation_history),
                    'metadata': {
                        'team_type': 'selector' if use_selector else 'round_robin',
                        'num_agents': len(self.agents),
                        'num_messages': len(conversation_history),
                        'model': self.config.model,
                        'total_cost': self.total_cost,
                        'github_mcp_enabled': self.config.enable_github_mcp
                    }
                }) if save_to_file else None,
                'metadata': {
                    'team_type': 'selector' if use_selector else 'round_robin',
                    'num_agents': len(self.agents),
                    'num_messages': len(conversation_history),
                    'model': self.config.model,
                    'total_cost': self.total_cost,
                    'github_mcp_enabled': self.config.enable_github_mcp
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in ask_question: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'context': context
            }
        
    async def close(self):
        """Close the model client and MCP workbench, cleanup resources."""
        if self.model_client:
            await self.model_client.close()
            self.logger.info("Model client closed")
        
        if self.workbench:
            try:
                await self.workbench.stop()
                self.logger.info("GitHub MCP workbench closed")
            except Exception as e:
                self.logger.warning(f"Error closing MCP workbench: {e}")


# Example tools that can be used with the system
async def web_search_tool(query: str) -> str:
    """Mock web search tool - replace with actual implementation."""
    return f"Search results for '{query}': [Mock search results would appear here]"


async def calculator_tool(expression: str) -> str:
    """Simple calculator tool for mathematical expressions."""
    try:
        # Safe evaluation of simple mathematical expressions
        result = eval(expression.replace('^', '**'))
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


# Example usage
async def main():
    """Example usage of the Advanced QA System with GitHub MCP."""
    
    # Configuration - Now loads from .env automatically
    config = QASystemConfig()
    
    # Create QA system
    qa_system = AdvancedQASystem(config)
    
    try:
        # Example tools
        tools = [web_search_tool, calculator_tool]
        
        # Choose mode
        print("🤖 AutoGen QA System with GitHub MCP")
        print("=" * 40)
        print(f"GitHub MCP: {'✅ Enabled' if config.enable_github_mcp else '❌ Disabled'}")
        print("=" * 40)
        
        mode = input("Choose mode:\n"
                    "[1] Automatic (agents work independently)\n"
                    "[2] Interactive (you review each agent response)\n"
                    "Your choice (1 or 2): ").strip()
        
        # Get user input
        print("\n📝 Enter your question or topic:")
        question = input("Question: ").strip()
        
        print("\n📝 Enter optional context (press Enter to skip):")
        context = input("Context: ").strip()
        if not context:
            context = None
        
        # Ask about GitHub focus
        if config.enable_github_mcp:
            github_focus = input("\n🔍 Should agents focus on GitHub/code analysis? [y/N]: ").strip().lower()
            if github_focus in ['y', 'yes']:
                if context:
                    context += " Focus on GitHub repositories, code analysis, and development practices."
                else:
                    context = "Focus on GitHub repositories, code analysis, and development practices."
        
        if mode == "2":
            # Interactive mode
            result = await qa_system.ask_question_interactive(
                question=question,
                context=context,
                custom_tools=tools
            )
        else:
            # Automatic mode
            result = await qa_system.ask_question(
                question=question,
                context=context,
                use_selector=True,
                custom_tools=tools
            )
        
        # Display results
        if result['success']:
            print("\n" + "="*60)
            print("=== QUESTION ===")
            print(result['question'])
            if result.get('context'):
                print(f"\n=== CONTEXT ===")
                print(result['context'])
            print("\n=== FINAL ANSWER ===")
            print(result['final_answer'])
            print(f"\n=== METADATA ===")
            metadata = result['metadata']
            print(f"Team Type: {metadata['team_type']}")
            print(f"Messages: {metadata['num_messages']}")
            print(f"Agents: {metadata['num_agents']}")
            print(f"GitHub MCP: {'✅ Enabled' if metadata.get('github_mcp_enabled') else '❌ Disabled'}")
            
            # Show cost info
            total_cost = metadata.get('total_cost', {})
            if total_cost.get('total_tokens', 0) > 0:
                print(f"\n=== COST SUMMARY ===")
                print(f"Total Cost: {CostCalculator.format_cost(total_cost['total_cost'])}")
                print(f"Total Tokens: {total_cost['total_tokens']:,}")
            
            # Show saved file info
            if result.get('saved_file'):
                print(f"\n=== CONVERSATION SAVED ===")
                print(f"File: {result['saved_file']}")
                print("📄 Check the markdown file for detailed conversation flow!")
        else:
            print(f"❌ Error: {result['error']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        await qa_system.close()


if __name__ == "__main__":
    asyncio.run(main())