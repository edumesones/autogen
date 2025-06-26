"""
Advanced AutoGen v0.4 Question Answering Agent System
A modular and efficient implementation using the latest AutoGen AgentChat API.
Based on Microsoft's AutoGen v0.4 architecture with async support.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AutoGen v0.4 imports
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
        if cost < 0.001:
            return f"${cost*1000:.3f}m"  # Show in millidollars
        elif cost < 0.01:
            return f"${cost:.4f}"
        else:
            return f"${cost:.3f}"


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
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 300
    seed: Optional[int] = 42
    log_level: str = "INFO"
    max_rounds: int = 10
    enable_code_execution: bool = False  # Disabled by default
    work_dir: str = "qa_workspace"
    
    def __post_init__(self):
        """Load configuration from environment variables if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Load optional environment variables
        self.model = os.getenv("MODEL", self.model)
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))
        self.max_rounds = int(os.getenv("MAX_ROUNDS", self.max_rounds))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.work_dir = os.getenv("QA_WORK_DIR", self.work_dir)
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided either as parameter or environment variable")


class SystemMessageTemplates:
    """Templates for system messages for different agent roles."""
    
    RESEARCHER = """
    You are a Research Agent specializing in gathering comprehensive information.
    Your responsibilities:
    - Search for relevant information and gather facts
    - Identify key statistics, evidence, and sources
    - Provide well-researched and detailed findings
    - Flag any information gaps or uncertainties
    - Use web search tools when available
    
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
    - Use code and data analysis when beneficial
    
    Use clear reasoning and explain your analytical approach step by step.
    """
    
    FACT_CHECKER = """
    You are a Fact-Checking Agent ensuring accuracy and reliability.
    Your responsibilities:
    - Verify claims and statements for accuracy
    - Cross-reference information from multiple sources
    - Identify potential biases, inconsistencies, or misinformation
    - Rate the credibility and reliability of sources
    - Highlight conflicting information
    
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
    
    Be thorough but constructive. Focus on improving the overall quality of the response.
    """


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
        """Format cost in a readable way with full decimal precision."""
        # if cost < 0.000001:  # Less than $0.000001
        #     return f"${cost*1000000:.3f}μ"  # Show in microdollars
        # elif cost < 0.001:  # Less than $0.001
        #     return f"${cost*1000:.6f}m"  # Show in millidollars with 6 decimals
        # elif cost < 1.0:  # Less than $1
        #     return f"${cost:.8f}"  # Show with 8 decimal places
        # else:
        return f"${cost:.9f}"
              # Show with 6 decimal places for larger amounts
    @classmethod
    def show_total_session_cost(self):
        """Show total session cost summary."""
        if self.total_cost["total_tokens"] > 0:
            print(f"\n💰 TOTAL SESSION COST")
            print("-" * 40)
            print(f"Total Cost: {CostCalculator.format_cost(self.total_cost['total_cost'])}")
            print(f"Input: {self.total_cost['prompt_tokens']:,} tokens ({CostCalculator.format_cost(self.total_cost['input_cost'])})")
            print(f"Output: {self.total_cost['completion_tokens']:,} tokens ({CostCalculator.format_cost(self.total_cost['output_cost'])})")
            print(f"Total Tokens: {self.total_cost['total_tokens']:,}")
            print(f"Model: {self.config.model}")
            print("-" * 40)

class AdvancedQASystem:
    """Advanced Question Answering System using AutoGen v0.4."""
    
    def __init__(self, config: QASystemConfig):
        self.config = config
        self.model_client: Optional[OpenAIChatCompletionClient] = None
        self.agents: Dict[str, Any] = {}
        self.team: Optional[Any] = None
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
        
    async def create_agents(self, custom_tools: Optional[List[Callable]] = None):
        """Create specialized agents for the QA system."""
        await self._setup_model_client()
        
        # Create research agent with potential web search capability
        tools = custom_tools or []
        
        self.agents["researcher"] = AssistantAgent(
            name="researcher",
            model_client=self.model_client,
            system_message=SystemMessageTemplates.RESEARCHER,
            tools=tools
        )
        
        # Create analyst agent
        self.agents["analyst"] = AssistantAgent(
            name="analyst", 
            model_client=self.model_client,
            system_message=SystemMessageTemplates.ANALYST,
            tools=tools
        )
        
        # Create fact checker agent
        self.agents["fact_checker"] = AssistantAgent(
            name="fact_checker",
            model_client=self.model_client,
            system_message=SystemMessageTemplates.FACT_CHECKER,
            tools=tools
        )
        
        # Create synthesizer agent
        self.agents["synthesizer"] = AssistantAgent(
            name="synthesizer",
            model_client=self.model_client,
            system_message=SystemMessageTemplates.SYNTHESIZER
        )
        
        # Create critic agent
        self.agents["critic"] = AssistantAgent(
            name="critic",
            model_client=self.model_client,
            system_message=SystemMessageTemplates.CRITIC
        )
        
        # Create code executor agent if enabled
        if self.config.enable_code_execution:
            from autogen_ext.code_executors import DockerCommandLineCodeExecutor
            
            try:
                # Try to create Docker executor first
                code_executor = DockerCommandLineCodeExecutor(work_dir=self.config.work_dir)
                self.agents["code_executor"] = CodeExecutorAgent(
                    name="code_executor",
                    code_executor=code_executor
                )
            except Exception as e:
                self.logger.warning(f"Docker not available, skipping code executor: {e}")
                # Don't create code executor if Docker is not available
        
        # Create user proxy agent
        self.agents["user_proxy"] = UserProxyAgent(
            name="user_proxy"
        )
        
        self.logger.info(f"Created {len(self.agents)} agents")
        
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
            agent_order.insert(2, self.agents["code_executor"])  # After analyst
        
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
            
        # Default selector prompt
        if not selector_prompt:
            selector_prompt = """
            You are a team coordinator selecting the next agent to respond.
            
            Available agents:
            - researcher: Gathers information and facts
            - analyst: Analyzes data and patterns  
            - fact_checker: Verifies accuracy and credibility
            - synthesizer: Combines insights into coherent responses
            - critic: Reviews quality and completeness
            - code_executor: Executes code and analysis (if available)
            
            Select the most appropriate agent based on:
            1. The current conversation context
            2. What type of contribution is needed next
            3. The overall progress toward answering the question
            
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
            filename = f"qa_conversation_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# 🤖 AutoGen QA System - Interactive Session\n\n")
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
                f.write(f"- **Number of Agents:** {metadata.get('num_agents', 'Unknown')}\n")
                f.write(f"- **Total Messages:** {metadata.get('num_messages', 'Unknown')}\n")
                f.write(f"- **Session Status:** {metadata.get('status', 'Unknown')}\n\n")
                
                # Interactive conversation flow
                f.write(f"## 🔄 Interactive Conversation Flow\n\n")
                conversation = result.get('conversation_history', [])
                
                if not conversation:
                    f.write("*No conversation history available*\n\n")
                else:
                    current_step = 0
                    for i, msg in enumerate(conversation):
                        sender = msg.get('sender', 'Unknown')
                        content = msg.get('content', '')
                        timestamp = msg.get('timestamp', '')
                        msg_type = msg.get('type', 'Unknown')
                        approved = msg.get('approved', None)
                        feedback = msg.get('human_feedback', '')
                        
                        # Agent role emoji mapping
                        agent_emoji = {
                            'user': '👤',
                            'researcher': '🔍',
                            'analyst': '📊', 
                            'fact_checker': '✅',
                            'synthesizer': '🧠',
                            'critic': '🎯',
                            'code_executor': '💻',
                            'synthesizer_final': '🎯'
                        }
                        
                        emoji = agent_emoji.get(sender, '🤖')
                        
                        # Determine message category for formatting
                        if sender == 'user':
                            current_step += 1
                            f.write(f"### {emoji} Step {current_step}: Initial Question\n\n")
                            f.write(f"**Time:** {timestamp}\n\n")
                            f.write(f"```\n{content}\n```\n\n")
                            
                        elif msg_type in ['AgentMessage', 'RevisedAgentMessage', 'SkippedAgentMessage', 'FinalAnswer']:
                            current_step += 1
                            
                            # Header with status indicators
                            status_indicators = []
                            if approved is True:
                                status_indicators.append("✅ APPROVED")
                            elif approved is False:
                                status_indicators.append("❌ REJECTED" if msg_type != 'SkippedAgentMessage' else "⏭️ SKIPPED")
                            
                            if msg_type == 'RevisedAgentMessage':
                                status_indicators.append("🔄 REVISED")
                            elif msg_type == 'FinalAnswer':
                                status_indicators.append("🎯 FINAL")
                            
                            status_text = " | ".join(status_indicators) if status_indicators else ""
                            
                            f.write(f"### {emoji} Step {current_step}: {sender.title()}")
                            if status_text:
                                f.write(f" | {status_text}")
                            f.write(f"\n\n")
                            
                            if timestamp:
                                f.write(f"**Time:** {timestamp}\n")
                            
                            # Add cost information to message display
                            if msg_type and msg_type != 'Unknown':
                                f.write(f"**Type:** {msg_type}\n")
                            
                            # Show cost information if available
                            cost_info = msg.get('cost', {})
                            if cost_info and cost_info.get('total_tokens', 0) > 0:
                                f.write(f"**Cost:** {CostCalculator.format_cost(cost_info['total_cost'])} ({cost_info['total_tokens']} tokens)\n")
                            
                            if approved is not None:
                                f.write(f"**Status:** {'Approved' if approved else 'Rejected/Skipped'}\n")
                            f.write(f"\n")
                            
                            # Agent response content
                            if content.strip():
                                f.write(f"**Agent Response:**\n")
                                f.write(f"```\n{content.strip()}\n```\n\n")
                                
                                # Add length info for long messages
                                if len(content) > 1000:
                                    f.write(f"*Response length: {len(content)} characters*\n\n")
                            
                            # Human feedback section
                            if feedback:
                                f.write(f"**👤 Human Feedback:**\n")
                                if approved is False and msg_type != 'SkippedAgentMessage':
                                    f.write(f"```\n{feedback}\n```\n\n")
                                elif msg_type == 'RevisedAgentMessage':
                                    f.write(f"```\n{feedback}\n```\n\n")
                                else:
                                    f.write(f"*{feedback}*\n\n")
                        
                        # Add separator between major steps
                        if current_step > 0 and i < len(conversation) - 1:
                            f.write("---\n\n")
                
                # Final answer section
                f.write(f"## 🎯 Final Result\n\n")
                final_answer = result.get('final_answer', 'No final answer generated')
                f.write(f"**Final Answer:**\n")
                f.write(f"```\n{final_answer}\n```\n\n")
                
                # Comprehensive analysis section
                f.write(f"## 📊 Session Analysis\n\n")
                
                # Basic stats
                f.write(f"### 📈 Basic Statistics\n")
                f.write(f"- **Total interactions:** {len(conversation)}\n")
                f.write(f"- **Conversation steps:** {current_step}\n")
                
                # Agent participation analysis
                agent_stats = {}
                approval_stats = {'approved': 0, 'rejected': 0, 'skipped': 0, 'revised': 0}
                
                for msg in conversation:
                    sender = msg.get('sender', 'Unknown')
                    msg_type = msg.get('type', 'Unknown')
                    approved = msg.get('approved', None)
                    
                    # Count agent participation
                    if sender != 'user':
                        if sender not in agent_stats:
                            agent_stats[sender] = {'messages': 0, 'approved': 0, 'rejected': 0, 'revised': 0}
                        agent_stats[sender]['messages'] += 1
                        
                        # Count approval status
                        if approved is True:
                            agent_stats[sender]['approved'] += 1
                            approval_stats['approved'] += 1
                        elif approved is False:
                            if msg_type == 'SkippedAgentMessage':
                                approval_stats['skipped'] += 1
                            else:
                                agent_stats[sender]['rejected'] += 1
                                approval_stats['rejected'] += 1
                        
                        if msg_type == 'RevisedAgentMessage':
                            agent_stats[sender]['revised'] += 1
                            approval_stats['revised'] += 1
                if self.total_cost["total_tokens"] > 0:
                    print(f"\n💰 TOTAL SESSION COST")
                    print("-" * 30)
                    print(f"Total Cost: {CostCalculator.format_cost(self.total_cost['total_cost'])}")
                    print(f"Input Tokens: {self.total_cost['prompt_tokens']} ({CostCalculator.format_cost(self.total_cost['input_cost'])})")
                    print(f"Output Tokens: {self.total_cost['completion_tokens']} ({CostCalculator.format_cost(self.total_cost['output_cost'])})")
                    print(f"Total Tokens: {self.total_cost['total_tokens']}")
                    print("-" * 30)
                # Agent participation breakdown
                if agent_stats:
                    f.write(f"\n### 🤖 Agent Performance\n")
                    for agent, stats in agent_stats.items():
                        f.write(f"- **{agent}:**\n")
                        f.write(f"  - Total responses: {stats['messages']}\n")
                        f.write(f"  - ✅ Approved: {stats['approved']}\n")
                        f.write(f"  - ❌ Rejected: {stats['rejected']}\n")
                        f.write(f"  - 🔄 Revised: {stats['revised']}\n")
                
                # Overall approval rates
                f.write(f"\n### 👤 Human Review Summary\n")
                total_reviews = approval_stats['approved'] + approval_stats['rejected'] + approval_stats['skipped']
                if total_reviews > 0:
                    f.write(f"- **Total reviews:** {total_reviews}\n")
                    f.write(f"- **✅ Approved:** {approval_stats['approved']} ({approval_stats['approved']/total_reviews*100:.1f}%)\n")
                    f.write(f"- **❌ Rejected:** {approval_stats['rejected']} ({approval_stats['rejected']/total_reviews*100:.1f}%)\n")
                    f.write(f"- **⏭️ Skipped:** {approval_stats['skipped']} ({approval_stats['skipped']/total_reviews*100:.1f}%)\n")
                    f.write(f"- **🔄 Revisions:** {approval_stats['revised']}\n")
                
                # Workflow analysis
                f.write(f"\n### 🔄 Workflow Analysis\n")
                agent_flow = []
                for msg in conversation:
                    sender = msg.get('sender', 'Unknown')
                    approved = msg.get('approved', None)
                    if sender != 'user' and approved is not False:  # Include approved and final messages
                        agent_flow.append(sender)
                
                if agent_flow:
                    f.write(f"- **Execution flow:** {' → '.join(agent_flow)}\n")
                
                # Message types breakdown
                type_counts = {}
                for msg in conversation:
                    msg_type = msg.get('type', 'Unknown')
                    type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
                
                if type_counts:
                    f.write(f"\n### 📝 Message Types\n")
                    for msg_type, count in sorted(type_counts.items()):
                        f.write(f"- **{msg_type}:** {count}\n")
                
                # Cost analysis section
                f.write(f"\n### 💰 Cost Analysis\n")
                total_cost_info = metadata.get('total_cost', {})
                if total_cost_info and total_cost_info.get('total_tokens', 0) > 0:
                    f.write(f"- **Total Cost:** {CostCalculator.format_cost(total_cost_info['total_cost'])}\n")
                    f.write(f"- **Input Cost:** {CostCalculator.format_cost(total_cost_info['input_cost'])} ({total_cost_info['prompt_tokens']} tokens)\n")
                    f.write(f"- **Output Cost:** {CostCalculator.format_cost(total_cost_info['output_cost'])} ({total_cost_info['completion_tokens']} tokens)\n")
                    f.write(f"- **Total Tokens:** {total_cost_info['total_tokens']:,}\n")
                    f.write(f"- **Model:** {metadata.get('model', 'Unknown')}\n")
                    
                    # Cost per interaction
                    if total_reviews > 0:
                        cost_per_interaction = total_cost_info['total_cost'] / total_reviews
                        f.write(f"- **Average cost per interaction:** {CostCalculator.format_cost(cost_per_interaction)}\n")
                
                # Individual agent costs
                agent_costs = {}
                total_agent_cost = 0.0
                for msg in conversation:
                    sender = msg.get('sender', 'Unknown')
                    cost_info = msg.get('cost', {})
                    if cost_info and cost_info.get('total_cost', 0) > 0:
                        if sender not in agent_costs:
                            agent_costs[sender] = {'total_cost': 0.0, 'total_tokens': 0, 'calls': 0}
                        agent_costs[sender]['total_cost'] += cost_info['total_cost']
                        agent_costs[sender]['total_tokens'] += cost_info['total_tokens']
                        agent_costs[sender]['calls'] += 1
                        total_agent_cost += cost_info['total_cost']
                
                if agent_costs:
                    f.write(f"\n### 💳 Cost Breakdown by Agent\n")
                    for agent, cost_data in sorted(agent_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True):
                        percentage = (cost_data['total_cost'] / total_agent_cost * 100) if total_agent_cost > 0 else 0
                        f.write(f"- **{agent}:** {CostCalculator.format_cost(cost_data['total_cost'])} ({percentage:.1f}%)\n")
                        f.write(f"  - Tokens: {cost_data['total_tokens']:,}\n")
                        f.write(f"  - API calls: {cost_data['calls']}\n")
                status = metadata.get('status', 'unknown')
                if status == 'completed':
                    f.write(f"- **Status:** ✅ Successfully completed\n")
                elif status == 'terminated_by_user':
                    f.write(f"- **Status:** 🛑 Terminated by user\n")
                else:
                    f.write(f"- **Status:** ❓ {status}\n")
                
                # Success metrics
                if total_reviews > 0:
                    success_rate = approval_stats['approved'] / total_reviews * 100
                    f.write(f"- **Approval rate:** {success_rate:.1f}%\n")
                    f.write(f"- **Revision rate:** {approval_stats['revised'] / total_reviews * 100:.1f}%\n")
                
                # Session summary
                f.write(f"\n---\n")
                f.write(f"**Session Summary:** ")
                if status == 'completed':
                    f.write(f"Interactive session completed successfully with {approval_stats['approved']} approved responses")
                    if approval_stats['revised'] > 0:
                        f.write(f" and {approval_stats['revised']} revisions")
                elif status == 'terminated_by_user':
                    f.write(f"Session was terminated early by user after {current_step} steps")
                f.write(f".\n\n")
                
                f.write(f"*Generated by AutoGen QA System - Interactive Mode*\n")
            
            self.logger.info(f"Interactive conversation saved to: {filename}")
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
        
        Args:
            question: The question to ask
            context: Optional context or background information
            custom_tools: Optional list of custom tools for agents
            save_to_file: Whether to save conversation to markdown file
            
        Returns:
            Dictionary containing the conversation result and metadata
        """
        try:
            # Setup agents
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
            
            print("🎯 INTERACTIVE QA MODE STARTED")
            print("=" * 50)
            print(f"Question: {question}")
            if context:
                print(f"Context: {context}")
            print("=" * 50)
            
            # Define agent workflow
            agent_workflow = [
                ('researcher', '🔍', 'Research and gather information'),
                ('analyst', '📊', 'Analyze the findings'),
                ('fact_checker', '✅', 'Verify accuracy and credibility'),
                ('synthesizer', '🧠', 'Synthesize final answer'),
                ('critic', '🎯', 'Review and improve quality')
            ]
            
            # Execute interactive workflow
            current_context = [initial_msg]
            
            for agent_name, emoji, description in agent_workflow:
                if agent_name not in self.agents:
                    continue
                    
                print(f"\n{emoji} {agent_name} Turn")
                print(f"Role: {description}")
                print("-" * 30)
                
                # Get the agent
                agent = self.agents[agent_name]
                
                # Prepare context for agent
                context_text = "\n\n".join([
                    f"{msg['sender']}: {msg['content']}" 
                    for msg in current_context 
                    if msg.get('approved', True)
                ])
                
                # Create task for agent
                agent_task = f"""
Previous conversation:
{context_text}

Your role: {description}

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
                    # Last message is the agent's final response
                    last_message = result.messages[-1]
                    agent_content = getattr(last_message, 'content', str(last_message))
                else:
                    agent_content = str(result)
                
                # Show agent response to human
                print(f"\n{emoji} {agent_name} Response:")
                print("-" * 40)
                print(agent_content)
                print("-" * 40)
                
                # Show cost information
                if call_cost["total_tokens"] > 0:
                    print(f"💰 Cost: {CostCalculator.format_cost(call_cost['total_cost'])} "
                          f"({call_cost['total_tokens']} tokens)")
                    print(f"   Input: {call_cost['prompt_tokens']} tokens ({CostCalculator.format_cost(call_cost['input_cost'])})")
                    print(f"   Output: {call_cost['completion_tokens']} tokens ({CostCalculator.format_cost(call_cost['output_cost'])})")
                
                # Get human approval
                while True:
                    choice = input(f"\n🤔 Review {agent_name}'s response:\n"
                                 f"[A]pprove and continue\n"
                                 f"[R]equest changes (provide feedback)\n"
                                 f"[S]kip this agent\n"
                                 f"[Q]uit session\n"
                                 f"Your choice: ").strip().lower()
                    
                    if choice == 'a':
                        # Approve and continue
                        msg = {
                            'sender': agent_name,
                            'content': agent_content,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'AgentMessage',
                            'approved': True,
                            'human_feedback': 'Approved',
                            'cost': call_cost,
                            'cost': call_cost
                        }
                        conversation_history.append(msg)
                        current_context.append(msg)
                        print(f"✅ {agent_name} response approved!")
                        break
                        
                    elif choice == 'r':
                        # Request changes
                        feedback = input("📝 What changes would you like? (Enter your feedback): ").strip()
                        if feedback:
                            # Record original response
                            msg = {
                                'sender': agent_name,
                                'content': agent_content,
                                'timestamp': datetime.now().isoformat(),
                                'type': 'AgentMessage',
                                'approved': False,
                                'human_feedback': feedback,
                                'cost': call_cost,
                                'cost': call_cost
                            }
                            conversation_history.append(msg)
                            
                            # Get revised response
                            revision_task = f"""
Original task: {agent_task}

Your previous response:
{agent_content}

Human feedback: {feedback}

Please provide a revised response addressing the feedback.
"""
                            
                            print(f"\n🔄 Getting revised response from {agent_name}...")
                            revised_result = await agent.run(task=revision_task, cancellation_token=cancellation_token)
                            
                            # Extract cost for revision
                            revision_cost = self._extract_and_accumulate_costs(revised_result)
                            
                            # Extract revised content from TaskResult
                            revised_content = ""
                            if hasattr(revised_result, 'messages') and revised_result.messages:
                                revised_content = getattr(revised_result.messages[-1], 'content', str(revised_result.messages[-1]))
                            else:
                                revised_content = str(revised_result)
                            
                            print(f"\n{emoji} {agent_name} Revised Response:")
                            print("-" * 40)
                            print(revised_content)
                            print("-" * 40)
                            
                            # Show revision cost
                            if revision_cost["total_tokens"] > 0:
                                print(f"💰 Revision Cost: {CostCalculator.format_cost(revision_cost['total_cost'])} "
                                      f"({revision_cost['total_tokens']} tokens)")
                            
                            # Auto-approve revised response (or ask again)
                            revised_msg = {
                                'sender': agent_name,
                                'content': revised_content,
                                'timestamp': datetime.now().isoformat(),
                                'type': 'RevisedAgentMessage',
                                'approved': True,
                                'human_feedback': f'Revised based on: {feedback}',
                                'cost': revision_cost
                            }
                            conversation_history.append(revised_msg)
                            current_context.append(revised_msg)
                            print(f"✅ Revised response from {agent_name} accepted!")
                            break
                        
                    elif choice == 's':
                        # Skip this agent
                        msg = {
                            'sender': agent_name,
                            'content': agent_content,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'SkippedAgentMessage',
                            'approved': False,
                            'human_feedback': 'Skipped by user',
                            'cost': call_cost
                        }
                        conversation_history.append(msg)
                        print(f"⏭️ Skipped {agent_name}")
                        break
                        
                    elif choice == 'q':
                        # Quit session
                        print("🛑 Session terminated by user")
                        final_answer = self._extract_final_answer(conversation_history)
                        return {
                            'success': True,
                            'question': question,
                            'context': context,
                            'conversation_history': conversation_history,
                            'final_answer': final_answer,
                            'saved_file': self._save_conversation_to_markdown({
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
                                    'status': 'terminated_by_user'
                                }
                            }) if save_to_file else None,
                            'metadata': {
                                'team_type': 'interactive',
                                'num_agents': len(self.agents),
                                'num_messages': len(conversation_history),
                                'model': self.config.model,
                                'status': 'terminated_by_user'
                            }
                        }
                    else:
                        print("❌ Invalid choice. Please enter A, R, S, or Q")
            
            # Final review
            print(f"\n🎯 FINAL REVIEW")
            print("=" * 50)
            
            final_answer = self._extract_final_answer(conversation_history)
            print("Final Answer:")
            print(final_answer)
            print("=" * 50)
            
            # Show total session cost
            if self.total_cost["total_tokens"] > 0:
                print(f"\n💰 TOTAL SESSION COST")
                print("-" * 30)
                print(f"Total Cost: {CostCalculator.format_cost(self.total_cost['total_cost'])}")
                print(f"Input Tokens: {self.total_cost['prompt_tokens']} ({CostCalculator.format_cost(self.total_cost['input_cost'])})")
                print(f"Output Tokens: {self.total_cost['completion_tokens']} ({CostCalculator.format_cost(self.total_cost['output_cost'])})")
                print(f"Total Tokens: {self.total_cost['total_tokens']}")
                print("-" * 30)
            
            # Show total session cost
            if self.total_cost["total_tokens"] > 0:
                print(f"\n💰 TOTAL SESSION COST")
                print("-" * 30)
                print(f"Total Cost: {CostCalculator.format_cost(self.total_cost['total_cost'])}")
                print(f"Input Tokens: {self.total_cost['prompt_tokens']} ({CostCalculator.format_cost(self.total_cost['input_cost'])})")
                print(f"Output Tokens: {self.total_cost['completion_tokens']} ({CostCalculator.format_cost(self.total_cost['output_cost'])})")
                print(f"Total Tokens: {self.total_cost['total_tokens']}")
                print("-" * 30)
            
            final_approval = input("Final approval:\n"
                                 "[A]pprove final answer\n"
                                 "[R]equest final changes\n"
                                 "Your choice: ").strip().lower()
            
            if final_approval == 'r':
                final_feedback = input("What final changes would you like?: ").strip()
                if final_feedback:
                    # Get final revision from synthesizer
                    synthesizer = self.agents.get('synthesizer')
                    if synthesizer:
                        final_task = f"""
Please provide a final, polished answer to this question:
{question}

Based on all the previous conversation and this final feedback:
{final_feedback}

Conversation history:
{chr(10).join([f"{msg['sender']}: {msg['content']}" for msg in conversation_history if msg.get('approved')])}
"""
                        final_result = await synthesizer.run(task=final_task, cancellation_token=CancellationToken())
                        
                        # Extract final cost
                        final_cost = self._extract_and_accumulate_costs(final_result)
                        
                        if hasattr(final_result, 'messages') and final_result.messages:
                            final_answer = getattr(final_result.messages[-1], 'content', str(final_result.messages[-1]))
                        
                        conversation_history.append({
                            'sender': 'synthesizer_final',
                            'content': final_answer,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'FinalAnswer',
                            'approved': True,
                            'human_feedback': f'Final revision: {final_feedback}',
                            'cost': final_cost
                        })
            
            print(f"\n✅ Interactive QA session completed!")
            
            return {
                'success': True,
                'question': question,
                'context': context,
                'conversation_history': conversation_history,
                'final_answer': final_answer,
                'saved_file': self._save_conversation_to_markdown({
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
                        'status': 'completed',
                        'total_cost': self.total_cost
                    }
                }) if save_to_file else None,
                'metadata': {
                    'team_type': 'interactive',
                    'num_agents': len(self.agents),
                    'num_messages': len(conversation_history),
                    'model': self.config.model,
                    'status': 'completed',
                    'total_cost': self.total_cost
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in interactive session: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'context': context
            }
    async def ask_question(self, 
                          question: str,
                          context: Optional[str] = None,
                          use_selector: bool = True,
                          custom_tools: Optional[List[Callable]] = None,
                          save_to_file: bool = True) -> Dict[str, Any]:
        """
        Ask a question using the multi-agent system (automatic mode).
        
        Args:
            question: The question to ask
            context: Optional context or background information
            use_selector: Whether to use selector-based team (True) or round-robin (False)
            custom_tools: Optional list of custom tools for agents
            save_to_file: Whether to save conversation to markdown file (default: True)
            
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
            
            # Prepare the initial message
            if context:
                initial_message = f"""
                Context: {context}
                
                Question: {question}
                
                Please provide a comprehensive, well-researched answer. 
                Work collaboratively to ensure accuracy, completeness, and quality.
                """
            else:
                initial_message = f"""
                Question: {question}
                
                Please provide a comprehensive, well-researched answer.
                Work collaboratively to ensure accuracy, completeness, and quality.
                """
            
            # Create cancellation token
            cancellation_token = CancellationToken()
            
            # Run the team conversation
            self.logger.info("Starting multi-agent conversation")
            
            # Run the conversation directly
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
                        'total_cost': self.total_cost
                    }
                }) if save_to_file else None,
                'metadata': {
                    'team_type': 'selector' if use_selector else 'round_robin',
                    'num_agents': len(self.agents),
                    'num_messages': len(conversation_history),
                    'model': self.config.model,
                    'total_cost': self.total_cost
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
        
    async def close(self):
        """Close the model client and cleanup resources."""
        if self.model_client:
            await self.model_client.close()
            self.logger.info("Model client closed")


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
    """Example usage of the Advanced QA System."""
    
    # Configuration - Now loads from .env automatically
    config = QASystemConfig()
    
    # Create QA system
    qa_system = AdvancedQASystem(config)
    
    try:
        # Example tools
        tools = [web_search_tool, calculator_tool]
        
        # Choose mode
        print("🤖 AutoGen QA System")
        print("=" * 30)
        mode = input("Choose mode:\n"
                    "[1] Automatic (agents work independently)\n"
                    "[2] Interactive (you review each agent response)\n"
                    "Your choice (1 or 2): ").strip()
        
        if mode == "2":
            # Interactive mode
            result = await qa_system.ask_question_interactive(
                question=input("Enter your question / topic: "),
                context="Focus on practical and understable insights.",
                custom_tools=tools
            )
        else:
            # Automatic mode
            result = await qa_system.ask_question(
                question="What are the main benefits of using Python for data science?",
                context="Focus on practical advantages for beginners.",
                use_selector=True,
                custom_tools=tools
            )
        
        # Display results
        if result['success']:
            print("=== QUESTION ===")
            print(result['question'])
            print("\n=== FINAL ANSWER ===")
            print(result['final_answer'])
            print(f"\n=== METADATA ===")
            print(f"Team Type: {result['metadata']['team_type']}")
            print(f"Messages: {result['metadata']['num_messages']}")
            print(f"Agents: {result['metadata']['num_agents']}")
            
            # Show saved file info
            if result.get('saved_file'):
                print(f"\n=== CONVERSATION SAVED ===")
                print(f"File: {result['saved_file']}")
                print("📄 Check the markdown file for detailed conversation flow!")
        else:
            print(f"Error: {result['error']}")
            
    finally:
        await qa_system.close()


if __name__ == "__main__":
    asyncio.run(main())