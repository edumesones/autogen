"""
Fix específico para el error 'list' object has no attribute 'strip' 
en MultimodalWebSurfer cuando toma screenshots
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# IMPORTS CORRECTOS para AutoGen v4.0
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

# MCP imports
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# WebSurfer import
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# Magentic-One prebuilt system
from autogen_ext.teams.magentic_one import MagenticOne
from pathlib import Path

# ============================
# CLASE CUSTOM PARA FIX DEL WEBSURFER
# ============================

class FixedMultimodalWebSurfer(MultimodalWebSurfer):
    """
    WebSurfer con fix para el manejo de contenido multimodal.
    Corrige el error 'list' object has no attribute 'strip'.
    """
    
    def __init__(self, *args, **kwargs):
        # Forzar to_save_screenshots=True para debugging
        kwargs['to_save_screenshots'] = True
        super().__init__(*args, **kwargs)
        
        # Logger para debugging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _safe_process_multimodal_content(self, content: Any) -> str:
        """
        Procesa contenido multimodal de forma segura.
        Fix principal para el error de .strip() en listas.
        """
        try:
            if content is None:
                return ""
            
            # Si es una lista (contenido multimodal)
            if isinstance(content, list):
                # Buscar el primer elemento de texto
                text_content = ""
                for item in content:
                    if isinstance(item, str):
                        text_content = item
                        break
                    elif hasattr(item, 'text'):  # Objeto con atributo text
                        text_content = str(item.text)
                        break
                    elif hasattr(item, 'content'):  # Objeto con atributo content
                        text_content = str(item.content)
                        break
                
                # Si no encontramos texto, convertir toda la lista a string
                if not text_content:
                    text_content = str(content)
                
                return text_content.strip() if hasattr(text_content, 'strip') else str(text_content)
            
            # Si es un string normal
            elif isinstance(content, str):
                return content.strip()
            
            # Si es un objeto con atributos
            elif hasattr(content, 'text'):
                return str(content.text).strip()
            elif hasattr(content, 'content'):
                return str(content.content).strip()
            
            # Fallback: convertir a string
            else:
                return str(content).strip() if hasattr(str(content), 'strip') else str(content)
                
        except Exception as e:
            self.logger.error(f"Error processing multimodal content: {e}")
            return f"[Error processing content: {str(e)}]"

    async def on_messages(self, messages, *args, **kwargs):
        """
        Override del método principal con fix para contenido multimodal.
        """
        try:
            # Procesar mensajes de entrada de forma segura
            safe_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    # Aplicar fix al contenido
                    safe_content = self._safe_process_multimodal_content(msg.content)
                    
                    # Crear una copia del mensaje con contenido seguro
                    if isinstance(msg, MultiModalMessage):
                        # Para MultiModalMessage, mantener la estructura pero con contenido seguro
                        safe_msg = TextMessage(
                            content=safe_content,
                            source=getattr(msg, 'source', 'user')
                        )
                    else:
                        safe_msg = msg
                    
                    safe_messages.append(safe_msg)
                else:
                    safe_messages.append(msg)
            
            # Llamar al método padre con mensajes seguros
            result = await super().on_messages(safe_messages, *args, **kwargs)
            
            # Verificar si el resultado tiene contenido multimodal y aplicar fix
            if hasattr(result, 'content'):
                result.content = self._safe_process_multimodal_content(result.content)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in on_messages: {e}")
            # Devolver un mensaje de error seguro
            return TextMessage(
                content=f"Error processing web request: {str(e)}",
                source=self.name
            )

# ============================
# CONFIGURACIÓN CON FIX
# ============================

@dataclass
class MagenticQAConfig:
    """Configuration con fix para WebSurfer."""
    openai_api_key: str = None
    github_token: str = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_rounds: int = 3
    enable_github_mcp: bool = False
    enable_web_surfing: bool = True
    enable_coding: bool = True
    enable_file_operations: bool = True
    log_level: str = "INFO"
    screenshots_dir: str = "screenshots"
    save_screenshots: bool = True
    use_fixed_websurfer: bool = True  # Nuevo flag

    def __post_init__(self):
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.github_token is None:
            self.github_token = os.getenv("GITHUB_TOKEN")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required")
        if self.enable_github_mcp and not self.github_token:
            raise ValueError("GITHUB_TOKEN required for GitHub MCP")
        
        self.screenshots_dir = os.getenv("SCREENSHOTS_DIR", self.screenshots_dir)
        
        # Crear directorio si no existe
        if self.save_screenshots:
            os.makedirs(self.screenshots_dir, exist_ok=True)

# ============================
# SISTEMA PRINCIPAL CON FIX
# ============================

class MagenticQASystemFixed:
    """Sistema QA con fix específico para MultimodalWebSurfer."""
    
    def __init__(self, config: MagenticQAConfig):
        self.config = config
        self.model_client: Optional[OpenAIChatCompletionClient] = None
        self.github_workbench: Optional[McpWorkbench] = None
        self.magentic_team: Optional[MagenticOneGroupChat] = None
        self.custom_agents_count = 0
        self._stored_agents = []  # ✅ NUEVO: Almacenar referencia a los agentes
        
        # Sistema de logging
        self.session_log = []
        self.start_time = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def _setup_model_client(self):
        """Setup OpenAI model client."""
        if not self.model_client:
            self.model_client = OpenAIChatCompletionClient(
                model=self.config.model,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature
            )
            self.logger.info(f"Model client initialized: {self.config.model}")

    def _create_fixed_websurfer_agents(self, screenshots_dir: str) -> List:
        """Crear agentes con WebSurfer corregido."""
        
        # Usar el WebSurfer con fix
        if self.config.use_fixed_websurfer:
            web_surfer = FixedMultimodalWebSurfer(
                name="WebSurfer",
                model_client=self.model_client,
                downloads_folder=screenshots_dir,
                to_save_screenshots=True,  # Asegurar que guarde screenshots
                headless=True,  # Ejecutar en modo headless
                debug_dir=screenshots_dir  # Directorio para debugging
            )
            self.logger.info("✅ Using FIXED MultimodalWebSurfer")
        else:
            # WebSurfer original (mantenido para comparación)
            web_surfer = MultimodalWebSurfer(
                name="WebSurfer",
                model_client=self.model_client,
                downloads_folder=screenshots_dir,
                to_save_screenshots=True
            )
            self.logger.info("⚠️ Using ORIGINAL MultimodalWebSurfer")
        
        # Agente asistente especializado en screenshots
        screenshot_assistant = AssistantAgent(
            name="screenshot_assistant",
            model_client=self.model_client,
            system_message=f"""You are a screenshot assistant working with Magentic-One and WebSurfer.

Your responsibilities:
1. Guide WebSurfer to navigate to the correct websites
2. Request screenshots at appropriate moments
3. Analyze and describe screenshot content
4. Handle multimodal content (text + images) properly
5. Report screenshot filenames and locations

Screenshots are saved to: {screenshots_dir}

IMPORTANT: When processing responses from WebSurfer:
- Handle both text and image content
- Extract meaningful information from screenshots
- Always mention the screenshot filename when one is taken
- Process multimodal content as strings, never as lists
- Be patient with loading times and browser actions

Example workflow:
1. Ask WebSurfer to navigate to a website
2. Request a screenshot of the page
3. Analyze what's visible in the screenshot
4. Provide specific details about the page content
"""
        )
        
        return [screenshot_assistant, web_surfer]

    async def _setup_magentic_one_safe(self):
        """Setup Magentic-One con WebSurfer corregido."""
        await self._setup_model_client()
        
        # Crear directorio para screenshots
        screenshots_dir = self.config.screenshots_dir
        if not screenshots_dir:
            screenshots_dir = "screenshots"
        
        full_screenshots_path = os.path.join(os.getcwd(), screenshots_dir)
        os.makedirs(full_screenshots_path, exist_ok=True)
        
        # Crear agentes con WebSurfer corregido
        agents = self._create_fixed_websurfer_agents(full_screenshots_path)
        
        # Guardar cuenta de agentes y almacenar referencia para acceso posterior
        self.custom_agents_count = len(agents)
        self._stored_agents = agents  # ✅ NUEVO: Almacenar referencia a los agentes
        
        # Crear MagenticOneGroupChat
        self.magentic_team = MagenticOneGroupChat(
            participants=agents,
            model_client=self.model_client,
            termination_condition=MaxMessageTermination(self.config.max_rounds)
        )
        
        agent_count = self.custom_agents_count + 4  # +4 agentes built-in de Magentic-One
        self.logger.info(f"✅ Magentic-One team initialized with {agent_count} total agents")
        self.logger.info(f"📁 Screenshots will be saved to: {full_screenshots_path}")
        self.logger.info(f"🔧 Using fixed WebSurfer: {self.config.use_fixed_websurfer}")

    def _enhance_question_for_screenshots(self, question: str, context: Optional[str] = None) -> str:
        """
        Preparar la pregunta con instrucciones específicas para screenshots.
        """
        enhanced = f"Question: {question}\n\n"
        
        if context:
            enhanced += f"Context: {context}\n\n"
        
        # Detectar si se requieren screenshots
        needs_screenshot = any(keyword in question.lower() for keyword in [
            'screenshot', 'capture', 'image', 'visual', 'see', 'show', 'picture'
        ])
        
        if needs_screenshot or 'toma' in question.lower():
            enhanced += """SCREENSHOT INSTRUCTIONS:
1. Use WebSurfer to navigate to the target website
2. Wait for the page to fully load
3. Take a screenshot using WebSurfer's screenshot functionality
4. Save the screenshot to the local directory
5. Provide the screenshot filename in your response
6. Describe what you see in the screenshot with specific details

"""
        
        enhanced += """Instructions for Magentic-One Team:
1. WebSurfer: Handle all web navigation and screenshot capture
2. Screenshot Assistant: Guide the process and analyze results
3. Built-in agents: Coordinate and orchestrate the workflow
4. Always handle multimodal content (text + images) properly
5. Process all content as strings, never as lists
6. Be patient with browser loading times

Provide comprehensive answers with visual evidence when screenshots are taken.
"""
        
        return enhanced

    async def ask_question_with_screenshot_support(self, question: str, context: Optional[str] = None):
        """
        Ask question con soporte específico para screenshots.
        """
        self.start_time = datetime.now()
        
        try:
            print(f"👤 USER: {question}")
            
            if not self.magentic_team:
                await self._setup_magentic_one_safe()
                print(f"⚙️ SYSTEM: Magentic-One team initialized with fixed WebSurfer")
            
            enhanced_question = self._enhance_question_for_screenshots(question, context)
            
            print("⚙️ SYSTEM: Starting screenshot-enabled workflow...")
            
            cancellation_token = CancellationToken()
            
            # Ejecutar con manejo específico para screenshots
            result = await self.magentic_team.run(
                task=enhanced_question,
                cancellation_token=cancellation_token
            )
            
            # Procesar resultados con manejo multimodal
            conversation_history = []
            final_answer = "No response generated"
            screenshot_files = []
            
            if hasattr(result, 'messages') and result.messages:
                for msg in result.messages:
                    sender = getattr(msg, 'source', 'Unknown')
                    content = getattr(msg, 'content', '')
                    
                    # Manejar contenido multimodal de forma segura
                    if isinstance(msg, MultiModalMessage):
                        # Buscar el websurfer en los agentes almacenados
                        websurfer = None
                        for agent in self._stored_agents:  # Usar agentes almacenados
                            if isinstance(agent, (MultimodalWebSurfer, FixedMultimodalWebSurfer)):
                                websurfer = agent
                                break
                        
                        if websurfer and hasattr(websurfer, '_safe_process_multimodal_content'):
                            content = websurfer._safe_process_multimodal_content(content)
                        else:
                            # Fallback si no tenemos el método
                            if isinstance(content, list):
                                content = ' '.join(str(item) for item in content if isinstance(item, str))
                            content = str(content)
                    
                    # Buscar menciones de archivos de screenshot
                    if 'screenshot' in content.lower() or '.png' in content.lower():
                        # Extraer nombres de archivos de screenshot
                        import re
                        screenshot_matches = re.findall(r'[\w\-_\.]+\.png', content)
                        screenshot_files.extend(screenshot_matches)
                    
                    if content.strip():
                        print(f"🤖 {sender.upper()}: {content[:100]}...")
                        conversation_history.append({
                            'sender': sender,
                            'content': content,
                            'timestamp': datetime.now().isoformat(),
                            'type': type(msg).__name__
                        })
                        final_answer = content
            
            print("✅ SYSTEM: Screenshot workflow completed")
            
            # Verificar si se generaron screenshots
            screenshots_dir = os.path.join(os.getcwd(), self.config.screenshots_dir)
            if os.path.exists(screenshots_dir):
                screenshot_files_found = [f for f in os.listdir(screenshots_dir) if f.endswith('.png')]
                if screenshot_files_found:
                    print(f"📸 Screenshots generated: {len(screenshot_files_found)} files")
                    for file in screenshot_files_found[-3:]:  # Mostrar últimos 3
                        print(f"   📁 {file}")
            
            return {
                'success': True,
                'question': question,
                'context': context,
                'conversation_history': conversation_history,
                'final_answer': final_answer,
                'screenshot_files': screenshot_files,
                'screenshots_directory': screenshots_dir,
                'metadata': {
                    'system_type': 'magentic_one_fixed_websurfer',
                    'num_agents': self.custom_agents_count + 4,
                    'num_messages': len(conversation_history),
                    'fixed_websurfer_used': self.config.use_fixed_websurfer,
                    'screenshots_taken': len(screenshot_files),
                    'duration_seconds': (datetime.now() - self.start_time).total_seconds()
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ ERROR in system: {error_msg}")
            
            self.logger.error(f"Error in screenshot workflow: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'question': question,
                'context': context,
            }

    async def close(self):
        """Close all resources safely."""
        if self.model_client:
            await self.model_client.close()
            self.logger.info("Model client closed")

# ============================
# FUNCIÓN PRINCIPAL
# ============================

async def main():
    """Test del sistema con fix para screenshots."""
    
    # Cargar variables de entorno
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / 'report-auditor/.env'
    if env_path.exists():
        print(f"🔍 Loading .env from: {env_path}")
        load_dotenv(env_path)
    else:
        print("⚠️ No .env file found, using system environment variables")
        load_dotenv()
    
    # Configuration con fix habilitado
    config = MagenticQAConfig(
        use_fixed_websurfer=True,  # Usar el WebSurfer corregido
        save_screenshots=True,
        screenshots_dir="screenshots"
    )
    
    # Create system con fix
    qa_system = MagenticQASystemFixed(config)
    
    try:
        print("🤖 Magentic-One QA System with FIXED WebSurfer")
        print("=" * 60)
        print(f"Fixed WebSurfer: {'✅' if config.use_fixed_websurfer else '❌'}")
        print(f"Screenshots: {'✅' if config.save_screenshots else '❌'}")
        print(f"Screenshots Dir: {config.screenshots_dir}")
        print("=" * 60)
        
        # Test con la pregunta original que causaba el error
        test_question = "Abre el repo microsoft/autogen, ve a la documentación y toma un screenshot de la página principal"
        
        print(f"\n🧪 Testing with: {test_question}")
        
        # Ejecutar con soporte para screenshots
        result = await qa_system.ask_question_with_screenshot_support(
            question=test_question,
            context="un screenshot de la página principal"
        )
        
        # Mostrar resultados
        if result['success']:
            print("\n" + "="*80)
            print("=== FINAL ANSWER ===")
            print(result['final_answer'])
            
            if result.get('screenshot_files'):
                print(f"\n=== SCREENSHOTS GENERATED ===")
                for file in result['screenshot_files']:
                    print(f"📸 {file}")
            
            print(f"\n=== METADATA ===")
            metadata = result['metadata']
            print(f"System: {metadata['system_type']}")
            print(f"Fixed WebSurfer: {metadata['fixed_websurfer_used']}")
            print(f"Screenshots Taken: {metadata['screenshots_taken']}")
            print(f"Duration: {metadata['duration_seconds']:.1f}s")
            
        else:
            print(f"❌ Error: {result['error']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await qa_system.close()

if __name__ == "__main__":
    asyncio.run(main())