import os
import re
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
import logging

logging = logging.getLogger("VGA") 

# Add the parent directory to sys.path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from prompts.save_and_build_agent_system_prompt import SAVE_AND_BUILD_AGENT_SYSTEM_PROMPT

# Import the utilities required for the agents
from utils.build import save_build_run
from utils.expected_patterns import LANGUAGE_PATTERNS

from utils.azure_key import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_INFERENCE_URL,
    AZURE_OPENAI_INFERENCE_API_VERSION,
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    AZURE_OPENAI_INFERENCE_MODEL,
    AZURE_OPENAI_EMBEDDING_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_API_VERSION,
    AZURE_OPENAI_INFERENCE_DEPLOYMENT
)

from utils.file_io import (
    FileIO, 
    TestPlanParser
)

# Conditional import for vector index to handle missing dependencies
try:
    from vector_index.generate_vector_db import KnowledgeBase
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vector database not available due to missing dependencies: {e}")
    VECTOR_DB_AVAILABLE = False
    KnowledgeBase = None


# Load environment variables
load_dotenv()

# Configure autogen for Intel's internal API
config_list = [
     {
        "model": AZURE_OPENAI_INFERENCE_MODEL,  
        "api_type": "azure",
        "base_url": AZURE_OPENAI_INFERENCE_URL,
        "api_key": AZURE_OPENAI_API_KEY,
        "api_version": AZURE_OPENAI_INFERENCE_API_VERSION,
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.1,
}

INPUT_DIR = 'input_dirs'
URLS_FILE = f"{INPUT_DIR}/public_urls.txt"


@dataclass
class TestCase:
    title: str
    description: str
    steps: List[str]
    expected_results: str
    implementation_file: Optional[str] = None
    data_types: List[str] = field(default_factory=list)

# Agent 3: Smart Test Execution Agent
class TestRunnerUserProxy(autogen.UserProxyAgent):
    def __init__(self, logger, output_dir="generated_tests"):
        # The executor will only run code received in conversation
        self.executor = LocalCommandLineCodeExecutor(
            timeout=120,
            work_dir=output_dir,
        )

        super().__init__(
            name="TestExecutionProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=50,
            code_execution_config={"executor": self.executor},
        )
        self.output_dir = output_dir

# Function to save code to a file.
# The function is registered with the UserProxyAgent to handle code saving requests
def save_code_to_file(code: str, filename: str, directory: str) -> str:
    """Save code to a file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

    return f"Code saved successfully to {filepath}"

class ContextManagedGroupChat(autogen.GroupChat):
    """GroupChat with automatic context management"""

    def __init__(self, agents, messages, max_round, max_context_messages=25):
        super().__init__(agents=agents, messages=messages, max_round=max_round)
        self.max_context_messages = max_context_messages

    def append(self, message, speaker):
        """Override append to manage context automatically"""
        # Call parent's append method with correct signature
        super().append(message, speaker)

        # Trigger context management when approaching limit
        if len(self.messages) > self.max_context_messages * 0.9:
            logging.info(f"[GroupChat]- Auto-managing context at {len(self.messages)} messages")
            self._auto_manage_context()

    def _auto_manage_context(self):
        """Automatically manage context using the same strategy as the orchestrator"""
        if len(self.messages) <= self.max_context_messages:
            return

        # Keep initial message
        initial_message = self.messages[0] if self.messages else None
        if not initial_message:
            return

        # Find important messages
        important_keywords = [
            'APPROVED FOR EXECUTION', 'CODE IS READY FOR EXECUTION', 'APPROVE THIS CODE FOR TESTING',
            'FAILED:', 'SUCCESS:', 'ERROR:', 'filename:', 'def test_', 'import pytest',
            'subprocess.run', 'if __name__ == "__main__":', 'Exit code:', 'Test execution'
        ]

        important_messages = []
        for msg in self.messages[1:]:
            content = msg.get('content', '')
            if any(keyword in content for keyword in important_keywords) or '```python' in content or '```bash' in content:
                important_messages.append(msg)

        # Keep recent messages
        recent_count = min(8, self.max_context_messages // 4)
        recent_messages = self.messages[-recent_count:]

        # Combine and deduplicate
        managed_messages = [initial_message]
        seen_contents = {initial_message.get('content', '')[:100]}

        for msg in important_messages + recent_messages:
            content_key = msg.get('content', '')[:100]
            if content_key not in seen_contents:
                managed_messages.append(msg)
                seen_contents.add(content_key)

        # Final limit check
        if len(managed_messages) > self.max_context_messages:
            managed_messages = [managed_messages[0]] + managed_messages[-(self.max_context_messages-1):]

        self.messages = managed_messages
        logging.info(f"[GroupChat]- Context auto-managed: reduced to {len(self.messages)} messages")

class CustomUserProxyAgent(autogen.UserProxyAgent):
    def get_human_input(self, prompt: str = "Your input: ") -> str:
        # Customize message shown to user
        custom_message = "[Human Feedback Requested] Please provide your feedback. Flow for agent is [Code-generation -> Review -> Execution/saving]. Human feedback agent can ask how to proceed: > "
        return input(custom_message)


class MultiAgentTestOrchestrator:
    def __init__(self, args, output_dir: str,
                 max_retries: int = 2,
                 max_context_messages: int = 25,
                 code_agent_prompt: str = "",
                 review_agent_prompt: str = "",
                 test_coordinator_prompt: str = ""):
        self.args=args
        self.build=args.build
        self.build_dir=args.build_dir or ""
        self.build_cmd=args.build_cmd or ""
        self.execute_cpp=args.execute_cpp
        self.execute_dir=args.execute_dir or ""
        self.execute_args=args.execute_args
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.max_context_messages = max_context_messages
        self.kb = None # Knowledge base instance if available
        self.execute_python = args.execute_python
        self.code_agent_prompt = code_agent_prompt
        self.review_agent_prompt = review_agent_prompt
        self.test_coordinator_prompt = test_coordinator_prompt

        # check if source code dir exists
        if not os.path.exists(INPUT_DIR):
            raise FileNotFoundError(f"The source code directory '{INPUT_DIR}' does not exist.")

        # Initialize the agents
        # Create the code generation agent
        self.codegen_agent = autogen.AssistantAgent(
            name="TestGenerationAgent",
            llm_config=llm_config,
            system_message=self.code_agent_prompt)

        # Create the review agent
        self.review_agent = autogen.AssistantAgent(
            name="TestCodeReviewAgent",
            llm_config=llm_config,
            system_message=self.review_agent_prompt)
        
        if args.human_feedback:
            self.user_proxy = CustomUserProxyAgent(
                name="HumanFeedbackAgent",
                human_input_mode="ALWAYS",
                code_execution_config=False,
                description="A human user capable of working with Autonomous AI Agents.",
            )

        # Create a runner agent to execute tests
        if self.execute_python:
            # Full execution capability when tests should be executed
            self.runner_agent = autogen.ConversableAgent(
                name="TestExecutionAgent",
                system_message="You are responsible for executing test code only after approval. Also summarize the execution and then call the user_proxy agent.",
                human_input_mode="NEVER",  # fully automated, no manual input
                max_consecutive_auto_reply=50,
                code_execution_config={"executor": LocalCommandLineCodeExecutor(
                                                    timeout=120,
                                                    work_dir=output_dir)
                },
            )
        elif self.build:
            def wrapped_build_code(code: str, filename: str) -> str:
                return save_build_run(
                    code=code,
                    filename=filename,
                    directory=self.output_dir,
                    build=self.build,
                    build_cmd=self.build_cmd,
                    build_dir=self.build_dir,
                    execute=self.execute_cpp,
                    execute_dir=self.execute_dir,
                    execute_args=self.execute_args,
                    llm_config=llm_config,
                    args=self.args
                )

            self.runner_agent = autogen.ConversableAgent(
                name="TestBuildAndExecuteProxy",
                system_message=SAVE_AND_BUILD_AGENT_SYSTEM_PROMPT,
                llm_config={"config_list": config_list},  # keep your previous LLM configs
                human_input_mode="NEVER",
                max_consecutive_auto_reply=50
            )
            # Execution side
            self.runner_agent.register_for_execution(name="save_code")(wrapped_build_code)

            # LLM side
            self.runner_agent.register_for_llm(
                name="save_code",
                description="Save the provided code into the given filename, then build and run if build is true."
            )(wrapped_build_code)
        else:
            self.runner_agent = autogen.ConversableAgent(
                name="TestFileSaveAgent",
                system_message="""You are a file manager. Your job is to save code to files when requested.
                When you receive reviewed code, save it to the specified directory with an appropriate filename.""",
                llm_config={"config_list": config_list},
                human_input_mode="NEVER",
                max_consecutive_auto_reply=2
            )

            self.runner_agent.register_for_execution(name="save_code")(save_code_to_file)
            self.runner_agent.register_for_llm(name="save_code", description=f"Save code to a file to the path mentioned in The Directory={output_dir}. Do not change/modify the path where the output files should be saved. the Directory value when making the tool call should always be the value of {output_dir}")(save_code_to_file)

        # Create a coordinator agent to manage the conversation
        self.coordinator = autogen.UserProxyAgent(
            name="TestCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,  # Allow coordinator to participate in conversation
            code_execution_config=False,
            
            system_message="Coordinate the test generation and execution process between agents."
        )

        # Set up GroupChat - always include runner_agent but with different roles
        self.group_chat = ContextManagedGroupChat(
            agents=[ self.coordinator, self.codegen_agent, *( [self.user_proxy] if args.human_feedback else [] ), self.review_agent, self.runner_agent],
            messages=[],
            max_round=50,  # Allow enough rounds for iterations
            max_context_messages=self.max_context_messages,
        )
        # GroupChat manager with custom speaker selection logic
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            human_input_mode="NEVER",
            llm_config=llm_config,
            system_message=self.test_coordinator_prompt,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Initialize knowledge base only if available
        if VECTOR_DB_AVAILABLE and KnowledgeBase:
            self.kb = KnowledgeBase(
                aoai_api_key=AZURE_OPENAI_API_KEY,
                aoai_inf_endpoint_version=AZURE_OPENAI_INFERENCE_API_VERSION,
                aoai_inf_endpoint=AZURE_OPENAI_INFERENCE_ENDPOINT,
                aoai_embd_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
                aoai_embd_endpoint_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
                model_name=AZURE_OPENAI_INFERENCE_MODEL,
                aoai_api_version=AZURE_OPENAI_INFERENCE_DEPLOYMENT,
            )

        kb_success = self.build_knowledge_base(
            input_dirs=INPUT_DIR,
            urls=URLS_FILE
        )
        if not kb_success:
            logging.warning("[Orchestrator]- Knowledge base initialization failed, proceeding without it")

        self.FileIO=FileIO(self.output_dir)

    def _get_context(self,context_input: str) -> str:
        context = ""
        if self.args.add_context_dir is not None:
            logging.info(f"[Orchestrator]- Using the files at the directory {self.args.add_context_dir}")
            combined_content = ""
            for filename in os.listdir(self.args.add_context_dir):
                file_path = os.path.join(self.args.add_context_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        combined_content += f.read() + "\n"
        
            context=combined_content

        elif self.kb:
            try:
                if self.args.human_feedback:
                    context_input=''
                    context_input=input("\n\nEnter the input to fetch the context input from index_db. Flow for agent is {context_retrieval->[Code-generation -> Review -> Execution/saving]}: ")
                    if context_input=='':
                        logging.info('Skipping retrieving context from index_db')
                        return ''

                context = self.kb.retrive_document_chunks(context_input)
                if re.search(r'error', str(context), re.IGNORECASE):
                    logging.warning(f"[Orchestrator]-  Failed to retrieve doc chunks for , proceeding without context")
                    context = ""
            except Exception as e:
                logging.warning(f"[Orchestrator]- retrieving doc chunks: {e}, proceeding without context")
                context = ""
        else:
            logging.warning(f"[Orchestrator]- Knowledge base not available, proceeding without context")

        return context
    
    def human_in_loop(self) -> bool:
        """Run human-in-the-loop flow with UserProxyAgent and GroupChat."""
        logging.info("[Orchestrator] - Starting GroupChat with human-in-the-loop")

        try:
            initial_proxy_agent = autogen.UserProxyAgent(
                name="User Proxy Agent",
                system_message="You are required to take initial message from user.",
                llm_config=False,
                code_execution_config=False,
                human_input_mode="ALWAYS",
            )
        except Exception as e:
            logging.error(f"[Orchestrator] - Failed to initialize UserProxyAgent: {e}", exc_info=True)
            return False

        while True:
            try:
                reply = initial_proxy_agent.generate_reply(
                    messages=[{"role": "system", "content": "Please provide your Prompt here to start the code generation process (type 'exit' to quit). Flow for agent is {initial_prompt -> context_retrieval -> [Code-generation -> Review -> Execution/saving]}: "}]
                )
            except Exception as e:
                logging.error(f"[Orchestrator] - Error while getting user input: {e}", exc_info=True)
                break

            if not reply or not isinstance(reply, dict) or "content" not in reply or reply==None:
                logging.info("[Orchestrator] - No valid reply received, ending loop")
                break

            user_input = reply["content"].strip()
            if user_input.lower() == "exit":
                logging.info("[Orchestrator] - User requested exit, ending loop")
                break

            logging.debug(f"[Orchestrator] - Received user input: {user_input}")

            context = self._get_context(user_input)
            full_message = f"{context}\n{user_input}" if context else user_input

            try:
                self.coordinator.initiate_chat(
                    self.manager,
                    message=full_message,
                    max_turns=20
                )
                logging.info("[Orchestrator] - GroupChat session completed")
            except Exception as e:
                logging.error(f"[Orchestrator] - Error during GroupChat initiation: {e}", exc_info=True)

        return True

    def orchestrate_test_generation(self, test_plan_path: str):
        """Main orchestration method using GroupChat for natural agent communication"""
        logging.info(f"[Orchestrator]- Starting multi-agent test generation from {test_plan_path}")

        # Parse test plan
        parser = TestPlanParser(test_plan_path)
        success, test_cases, implementation_files = parser.extract_test_cases()

        if not success:
            logging.error("[Orchestrator]- Failed to parse test plan")
            return False

        if not test_cases:
            logging.error("[Orchestrator]- No test cases found in the test plan")
            return False

        logging.info(f"[Orchestrator]- Found {len(test_cases)} test cases across {len(implementation_files)} files")
        logging.info(f"[Orchestrator]- Expected implementation files: {implementation_files}")

        # Process each implementation file using GroupChat
        successful_files = []
        failed_files = []

        for impl_file in implementation_files:
            if self._process_implementation_file_with_groupchat(impl_file, test_cases):
                successful_files.append(impl_file)
            else:
                failed_files.append(impl_file)

        # Check if all expected files were generated successfully
        all_files_generated = self.FileIO._validate_all_files_generated(implementation_files, successful_files, failed_files)

        # If execute_python is False, skip the execution step but still validate file generation
        if not self.execute_python and not self.execute_cpp:
            logging.info("[Orchestrator]- Skipping test execution as per configuration.")
            return all_files_generated

        # Summary
        logging.info(f"[Orchestrator]- Successfully generated tests for {len(successful_files)}/{len(implementation_files)} files")
        return all_files_generated

    def _process_implementation_file_with_groupchat(self, impl_file: str, all_test_cases: List[Dict]) -> bool:
        """Process a single implementation file using GroupChat for dynamic agent interaction"""
        # Filter relevant test cases
        relevant_tests = [tc for tc in all_test_cases if tc.get('implementation_file') == impl_file]
        if not relevant_tests:
            logging.warning(f"[Orchestrator]-  No relevant tests found for {impl_file}")
            return False

        logging.info(f"[Orchestrator]- Processing {impl_file} with {len(relevant_tests)} test cases using GroupChat")

        # Format test cases for the group chat
        test_cases_text = self._format_test_cases_for_chat(relevant_tests)

        # Create initial message to start the group chat
        execution_note = ""
        if self.build and not self.execute_cpp:
            execution_note = "\n\nGenerate, review, save, and build the test code."
        elif not self.execute_cpp and not self.execute_python:
            execution_note = "\n\nNOTE: Test execution is disabled. Only generate, review, and save the test code to files. Do not execute the tests."
        else:
            execution_note = "\n\nGenerate, review, save, and execute the test code."

        initial_message = f"""
            We need to generate and review tests for: {impl_file}
            {execution_note}

            Test Cases to implement:
            {test_cases_text}
            """
        try:
            # Start the group chat
            logging.info(f"[Orchestrator]- Starting GroupChat for {impl_file}")

            # Initialize a fresh group chat for this file
            self.group_chat.messages = []  # Clear previous messages

            # Get context from knowledge base if available
            test_categories_string = ' '.join({tc['test_category'] for tc in relevant_tests})
            context_input=f"Test is {test_categories_string}"
            context = self._get_context(context_input)

            if context:
                prompt_with_context = f"Based on the following code context:\n\n{context}\n\n {initial_message}"
            else:
                prompt_with_context = initial_message

            # Start the conversation
            self.coordinator.initiate_chat(
                self.manager,
                message=prompt_with_context,
                max_turns=20  # Limit turns to prevent infinite loops
            )

            # Manage context after conversation to keep it within limits
            self._manage_context_length()

            _, ext = os.path.splitext(impl_file)
            language=ext.lstrip(".")

            # Check if we have successful test generation
            success = self._extract_success_from_chat(language)

            # Additionally verify that the specific file was actually created
            file_actually_created = self.FileIO._verify_file_created(impl_file)

            # Debug logging
            logging.info(f"[Orchestrator]- SUCCESS DETECTION: Found {len(self.group_chat.messages)} messages in chat")
            logging.info(f"[Orchestrator]- Chat success detected: {success}")
            logging.info(f"[Orchestrator]- File actually created: {file_actually_created}")

            for i, msg in enumerate(self.group_chat.messages[-3:]):  # Log last 3 messages for debugging
                content_preview = str(msg.get('content', ''))[:200]  # First 200 chars
                logging.info(f"[Orchestrator]- Message {i}: {content_preview}...")

            # Final success is both chat success AND file creation
            final_success = success and file_actually_created

            if final_success:
                logging.info(f"[Orchestrator]- SUCCESS: GroupChat completed and file created for {impl_file}")
                # Save artifacts from the group chat
                artifact_saved = self._save_artifacts_from_chat(impl_file)
                if not artifact_saved:
                    logging.warning(f"[Orchestrator]- Failed to save artifacts for {impl_file}")

                return True
            else:
                if success and not file_actually_created:
                    logging.warning(f"[Orchestrator]- PARTIAL FAILURE: GroupChat succeeded but file {impl_file} was not created")
                elif not success and file_actually_created:
                    logging.warning(f"[Orchestrator]- PARTIAL FAILURE: File {impl_file} was created but chat did not complete successfully")
                else:
                    logging.error(f"[Orchestrator]- COMPLETE FAILURE: Neither chat success nor file creation for {impl_file}")
                return False

        except Exception as e:
            logging.error(f"[Orchestrator]- ERROR: GroupChat failed for {impl_file}: {str(e)}")
            return False


    def _format_test_cases_for_chat(self, test_cases: List[Dict]) -> str:
        """Format test cases for group chat message"""
        formatted = []
        for i, tc in enumerate(test_cases, 1):
            formatted.append(f"Test Case {i}")
            formatted.append(f"Description: {tc.get('description', 'N/A')}")
            if tc.get('steps'):
                formatted.append(f"Steps: {'; '.join(tc['steps'])}")
            if tc.get('data_types'):
                formatted.append(f"Data Types: {', '.join(tc['data_types'])}")
            formatted.append("")  # Empty line for separation
        return '\n'.join(formatted)

    def _extract_success_from_chat(self, language='py') -> bool:
        """Extract success status from the group chat messages and logger for a specific language"""
        if language not in LANGUAGE_PATTERNS:
            logging.error(f"[Orchestrator]- Unsupported language: {language}")
            return False

        patterns = LANGUAGE_PATTERNS[language]
        file_extension = language

        # Check for test generation patterns
        for message in self.group_chat.messages:
            content = str(message.get('content', ''))
            for pattern in patterns['generation']:
                if re.search(pattern, content, re.IGNORECASE):
                    logging.info(f"[Orchestrator]- TEST GENERATION PATTERN MATCHED: '{pattern}' in GroupChat message")
                    return True

        # Check for file saving patterns if execute_python is False
        if not self.execute_python or not self.execute_cpp:
            for message in self.group_chat.messages:
                content = str(message.get('content', ''))
                for pattern in patterns['file_saving']:
                    if re.search(pattern, content, re.IGNORECASE):
                        logging.info(f"[Orchestrator]- FILE SAVING PATTERN MATCHED: '{pattern}' in GroupChat message")
                        return True

        # Check for execution success patterns if execute_python is True
        if self.execute_python or self.execute_cpp:
            for message in self.group_chat.messages:
                content = str(message.get('content', ''))
                for pattern in patterns['execution']:
                    if re.search(pattern, content, re.IGNORECASE):
                        if 'failed' not in content.lower() or 'passed' in content.lower():
                            logging.info(f"[Orchestrator]- EXECUTION SUCCESS PATTERN MATCHED: '{pattern}' in GroupChat message")
                            return True

        # Check if any files exist in output directory
        try:
            if os.path.exists(self.output_dir):
                files = []
                files.extend([f for f in os.listdir(self.output_dir) if f.endswith(f'.{file_extension}')])
                if files:
                    logging.info(f"[Orchestrator]- SUCCESS: Found {len(files)} {language} files in output directory: {files}")
                    return True
        except Exception as e:
            logging.error(f"[Orchestrator]- checking for {language} files: {e}")

        # Fallback: check if any meaningful conversation happened
        meaningful_messages = [msg for msg in self.group_chat.messages
                               if len(str(msg.get('content', ''))) > 100]
        if len(meaningful_messages) >= 3:
            logging.info(f"[Orchestrator]- SUCCESS: Meaningful conversation detected with {len(meaningful_messages)} substantial messages")
            return True

        logging.error("[Orchestrator]- No success patterns found in chat messages")
        return False

    def _save_artifacts_from_chat(self, impl_file: str) -> bool:
        """Save artifacts from the group chat conversation

        Returns:
            bool: True if artifacts were saved successfully, False otherwise
        """
        try:
            base_name = os.path.splitext(impl_file)[0]

            # Save the entire conversation log
            chat_log_file = os.path.join(self.output_dir, f"{base_name}_chat_log.txt")
            with open(chat_log_file, 'w') as f:
                f.write("=== GROUP CHAT CONVERSATION LOG ===\n\n")
                for i, message in enumerate(self.group_chat.messages):
                    f.write(f"Message {i+1} - {message.get('name', 'Unknown')}:\n")
                    f.write(f"{message.get('content', '')}\n")
                    f.write("-" * 50 + "\n")

            logging.info(f"[Orchestrator]- Saved chat log for {impl_file}")
            return True

        except Exception as e:
            logging.error(f"[Orchestrator]- saving artifacts for {impl_file}: {e}")
            return False

    def _manage_context_length(self):
        """Manage GroupChat context to prevent token overflow while preserving important information"""
        if len(self.group_chat.messages) <= self.max_context_messages:
            return  # No management needed

        logging.info(f"[Orchestrator]- Managing context: {len(self.group_chat.messages)} messages -> target: {self.max_context_messages}")

        # Strategy: Keep initial prompt + important messages + recent messages
        if not self.group_chat.messages:
            return

        initial_message = self.group_chat.messages[0]

        # Find important messages (approvals, code blocks, errors, successes)
        important_messages = []
        important_keywords = [
            'APPROVED FOR EXECUTION', 'CODE IS READY FOR EXECUTION', 'APPROVE THIS CODE FOR TESTING',
            'FAILED:', 'SUCCESS:', 'ERROR:', 'filename:', 'def test_', 'import pytest',
            'subprocess.run', 'if __name__ == "__main__":', 'Exit code:', 'Test execution',
            '#include ', 'TEST(', 'TEST_F(', 'CU_ASSERT(', 'CU_ADD_TEST(', 'global _?main',
            'section .text', 'mov ', 'assert.h', 'CUnit/CUnit.h', 'main'
        ]

        for msg in self.group_chat.messages[1:]:
            content = msg.get('content', '')
            # Check if message contains important keywords
            if any(keyword in content for keyword in important_keywords):
                important_messages.append(msg)
            # Also keep messages with code blocks
            elif any(code_block in content for code_block in ['```python', '```cpp', '```c', '```asm', '```bash']):
                important_messages.append(msg)

        # Keep most recent messages (last 8-10 messages are usually most relevant)
        recent_count = min(10, self.max_context_messages // 3)
        recent_messages = self.group_chat.messages[-recent_count:]

        # Combine and deduplicate while preserving order
        managed_messages = [initial_message]
        seen_contents = {initial_message.get('content', '')[:100]}  # Use first 100 chars as key

        # Add important messages first
        for msg in important_messages:
            content_key = msg.get('content', '')[:100]
            if content_key not in seen_contents:
                managed_messages.append(msg)
                seen_contents.add(content_key)

        # Add recent messages
        for msg in recent_messages:
            content_key = msg.get('content', '')[:100]
            if content_key not in seen_contents:
                managed_messages.append(msg)
                seen_contents.add(content_key)

        # If still too many, keep only most recent within limit
        if len(managed_messages) > self.max_context_messages:
            managed_messages = [managed_messages[0]] + managed_messages[-(self.max_context_messages-1):]

        # Update the group chat messages
        self.group_chat.messages = managed_messages
        logging.info(f"[Orchestrator]- Context managed: reduced to {len(self.group_chat.messages)} messages")

    def build_knowledge_base(self, input_dirs, urls):
        """Build the knowledge base

        Args:
            code_dirs (list): List of directories to index
            urls (list): List of URLs to index

        Returns:
            bool: True if knowledge base was built successfully, False otherwise
        """
        if self.kb:
            try:
                self.kb.build_index(input_dirs, urls)
                return True
            except Exception as e:
                logging.warning(f"Failed to build knowledge base: {e}")
                return False
        else:
            logging.warning("Cannot build knowledge base - dependencies not available")
            return False

def run_test_automation(args, test_plan_path: str,
                        output_dir: str = "generated_tests",
                       max_retries: int = 20,
                       max_context: int = 25,
                       code_agent_prompt: str = "",
                       review_agent_prompt: str = "",
                       test_coordinator_prompt: str = "") -> bool:
    """
    Run the multi-agent test automation system.

    Args:
        args: arguments passed by user
        test_plan_path: Path to test plan document (JSON or DOCX format)
        output_dir: Output directory for generated tests
        max_retries: Maximum retries for code correction
        max_context: Maximum context messages in GroupChat

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input file
        if not args.human_feedback and not os.path.exists(test_plan_path):
            logging.error(f"ERROR: Test plan file '{test_plan_path}' not found")
            return False

        if not args.human_feedback and not test_plan_path.endswith(('.json', '.docx')):
            logging.error("ERROR: Test plan must be a JSON or DOCX file")
            return False

        if args.verbose:
            logging.info("Multi-Agent Test Automation System")
            logging.info("=" * 40)
            logging.info(f"Test Plan: {test_plan_path}")
            logging.info(f"Output Directory: {output_dir}")
            logging.info(f"Max Retries: {max_retries}")
            logging.info(f"Max Context Messages: {max_context}")
            logging.info("=" * 40)

        # Initialize and run the multi-agent orchestrator
        orchestrator = MultiAgentTestOrchestrator(
            args=args,
            output_dir=output_dir,
            max_retries=max_retries,
            max_context_messages=max_context,
            code_agent_prompt=code_agent_prompt,
            review_agent_prompt=review_agent_prompt,
            test_coordinator_prompt=test_coordinator_prompt
        )
        success = orchestrator.human_in_loop() if (args.human_feedback and test_plan_path is None) else orchestrator.orchestrate_test_generation(test_plan_path)

        if success:
            if args.verbose:
                logging.info("\n" + "="*50)
                logging.info("COMPLETE SUCCESS: All expected test files generated!")
                logging.info(f"Generated tests are available in: {output_dir}")
                logging.info("="*50)
            else:
                logging.info("SUCCESS: All expected test files generated successfully!")
        else:
            if args.verbose:
                logging.error("="*50)
                logging.error("FAILURE: Not all expected test files were generated!")
                logging.error("Check the logs above for details on which files failed.")
                logging.error("="*50)
            else:
                logging.error("\nFAILURE: Test generation incomplete - not all files generated!")

        return success

    except KeyboardInterrupt:
        logging.error("Process interrupted by user")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
