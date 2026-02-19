"""LangGraph base class for conversational AI."""

import logging
import inspect
from abc import ABC, abstractmethod
from langchain_core.tools import StructuredTool
from langgraph_base_ros.chat_template_render import Messages
from langgraph_base_ros.ollama_utils import Ollama


class LangGraphBase(ABC):
    """
    Base class for LangGraph-based conversational AI.

    This class serves as a base for heritage, encapsulating common functionality and attribures.
    Child classes should implement specific workflow logic.
    Attributes:
        graph (StateGraph): The compiled LangGraph workflow.
        logger: Optional ROS2 logger for logging messages.
        ollama_agent (Ollama): Instance of the Ollama agent for LLM interactions.
        steps (int): Counter for the number of steps taken in the conversation.
        messages_count (int): Counter for the number of messages exchanged.
        max_steps (int): Maximum allowed steps before finishing interaction.
    Methods:
        make_graph() -> None:
            Initializes and compiles the LangGraph workflow.
    """

    def __init__(
        self,
        logger=None,
        ollama_agent: Ollama | None = None,
        max_steps: int = 5
    ) -> None:
        """
        Initialize the LangGraph Manager.

        Creates the LLM instance with default configuration.

        Parameters:
            logger: Optional ROS2 logger to use for logging (default: None).

        Returns:
            None
        """
        self.graph = None
        self.logger = logger
        self.ollama_agent = ollama_agent
        self.steps = 0
        self.messages_count = 0
        self.max_steps = max_steps
        if self.ollama_agent is None:
            raise ValueError('Ollama agent instance must be provided to LangGraphManager.')

    def _log_info(self, msg: str) -> None:
        """Log info message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.info(msg)
        else:
            logging.info(msg)

    def _log_debug(self, msg: str) -> None:
        """Log debug message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            logging.debug(msg)

    def _log_warning(self, msg: str) -> None:
        """Log warning message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            logging.warning(msg)

    def _log_error(self, msg: str) -> None:
        """Log error message using ROS2 logger or Python logging.

        Parameters
        ----------
        msg : str
            Message to log.
        """
        if self.logger is not None:
            self.logger.error(msg)
        else:
            logging.error(msg)

    def _get_system_prompt(self, system_prompt_path: str | None = None) -> str:
        """
        Retrieve the system prompt for the agent.

        Returns:
            None: Sets the system prompt attribute.
        """
        try:
            with open(system_prompt_path, 'r') as f:  # type: ignore[arg-type]
                self.sys_prompt = f.read()
        except FileNotFoundError:
            self._log_error(
                'Supervisor system prompt template not found at path:'
                f'{system_prompt_path}'
            )
            self.sys_prompt = 'You are a helpful assistant designed to perform specific tasks.'
        return self.sys_prompt

    def _generate_tools_list(self):
        """
        Generate a list of available tools for the agent.

        Automatically discovers all methods decorated with @tool in the class
        and generates the tools list in the required format.

        Returns:
            None: Updates self.lang_tools with the discovered tools.
        """
        self.lang_tools = []

        # Iterate through all members of the class
        for name, method in inspect.getmembers(self):
            if isinstance(method, StructuredTool):
                self.lang_tools.append({
                    'name': method.name,
                    'description': method.description,
                    'inputSchema': method.args_schema,
                    'tool_object': method
                })

    def _track_step(self, state: Messages) -> tuple[bool, bool]:
        """
        Track step count and detect tool calls in conversation state.

        Common bookkeeping shared by agent and supervisor routing methods.
        Increments the step counter, checks whether there is a tool call
        in the last message, and detects max_steps violations. Also updates
        ``messages_count`` for logging.

        Parameters
        ----------
        state : Messages
            Current conversation state with message history.

        Returns
        -------
        tuple[bool, bool]
            A tuple of (has_tool_call, max_steps_reached).
            - has_tool_call: True if last message role is 'tool'.
            - max_steps_reached: True if steps exceed max_steps.
        """
        self.steps += 1
        self.messages_count = len(state.get('messages', []))
        has_tool_call = bool(
            state.get('messages')
            and state['messages'][-1].get('role') == 'tool'
        )
        max_steps_reached = self.steps > self.max_steps
        return has_tool_call, max_steps_reached

    @abstractmethod
    async def make_graph(self):
        """
        Initialize and compile the LangGraph workflow.

        To be implemented in child classes

        Returns:
            None: The compiled graph is stored in self.graph.
        """

        # Compile the graph workflow
        # self.graph = workflow.compile()
