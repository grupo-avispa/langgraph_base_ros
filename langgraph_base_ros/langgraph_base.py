"""
LangGraph base class for conversational AI.
"""

import logging
from abc import ABC, abstractmethod
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

    def __init__(self,
                 logger=None,
                 ollama_agent: Ollama = None,
                 max_steps: int = 5):
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
            raise ValueError("Ollama agent instance must be provided to LangGraphManager.")

    def _log(self, msg: str) -> None:
        """
        Log helper. Uses ROS2 logger if available, else Python logging.

        Parameters:
            msg (str): Message to log.

        Returns:
            None
        """
        if self.logger is not None:
            self.logger.info(msg)
        else:
            logging.info(msg)

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
