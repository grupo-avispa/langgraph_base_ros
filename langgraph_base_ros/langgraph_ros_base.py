"""ROS2 server for LangGraph-based conversational AI."""

# asyncio for async operations
import asyncio
# JSON import for reading MCP server configurations
import json
# abc for abstract class methods
from abc import abstractmethod
# Custom Ollama agent and message utilities
from langgraph_base_ros.ollama_utils import Ollama
# MCP client for tool retrieval
from fastmcp import Client

# ROS2 imports
from rclpy.node import Node

# ============= ROS2 NODE =============


class LangGraphRosBase(Node):
    """
    ROS2 server for LangGraph-based conversational AI.

    This class sets up a ROS2 node that listens for user queries, processes them
    using a LangGraph workflow, and publishes the generated responses.
    Attributes:
        query_topic (str): Topic name for receiving user queries.
        response_topic (str): Topic name for publishing LLM responses.
        mcp_servers (str): MCP server configuration for tool retrieval.
        system_prompt (str): System prompt template for LLM interactions.
        loop: Asyncio event loop for asynchronous operations.
        ollama_agent (Ollama): Instance of the Ollama agent for LLM interactions.
        graph_manager (LangGraphManager): Instance of the LangGraph manager for workflow control.
    """

    # ============= INITIALIZATION =============

    def __init__(self) -> None:
        """Initialize the LangGraph Node."""
        super().__init__('langgraph_agent_node')

        # Retrieve ROS2 parameters (topic name, MCP servers, system prompt, etc.)
        self.get_params()

        # Create a persistent event loop for async operations
        # Using asyncio.new_event_loop() to avoid deprecation warning
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize Ollama agent
        try:
            self.loop.run_until_complete(self.initialize_ollama_agent())
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Ollama agent: {e}')
            raise

    @abstractmethod
    def build_graph(self) -> None:
        """Initialize and compile the LangGraph workflow."""
        # Initialize and compile the LangGraph workflow using event loop
        # try:
        #     self.loop.run_until_complete(self.graph_workflow.make_graph())
        # except Exception as e:
        #     self.get_logger().error(f'Failed to create LangGraph workflow: {e}')
        #     raise
        # -- to be implemented in child classes --

        self.get_logger().info('LangGraphManager graph created successfully...')

    async def initialize_ollama_agent(self) -> None:
        """
        Initialize the mcp client and Ollama agent with the required parameters.

        Returns:
            None
        """
        # Initialize MCP client
        mcp_servers_config = {}
        try:
            self.get_logger().info(f'Loading MCP servers from: {self.mcp_servers}')
            with open(self.mcp_servers, 'r') as f:
                mcp_servers_config = json.load(f)
            self.get_logger().info(f'MCP servers config: {mcp_servers_config}')
        except FileNotFoundError:
            self.get_logger().error(f'MCP servers file not found: {self.mcp_servers}')
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in MCP servers file: {e}')

        # Retrieve available tools from MCP
        self.get_logger().info('Initializing MCP client...')
        try:
            self.mcp_client = Client(mcp_servers_config)
        except Exception as e:
            self.get_logger().error(f'Error initializing MCP client: {e}')
            self.mcp_client = None  # type: ignore[assignment]

        # # Get system prompt template content
        # try:
        #     self.get_logger().info(f'Loading system prompt from: {self.system_prompt_file}')
        #     with open(self.system_prompt_file, 'r') as f:
        #         self.system_prompt = f.read()
        # except Exception as e:
        #     self.get_logger().error(
        #         f'System prompt could not be loaded from {self.system_prompt_file}: {e}')
        #     self.system_prompt = 'You are a helpful assistant.'
        #     self.get_logger().error(f'System prompt set to default: {self.system_prompt}')

        # Initialize Ollama agent with retrieved tools and parameters
        self.get_logger().info(f'Initializing Ollama agent with model: {self.llm_model}')
        self.ollama_agent = Ollama(
            model=self.llm_model,
            tool_call_pattern=self.tool_call_pattern,
            template_type=self.template_type,
            mcp_client=self.mcp_client,
            think=self.enable_thinking,
            raw=self.raw_mode,
            temperature=self.temperature,
            debug=self.debug_mode,
            repeat_penalty=self.repeat_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        self.get_logger().info('Ollama agent initialized successfully')

    def get_params(self) -> None:
        """
        Retrieve and configure ROS2 parameters.

        Declares and retrieves parameters from the ROS2 parameter server,
        Logs each parameter value for verification.

        Parameters:
            None

        Returns:
            None
        """
        # Declare and retrieve topic parameters
        self.declare_parameter('service_name', 'agent_service')
        self.service_name = self.get_parameter(
            'service_name').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter service_name is set to: [{self.service_name}]')

        # Declare and retrieve MCP servers parameter
        self.declare_parameter('mcp_servers', 'mcp.json')
        self.mcp_servers = self.get_parameter(
            'mcp_servers').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter mcp_servers is set to: [{self.mcp_servers}]')

        # # Declare and retrieve system prompt template path parameter
        # self.declare_parameter('system_prompt_file', 'system_prompt.jinja')
        # self.system_prompt_file = self.get_parameter(
        #     'system_prompt_file').get_parameter_value().string_value
        # self.get_logger().info(
        #     f'The parameter system_prompt_file is set to: [{self.system_prompt_file}]')

        # Declare and retrieve model chat template file path parameter
        self.declare_parameter('template_type', 'qwen3')
        self.template_type = self.get_parameter(
            'template_type').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter template_type is set to: [{self.template_type}]')

        # Declare and retrieve LLM model name parameter
        self.declare_parameter('llm_model', 'qwen3:0.6b')
        self.llm_model = self.get_parameter(
            'llm_model').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter llm_model is set to: [{self.llm_model}]')

        # Declare tool call regex pattern to extract tool calls from LLM response
        self.declare_parameter('tool_call_pattern', '<tool_call>(.*?)</tool_call>')
        self.tool_call_pattern = self.get_parameter(
            'tool_call_pattern').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter tool_call_pattern is set to: [{self.tool_call_pattern}]')

        # Declare and retrieve Ollama generation parameters
        self.declare_parameter('raw_mode', False)
        self.raw_mode = self.get_parameter(
            'raw_mode').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter raw_mode is set to: [{self.raw_mode}]')

        self.declare_parameter('debug_mode', False)
        self.debug_mode = self.get_parameter(
            'debug_mode').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter debug_mode is set to: [{self.debug_mode}]')

        self.declare_parameter('temperature', 0.0)
        self.temperature = self.get_parameter(
            'temperature').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter temperature is set to: [{self.temperature}]')

        self.declare_parameter('repeat_penalty', 1.1)
        self.repeat_penalty = self.get_parameter(
            'repeat_penalty').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter repeat_penalty is set to: [{self.repeat_penalty}]')

        self.declare_parameter('top_k', 10)
        self.top_k = self.get_parameter(
            'top_k').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter top_k is set to: [{self.top_k}]')

        self.declare_parameter('top_p', 0.25)
        self.top_p = self.get_parameter(
            'top_p').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter top_p is set to: [{self.top_p}]')

        self.declare_parameter('num_ctx', 8192)
        self.num_ctx = self.get_parameter(
            'num_ctx').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter num_ctx is set to: [{self.num_ctx}]')

        self.declare_parameter('num_predict', 256)
        self.num_predict = self.get_parameter(
            'num_predict').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter num_predict is set to: [{self.num_predict}]')

        # Declare and retrieve LangGraph workflow parameters
        self.declare_parameter('max_steps', 5)
        self.max_steps = self.get_parameter(
            'max_steps').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter max_steps is set to: [{self.max_steps}]')

        self.declare_parameter('enable_thinking', False)
        self.enable_thinking = self.get_parameter(
            'enable_thinking').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter enable_thinking is set to: [{self.enable_thinking}]')

    @abstractmethod
    def agent_callback(self, request, response) -> None:
        """
        Handle incoming user queries via callback.

        To be implemented in child classes.
        Parameters:
            request: The incoming user query request.
            response: The response object to populate with the generated response.
        Returns:
            response: The generated response to the user query.
        """
