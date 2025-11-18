# langgraph_base_ros

ROS2 package providing an intelligent conversational agent framework for robots using language models (LLMs) and tools via the Model Context Protocol (MCP).

## Overview

This package offers a flexible framework for implementing conversational agents in ROS2 using LangGraph workflows and Ollama as the backend for running LLM models locally. The architecture is designed with extensibility in mind, allowing developers to create custom use cases by inheriting from base classes.

## Architecture

### Core Components

- **Base Classes**: Abstract base classes for creating custom agents
  - `LangGraphBase`: Base class for LangGraph workflow management
  - `LangGraphRosBase`: Base class for ROS2 integration

- **Utilities**:
  - `Ollama`: Client for Ollama LLM server with MCP integration
  - MCP Client: Tool retrieval and execution via Model Context Protocol
  - Prompt System: Jinja2 templates for customizable prompts

### Communication Interface

The agent operates as a ROS2 service:

- **Service** (default: `agent_service`): Receives user queries and returns agent responses
  - Service Type: `llm_interactions_msgs/UserQueryResponse`
  - Request Fields:
    - `user_query` (string): The user's question or command
    - `user_name` (string): Optional username for personalization
  - Response Fields:
    - `response_text` (string): The agent's generated response

### File Structure

```
langgraph_base_ros/
├── test/
│   └── test_langgraph.py          # Unit tests for the package
├── resource/
│   └── langgraph_base_ros           # Resource files for ROS2
├── langgraph_base_ros/
│   ├── langgraph_base.py           # Base class for LangGraph workflows
│   ├── langgraph_ros_base.py       # Base class for ROS2 integration
│   ├── ollama_utils.py             # Ollama client utilities
├── package.xml
├── requirements.txt
├── README.md
├── setup.cfg
└── setup.py
```

## Prerequisites

### Installation

1. **Ollama installed and running**:
   ```bash
   # Install Ollama (if not installed)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Download the model
   ollama pull qwen3:0.6b
   
   # Verify Ollama is running
   ollama list
   ```

2. **Configured MCP server**:
   - Ensure the MCP server specified in `langgraph_mcp.json` is accessible
   - Server must implement the MCP protocol correctly
   - A fake MCP server (`fake_mcp_server.py`) is provided for testing

3. **Python virtual environment** with dependencies:
   ```bash
   python -m venv agent-venv
   source agent-venv/bin/activate
   pip install -r requirements.txt
   ```
   ```bash
   cd uv
   uv sync
   source .venv/bin/activate
   ```

4. **Environment variables** (optional, for LangSmith tracing):
   Edit the `.env` file in the package:
   ```bash
   LANGSMITH_API_KEY=your_api_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=project_name
   ```
    
5. **Building**:
   ```bash
   cd ~/colcon_ws
   source /opt/ros/humble/setup.bash
   colcon build --packages-select pisito_agent --symlink-install
   source install/setup.bash
   ```

## Base Classes

### LangGraphBase

Abstract base class for implementing LangGraph workflows. Provides common functionality and attributes for conversation management.

**Key Features**:
- Workflow state management
- Step and message counting
- Logging abstraction
- Abstract `make_graph()` method for custom workflow definition

**Usage**:
```python
from langgraph_base_ros.langgraph_base import LangGraphBase

class CustomWorkflow(LangGraphBase):
    async def make_graph(self):
        # Define your custom workflow here
        workflow = StateGraph(Messages)
        # Add nodes, edges, etc.
        self.graph = workflow.compile()
```

### LangGraphRosBase

Abstract base class for ROS2 integration. Handles ROS2 parameter management, Ollama agent initialization, and MCP client setup.

**Key Features**:
- ROS2 parameter declaration and retrieval
- Async Ollama agent initialization
- MCP client configuration and tool retrieval
- Abstract `agent_callback()` method for custom service handling
- Abstract `build_graph()` method for workflow compilation based on the defined LangGraph workflow
- Persistent asyncio event loop management

**Usage**:
```python
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase

class CustomRosAgent(LangGraphRosBase):
    def __init__(self):
        super().__init__()
        # Initialize your custom workflow manager
        self.graph_manager = CustomWorkflow(...)
        self.build_graph(self.graph_manager)
        
        # Create your service/subscriber/etc.
        self.create_service(...)
    
    def agent_callback(self, request, response):
        # Handle incoming requests
        return response
```

## Creating Custom Use Cases

### Step 1: Define Your Workflow

Create a class inheriting from `LangGraphBase`:

```python
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph.graph import START, StateGraph, END

class MyCustomWorkflow(LangGraphBase):
    async def my_custom_node(self, state):
        # Your custom logic
        return state
    
    def my_decision_function(self, state):
        # Decision logic for conditional edges
        return "next_node"
    
    async def make_graph(self):
        workflow = StateGraph(Messages)
        
        # Add nodes
        workflow.add_node('my_node', self.my_custom_node)
        
        # Add edges
        workflow.add_edge(START, 'my_node')
        workflow.add_conditional_edges(
            'my_node',
            self.my_decision_function,
            {'next_node': 'my_node', 'finish': END}
        )
        
        self.graph = workflow.compile()
```

### Step 2: Create ROS2 Integration

Create a class inheriting from `LangGraphRosBase`:

```python
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from your_workflow_module import MyCustomWorkflow
from your_msgs.srv import YourServiceType

class MyCustomAgent(LangGraphRosBase):
    def __init__(self):
        super().__init__()
        
        # Initialize your workflow
        self.graph_manager = MyCustomWorkflow(
            logger=self.get_logger(),
            ollama_agent=self.ollama_agent,
            max_steps=self.max_steps
        )

        self.build_graph()
        
        # Create your service
        self.srv = self.create_service(
            YourServiceType,
            'your_service_name',
            self.agent_callback
        )
    
    def agent_callback(self, request, response):
        # Process request using your workflow
        result = self.loop.run_until_complete(
            self.graph_manager.graph.ainvoke(initial_state)
        )
        response.field = result['messages'][-1]['content']
        return response
    def build_graph(self) -> None:
        # Initialize and compile the LangGraph workflow
        try:
            self.loop.run_until_complete(self.graph_manager.make_graph())
        except Exception as e:
            self.get_logger().error(f'Failed to create LangGraph workflow: {e}')
            raise

        self.get_logger().info('MyCustomWorkflow graph created successfully...')
```

## License

Apache License 2.0

## Maintainer

Oscar Pons Fernandez (opfernandez@uma.es)

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [LangSmith](https://www.langchain.com/langsmith)