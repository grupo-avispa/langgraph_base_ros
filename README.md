# pisito_agent

ROS2 package providing an intelligent conversational agent framework for robots using language models (LLMs) and tools via the Model Context Protocol (MCP).

## Overview

This package offers a flexible framework for implementing conversational agents in ROS2 using LangGraph workflows and Ollama as the backend for running LLM models locally. The architecture is designed with extensibility in mind, allowing developers to create custom use cases by inheriting from base classes.

The package includes a complete home assistant implementation as a reference use case, demonstrating how to build domain-specific agents on top of the provided base classes.

## Architecture

### Core Components

- **Base Classes**: Abstract base classes for creating custom agents
  - `LangGraphBase`: Base class for LangGraph workflow management
  - `LangGraphRosBase`: Base class for ROS2 integration
  
- **Use Case Implementations**: Concrete implementations for specific scenarios
  - `LangGraphManager`: Home assistant workflow implementation
  - `RosHomeAssistantAgent`: Home assistant ROS2 node

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
pisito_agent/
├── launch/
│   └── langgraph.launch.py          # Launcher for LangGraph agent
├── params/
│   ├── default_params.yaml          # Default parameters
│   └── langgraph_mcp.json          # MCP server configuration
├── templates/
│   ├── system_prompt.jinja         # System prompt template
│   └── qwen3.jinja                 # Chat template for Qwen3
├── pisito_agent/
│   ├── langgraph_base.py           # Base class for LangGraph workflows
│   ├── langgraph_ros_base.py       # Base class for ROS2 integration
│   ├── langgraph_home_assistant.py # Home assistant workflow implementation
│   ├── langgraph_ros_home_assistant_agent.py # Home assistant ROS2 node
│   ├── ollama_utils.py             # Ollama client utilities
│   └── fake_mcp_server.py          # Test MCP server
├── .env                            # Environment variables (LangSmith)
├── package.xml
└── setup.py
```

## Prerequisites

### Option A: Native Installation

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

### Option B: Docker Installation

Alternatively, you can run the agent using Docker, which includes all dependencies pre-configured.

1. **Build Docker image**:
   ```bash
   cd ~/colcon_ws/src/interaction/pisito_agent/Docker
   docker compose build
   ```

2. **Run with VSCode Dev Containers**:
   - Open the `pisito_agent` folder in VSCode
   - Reopen in container when prompted
   - Use the integrated terminal to run the agent

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
from pisito_agent.langgraph_base import LangGraphBase

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
- Persistent asyncio event loop management

**Usage**:
```python
from pisito_agent.langgraph_ros_base import LangGraphRosBase

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
from pisito_agent.langgraph_base import LangGraphBase
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
from pisito_agent.langgraph_ros_base import LangGraphRosBase
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
        self.build_graph(self.graph_manager)
        
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
```

### Step 3: Create Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pisito_agent',
            executable='your_custom_agent_executable',
            name='your_agent_node',
            parameters=[your_params_file]
        )
    ])
```

## Home Assistant Use Case (Reference Implementation)

### Key Features

- **LangGraph Workflow**: State and conversation flow management via graphs
- **Ollama Backend**: Local execution of optimized LLM models
- **Raw Mode**: Direct control over message format using custom templates
- **Chat Templates**: Support for custom Jinja2 templates
- **Async Operations**: Asynchronous processing for better performance
- **MCP Integration**: External tool execution via Model Context Protocol

### Configuration Parameters

File: `params/default_params.yaml`

```yaml
langgraph_agent_node:
  ros__parameters:
    # Service configuration
    service_name: "agent_service"
    
    # Configuration files
    mcp_servers: "langgraph_mcp.json"
    system_prompt_file: "system_prompt.jinja"
    model_chat_template_file: "qwen3.jinja"
    
    # LLM Model
    llm_model: "qwen3:0.6b"
    
    # Tool call extraction
    tool_call_pattern: "<tool_call>(.*?)</tool_call>"
    
    # Ollama generation parameters
    raw_mode: true
    debug_mode: false
    temperature: 0.0
    repeat_penalty: 1.1
    top_k: 10
    top_p: 1.0
    num_ctx: 8192
    num_predict: 128
    
    # LangGraph workflow parameters
    max_steps: 5
    enable_thinking: false
```

### MCP Configuration

File: `params/langgraph_mcp.json` defines MCP servers to connect to:

```json
{
    "mcpServers": {
        "server_name": {
            "url": "http://localhost:3002/mcp"
        }
    }
}
```

### Running the Home Assistant

```bash
# Activate virtual environment
source agent-venv/bin/activate

# Launch agent with default configuration
ros2 launch pisito_agent langgraph.launch.py

# Test the agent
ros2 service call /agent_service llm_interactions_msgs/srv/UserQueryResponse "{user_query: 'Turn on the living room lights', user_name: 'John'}"
```

## Customization

### System Prompts

Edit `templates/system_prompt.jinja` to change agent behavior:

```jinja
You are a helpful assistant specialized in [your domain].
You can use the following tools: [list tools]
Always respond politely and concisely.
```

### Chat Templates

Add custom model templates to `/templates` folder. The template must be compatible with the Jinja2 format expected by your model.

### Adding New Tools

Tools are automatically retrieved from the MCP server:

1. Implement the tool on your MCP server
2. Restart the MCP server
3. Relaunch the ROS2 agent

The agent will automatically discover and use the new tools.

## Development

### Testing with Fake MCP Server

For development and testing, use the provided fake MCP server:

```bash
python pisito_agent/fake_mcp_server.py
```

This server provides mock home automation tools for testing agent interactions.

### Modifying Agent Flow

Edit your workflow class (e.g., `langgraph_home_assistant.py`) to change the graph structure:

- Add new nodes for additional processing steps
- Modify conditional edges to change decision logic
- Adjust max_steps and other parameters in the configuration

### Debugging

Enable debug mode in parameters to see detailed LLM interactions:

```yaml
debug_mode: true
```

Enable LangSmith tracing by setting environment variables in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name
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
