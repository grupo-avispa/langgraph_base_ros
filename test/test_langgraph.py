import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch, mock_open
import json
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase
from langgraph_base_ros.langgraph_base import LangGraphBase
from langgraph_base_ros.ollama_utils import Ollama


class ConcreteLangGraphRosBase(LangGraphRosBase):
    """Concrete implementation for testing abstract class"""
    def build_graph(self):
        pass
    
    def agent_callback(self, request, response):
        pass


class ConcreteLangGraphBase(LangGraphBase):
    """Concrete implementation for testing abstract class"""
    async def make_graph(self):
        self.graph = "compiled_graph"


@pytest.fixture
def mock_node():
    """Fixture to mock ROS2 Node initialization"""
    with patch('langgraph_base_ros.langgraph_ros_base.Node.__init__', return_value=None):
        yield


@pytest.fixture
def mock_ollama_agent():
    """Fixture to create a mock Ollama agent"""
    agent = Mock(spec=Ollama)
    agent.model = "qwen3:0.6b"
    return agent


@pytest.fixture
def mock_mcp_client():
    """Fixture to create a mock MCP client"""
    client = AsyncMock()
    client.list_tools = AsyncMock(return_value=[{"name": "tool1"}, {"name": "tool2"}])
    return client


class TestLangGraphBase:
    """Test suite for LangGraphBase class"""
    
    def test_init_with_logger(self, mock_ollama_agent):
        """Test initialization with logger"""
        logger = Mock()
        graph_base = ConcreteLangGraphBase(logger=logger, ollama_agent=mock_ollama_agent, max_steps=10)
        
        assert graph_base.logger == logger
        assert graph_base.ollama_agent == mock_ollama_agent
        assert graph_base.max_steps == 10
        assert graph_base.steps == 0
        assert graph_base.messages_count == 0
        assert graph_base.graph is None
    
    def test_init_without_logger(self, mock_ollama_agent):
        """Test initialization without logger"""
        graph_base = ConcreteLangGraphBase(ollama_agent=mock_ollama_agent)
        
        assert graph_base.logger is None
        assert graph_base.ollama_agent == mock_ollama_agent
        assert graph_base.max_steps == 5
    
    def test_init_without_ollama_agent(self):
        """Test initialization fails without Ollama agent"""
        with pytest.raises(ValueError, match="Ollama agent instance must be provided"):
            ConcreteLangGraphBase(ollama_agent=None)
    
    def test_log_with_ros_logger(self, mock_ollama_agent):
        """Test logging with ROS2 logger"""
        logger = Mock()
        graph_base = ConcreteLangGraphBase(logger=logger, ollama_agent=mock_ollama_agent)
        
        graph_base._log("test message")
        logger.info.assert_called_once_with("test message")
    
    @patch('langgraph_base_ros.langgraph_base.logging')
    def test_log_without_ros_logger(self, mock_logging, mock_ollama_agent):
        """Test logging with Python logging"""
        graph_base = ConcreteLangGraphBase(ollama_agent=mock_ollama_agent)
        
        graph_base._log("test message")
        mock_logging.info.assert_called_once_with("test message")
    
    @pytest.mark.asyncio
    async def test_make_graph(self, mock_ollama_agent):
        """Test make_graph abstract method implementation"""
        graph_base = ConcreteLangGraphBase(ollama_agent=mock_ollama_agent)
        
        await graph_base.make_graph()
        assert graph_base.graph == "compiled_graph"


class TestLangGraphRosBase:
    """Test suite for LangGraphRosBase class"""
    
    @patch('langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop')
    @patch('langgraph_base_ros.langgraph_ros_base.asyncio.set_event_loop')
    def test_get_params(self, mock_set_loop, mock_new_loop, mock_node):
        """Test parameter retrieval"""
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop
        
        with patch.object(ConcreteLangGraphRosBase, 'initialize_ollama_agent', new_callable=AsyncMock):
            with patch.object(ConcreteLangGraphRosBase, 'get_logger') as mock_logger:
                mock_logger.return_value = Mock()
                with patch.object(ConcreteLangGraphRosBase, 'declare_parameter'):
                    with patch.object(ConcreteLangGraphRosBase, 'get_parameter') as mock_get_param:
                        mock_param = Mock()
                        mock_param.get_parameter_value().string_value = "test_value"
                        mock_param.get_parameter_value().bool_value = True
                        mock_param.get_parameter_value().double_value = 0.5
                        mock_param.get_parameter_value().integer_value = 10
                        mock_get_param.return_value = mock_param
                        
                        node = ConcreteLangGraphRosBase()
                        
                        assert node.service_name == "test_value"
                        assert node.raw_mode == True
                        assert node.temperature == 0.5
    
    @pytest.mark.asyncio
    @patch('builtins.open', new_callable=mock_open, read_data='{"servers": {}}')
    @patch('langgraph_base_ros.langgraph_ros_base.Client')
    async def test_initialize_ollama_agent_success(self, mock_client_class, mock_file, mock_ollama_agent, mock_mcp_client):
        """Test successful Ollama agent initialization"""
        mock_client_class.return_value = mock_mcp_client
        
        with patch('langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop'):
            with patch.object(ConcreteLangGraphRosBase, 'get_params'):
                with patch.object(ConcreteLangGraphRosBase, 'get_logger') as mock_logger:
                    mock_logger.return_value = Mock()
                    
                    node = ConcreteLangGraphRosBase.__new__(ConcreteLangGraphRosBase)
                    node.mcp_servers = "mcp.json"
                    node.system_prompt_file = "prompt.jinja"
                    node.llm_model = "qwen3:0.6b"
                    node.tool_call_pattern = "<tool_call>(.*?)</tool_call>"
                    node.chat_template_path = "qwen3.jinja"
                    node.enable_thinking = False
                    node.raw_mode = True
                    node.temperature = 0.0
                    node.debug_mode = False
                    node.repeat_penalty = 1.1
                    node.top_k = 10
                    node.top_p = 0.25
                    node.num_ctx = 8192
                    node.num_predict = 256
                    
                    with patch.object(node, 'get_logger') as mock_get_logger:
                        mock_get_logger.return_value = Mock()
                        
                        with patch('langgraph_base_ros.langgraph_ros_base.Ollama') as mock_ollama:
                            await node.initialize_ollama_agent()
                            
                            assert mock_ollama.called
    
    @pytest.mark.asyncio
    @patch('builtins.open', side_effect=FileNotFoundError)
    async def test_initialize_ollama_agent_file_not_found(self, mock_file):
        """Test Ollama agent initialization with missing MCP config"""
        with patch('langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop'):
            with patch.object(ConcreteLangGraphRosBase, 'get_params'):
                node = ConcreteLangGraphRosBase.__new__(ConcreteLangGraphRosBase)
                node.mcp_servers = "missing.json"
                
                with patch.object(node, 'get_logger') as mock_logger:
                    mock_logger.return_value = Mock()
                    
                    with pytest.raises(FileNotFoundError):
                        await node.initialize_ollama_agent()
    
    @pytest.mark.asyncio
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    async def test_initialize_ollama_agent_invalid_json(self, mock_file):
        """Test Ollama agent initialization with invalid JSON"""
        with patch('langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop'):
            with patch.object(ConcreteLangGraphRosBase, 'get_params'):
                node = ConcreteLangGraphRosBase.__new__(ConcreteLangGraphRosBase)
                node.mcp_servers = "invalid.json"
                
                with patch.object(node, 'get_logger') as mock_logger:
                    mock_logger.return_value = Mock()
                    
                    with pytest.raises(json.JSONDecodeError):
                        await node.initialize_ollama_agent()
    
    def test_build_graph_abstract(self, mock_node):
        """Test that build_graph is abstract"""
        assert hasattr(LangGraphRosBase, 'build_graph')
        assert getattr(LangGraphRosBase.build_graph, '__isabstractmethod__', False)
    
    def test_agent_callback_abstract(self, mock_node):
        """Test that agent_callback is abstract"""
        assert hasattr(LangGraphRosBase, 'agent_callback')
        assert getattr(LangGraphRosBase.agent_callback, '__isabstractmethod__', False)