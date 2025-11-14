import pytest
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
def mock_node(mocker):
    """Fixture to mock ROS2 Node initialization"""
    mocker.patch("langgraph_base_ros.langgraph_ros_base.Node.__init__", return_value=None)


@pytest.fixture
def ollama_fixture():
    """Fixture to create a Ollama agent"""
    ollama_fixture = Ollama(raw=False)
    return ollama_fixture


@pytest.fixture
def mock_mcp_client(mocker):
    """Fixture to create a mock MCP client"""
    client = mocker.AsyncMock()
    client.list_tools = mocker.AsyncMock(return_value=[{"name": "tool1"}, {"name": "tool2"}])
    return client

@pytest.fixture
def manual_node():
    # Create node manually
    node = ConcreteLangGraphRosBase.__new__(ConcreteLangGraphRosBase)
    node.mcp_servers = "missing.json"
    node.system_prompt_file = ""
    node.llm_model = "qwen3:0.6b"
    node.tool_call_pattern = "<tool_call>(.*?)</tool_call>"
    node.chat_template_path = ""
    node.enable_thinking = False
    node.raw_mode = False
    node.temperature = 0.0
    node.debug_mode = False
    node.repeat_penalty = 1.1
    node.top_k = 10
    node.top_p = 0.25
    node.num_ctx = 8192
    node.num_predict = 256
    return node


# ============================================================
#                     TEST LangGraphBase
# ============================================================

class TestLangGraphBase:
    def test_init_with_logger(self, mocker, ollama_fixture):
        logger = mocker.Mock()
        graph_base = ConcreteLangGraphBase(logger=logger, ollama_agent=ollama_fixture, max_steps=10)
        
        assert graph_base.logger == logger
        assert graph_base.ollama_agent == ollama_fixture
        assert graph_base.max_steps == 10
        assert graph_base.steps == 0
        assert graph_base.messages_count == 0
        assert graph_base.graph is None
    
    def test_init_without_logger(self, ollama_fixture):
        graph_base = ConcreteLangGraphBase(ollama_agent=ollama_fixture)
        
        assert graph_base.logger is None
        assert graph_base.ollama_agent == ollama_fixture
        assert graph_base.max_steps == 5
    
    def test_init_without_ollama_agent(self):
        with pytest.raises(ValueError, match="Ollama agent instance must be provided"):
            ConcreteLangGraphBase(ollama_agent=None)
    
    def test_log_with_ros_logger(self, mocker, ollama_fixture):
        logger = mocker.Mock()
        graph_base = ConcreteLangGraphBase(logger=logger, ollama_agent=ollama_fixture)
        
        graph_base._log("test message")
        logger.info.assert_called_once_with("test message")
    
    def test_log_without_ros_logger(self, mocker, ollama_fixture):
        mock_logging = mocker.patch("langgraph_base_ros.langgraph_base.logging")
        graph_base = ConcreteLangGraphBase(ollama_agent=ollama_fixture)
        
        graph_base._log("test message")
        mock_logging.info.assert_called_once_with("test message")
    
    @pytest.mark.asyncio
    async def test_make_graph(self, ollama_fixture):
        graph_base = ConcreteLangGraphBase(ollama_agent=ollama_fixture)
        
        await graph_base.make_graph()
        assert graph_base.graph == "compiled_graph"


# ============================================================
#                 TEST LangGraphRosBase
# ============================================================

class TestLangGraphRosBase:
    
    def test_get_params(self, mocker, mock_node):
        # Patch asyncio loop
        mock_new_loop = mocker.patch("langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop")
        mock_set_loop = mocker.patch("langgraph_base_ros.langgraph_ros_base.asyncio.set_event_loop")
        mock_new_loop.return_value = mocker.Mock()

        # Patch methods
        mocker.patch.object(ConcreteLangGraphRosBase, "initialize_ollama_agent", new_callable=mocker.AsyncMock)
        mocker.patch.object(ConcreteLangGraphRosBase, "declare_parameter")
        mocker.patch.object(ConcreteLangGraphRosBase, "get_logger", return_value=mocker.Mock())

        # Mock get_parameter + different value types
        mock_param = mocker.Mock()
        mock_param.get_parameter_value.return_value.string_value = "test_value"
        mock_param.get_parameter_value.return_value.bool_value = True
        mock_param.get_parameter_value.return_value.double_value = 0.5
        mock_param.get_parameter_value.return_value.integer_value = 10

        mocker.patch.object(ConcreteLangGraphRosBase, "get_parameter", return_value=mock_param)

        node = ConcreteLangGraphRosBase()

        assert node.service_name == "test_value"
        assert node.raw_mode is True
        assert node.temperature == 0.5
    

    @pytest.mark.asyncio
    async def test_initialize_ollama_agent_success(self, mocker, manual_node, mock_mcp_client):
        # Mock file open
        mocker.patch("builtins.open", mocker.mock_open(read_data='{"servers": {}}'))
        mocker.patch("json.load", return_value={"servers": {}})

        # Mock MCP Client
        mocker.patch("langgraph_base_ros.langgraph_ros_base.Client", return_value=mock_mcp_client)

        # Patch misc methods
        mocker.patch("langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop")
        mocker.patch.object(ConcreteLangGraphRosBase, "get_params")
        mocker.patch.object(ConcreteLangGraphRosBase, "get_logger", return_value=mocker.Mock())

        mock_ollama = mocker.patch("langgraph_base_ros.langgraph_ros_base.Ollama")

        await manual_node.initialize_ollama_agent()
        assert mock_ollama.called
    

    @pytest.mark.asyncio
    async def test_initialize_ollama_agent_file_not_found(self, mocker, manual_node):
        mocker.patch("builtins.open", side_effect=FileNotFoundError)
        mocker.patch("langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop")
        mocker.patch.object(ConcreteLangGraphRosBase, "get_params")

        mocker.patch.object(manual_node, "get_logger", return_value=mocker.Mock())

        await manual_node.initialize_ollama_agent()

        assert manual_node.tools == []
        assert manual_node.mcp_client is None
        assert manual_node.ollama_agent is not None

    @pytest.mark.asyncio
    async def test_initialize_ollama_agent_invalid_json(self, mocker, manual_node):
        mocker.patch("builtins.open", mocker.mock_open(read_data="invalid json"))
        mocker.patch("json.load", side_effect=json.JSONDecodeError("err", "doc", 0))
        mocker.patch("langgraph_base_ros.langgraph_ros_base.asyncio.new_event_loop")
        mocker.patch.object(ConcreteLangGraphRosBase, "get_params")

        mocker.patch.object(manual_node, "get_logger", return_value=mocker.Mock())

        await manual_node.initialize_ollama_agent()

        assert manual_node.tools == []
        assert manual_node.mcp_client is None
        assert manual_node.ollama_agent is not None
