
# tests/test_ollama.py
import pytest
import json
from langgraph_base_ros.ollama_utils import Ollama, Messages, Message
import ollama
from ollama import generate

# --------------------------------------------------
# FIXTURE: Creates a ready-to-use Ollama instance with a temporary Jinja template
# --------------------------------------------------
@pytest.fixture
def ollama_fixture(tmp_path):
    # Create a temporary fake Jinja template
    template = tmp_path / "template.jinja"
    template.write_text("{{ messages }}")
    return Ollama(raw=True, jinja_template_path=str(template), debug=True)


# --------------------------------------------------
# TEST 1: Initialization
# --------------------------------------------------
def test_init_requires_template():
    """If raw=True but no Jinja template path is provided, it should raise ValueError."""
    with pytest.raises(ValueError):
        Ollama(raw=True)


def test_init_with_template(ollama_fixture):
    """Ensure Ollama initializes correctly with a valid Jinja template."""
    assert ollama_fixture.model == "qwen3:0.6b"
    assert "messages" in ollama_fixture.state
    assert ollama_fixture.state["messages"][0]["role"] == "system"


# --------------------------------------------------
# TEST 2: create_message
# --------------------------------------------------
def test_create_message_fields(ollama_fixture):
    """create_message should return a valid Message dict with all expected fields."""
    msg = ollama_fixture.create_message(role="user", content="Hello world")
    assert msg["role"] == "user"
    assert msg["content"] == "Hello world"
    assert msg["reasoning_content"] == ""
    assert isinstance(msg["tool_calls"], list)


# --------------------------------------------------
# TEST 3: parse_tool_calls
# --------------------------------------------------
def test_parse_tool_calls_valid(ollama_fixture):
    """parse_tool_calls should correctly extract JSON tool call data from response text."""
    response = """
    <tool_call>
    {"name": "turn_on_light", "arguments": {"room": "kitchen"}}
    </tool_call>
    """
    result = ollama_fixture.parse_tool_calls(response)
    assert result[0]["tool_name"] == "turn_on_light"
    assert result[0]["tool_arguments"] == {"room": "kitchen"}
    # The assistant message should be added to memory
    assert ollama_fixture.state["messages"][-1]["role"] == "assistant"
    # The tool_calls field should be a non-empty list
    assert ollama_fixture.state["messages"][-1]["tool_calls"]


def test_parse_tool_calls_no_tag(ollama_fixture):
    """If there is no <tool_call> tag, a ValueError should be raised."""
    with pytest.raises(ValueError, match="No tool call found in the model response."):
        ollama_fixture.parse_tool_calls("no tags here")

def test_parse_tool_calls_invalid_json(ollama_fixture):
    """If JSON inside <tool_call> is invalid, ValueError should be raised."""
    bad_json = "<tool_call>{invalid_json}</tool_call>"
    with pytest.raises(ValueError, match="Found tool call tags but failed to parse them."):
        ollama_fixture.parse_tool_calls(bad_json)

def test_parse_tool_calls_no_json(ollama_fixture):
    """If JSON inside <tool_call> is invalid, ValueError should be raised."""
    bad_json = "<tool_call>no_json</tool_call>"
    with pytest.raises(ValueError, match="Found tool call tags but failed to parse them."):
        ollama_fixture.parse_tool_calls(bad_json)

def test_parse_tool_calls_invalid_json(ollama_fixture):
    """If JSON args inside <tool_call> is invalid, ValueError should be raised."""
    response = """
    <tool_call>
    {"namo": "turn_on_light", "argumento": {"room": "kitchen"}}
    </tool_call>
    """
    with pytest.raises(ValueError, match="Found tool call tags but failed to parse them."):
        ollama_fixture.parse_tool_calls(response)

# --------------------------------------------------
# TEST 4: reset_memory
# --------------------------------------------------
def test_reset_memory_clears_messages(ollama_fixture):
    """reset_memory should empty the conversation message list."""
    ollama_fixture.state["messages"].append({"role": "user", "content": "hello"})
    ollama_fixture.reset_memory()
    assert ollama_fixture.state["messages"] == []


# --------------------------------------------------
# TEST 5: invoke (mocked)
# --------------------------------------------------
@pytest.mark.asyncio
async def test_invoke_raw_with_mocked_generate(monkeypatch, ollama_fixture):
    """invoke should call generate() and parse tool calls correctly when mocked."""
    # Fake generate() function to avoid real API calls
    def fake_generate(**kwargs):
        return {"response": '<tool_call>{"name":"sum","arguments":{"a":1,"b":2}}</tool_call>'}

    # Replace ollama.generate with our fake function
    monkeypatch.setattr(ollama, "generate", fake_generate)

    result = await ollama_fixture.invoke(user_query="What is 1+2?")
    messages = result["messages"]
    assert any(m["role"] == "assistant" for m in messages)

@pytest.mark.asyncio
async def test_invoke_with_mocked_generate(monkeypatch):
    """invoke should call generate() and parse tool calls correctly when mocked."""
    # Fake generate() function to avoid real API calls
    def fake_generate(**kwargs):
        return {"response": '<tool_call>{"name":"sum","arguments":{"a":1,"b":2}}</tool_call>'}

    # Replace ollama.generate with our fake function
    monkeypatch.setattr(ollama, "generate", fake_generate)

    ollama_not_raw = Ollama(raw=False, debug=True)

    result = await ollama_not_raw.invoke(user_query="What is 1+2?")
    messages = result["messages"]
    assert any(m["role"] == "assistant" for m in messages)

@pytest.mark.asyncio
async def test_invoke_state_query_mocked_generate(monkeypatch, ollama_fixture):
    """invoke should call generate() and parse tool calls correctly when mocked."""
    # Fake generate() function to avoid real API calls
    def fake_generate(**kwargs):
        return {"response": '<tool_call>{"name":"sum","arguments":{"a":1,"b":2}}</tool_call>'}

    # Replace ollama.generate with our fake function
    monkeypatch.setattr(ollama, "generate", fake_generate)
    state_msg = Message(role="user", content="Previous message")
    state = Messages(messages=[state_msg])
    result = await ollama_fixture.invoke(user_query="What is 1+2?", state=state)
    messages = result["messages"]
    assert any(m["role"] == "assistant" for m in messages)

@pytest.mark.asyncio
async def test_invoke_no_state_mocked_generate(monkeypatch, ollama_fixture):
    """invoke should call generate() and parse tool calls correctly when mocked."""
    # Fake generate() function to avoid real API calls
    def fake_generate(**kwargs):
        return {"response": '<tool_call>{"name":"sum","arguments":{"a":1,"b":2}}</tool_call>'}

    # Replace ollama.generate with our fake function
    monkeypatch.setattr(ollama, "generate", fake_generate)
    
    with pytest.raises(ValueError):
        result = await ollama_fixture.invoke()