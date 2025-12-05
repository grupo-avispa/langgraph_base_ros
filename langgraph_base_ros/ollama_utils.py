import json
import re
from typing import TypedDict, NotRequired
from ollama import generate
from jinja2 import Template
from fastmcp import Client
from rich.console import Console
from rich.panel import Panel

console = Console()


class Message(TypedDict):
    """
    This class contains all the message fields required to render model prompt templates.

    Attributes:
        role (str): The role of the message (system, user, assistant, tool)
        content (str): The content of the message
        reasoning_content (str): The reasoning content of the message (for thinking models)
        tool_calls (list): The tool calls associated with the message
    """

    role: str
    content: NotRequired[str]
    reasoning_content: NotRequired[str]
    tool_calls: NotRequired[list]


class Messages(TypedDict):
    """
    Class to hold conversation messages for LangGraph workflow.

    Attributes:
        messages (list[Message]): List of message objects in the conversation.
    """

    messages: list[Message]


class Ollama:
    """
    This class is responsible for communicating with the Ollama Server.

    The conversation memory is stored inside this class.
    """

    def __init__(self,
                 model: str = 'qwen3:0.6b',
                 tool_call_pattern: str = '<tool_call>(.*?)</tool_call>',
                 mcp_client: Client | None = None,
                 think: bool = False,
                 raw: bool = False,
                 temperature: float = 0.0,
                 repeat_penalty: float = 1.1,
                 top_k: int = 10,
                 top_p: float = 0.25,
                 num_ctx: int = 8192,
                 num_predict: int = 256,
                 jinja_template_path: str = '',
                 system_prompt: str = 'You are a helpful assistant.',
                 debug: bool = False):
        """
        Initialize the Ollama class.

        Parameters
        ----------
        model       :   str
            the model name
        lang_tools       :   list
            the list of langchain tools available for use.
        state       :   Messages
            the conversation history state
        mcp_client : Client
            the MCP client to use for tool calls besides the langchain tools.
        tool_call_pattern : str
            the regex pattern to identify tool calls in the model response
        think       :   bool
            (for thinking models) should the model think before responding?
        stream      :   bool
            if false the response will be returned as a single response object,
            rather than a stream of objects.
        raw         :   bool
            if true no formatting will be applied to the prompt. You may choose to use the raw
            parameter if you are specifying a full templated prompt in your request to the API
        jinja_template_path : str
            the jinja template file path to use for prompt formatting.
        system_prompt : str
            the system prompt to use.
        options     :   dict
            a dictionary of options to configure model inference.
        """
        self.model = model
        self.lang_tools: list = []
        self.tool_call_pattern = tool_call_pattern
        self.think = think
        self.raw = raw
        self.system_prompt = system_prompt
        self.options = {
            'temperature': temperature,
            'repeat_penalty': repeat_penalty,
            'top_k': top_k,
            'top_p': top_p,
            'num_ctx': num_ctx,
            'num_predict': num_predict
        }
        self.debug = debug

        # Initialize MCP client if provided, else None
        self.mcp_client = mcp_client

        # Load Jinja2 template if raw mode is enabled
        if raw and jinja_template_path != '':
            with open(jinja_template_path, 'r') as f:
                template_content = f.read()
                self.template = Template(template_content)
        elif raw and jinja_template_path == '':
            raise ValueError(
                'If raw mode is true, a jinja template must be provided for prompt '
                'formatting. The jinja template only applies in raw mode.')

        # Initialize conversation state with system prompt
        self.state: Messages = {
            'messages': [
                self.create_message(
                    role='system',
                    content=self.system_prompt
                )
            ]
        }

    async def retrieve_tools(self, lang_tools: list = []):
        """
        Initialize tool list with langchain tools received as class parameter.

        If the mcp client object is not None tries to asynchronously retrieve the
        list of tools from the MCP server and update the tools attribute.
        The LangChain tools must be provided as a list of dictionaries with the following keys:
        - name: (str) the tool name
        - description : (str) the tool description
        - inputSchema : (dict) the tool input schema
        - tool_object : (object) the langchain tool object with an invoke method.

        Parameters
        ----------
        lang_tools : list
            the list of langchain tools available for use.

        Returns
        None
        """
        self.lang_tools = lang_tools
        self.tools = []
        for tool in self.lang_tools:
            try:
                self.tools.append({
                    'name': tool['name'],
                    'description': tool['description'],
                    'inputSchema': tool['inputSchema'],
                })
            except AttributeError as e:
                console.print(f'[red]Error retrieving langchain tool attributes: {e}[/red]')
        if self.mcp_client is not None:
            async with self.mcp_client:
                tools = await self.mcp_client.list_tools()
                for tool in tools:
                    self.tools.append({
                        'name': tool.name,
                        'description': tool.description,
                        'inputSchema': tool.inputSchema,
                    })
        else:
            console.print('[red]MCP client is not initialized. Cannot retrieve tools[/red]')

    def create_message(
            self,
            role: str,
            content: str = '',
            reasoning_content: str = '',
            tool_calls: list | None = None
    ) -> Message:
        """
        Create a message object.

        Parameters
        ----------
        role : str
            The role of the message (system, user, assistant, tool)
        content : str
            The content of the message
        reasoning_content : str
            The reasoning content of the message (for thinking models)
        tool_calls : list
            The tool calls associated with the message

        Returns
        -------
        Message
            The created message dictionary
        """
        if tool_calls is None:
            tool_calls = []

        msg: Message = {
            'role': role,
            'content': content,
            'reasoning_content': reasoning_content,
            'tool_calls': tool_calls
        }
        return msg

    def parse_tool_calls(self, response: str):
        """
        Parse tool calls from the model response.

        Parameters
        response : str
            the response from the model

        Returns
        tool_calls : list
            the list of parsed tool calls
        """
        # Look for tool call patterns in the response
        tool_call_matches = re.findall(self.tool_call_pattern, response, re.DOTALL)
        if tool_call_matches:
            tool_calls_list = []
            all_actions = []
            # Iterate over all matches
            for match in tool_call_matches:
                parsed_response = match.strip()
                # Create JSON object from the parsed response
                try:
                    action = json.loads(parsed_response)
                except json.JSONDecodeError as e:
                    console.print(f'[red]JSON decode error while parsing tool call: {e}[/red]')
                    continue
                # Extract tool name and parameters
                try:
                    tool_name = action['name']
                    tool_arguments = action['arguments']
                    # Append the tool call to the list
                    tool_calls_list.append({
                        'tool_name': tool_name,
                        'tool_arguments': tool_arguments,
                        'raw': parsed_response
                    })
                    all_actions.append(action)
                except KeyError as e:
                    console.print(f'[red]Error parsing tool call: {e}[/red]')
                    continue
            # Check if any tool calls were successfully parsed
            if tool_calls_list:
                # Append the tool calls to the conversation memory
                self.state['messages'].append(self.create_message(
                    role='assistant',
                    tool_calls=all_actions)
                )
                return tool_calls_list
            else:
                # Append the response without tool call to the conversation memory
                self.state['messages'].append(self.create_message(
                    role='assistant',
                    content=response))
                raise ValueError('Found tool call tags but failed to parse them.')
        else:
            # Append the response without tool call to the conversation memory
            self.state['messages'].append(self.create_message(
                role='assistant',
                content=response))
            raise ValueError('No tool call found in the model response.')

    async def invoke(self, user_query: str = '', state: Messages | None = None) -> Messages:
        """
        Send the request to the ollama server and return the response.

        The state messages are updated with the new response.

        Parameters
        user_query : str
            the user query to send to the model. If empty, the last user message in state is used.
        state : Messages
            Optional Messages object containing conversation history.
            If provided, replaces current state.

        Returns
        Messages : the updated state with new messages.
        """
        # If a state is provided, use it to replace the current state
        if state is not None:
            self.state = state

        # Check if any of the message roles is 'user' if no user_query is provided
        has_user_message = any(
            (msg['role'] == 'user' and msg.get('content', ''))
            for msg in self.state['messages'])
        if not user_query and not has_user_message:
            raise ValueError(
                'If no user query is provided, the state must contain at least one user message.')
        elif not has_user_message and user_query:
            # Add user message to conversation memory
            self.state['messages'].append(self.create_message(
                role='user',
                content=user_query)
            )
        elif has_user_message and user_query:
            console.print(
                '[yellow]Warning: Both user_query and user message in state provided. '
                'Ignoring user_query.[/yellow]')

        # Prepare the prompt
        if self.raw:
            prompt = self.template.render(
                messages=self.state['messages'],
                tools=self.tools,
                add_generation_prompt=True,
                enable_thinking=self.think
            )

            # Uncomment to see rendered prompt
            console.print(Panel(
                prompt,
                title='[cyan bold]RENDERED PROMPT[/cyan bold]',
                border_style='cyan',
                expand=False
            ))

            response = generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                raw=self.raw,
                options=self.options
            )
            if self.debug:
                console.print(Panel(
                    response['response'],
                    title='[yellow bold]RAW RESPONSE TEXT[/yellow bold]',
                    border_style='yellow',
                    expand=False
                ))
        else:
            response = generate(
                model=self.model,
                prompt=self.state['messages'][-1]['content'],
                stream=False,
                raw=self.raw,
                system=self.system_prompt,
                options=self.options
            )
        # Check if tool calls are present in the response
        try:
            # Parse tool calls from the response and iterate over them
            tool_calls = self.parse_tool_calls(response['response'])
            print(f'Parsed tool calls: {tool_calls}')
            for tool_call in tool_calls:
                tool_call_done = False
                # First check if the tool is a langchain tool
                if self.lang_tools is not None:
                    for lang_tool in self.lang_tools:
                        if lang_tool['name'] == tool_call['tool_name']:
                            # Call the langchain tool
                            tool_response = lang_tool['tool_object'].invoke(
                                tool_call['tool_arguments']
                            )
                            tool_call_done = True
                            break
                # If not found, check if MCP client is available to call the tool
                if not tool_call_done and self.mcp_client is not None:
                    async with self.mcp_client:
                        tool_response = await self.mcp_client.call_tool(
                            tool_call['tool_name'],
                            tool_call['tool_arguments']
                        )
                        tool_call_done = True
                # If tool call was successful, add the response to the conversation memory
                if tool_call_done:
                    if self.debug:
                        console.print(Panel(
                            str(tool_response),
                            title='[green bold]TOOL RESPONSE[/green bold]',
                            border_style='green',
                            expand=False
                        ))
                    # Add observation to conversation memory
                    content = tool_response.content[0].text \
                        if hasattr(tool_response, 'content') else str(tool_response)
                    self.state['messages'].append(self.create_message(
                        role='tool',
                        content=content)
                    )
        except ValueError:
            pass
        return self.state

    def reset_memory(self) -> None:
        """Reset the conversation memory."""
        self.state['messages'] = []
