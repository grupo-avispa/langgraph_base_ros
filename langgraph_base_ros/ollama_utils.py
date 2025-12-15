import json
import re
from typing import Optional
from ollama import generate, chat, Message
from fastmcp import Client
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

from chat_template_render import TemplateRenderer, Messages

console = Console()


class Ollama:
    """
    This class is responsible for communicating with the Ollama Server.

    The conversation memory is stored inside this class.
    """

    def __init__(
        self,
        model: str = 'qwen3:0.6b',
        tool_call_pattern: str = '',
        template_type: str = '',
        mcp_client: Client | None = None,
        think: bool = False,
        raw: bool = False,
        temperature: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 10,
        top_p: float = 0.25,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        debug: bool = False
    ) -> None:
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
        render :   TemplateRenderer
            the template renderer object to format prompts in raw mode.
        options     :   dict
            a dictionary of options to configure model inference.
        """
        self.model = model
        self.lang_tools: list = []
        self.tool_call_pattern = tool_call_pattern
        self.think = think
        self.raw = raw
        self.options = {
            'temperature': temperature,
            'repeat_penalty': repeat_penalty,
            'top_k': top_k,
            'top_p': top_p,
            'num_ctx': num_ctx,
            'num_predict': num_predict
        }
        self.debug = debug
        self.tools: list = []

        # Initialize MCP client if provided, else None
        self.mcp_client = mcp_client

        # Load Jinja2 template if raw mode is enabled
        if raw and template_type != '':
            self.renderer = TemplateRenderer(
                template_type=template_type,
                think=think
            )
        if raw and template_type == '':
            raise ValueError(
                'If raw mode is true, a jinja template type must be provided for prompt '
                'formatting. The jinja template only applies in raw mode.')
        if raw and self.tool_call_pattern == '':
            self.tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
            raise ValueError(
                'Raw mode is true but no tool call pattern is provided. '
                'A default pattern will be used but probably won\'t match your template.')

    async def retrieve_tools(self, lang_tools: list = []):
        """
        Initialize tool list with langchain tools received as class parameter.

        If the mcp client object is not None tries to asynchronously retrieve the
        list of tools from the MCP server and update the tools attribute.
        The LangChain tools must be provided as a list of dictionaries with the following format:
        {
            'type': 'function',
            'function': {
                'name': tool["name"],
                'description': tool["description"],
                'parameters': tool["inputSchema"],
                'tool_object': tool
            } 
        }

        Parameters
        ----------
        lang_tools : list
            the list of langchain tools available for use.

        Returns
        None
        """
        self.lang_tools = lang_tools
        for tool in self.lang_tools:
            try:
                if not all(k in tool for k in ("name", "description", "inputSchema", "tool_object")):
                    raise KeyError(
                        "One or more required keys are missing in the langchain tool dictionary.")
                self.tools.append({
                    'type': 'function',
                    'function': {
                        'name': tool["name"],
                        'description': tool["description"],
                        'parameters': tool["inputSchema"]
                    }
                })
            except Exception as e:
                console.print(f'[yellow]Error retrieving langchain tool attributes: {e}[/yellow]')
        if self.mcp_client is not None:
            async with self.mcp_client:
                tools = await self.mcp_client.list_tools()
                for tool in tools:
                    self.tools.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': tool.description,
                            'parameters': tool.inputSchema
                        }
                    })
        else:
            console.print('[yellow]MCP client is not initialized. Cannot retrieve tools[/yellow]')

    def create_message(
            self,
            role: str,
            content: Optional[str] = None,
            thinking: Optional[str] = None,
            tool_calls: list | None = None,
            tool_name: Optional[str] = None
    ) -> Message:
        """
        Create a message object using Ollama's native Message class.
        Only assigns optional attributes when they are provided and not None.

        Parameters
        ----------
        role : str
            The role of the message (system, user, assistant, tool)
        content : str, optional
            The content of the message
        thinking : str, optional
            The thinking content of the message (for thinking models)
        tool_calls : list, optional
            The tool calls associated with the message
        tool_name : str, optional
            Name of the executed tool (for tool role messages)

        Returns
        -------
        Message
            The created message object with only the provided attributes
        """
        # Create base message with only role
        kwargs = {'role': role}

        # Add optional attributes only if provided
        if content is not None:
            kwargs['content'] = content
        if thinking is not None:
            kwargs['thinking'] = thinking
        if tool_calls is not None:
            kwargs['tool_calls'] = tool_calls
        if tool_name is not None:
            kwargs['tool_name'] = tool_name

        return Message(**kwargs)

    def parse_tool_calls(self, response: str):
        """
        Parse tool calls from the model response (for raw mode).
        Returns a list of Message.ToolCall objects matching Ollama's structure.

        Parameters
        ----------
        response : str
            the response from the model

        Returns
        -------
        tool_calls : Sequence[Message.ToolCall] | None
            the list of parsed tool calls in Ollama's format
        """
        # Look for tool call patterns in the response
        tool_call_matches = re.findall(self.tool_call_pattern, response, re.DOTALL)
        if tool_call_matches:
            tool_calls_list = []

            # Iterate over all matches
            for match in tool_call_matches:
                parsed_response = match.strip()
                # Create JSON object from the parsed response
                try:
                    action = json.loads(parsed_response)
                    console.print(f'[cyan]Parsed tool call: {action}[/cyan]')
                except json.JSONDecodeError as e:
                    console.print(
                        f'[yellow]JSON decode error while parsing tool call: {e}[/yellow]')
                    continue

                # Extract tool name and parameters
                try:
                    tool_name = action['name']
                    tool_arguments = action['arguments']

                    # Create ToolCall object matching Ollama's structure
                    tool_call = Message.ToolCall(
                        function=Message.ToolCall.Function(
                            name=tool_name,
                            arguments=tool_arguments
                        )
                    )
                    tool_calls_list.append(tool_call)

                except KeyError as e:
                    console.print(f'[yellow]Error parsing tool call: {e}[/yellow]')
                    continue

            # Check if any tool calls were successfully parsed
            if tool_calls_list:
                # Append the tool calls to the conversation memory
                self.state["messages"].append(
                    Message(
                        role='assistant',
                        tool_calls=tool_calls_list
                    )
                )
                console.print(
                    f'[cyan]Appended {tool_calls_list} tool calls to conversation memory[/cyan]')
                return tool_calls_list
            else:
                # Append the response without tool call to the conversation memory
                self.state["messages"].append(
                    Message(
                        role='assistant',
                        content=response
                    )
                )
                raise ValueError('Found tool call tags but failed to parse them.')
        else:
            # Append the response without tool call to the conversation memory
            self.state["messages"].append(
                Message(
                    role='assistant',
                    content=response
                )
            )
            raise ValueError('No tool call found in the model response.')

    async def invoke(self, state: Messages = None) -> Messages:
        """
        Send the request to the ollama server and return the response.

        The state messages are updated with the new response.

        Parameters
        state : Messages
            List of message objects in the conversation history.

        Returns
        Messages : the updated state with new messages.
        """
        # If a state is provided, use it to replace the current state
        if state is not None:
            self.state = state

        # Check if any of the message roles is 'user' if no user_query is provided
        has_user_message = any(
            (msg.role == 'user' and msg.content is not None)
            for msg in self.state['messages'])
        if not has_user_message:
            raise ValueError(
                'The state must contain at least one user message.')

        # Prepare the prompt
        if self.raw:
            prompt = self.renderer.render(
                state=self.state,
                tools=self.tools)

            if self.debug:
                console.print(Panel(
                    Text(prompt),
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
            response = chat(
                model=self.model,
                stream=False,
                messages=self.state['messages'],
                think=self.think,
                tools=self.tools,
                options=self.options,
            )
        # Check if tool calls are present in the response
        try:
            if self.raw:
                # Parse tool calls from the response (returns Sequence[Message.ToolCall])
                tool_calls = self.parse_tool_calls(response['response'])
            else:
                # Add the assistant message with tool calls to messages
                # Only include attributes that are present in the response
                msg_kwargs = {'role': 'assistant'}

                if hasattr(response.message, 'content') and response.message.content is not None:
                    msg_kwargs['content'] = response.message.content
                if hasattr(response.message, 'thinking') and response.message.thinking is not None:
                    msg_kwargs['thinking'] = response.message.thinking
                if hasattr(response.message, 'tool_calls') and response.message.tool_calls is not None:
                    msg_kwargs['tool_calls'] = response.message.tool_calls

                self.state["messages"].append(Message(**msg_kwargs))
                tool_calls = response.message.tool_calls if hasattr(
                    response.message, 'tool_calls') else None

            if not tool_calls:
                return self.state

            if self.debug:
                console.print(f'[cyan]Tool calls to execute: {len(tool_calls)}[/cyan]')

            for tool_call in tool_calls:
                if self.debug:
                    console.print(f'[cyan]Executing tool: {tool_call.function.name}[/cyan]')
                    console.print(f'[cyan]Arguments: {tool_call.function.arguments}[/cyan]')

                tool_call_done = False

                # First check if the tool is a langchain tool
                if self.lang_tools:
                    for lang_tool in self.lang_tools:
                        if lang_tool['name'] == tool_call.function.name:
                            # Call the langchain tool
                            tool_response = lang_tool['tool_object'].invoke(
                                tool_call.function.arguments
                            )
                            tool_call_done = True
                            break
                # If not found, check if MCP client is available to call the tool
                if not tool_call_done and self.mcp_client is not None:
                    async with self.mcp_client:
                        # Call MCP tool with name and arguments from tool_call.function
                        tool_response = await self.mcp_client.call_tool(
                            tool_call.function.name,
                            arguments=tool_call.function.arguments
                        )
                        tool_call_done = True
                # If tool call was successful, add the response to the conversation memory
                if tool_call_done:
                    if self.debug:
                        console.print(Panel(
                            str(tool_response),
                            title=f'[green bold]TOOL RESPONSE: {tool_call.function.name}[/green bold]',
                            border_style='green',
                            expand=False
                        ))
                    # Add observation to conversation memory
                    content = tool_response.content[0].text \
                        if hasattr(tool_response, 'content') else str(tool_response)
                    self.state['messages'].append(
                        Message(
                            role='tool',
                            content=content
                        )
                    )
                else:
                    if self.debug:
                        console.print(
                            f'[red]Tool {tool_call.function.name} not found or failed to execute[/red]')
                    self.state['messages'].append(
                        Message(
                            role='tool',
                            content=f"Tool {tool_call.function.name} not found or failed to execute"
                        )
                    )
        except (ValueError, AttributeError) as e:
            # No tool calls or error parsing them
            if self.debug:
                console.print(f'[yellow]No tool calls or error: {e}[/yellow]')
            pass
        return self.state

    def reset_memory(self) -> None:
        """Reset the conversation memory."""
        self.state['messages'] = []
