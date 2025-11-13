import json
import re
from typing import TypedDict, NotRequired
from ollama import generate
from jinja2 import Template
from fastmcp import Client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class Message(TypedDict):
    '''
    This class contains all the message fields required to render model prompt templates.
    
    Attributes:
        role (str): The role of the message (system, user, assistant, tool)
        content (str): The content of the message
        reasoning_content (str): The reasoning content of the message (for thinking models)
        tool_calls (list): The tool calls associated with the message
    '''
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
    '''
    This class is responsible for communicating with the Ollama Server.
    The conversation memory is stored inside this class.
    '''

    def __init__(self, 
                 model: str = 'qwen3:0.6b',
                 tools: list = [],
                 tool_call_pattern: str = "<tool_call>(.*?)</tool_call>",
                 mcp_client: Client = None, 
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
        '''
        Initialize the Ollama class.
        Parameters
        ----------
        model       :   str
            the model name
        tools       :   list
            the list of tools available to the model
        state       :   Messages
            the conversation history state
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
        Returns
        -------
        '''
        self.model = model 
        self.tools = tools
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
            raise ValueError("If raw mode is true, a jinja template must be provided for prompt " + 
                             "formatting. The jinja template only applies in raw mode.")
        
        # Initialize conversation state with system prompt
        self.state: Messages = {
            'messages': [
                self.create_message(
                    role='system',
                    content=self.system_prompt
                )
            ]
        }

    def create_message(self, role: str, content: str = '', reasoning_content: str = '', 
                       tool_calls: list = None) -> Message:
        '''
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
        '''
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
        '''
        Parse tool calls from the model response.

        Parameters
        response : str
            the response from the model

        Returns
        tool_calls : list
            the list of parsed tool calls
        '''
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
                    console.print(f"[red]JSON decode error while parsing tool call: {e}[/red]")
                    continue
                # Extract tool name and parameters
                try:
                    tool_name = action["name"]
                    tool_arguments = action["arguments"]
                    # Append the tool call to the list
                    tool_calls_list.append({
                        "tool_name": tool_name,
                        "tool_arguments": tool_arguments,
                        "raw": parsed_response
                    })
                    all_actions.append(action)
                except KeyError as e:
                    console.print(f"[red]Error parsing tool call: {e}[/red]")
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
                raise ValueError("Found tool call tags but failed to parse them.")
        else:
            # Append the response without tool call to the conversation memory
            self.state['messages'].append(self.create_message(
                    role='assistant',
                    content=response))
            raise ValueError("No tool call found in the model response.")


    async def invoke(self, user_query: str = '', state: Messages = None):
        '''
        Send the request to the ollama server and return the response.
        The state messages are updated with the new response.

        Parameters
        user_query : str
            the user query to send to the model. If empty, the last user message in state is used.
        state : Messages
            Optional Messages object containing conversation history. If provided, replaces current state.

        Returns
        Messages : the updated state with new messages.        
        '''
        # If a state is provided, use it to replace the current state
        if state is not None:
            self.state = state
        
        # Check if any of the message roles is 'user' if no user_query is provided
        has_user_message = any((msg['role'] == 'user' and msg.get('content', '')) for msg in self.state['messages'])
        if not user_query and not has_user_message:
            raise ValueError("If no user query is provided, the state must contain at least one user message.")
        elif not has_user_message and user_query:
            # Add user message to conversation memory
            self.state['messages'].append(self.create_message(
                role='user',
                content=user_query)
            )
        elif has_user_message and user_query:
            console.print("[yellow]Warning: Both user_query and user message in state provided. Ignoring user_query.[/yellow]")

        # Prepare the prompt
        if self.raw:
            prompt = self.template.render(
                messages=self.state['messages'],
                tools=self.tools,
                add_generation_prompt=True,
                enable_thinking=self.think
            )

            # Uncomment to see rendered prompt
            # console.print(Panel(
            #     prompt,
            #     title="[cyan bold]RENDERED PROMPT[/cyan bold]",
            #     border_style="cyan",
            #     expand=False
            # ))

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
                    title="[yellow bold]RAW RESPONSE TEXT[/yellow bold]",
                    border_style="yellow",
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
            tool_calls = self.parse_tool_calls(response['response'])
            # If MCP client is initialized, call the tool via MCP
            if self.mcp_client is not None:
                async with self.mcp_client:
                    for tool_call in tool_calls:
                        tool_response = await self.mcp_client.call_tool(
                            tool_call['tool_name'],
                            tool_call['tool_arguments']
                        )
                        if self.debug:
                            console.print(Panel(
                            str(tool_response),
                            title="[green bold]TOOL RESPONSE[/green bold]",
                            border_style="green",
                            expand=False
                        ))
                        # Add observation to conversation memory
                        self.state['messages'].append(self.create_message(
                            role='tool',
                            content=tool_response.content[0].text)
                        )
        except ValueError:
            pass
        return self.state
    
    def reset_memory(self):
        '''
        Reset the conversation memory.
        Parameters
        None

        Returns
        None
        '''
        self.state['messages'] = []


# Uncomment the following code to run a standalone test of the Ollama class.
# IMPORTANT: 
#       - Make sure to have an Ollama server running and accessible.
#       - Make sure to have an MCP server running and accessible.
#       - Fake fake_mcp_server.py is provided in case you need a mock server.
#       - Update the mcp_config with your MCP server details if using a real server.
#       - max_steps defines the maximum number of steps for the conversation.
#       - qwen3:0.6b model is used in this example, ensure it's available on your Ollama server.

# async def main():

#     mcp_config = {
#         "mcpServers": {
#             "mcp_server": {"url": "http://localhost:3002/mcp"}
#         }
#     }

#     system_prompt = ("You are a polite and efficient home assistant."
#     "You can control home devices such as lights, doors, blinds, temperature, music, and presence sensors using the provided tools."
#     "If a user asks for an action, respond only by calling the right tool in JSON format inside <tool_call> tags."
#     "When the task is done, return the final answer.")

#     max_steps = 5

#     async with Client(mcp_config) as mcp_client:
#         tools = await mcp_client.list_tools()
#         tools_schemas = []
#         for tool in tools:
#             tools_schemas.append({
#                 "name": tool.name,
#                 "description": tool.description,
#                 "inputSchema": tool.inputSchema,
#             })
#         user_query = "Please turn on the living room lights and unlock the garage door."
#         ollama = Ollama(
#             model='qwen3:0.6b',
#             tools=tools_schemas,
#             tool_call_pattern="<tool_call>(.*?)</tool_call>",
#             mcp_client=mcp_client,
#             think=False,
#             raw=True,
#             temperature=0.0,
#             jinja_template_path='../templates/qwen3.jinja',
#             system_prompt=system_prompt,
#             debug=True
#         )
#         console.print(Panel(
#                     user_query,
#                     title="[dark_orange bold]User query[/dark_orange bold]",
#                     border_style="dark_orange"
#                 ))
#         import time
#         init_time = time.time()
#         for step in range(max_steps):
#             init_step_time = time.time()
#             console.rule(f"[bold white]STEP {step+1}/{max_steps}[/bold white]", style="white")

#             state = await ollama.invoke(user_query=user_query)
            
#             console.print(f"[red bold]Step {step+1} completed in {time.time() - init_step_time:.3f} seconds.[/red bold]")
            
#             # Check if the last message contains a tool call
#             if state['messages'][-1]['role'] == 'tool':
#                 # Plot all messages in conversation
                
#                 # Create table to display conversation messages
#                 table = Table(title=f"Conversation Messages - Step {step+1}", show_header=True, header_style="bold magenta")
#                 table.add_column("Index", style="cyan", width=8)
#                 table.add_column("Role", style="yellow", width=12)
#                 table.add_column("Content", style="white", width=50)
#                 table.add_column("Reasoning", style="blue", width=30)
#                 table.add_column("Tool Calls", style="green", width=30)
                
#                 for idx, msg in enumerate(state['messages']):
#                     table.add_row(
#                         str(idx),
#                         str(msg['role']),
#                         str(msg.get('content', ''))[:47] + "..." if len(str(msg.get('content', ''))) > 50 else str(msg.get('content', '')),
#                         str(msg.get('reasoning_content', ''))[:27] + "..." if len(str(msg.get('reasoning_content', ''))) > 30 else str(msg.get('reasoning_content', '')),
#                         str(msg.get('tool_calls', []))[:27] + "..." if len(str(msg.get('tool_calls', []))) > 30 else str(msg.get('tool_calls', []))
#                     )
                
#                 console.print(table)

#                 console.print(f"\n[yellow]Tool call detected, continuing to next step...\n[/yellow]")
#             else:
#                 console.print(f"\n[green]No tool call detected, ending interaction...[/green]")
#                 console.print(Panel(
#                     state['messages'][-1].get('content', ''),
#                     title="[green bold]Final response from assistant[/green bold]",
#                     border_style="green"
#                 ))
#                 break
#         console.print(f"[bold green]Interaction completed in {time.time() - init_time:.3f} seconds.[/bold green]")
#         ollama.reset_memory()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

