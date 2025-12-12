from typing import TypedDict, Optional
from jinja2 import Template
from ollama import Message

# class Message(SubscriptableBaseModel):
#   """
#   Chat message.
#   """

#   role: str
#   "Assumed role of the message. Response messages has role 'assistant' or 'tool'."

#   content: Optional[str] = None
#   'Content of the message. Response messages contains message fragments when streaming.'

#   thinking: Optional[str] = None
#   'Thinking content. Only present when thinking is enabled.'

#   images: Optional[Sequence[Image]] = None
#   """
#   Optional list of image data for multimodal models.

#   Valid input types are:

#   - `str` or path-like object: path to image file
#   - `bytes` or bytes-like object: raw image data

#   Valid image formats depend on the model. See the model card for more information.
#   """

#   tool_name: Optional[str] = None
#   'Name of the executed tool.'

#   class ToolCall(SubscriptableBaseModel):
#     """
#     Model tool calls.
#     """

#     class Function(SubscriptableBaseModel):
#       """
#       Tool call function.
#       """

#       name: str
#       'Name of the function.'

#       arguments: Mapping[str, Any]
#       'Arguments of the function.'

#     function: Function
#     'Function to be called.'

#   tool_calls: Optional[Sequence[ToolCall]] = None
#   """
#   Tools calls to be made by the model.
#   """

class Messages(TypedDict):
    """
    Class to hold conversation messages for LangGraph workflow.

    Attributes:
        messages (list[Message]): List of message objects in the conversation.
    """

    messages: list[Message]

class TemplateRenderer:
    """
    Renders chat prompts using Jinja2 templates for different LLMs.
    Attributes:
        template_type (str): Type of the template (e.g., 'mistral', 'qwen3', 'llama').
        template (Template): Jinja2 template object.
        render_params (dict): Parameters for rendering the template.
    Methods:
        render(state: Messages, tools: Optional[list] = None) -> str:
            Renders the template with the given state and tools.
    """
    
    def __init__(self, 
                 template_type: str,
                 think: bool = False):
        self.template_type = template_type
        self.available_templates = {
            "mistral": {
                "template_path": "../templates/mistral.jinja",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "pad_token": ""
            },
            "qwen3": {
                "template_path": "../templates/qwen3.jinja",
                "eos_token": "<|im_end|>",
                "bos_token": "",
                "pad_token": "<|endoftext|>"
            },
            "llama": {
                "template_path": "../templates/llama.jinja",
                "eos_token": "<|end_of_text|>",
                "bos_token": "<|begin_of_text|>",
                "pad_token": "<|end_of_text|>"
            }
        }
        if template_type not in self.available_templates:
            raise ValueError(
                f"Template type '{template_type}' is not supported." + 
                f" Supported types are: {list(self.available_templates.keys())}")

        try:
            with open(self.available_templates[template_type]["template_path"], 'r') as f:
                template_content = f.read()
                self.template = Template(template_content)
                self.template.globals['raise_exception'] = self._raise_exception
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Template file not found at path: {self.available_templates[template_type]['template_path']}")

        # Generate render params dict
        self.render_params = {
            'messages': [],
            'tools': [],
            'add_generation_prompt': True,
            'enable_thinking': think,
            'bos_token': self.available_templates[template_type]["bos_token"],
            'eos_token': self.available_templates[template_type]["eos_token"],
            'pad_token': self.available_templates[template_type]["pad_token"]
        }
        

    def _raise_exception(message: str):
        """Helper function for Jinja2 template to raise exceptions."""
        raise ValueError(message)
    
    def render(self, state: Messages, 
               tools: Optional[list] = None) -> str:
        # Populate render params
        self.render_params["messages"] = state['messages']
        self.render_params["tools"] = tools if tools is not None else []

        return self.template.render(**self.render_params)