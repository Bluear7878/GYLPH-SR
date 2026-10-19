# Modified for GLYPH-SR: retain only the released Llama3 image template; use the loaded tokenizer.
"""Conversation formatting for the released Llama3-LLaVA-NeXT checkpoint."""

import dataclasses
from enum import Enum
from typing import Any, List, Union


class SeparatorStyle(Enum):
    LLAMA_3 = 7


@dataclasses.dataclass
class Conversation:
    """The released Llama3 image conversation."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.LLAMA_3
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    tokenizer_id: str = ""
    tokenizer: Any = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)
        if self.tokenizer is None:
            raise ValueError(
                "Llama 3 tokenizer is not available. Make sure you have the necessary permissions."
            )
        chat_template_messages = [{"role": "system", "content": self.system}]
        for role, message in messages:
            if message:
                if type(message) is tuple:
                    message, images = message
                    message = "<image>" * len(images) + message
                chat_template_messages.append({"role": role, "content": message})
        return self.tokenizer.apply_chat_template(
            chat_template_messages, tokenize=False, add_generation_prompt=True
        )

    def append_message(self, role, message):
        self.messages.append([role, message])


conv_llava_llama_3 = Conversation(
    system="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    roles=("user", "assistant"),
    version="llama_v3",
    messages=[],
    offset=0,
    sep="<|eot_id|>",
    sep_style=SeparatorStyle.LLAMA_3,
    tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer=None,
    stop_token_ids=[128009],
)

conv_templates = {"llava_llama_3": conv_llava_llama_3}
