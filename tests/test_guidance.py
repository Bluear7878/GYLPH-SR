import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from glyph_sr import guidance


class Conversation:
    roles = ("user", "assistant")

    def __init__(self):
        self.messages = []

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        return repr(self.messages)


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (24, 16))
        self.images = [torch.zeros(3, 16, 24)]
        self.templates = {"test": Conversation()}
        self.model = Mock()
        self.model.generate.return_value = [torch.tensor([[10, 11]])]
        self.tokenizer = Mock()
        self.tokenizer.decode.return_value = "  scene caption"
        self.token_patch = patch.object(
            guidance, "_tokenizer_image_token", return_value=torch.tensor([1, 2])
        )
        self.tokenizer_image = self.token_patch.start()
        self.addCleanup(self.token_patch.stop)

    def test_caption_preserves_prompt_and_generation_options(self):
        caption = guidance.describe_image(
            self.images,
            self.image,
            self.model,
            self.tokenizer,
            "caption request",
            self.templates,
            -200,
            conv_template="test",
            device="cpu",
            temperature=0.3,
            max_new_tokens=17,
        )
        self.assertEqual(caption, ["scene caption"])
        self.assertIn("caption request", self.tokenizer_image.call_args.args[0])
        args, kwargs = self.model.generate.call_args
        self.assertEqual(args[0].shape, (1, 2))
        self.assertEqual(kwargs["image_sizes"], [(24, 16)])
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["max_new_tokens"], 17)
        self.assertEqual(self.templates["test"].messages, [])

    def test_ocr_keeps_leading_spaces_but_removes_leading_newlines(self):
        self.tokenizer.decode.return_value = "\n\n STOP\nSHOP"
        text, output = guidance.generate_ocr_text(
            "ocr request",
            self.images,
            [(24, 16)],
            self.model,
            self.tokenizer,
            self.templates,
            "test",
            -200,
            "cpu",
            max_new_tokens=19,
        )
        self.assertEqual(text, " STOP\nSHOP")
        self.assertIs(output, self.model.generate.return_value)
        kwargs = self.model.generate.call_args.kwargs
        self.assertEqual(kwargs["images"][0].dtype, torch.float16)
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["max_new_tokens"], 19)

    def test_location_deduplication_and_output_text_are_preserved(self):
        self.tokenizer.decode.side_effect = [" first position ", " second position "]
        text = guidance.describe_text_locations(
            self.images,
            "STOP, STOP go\nstop go\nSHOP",
            self.image,
            self.model,
            self.tokenizer,
            "{DEFAULT_IMAGE_TOKEN} Locate {pred_text}",
            self.templates,
            guidance.DEFAULT_IMAGE_TOKEN,
            guidance.IMAGE_TOKEN_INDEX,
            conv_template="test",
            device="cpu",
        )
        self.assertEqual(
            text,
            "There are a total of 2 words in the image. first position second position",
        )
        self.assertEqual(self.model.generate.call_count, 2)
        prompts = [call.args[0] for call in self.tokenizer_image.call_args_list]
        self.assertIn("Locate STOP go", prompts[0])
        self.assertIn("Locate SHOP", prompts[1])

    def test_empty_ocr_never_calls_the_model(self):
        text = guidance.describe_text_locations(
            self.images,
            "  ",
            self.image,
            self.model,
            self.tokenizer,
            "unused",
            self.templates,
            "<image>",
            -200,
            conv_template="test",
            device="cpu",
        )
        self.assertEqual(text, "  ")
        self.model.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
