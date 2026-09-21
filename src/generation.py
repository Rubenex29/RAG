# Standard library
from typing import List, cast

# Third-party dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenModel:
    """Load Qwen and generate answers from retrieved snippets."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """Initialize the configured Qwen model and tokenizer."""

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    def generate_answer(self, query: str, snippets: List[str]) -> str:
        """Generate a concise answer grounded in the supplied snippets."""

        context = "\n\n".join(
            f"[Snippet {i + 1}]\n{snippet[:600]}"
            for i, snippet in enumerate(snippets)
        )
        prompt = (
            "Answer the question using the provided context.\n\n"
            "        Use the context as your main source of information. "
            "You may combine information\n"
            "        from different parts of the context and make "
            "reasonable logical inferences.\n\n"
            "        Do not introduce facts that are unrelated to or "
            "unsupported by the context.\n"
            "        If the context does not contain enough information "
            "to answer the question,\n"
            '        say: "I don\'t have enough information in the provided '
            'context to answer this question."\n\n'
            "        Answer the question directly and concisely.\n\n"
            "        Question:\n"
            f"        {query}\n\n"
            "        Context:\n"
            f"        {context}\n\n"
            "        Answer:\n"
            "        "
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=150,
        )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):]

        answer = cast(
            str,
            self.tokenizer.decode(output_ids, skip_special_tokens=True),
        ).strip()

        return answer
