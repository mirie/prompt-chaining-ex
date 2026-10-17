"""Extract laptop specifications and turn them into JSON using two prompts."""

import json
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_chain(llm):
    """Feed the extraction response into the JSON transformation prompt."""
    prompt_extract = ChatPromptTemplate.from_template(
        "Extract the technical specifications from the following text. "
        "Use only information present in the text:\n\n{text_input}"
    )
    prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following specifications into a JSON object with "
        "'cpu', 'memory', and 'storage' as keys. Use string values for known "
        "specifications and null for missing ones. Return only valid JSON, "
        "without Markdown formatting:\n\n{specifications}"
    )

    extraction_chain = prompt_extract | llm | StrOutputParser()
    return (
        {"specifications": extraction_chain}
        | prompt_transform
        | llm
        | JsonOutputParser()
    )


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file.")

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0
    )
    full_chain = build_chain(llm)
    input_text = (
        "The new laptop model features a 3.5 GHz octa-core processor, "
        "16GB of RAM, and a 1TB NVMe SSD."
    )
    final_result = full_chain.invoke({"text_input": input_text})
    print("\n--- Final JSON Output ---")
    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()
