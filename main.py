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
        "'cpu', 'memory', 'display', and 'storage' as keys. Use string values for known "
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
    input_text1 = "The new laptop model features a 3.5 GHz octa-core processor, and 16GB of RAM, a 1TB NVMe SSD, and a 14-inch OLED input."
    input_text2 = "The new smartphone features a 3.2 GHz hexa-core processor, 8GB of RAM, a 256GB UFS storage drive, and a 6.5-inch AMOLED display."
    input_text3 = "The new gaming desktop features a 4.7 GHz 12-core processor, 32GB of RAM, a 2TB NVMe SSD, and a 27-inch QLED monitor."

    input_texts = [input_text1, input_text2, input_text3]

    for input_text in input_texts:
        final_result = full_chain.invoke({"text_input": input_text})
        print("\n--- Final JSON Output ---")
        print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()
