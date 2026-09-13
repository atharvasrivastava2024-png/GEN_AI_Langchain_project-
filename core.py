from dotenv import load_dotenv
load_dotenv()

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Optional
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# ---------------- MODEL ----------------

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    max_new_tokens=1024,
    temperature=0
)

model = ChatHuggingFace(llm=llm)


# ---------------- PYDANTIC MODEL ----------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]=[]
    rating: Optional[float] = None
    summary: str=""


# ---------------- OUTPUT PARSER ----------------

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are CineSage, an AI assistant that extracts movie
information from a paragraph.

Extract the movie information according to the following format:

{format_instructions}

Only use information available in the paragraph.
If information is not available, use null for optional fields
and an empty list for list fields.
"""
    ),
    (
        "human",
        """Extract the movie information from this paragraph:

{paragraph}
"""
    )
])


# ---------------- USER INPUT ----------------

paragraph = input("ENTER: ")


# ---------------- CREATE PROMPT ----------------

final_prompt = prompt.invoke({
    "paragraph": paragraph,
    "format_instructions": parser.get_format_instructions()
})


# ---------------- MODEL RESPONSE ----------------

response = model.invoke(final_prompt)


# ---------------- PARSE RESPONSE ----------------

try:
    movie = parser.parse(response.content)

    print("\nMovie Information:")
    print(movie)

except Exception as e:
    print("\nCould not parse model output.")
    print("Raw response:")
    print(response.content)
    print("\nError:")
    print(e)
