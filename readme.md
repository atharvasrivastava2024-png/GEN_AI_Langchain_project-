# 🎬 CineSage — AI Movie Information Extractor

CineSage is an AI-powered movie information extraction application built using **LangChain, Hugging Face, Pydantic, and Streamlit**.

It takes an unstructured movie description and extracts important movie information into a structured format.

---

## 🚀 Features

- 🎥 Extracts movie title
- 📅 Extracts release year
- 🎭 Identifies movie genres
- 🎬 Extracts director information
- 👥 Extracts cast members
- ⭐ Extracts movie rating when available
- 📖 Generates a clean movie summary
- 🧩 Uses Pydantic for structured output
- 🤖 Uses Hugging Face for LLM inference
- 🔗 Uses LangChain for prompt management and output parsing
- 🖥️ Provides an interactive Streamlit UI

---

## 🏗️ Architecture

```text
                Movie Description
                       │
                       ▼
              Streamlit Interface
                       │
                       ▼
             ChatPromptTemplate
                 ┌─────┴─────┐
                 │           │
              System       Human
               Prompt       Input
                 │           │
                 └─────┬─────┘
                       ▼
                Hugging Face LLM
                       │
                       ▼
             PydanticOutputParser
                       │
                       ▼
                 Movie Object
                       │
                       ▼
                Streamlit UI