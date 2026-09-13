import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎬 CineSage")
st.subheader("AI Movie Information Extractor")

st.write(
    "Enter a movie description and CineSage will extract "
    "structured information using an AI model."
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        max_new_tokens=1024,
        temperature=0
    )

    return ChatHuggingFace(llm=llm)


model = load_model()


# --------------------------------------------------
# PYDANTIC MODEL
# --------------------------------------------------

class Movie(BaseModel):

    title: str

    release_year: Optional[int] = None

    genre: List[str]

    director: Optional[str] = None

    cast: List[str] = []

    rating: Optional[float] = None

    summary: str = ""


# --------------------------------------------------
# OUTPUT PARSER
# --------------------------------------------------

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """You are CineSage, an AI assistant that extracts movie
information from a paragraph.

Extract the movie information according to the following format:

{format_instructions}

Only use information available in the paragraph.

If information is not available:
- use null for optional fields
- use an empty list for list fields
- use an empty string for summary

Return only the structured output.
"""
    ),

    (
        "human",
        """Extract the movie information from this paragraph:

{paragraph}
"""
    )
])


# --------------------------------------------------
# UI
# --------------------------------------------------

st.markdown("### 📝 Movie Description")

paragraph = st.text_area(
    "Enter a movie description",
    height=200,
    placeholder="""Example:
The Last Signal is a 2022 psychological thriller directed by
Daniel Brooks. It stars Ethan Cole as a radio technician who
receives a mysterious distress signal from the future."""
)


# --------------------------------------------------
# BUTTON
# --------------------------------------------------

if st.button("🎬 Analyze Movie", use_container_width=True):

    if not paragraph.strip():

        st.warning("Please enter a movie description.")

    else:

        with st.spinner("Analyzing movie..."):

            try:

                # Create prompt
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions":
                        parser.get_format_instructions()
                })

                # Get response
                response = model.invoke(final_prompt)

                # Parse response
                movie = parser.parse(response.content)


                # --------------------------------------------------
                # DISPLAY RESULT
                # --------------------------------------------------

                st.success("Movie information extracted successfully!")

                st.markdown("## 🎥 Movie Information")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### Basic Information")

                    st.write("**Title:**", movie.title)

                    st.write(
                        "**Release Year:**",
                        movie.release_year
                        if movie.release_year
                        else "Not mentioned"
                    )

                    st.write(
                        "**Director:**",
                        movie.director
                        if movie.director
                        else "Not mentioned"
                    )

                    st.write(
                        "**Rating:**",
                        movie.rating
                        if movie.rating
                        else "Not mentioned"
                    )


                with col2:

                    st.markdown("### 🎭 Genre")

                    if movie.genre:
                        for genre in movie.genre:
                            st.write(f"• {genre}")
                    else:
                        st.write("Not mentioned")


                    st.markdown("### 👥 Cast")

                    if movie.cast:
                        for actor in movie.cast:
                            st.write(f"• {actor}")
                    else:
                        st.write("Not mentioned")


                st.markdown("### 📖 Summary")

                st.info(movie.summary)


                # --------------------------------------------------
                # RAW STRUCTURED OUTPUT
                # --------------------------------------------------

                with st.expander("🔍 View Structured Output"):

                    st.json(movie.model_dump())


                # --------------------------------------------------
                # RAW MODEL RESPONSE
                # --------------------------------------------------

                with st.expander("🤖 View Raw AI Response"):

                    st.code(
                        response.content,
                        language="json"
                    )


            except Exception as e:

                st.error("Could not extract movie information.")

                with st.expander("Debug Information"):

                    st.write(e)

                    if "response" in locals():
                        st.code(response.content)
                        