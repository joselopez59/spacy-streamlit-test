import streamlit as st

import spacy
from spacy.language import Language
from spacy import displacy

import pandas as pd
import base64

model = "de_core_news_sm"
# default_text = "Helmut Schmidth war Kanzler von Deutschland."
default_text = "Hunde sind gut."

@st.cache_data
def load_model(name: str) -> spacy.language.Language:
  """Load a spaCy model."""
  return spacy.load(name)

@st.cache_data
def process_text(model_name: str, text: str) -> spacy.tokens.Doc:
  """Process a text and create a Doc object."""
  nlp = load_model(model_name)
  return nlp(text)

def get_svg(svg: str, style: str = "", wrap: bool = True):
  """Convert an SVG to a base64-encoded image."""
  b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
  html = f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
  return get_html(html) if wrap else html

def get_html(html: str):
  """Convert HTML so it can be rendered."""
  WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
  # Newlines seem to mess with the rendering
  html = html.replace("\n", " ")
  return WRAPPER.format(html)

nlp = load_model(model)

doc = nlp(default_text)

html = displacy.render(doc)

st.write(get_svg(html), unsafe_allow_html=True)



df = pd.DataFrame({
    "text": [token.text for token in doc],
    "lemma": [token.lemma_ for token in doc],
    "pos": [token.pos_ for token in doc],
    "tag": [token.tag_ for token in doc],
    "dep": [token.dep_ for token in doc],
    "ent_type": [token.ent_type_ for token in doc],
    "number": [token.morph.get("Number") for token in doc],
    "person": [token.morph.get("Person") for token in doc],
    "case": [token.morph.get("Case") for token in doc],
    "gender": [token.morph.get("Gender") for token in doc],
    "mood": [token.morph.get("Mood") for token in doc],
  })

print(df)

st.dataframe(df)

st.write(f"""{doc[0].morph}""")