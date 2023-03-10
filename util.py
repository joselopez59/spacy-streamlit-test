import streamlit as st
import spacy
import base64

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