import streamlit as st

import spacy
from spacy.language import Language
from spacy import displacy

import pandas as pd

import util

model = "de_core_news_sm"
nlp = util.load_model(model)

default_text = """Das Bundeskabinett 2024 hat den Gesetzentwurf für die Umstellung von Heizungen auf Erneuerbare Energien gebilligt in Deutschland auf Deutsch."""
# default_text = """Helmut Schmidth war Kanzler von Deutschland."""
# default_text = "Hunde sind gut."
# default_text = """Die wirtschaftliche Lage des Landes war angespannt, seit dem Krieg anfing."""
# default_text = """Gehts du nach dem Essen in die frische Luft spazieren?"""

input_text = st.text_area('Text to analyze:', default_text, help='Test')
# st.write('The current movie title is', title)

doc = nlp(input_text)

# st.divider()

st.markdown("**Entitäten Erkennung**")
html = displacy.render(doc, style="ent")
style = "<style>mark.entity { display: inline-block }</style>"
st.write(f"{style}{util.get_html(html)}", unsafe_allow_html=True)

# df = pd.DataFrame({
#     "Entität": [t.text for t in doc.ents],
#     "Typ": [t.label_ for t in doc.ents],
# })
# st.dataframe(df)

# Grammatik
st.markdown("**Grammatik**")

df = pd.DataFrame({
    "text": [token.text for token in doc],
    "pos": [token.pos_ for token in doc],
    "explain": [spacy.explain(token.pos_) for token in doc],
    "case": [token.morph.get("Case") for token in doc],
    "gender": [token.morph.get("Gender") for token in doc],
    "number": [token.morph.get("Number") for token in doc],
    "person": [token.morph.get("Person") for token in doc],
    })

st.dataframe(df)

# SVG
html = displacy.render(doc)
st.write(util.get_svg(html), unsafe_allow_html=True)


# TEXT PREPROCESSING
cleaned_list = [token.text for token in doc if not token.is_stop and not token.is_punct]
# for token in doc_cleaned:
#   print(token.text)
cleaned_str = ' '.join(cleaned_list)
st.markdown("**Text Cleaning** (Removed stop words and punctuation)")
st.write(cleaned_str)

