import streamlit as st

# import spacy
from spacy.language import Language
from spacy import displacy

import pandas as pd

# from .util import get_svg, load_model
import util

model = "de_core_news_sm"
# default_text = "Helmut Schmidth war Kanzler von Deutschland."
# default_text = "Hunde sind gut."
default_text = ""

nlp = util.load_model(model)

text = st.text_area("Text to analyze", default_text)

doc = nlp(text)

html = displacy.render(doc, style="ent")
style = "<style>mark.entity { display: inline-block }</style>"
st.write(f"{style}{util.get_html(html)}", unsafe_allow_html=True)

html = displacy.render(doc)
st.write(util.get_svg(html), unsafe_allow_html=True)

df = pd.DataFrame({
    "text": [token.text for token in doc],
    "lemma": [token.lemma_ for token in doc],
    "ent_type": [token.ent_type_ for token in doc],
    "pos": [token.pos_ for token in doc],
    # "tag": [token.tag_ for token in doc],
    "dep": [token.dep_ for token in doc],
    "case": [token.morph.get("Case") for token in doc],
    "gender": [token.morph.get("Gender") for token in doc],
    "number": [token.morph.get("Number") for token in doc],
    "person": [token.morph.get("Person") for token in doc],
    # "mood": [token.morph.get("Mood") for token in doc],
  })

# print(df)

st.dataframe(df)

# st.write(f"""{doc[0].morph}""")