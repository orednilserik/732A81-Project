from fetch import *

nlp = spacy.load("en_core_sci_sm") # scispacy for biomedical preprocessing of data


def preprocess_text(text):

    doc = nlp(text.lower())

    words = []
    for token in doc:
      if token.ent_type_: # named entities in the corpus are always kept
            words.append(token.text)
      # for non-entities preprocess as usual for punctuation, space, word length and stopword
      elif (not token.is_punct and not token.is_space and
              len(token.text) > 2 and not token.is_stop): #and token.text not in custom_stopwords):
            words.append(token.text)

    return ' '.join(words)

# apply preprocessing on abstract
df_papers['processed_text'] = df_papers['abstract'].apply(preprocess_text)