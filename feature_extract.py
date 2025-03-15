from preprocessing import *

# initialize SciBERT model which is optimized for scientific text
# https://huggingface.co/allenai/scibert_scivocab_uncased
model = SentenceTransformer('allenai/scibert_scivocab_uncased')

# generate embeddings
embeddings = model.encode(df_papers['processed_text'].tolist())