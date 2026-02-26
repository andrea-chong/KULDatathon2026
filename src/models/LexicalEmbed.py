import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

sentences_df = pd.read_parquet("data/tmp/sentences_df.parquet")

model = SentenceTransformer("gabrielloiseau/CALE-XLLEX", device='cpu')

sentences = sentences_df.embed_input.tolist()
embeddings = model.encode(sentences)

# similarities = model.similarity(embeddings, embeddings)

# Save the embeddings directly as a Parquet or NumPy file
# It's better to keep the word index associated with them
embeddings_df = pd.DataFrame(embeddings, index=sentences_df['token'])
embeddings_df.to_parquet("data/tmp/embeddings.parquet")