import pandas as pd
import numpy as np
import pickle

credits = pd.read_csv('./dataset/tmdb_5000_credits.csv')
movies_df = pd.read_csv("./dataset/tmdb_5000_movies.csv")

credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_df_merge = movies_df.merge(credits_column_renamed, on='id')

movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])


# NLP CONCEPT
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), stop_words='english')
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])

from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()

def give_rec(title, sig=sig):
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    movie_indices = [i[0] for i in sig_scores]
    return movies_cleaned_df[['original_title', 'genres', 'overview']].iloc[movie_indices].to_json(orient='records')

print(give_rec('Avatar'))


