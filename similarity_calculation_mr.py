#!/usr/bin/env python
# coding: utf-8
# using most recent publication of researchers as input to generate user profiles

import sys
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd

# load pre-train model
model = '/Users/sherry/Downloads/window_5/window_5.model.bin'
w2v_model = KeyedVectors.load_word2vec_format(model, binary=True)

# read all candidate papers info, contain two columns: paper ID and paper content
candidate_paper_df = pd.read_csv('/Users/sherry/Downloads/candidate_papers.csv')

# define DocSim class to calculate document similarities
class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = str(doc)
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self,user_profile,candidate_papers,threshold=0):
      # Calculate similarity between a given source document in user profile 
      # and all target documents in candidate papers 
      # candidate_papers is dataframe, user_profile is a one-line string

      # rename columns in user_profile and candidate_papers    
        candidate_papers.columns = ['paperID', 'paperText']

      # convert dataframe to dict
        candidate_paper_dict = candidate_papers.set_index('paperID').to_dict()

      # for each user profile doc as source doc, calculate similarity with each 
      # target doc
        source_doc = str(user_profile)
        source_vec = self.vectorize(source_doc)
        result = []
        i = 1
        for paperID,paperText in candidate_paper_dict['paperText'].items():
            target_doc = str(paperText)
            target_vec = self.vectorize(target_doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                result.append([paperID,sim_score])
        # Sort results by similar scores in desc order
        result.sort(key=lambda k : k[1] , reverse=True)
        return result       

ds = DocSim(w2v_model)

ranking = [1,2,3,4,5,6,7,8,9,10]
new_df = pd.DataFrame()
new_df.insert(0,'ranking',ranking)
for i in range(1,10,1):
    r = 'R'+ str(i)
    with open('user_profiles/user_profile_after_text_cleaning/cleaned_R{}-1.txt'.format(i), 'r') as f:
        user_profile = f.read()
  # cumputing sim scores
    sim_scores = ds.calculate_similarity(user_profile, candidate_paper_df)
    df = pd.DataFrame(sim_scores)
    df.columns = ['paperID', 'sim_score']
    # get the top-10 ranklist
    df = df.head(10)
    new_df[r] = df.iloc[:, 0]
# save ranking results for all researchers in rangking.csv
new_df.to_csv('rank_result_rm/ranking.csv', index=False)