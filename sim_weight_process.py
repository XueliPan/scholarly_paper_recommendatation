# python 3.7
# -*- coding: utf-8 -*-

import sys
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from datetime import datetime
import time
from threading import Thread
from multiprocessing import Process
import multiprocessing



# # load pre-train model on my own corpus
# model = '/content/drive/My Drive/scholarly_paper_recommendatation/models/window_5.bin'
# w2v_model = KeyedVectors.load_word2vec_format(model, binary=True)

# # read all candidate papers info, contain two columns: paper ID and paper content
# candidate_paper_df = pd.read_csv('/content/drive/My Drive/scholarly_paper_recommendatation/large data/candidate_papers.csv')

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
      # Computing similarity between a given source document in user profile
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

    def compute_sim_all_pubs(self, user_profile, candidate_papers,threshold=0):
        """
        Computing similarity between several given source documents in user profile (with equal weight) and all target
        documents in candidate
        papers
        :param user_profile: a list, all source docs of a researcher that used to construct one user profile
        :param candidate_papers: a dataframe, all target docs that used as candidate recommend doc
        :param threshold: filter recommend items according to threshold
        :return: Sort rank results by similar scores in desc order
        """
        # rename columns in user_profile and candidate_papers
        candidate_papers.columns = ['paperID', 'paperText']

        # convert dataframe to dict
        candidate_paper_dict = candidate_papers.set_index('paperID').to_dict()

        # for each user, source_doc_ls contains all his/her publications
        source_docs_vec_ls = []
        for pubished_seq,source_doc in enumerate(user_profile):
            # weight doc vector based on the order of published year
            c = 0.1
            source_doc_vec = self.vectorize(source_doc)*(1/np.log2(pubished_seq+1+c))
            # add each source doc vector into list source_docs_vec_ls
            source_docs_vec_ls.append(source_doc_vec)
        # compute user profile vector for each researcher based on all their publications with equal weight
        user_profile_vec = np.sum(source_docs_vec_ls,axis = 0)/len(source_docs_vec_ls)

        rank_result = []
        i = 1
        for paperID,paperText in candidate_paper_dict['paperText'].items():
            target_doc = str(paperText)
            target_vec = self.vectorize(target_doc)
            sim_score = self._cosine_sim(user_profile_vec, target_vec)
            if sim_score > threshold:
                rank_result.append([paperID,sim_score])
        # Sort results by similar scores in desc order
        rank_result.sort(key=lambda k : k[1] , reverse=True)
        return rank_result

global w2v_model,user_statistics_df,candidate_paper_df,output_dir

model_path = '/Users/sherry/Downloads/GoogleNews-vectors-negative300.bin.gz'
user_statistics_path = '/Users/sherry/git_project/scholarly_paper_recommendatation/user_profiles/user_profiles_statistics.csv'
candidate_paper_path = '/Users/sherry/Downloads/candidate_papers.csv'
output_dir =  'rank_result_weight/'

w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
# read all candidate papers info, contain two columns: paper ID and paper content
candidate_paper_df = pd.read_csv(candidate_paper_path)
# get the list of number of publications for each researcher
user_statistics_df = pd.read_csv(user_statistics_path)



def process_job(n):
  print('Process{} start\n'.format(n))
  # load pre-train model on my Google News corpus locally
  # w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
  ds = DocSim(w2v_model)

  # # get the list of number of publications for each researcher
  # user_statistics_df = pd.read_csv(user_statistics_path)
  num_pubs_ls = user_statistics_df.iloc[:, 1].tolist()
  new_df = pd.DataFrame()
  ranking = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  new_df.insert(0, 'ranking', ranking)
  # # read all candidate papers info, contain two columns: paper ID and paper content
  # candidate_paper_df = pd.read_csv(candidate_paper_path)

  # reverse all researchers publications
  for i in range(n, n + 9, 1):
      r = 'R' + str(i)
      print(datetime.now())
      user_profile = []

      # reverse all publications of one researcher, get a list of
      print('number of publication for researcher {} is {}'.format(r, num_pubs_ls[i - 1]))
      for j in range(1, num_pubs_ls[i - 1] + 1):
          with open('user_profiles/user_profile_after_text_cleaning/cleaned_R{}-{}.txt'.format(i, j), 'r') as f:
              each_doc = f.read()  # each_doc is a string

          # all source docs of a researcher that used to construct his/her user profile
          user_profile.append(each_doc)
      print('the len of user_profile list for this researcher is: {}'.format(len(user_profile)))

      # computing sim scores
      sim_scores = ds.compute_sim_all_pubs(user_profile, candidate_paper_df)
      df = pd.DataFrame(sim_scores)
      df.columns = ['paperID', 'sim_score']

      # get the top-10 rank list
      df = df.head(10)
      new_df[r] = df.iloc[:, 0]
      print(datetime.now())

      # save ranking results for all researchers
      new_df.to_csv('{}weight_GN_{}.csv'.format(output_dir, r), index=False)
  new_df.to_csv('{}weight_GN_{}.csv'.format(output_dir, r), index=False)
  print('Process{} finished\n'.format(n))

  return new_df


def multicore():
  print('start\n')
  startTime = time.time()
  pool = multiprocessing.Pool(processes=4)
  res = pool.map(process_job, range(15,51,9))
  pool.close()
  pool.join()
  endTime = time.time()
  print("time :", endTime - startTime)
  print(res)


if __name__=='__main__':
    multicore()

# def main():
#     print('start\n')
#     p1 = Process(target=job, name='R15+6',args=(15,)) #name this thread as T1
#     p1.start()
#     p2 = Process(target=job, name='R21+6',args=(21,)) #name this thread as T1
#     p2.start()
#     p3 = Process(target=job, name='R27+6',args=(27,)) #name this thread as T1
#     p3.start()
#     p1.join()
#     p2.join()
#     p3.join()
#     print('finished\n')

# if __name__=='__main__':
#     main()