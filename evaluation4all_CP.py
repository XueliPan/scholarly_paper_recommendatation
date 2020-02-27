# python 3.7
# -*- coding: utf-8 -*-
# Computing NDCG@10, P@10 and MRR when using user's most recent publication and word2vec model trained on scientific
# papers corpus to generate top-10 recommend papers to each researcher
# rank_result file:rank_result_rm/rank_result_mr_own_corpus.csv
# ground_truth file: user_profiles/ground_truth.csv


import re
import pandas as pd
import numpy as np

def standarize(input_str):
    """
    e.g:
    change "/AAMAS/AAMAS2005/p1067-sukthankar.pdf\n" to "AAMAS05-p1067-sukthankar"
    :param input_str: the conference info and paperID of the items in top-10 rank list
    :return: paper ID that match the ground truth
    """
    result = re.split(r'/', input_str)
    paperInfo = result[-1].replace('.pdf\n','')
    if '20' in result[-2]:
        conInfo = result[-2].replace('20','')
    if '19' in result[-2]:
        conInfo = result[-2].replace('19','')
    item_name = conInfo + '-' + paperInfo
    return item_name

def rename_rank_item(data):
    """rename all items name in the ranking result so as to check if the item in the result list hit the ground truth"""
    ranking = pd.read_csv(data)
    print(data)
    ranking_df = pd.DataFrame()
    # n denotes number of researchers, k denote top-k ranking list
    for n in range(1,51,1):
        r = 'R' + str(n)
        ranking_ls = []
        for k in range(len(ranking[r])):
            input_str = ranking[r][k]
            ranking_item = standarize(input_str)
            ranking_ls.append(ranking_item)
        ranking_df[r] = ranking_ls
    ranking_df.to_csv('rename_rank_result_all_CP.csv', index=False)
    return ranking_df

def iter_rank_ls(ranking_df, ground_truth):
    # reverse 50 researchers to see if the top 10 recommended item hit the ground truth
    rank_matrix = {}
    for r in range(50):
        hit_ls = []
        # the ranking result of researcher r
        rID = 'R' + str(r+1)
        r_result_ls = ranking_df.iloc[:,r].tolist()
        r_ground_truth_ls = ground_truth.iloc[r].tolist()
        for i in range(10):
            item_i  = r_result_ls[i]
            # to see if item i hit the ground truth
            if item_i in r_ground_truth_ls:
                hit_ls.append(1)
            else:
                hit_ls.append(0)
        rank_matrix[rID] = hit_ls
        pd.DataFrame.from_dict(rank_matrix).to_csv('rank_matrix_all_CP.csv', index=False)
    return rank_matrix


def get_dcg(rank_list):
    """
    NDCG computation
    rank_list: list, such as  [1,0,1,0,1,1,1], 1 denotes relevant item and 0 denotes irrelevant item
    """
    n = len(rank_list)
    dcg = 0
    for i in range(n):
        pos = i + 1
        # here gains is 1 or 0
        gains = rank_list[i]
        discounts = np.log2(pos + 1)
        if gains == 0:
            cg = 0
        else:
            cg = (gains / discounts)
        dcg += cg
    return dcg

def get_idcg(rank_list):
    """
    :param rank_list: list, such as  [1,0,1,0,1,1,1], 1 denotes relevant item and 0 denotes irrelevant item
    :return: the ideal dcg for the input rank list
    """
    ideal_rank_list = sorted(rank_list, reverse=True)
    idcg = get_dcg(ideal_rank_list)
    return idcg

def get_ndcg(rank_list):
    if get_dcg(rank_list) == 0:
        ndcg = 0
    else:
        ndcg = get_dcg(rank_list)/get_idcg(rank_list)
    return ndcg

def get_precision(rank_list):
    """
    input a top 10 result list such as [1,1,0,0,1,0,1,1,0,0]
    """
    tp = 0 # truth positive
    fp = 0 # flase positive
    for i in range(len(rank_list)):
        if rank_list[i] == 1:
            tp += 1
        else:
            fp += 1
    precision = tp/len(rank_list)
    return precision

def avg_precision(rank_matrix):
    """

    :param rank_matrix: a dataframe or a ndarray contains top-10 rank result for all users
    :return: average precision and precision list for all users
    """
    p_ls = []
    for (k,v) in  rank_matrix.items():
        p = get_precision(v)
        p_ls.append(p)
    avg_p = np.mean(p_ls)
    return avg_p,p_ls

def reciprocal_rank(rank_list):
    """ Computes the reciprocal rank.
        rank_list: list, prediction [1,0,1,0,1,1,1], 1 denotes relevant item and 0 denotes irrelevant item
    """

    score = 0.0
    for index,item in enumerate(rank_list):
        if item == 1:
            score = 1.0 / (index + 1.0)
            break
    return score

def mean_reciprocal_rank(rank_matrix):
    """
    :param rank_matrix: a dataframe or a ndarray contains top-10 rank result for all users
    :return: MRR scores and RR scores list for all users
    """
    rr_ls = []
    for (k, v) in rank_matrix.items():
        rr = reciprocal_rank(v)
        rr_ls.append(rr)
    mrr = np.mean(rr_ls)
    return mrr,rr_ls


# read ranking result for 50 researchers and the ground truth
data = 'rank_result_all_CP.csv'
ground_truth = pd.read_csv('/Users/sherry/git_project/scholarly_paper_recommendatation/user_profiles/ground_truth.csv', index_col=0)

# rename all items in the rank list for all users
ranking_df = rename_rank_item(data)
rank_matrix = iter_rank_ls(ranking_df, ground_truth)

# get mean ndcg for the ranking matrix of 50 researchers
ndcg_ls = []
for r in range(1,3,1):
    rank_list = rank_matrix['R{}'.format(r)]
    ndcg_r = get_ndcg(rank_list)
    ndcg_ls.append(ndcg_r)
    # print('the ndcg for researcher {} is: {}'.format(r,ndcg_r) )
mean_ndcg = np.mean(ndcg_ls)
print('the average ndcg for all researchers is: {}'.format(mean_ndcg))

# ge average precision for the ranking result of 50 researcher
average_precision = avg_precision(rank_matrix)
# for r in range(1,51,1):
#     print('the precision for researcher {} is: {}'.format(r, average_precision[1][r-1]))
print('the average precision for all researchers is: {}'.format(average_precision[0]))

# get mean reciprocal rank for the ranking result of 50 researcher
mrr = mean_reciprocal_rank(rank_matrix)
# for r in range(1,51,1):
#     print('the reciprocal rank  for researcher {} is: {}'.format(r, mrr[1][r-1]))
print('the mean reciprocal rank for all researchers is: {}'.format(mrr[0]))