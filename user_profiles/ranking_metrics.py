import numpy as np

# NDCG and MRR
def get_dcg(rank_list):
    n = len(rank_list)
    dcg = 0
    for i in range(n):
        pos = i + 1
        gains = rank_list[i]
        discounts = np.log2(pos + 1)
        if discounts == 0:
            cg = 0
        else:
            cg = (gains / discounts)
        dcg += cg
    return dcg

def get_idcg(rank_list):
    ideal_rank_list = sorted(rank_list, reverse=True)
    idcg = get_dcg(ideal_rank_list)
    return idcg

def get_ndcg(rank_list):
    ndcg = get_dcg(rank_list) / get_idcg(rank_list)
    print(ndcg)
    return ndcg

rank_list = [3,2,3,0,1,2]
ndcg6 = get_ndcg(rank_list)