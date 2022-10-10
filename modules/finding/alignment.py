import multiprocessing
import time
import numpy as np
from modules.finding.similarity import sim
from modules.utils.util import task_divide
import logging



def greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate, simi_matrix=None, return_simi=False, logger=None):
    """
    Search alignment with greedy strategy.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    top_k : list of integers
        Hits@k metrics for evaluating results.
    nums_threads : int
        The number of threads used to search alignment.
    metric : string
        The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
    normalize : bool, true or false.
        Whether to normalize the input embeddings.
    csls_k : int
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    alignment_rest :  list, pairs of aligned entities
    hits1 : float, hits@1 values for alignment results
    mr : float, MR values for alignment results
    mrr : float, MRR values for alignment results
    """
    if simi_matrix is not None:
        sim_mat = simi_matrix
    else:
        sim_mat = sim(embed1, embed2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    logger.info("Hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}".
                format(top_k, hits, mr, mrr))
    return top_k, hits, mr, mrr

def arg_sort(idx, sim_mat, prefix1, prefix2):
    candidates = dict()
    for i in range(len(idx)):
        x_i = prefix1 + str(idx[i])
        rank = (-sim_mat[i, :]).argsort()
        y_j = [prefix2 + str(r) for r in rank]
        candidates[x_i] = y_j
    return candidates


# def fast_match(index, sim_mat, total_num):
#     for i in range(index):
#         max_val = np.argmax(sim_mat[i])



def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    """
    idx is the list of true target
    """
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest


def galeshapley(suitor_pref_dict, reviewer_pref_dict, max_iteration):
    """ The Gale-Shapley algorithm. This is known to provide a unique, stable
    suitor-optimal matching. The algorithm is as follows:

    (1) Assign all suitors and reviewers to be unmatched.

    (2) Take any unmatched suitor, s, and their most preferred reviewer, r.
            - If r is unmatched, match s to r.
            - Else, if r is matched, consider their current partner, r_partner.
                - If r prefers s to r_partner, unmatch r_partner from r and
                  match s to r.
                - Else, leave s unmatched and remove r from their preference
                  list.
    (3) Go to (2) until all suitors are matched, then end.

    Parameters
    ----------
    suitor_pref_dict : dict
        A dictionary with suitors as keys and their respective preference lists
        as values
    reviewer_pref_dict : dict
        A dictionary with reviewers as keys and their respective preference
        lists as values
    max_iteration : int
        An integer as the maximum iterations

    Returns
    -------
    matching : dict
        The suitor-optimal (stable) matching with suitors as keys and the
        reviewer they are matched with as values
    """
    suitors = list(suitor_pref_dict.keys())
    matching = dict()
    rev_matching = dict()

    for i in range(max_iteration):
        if len(suitors) <= 0:
            break
        for s in suitors:
            r = suitor_pref_dict[s][0]
            if r not in matching.values():
                matching[s] = r
                rev_matching[r] = s
            else:
                r_partner = rev_matching.get(r)
                if reviewer_pref_dict[r].index(s) < reviewer_pref_dict[r].index(r_partner):
                    del matching[r_partner]
                    matching[s] = r
                    rev_matching[r] = s
                else:
                    suitor_pref_dict[s].remove(r)
        suitors = list(set(suitor_pref_dict.keys()) - set(matching.keys()))
    return matching

