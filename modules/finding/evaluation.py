import numpy as np

from modules.finding.alignment import greedy_alignment
import logging




def valid(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False, logger=None):
    if mapping is None:
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                      metric, normalize, csls_k, accurate, logger=logger)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                      metric, normalize, csls_k, accurate, logger=logger)
    return hits1_12, mrr_12


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True, simi_matrix=None, return_simi=False, logger=None):
    if mapping is None:
        top_k, hits, mr, mrr = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate, simi_matrix=simi_matrix, logger=logger)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        top_k, hits, mr, mrr = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate, simi_matrix=simi_matrix, logger=logger)
    return top_k, hits, mr, mrr

