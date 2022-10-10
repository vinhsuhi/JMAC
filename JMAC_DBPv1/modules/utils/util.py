import time
import torch 
import numpy as np

def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    return folder


"""Vinh Tong"""
'''Sample negative alignment entities! following RDGCN: https://www.ijcai.org/proceedings/2019/0733.pdf'''
def get_neg(ILL, output_layer, k):
    """
    Parameters
    ----------
    ILL : list
        List of some entities
    output_layer : numpy array
        Embeddings of all entities
    k : int
        The number of negative samples for each entities in ILL

    Returns
    -------

    neg: numpy array, size = (t * k, )
        Array of negative samples
    """
    neg = []
    t = len(ILL)
    ILL_vec = output_layer[ILL]
    KG_vec = output_layer
    sim = torch.mm(ILL_vec, KG_vec.t())
    neg = sim.topk(k, dim=1)[1].reshape((t * k, ))
    return neg
    


'''L2 norm of a 2D numpy array'''
def normalize(emb):
    return emb / np.sqrt((emb ** 2).sum(axis=1)).reshape(-1, 1)
