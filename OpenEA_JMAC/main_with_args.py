
import argparse
from modules.load.kgs import read_kgs_from_folder
from trainer.jmac_trainer import Trainer
import torch 
import numpy as np
import random 
import logging
logging.basicConfig(
     level=logging.INFO
 )

parser = argparse.ArgumentParser(description='JMAC for OpenEA dataset')
parser.add_argument('--training_data', type=str, default='OpenEA_dataset_v1.1/D_W_15K_V1/')
parser.add_argument('--output', type=str, default='output/results/')
parser.add_argument('--dataset_division', type=str, default='721_5fold/1/')

parser.add_argument('--alignment_module', type=str, default='mapping', choices=['sharing', 'mapping', 'swapping'])
parser.add_argument('--test_threads_num', type=int, default=16)
parser.add_argument('--pair_sample_weight', type=float, default=0.2, help='Beta vule controlling number of alignment candidates') # TODO: check this


parser.add_argument('--ordered', action='store_false') 
parser.add_argument('--top_k', type=list, default=[1, 5, 10])
parser.add_argument('--csls', type=int, default=10)

parser.add_argument('--search_module', type=str, default='greedy', choices=['greedy', 'global'])
parser.add_argument('--neg_sampling', type=str, default='uniform', choices=['uniform', 'truncated'])
parser.add_argument('--eval_norm', action="store_true")
parser.add_argument('--start_valid', type=int, default=10)
parser.add_argument('--stop_metric', type=str, default='mrr', choices=['hits1', 'mrr'])
parser.add_argument('--eval_metric', type=str, default='cosine', choices=['inner', 'cosine', 'euclidean', 'manhattan'])
parser.add_argument("--word_embed", type=str, default="../wiki-news-300d-1M.vec")
parser.add_argument("--seed", type=int, default=2)

parser.add_argument('--emb_dim', type=int, default=300) 
parser.add_argument('--num_negative', type=int, default=25) 
parser.add_argument('--alignment_learning_rate', type=float, default=0.0005) 
parser.add_argument("--margin_align", type=float, default=1)
parser.add_argument("--margin_completion", type=float, default=5)
parser.add_argument("--completion_dropout_rate", type=float, default=0.4)  
parser.add_argument("--num_gcn_layer", type=int, default=2) 
parser.add_argument("--leaky_relu_w", type=float, default=0.05)
parser.add_argument("--opn", type=str, default="sub") 

parser.add_argument('--completion_batch_size', type=int, default=1000) 
parser.add_argument('--completion_learning_rate', type=float, default=0.001)
parser.add_argument('--completion_map_weight', type=float, default=50)

parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--logger', type=str, default="")


args = parser.parse_args()
#logger.info(args)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic=True


import logging
import os 

class Logger:
    def __init__(self, name, level, tostdout=True, totxtfile=False):
        self.SAVE_FOLDER = "logging/"
        if not os.path.exists(self.SAVE_FOLDER):
            os.makedirs(self.SAVE_FOLDER)

        self.logger = logging.getLogger(name=name)
        self.logger.propagate = False # not propagate to root logger (print to sdtout)
        if tostdout:
            c_handler = logging.StreamHandler()
            c_format = logging.Formatter('%(asctime)s - %(module)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            c_handler.setFormatter(c_format)
            c_handler.setLevel(level)
            self.logger.addHandler(c_handler)
        if totxtfile:
            f_handler = logging.FileHandler(os.path.join(self.SAVE_FOLDER, name + ".logs"))
            f_format = logging.Formatter('%(asctime)s - %(module)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            f_handler.setFormatter(f_format)
            f_handler.setLevel(level)
            self.logger.addHandler(f_handler)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)


if __name__ == '__main__':
    args.name = args.training_data.split('/')[-2]
    logger = Logger(args.name, logging.INFO, True, True)
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered, linkpred=False, logger=logger)
    trainer = Trainer()
    trainer.set_args(args)
    trainer.set_kgs(kgs, logger)
    trainer.init()
    trainer.run()
