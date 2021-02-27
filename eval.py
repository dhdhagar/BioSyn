import argparse
import logging
import os
import json
from tqdm import tqdm
from utils import (
    evaluate
)
from src.biosyn import (
    DictionaryDataset,
    QueryDataset,
    BioSyn
)

from IPython import embed

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='BioSyn evaluation')

    # Required
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')

    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--normalize_vecs',  action="store_true")
    parser.add_argument('--type_given',  action="store_true")
    parser.add_argument('--topk',  type=int, default=20)
    parser.add_argument('--score_mode',  type=str, default='hybrid', help='hybrid/dense/sparse')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)
    
    args = parser.parse_args()
    return args
    
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def load_dictionary(dictionary_path): 
    dictionary = DictionaryDataset(
        dictionary_path = dictionary_path
    )
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data
                
def main(args):
    init_logging()
    print(args)

    # load dictionary and data
    eval_dictionary = load_dictionary(dictionary_path=args.dictionary_path)
    eval_queries = load_queries(
        data_dir=args.data_dir,
        filter_composite=args.filter_composite,
        filter_duplicate=args.filter_duplicate
    )

    biosyn = BioSyn().load_model(
            path=args.model_dir,
            max_length=args.max_length,
            normalize_vecs=args.normalize_vecs,
            use_cuda=args.use_cuda
    )
    
    result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=eval_dictionary,
        eval_queries=eval_queries,
        topk=args.topk,
        score_mode=args.score_mode,
        type_given=args.type_given
    )

    LOGGER.info("acc@1={}".format(result_evalset['acc1']))
    LOGGER.info("acc@2={}".format(result_evalset['acc2']))
    LOGGER.info("acc@4={}".format(result_evalset['acc4']))
    LOGGER.info("acc@8={}".format(result_evalset['acc8']))
    LOGGER.info("acc@16={}".format(result_evalset['acc16']))
    # LOGGER.info("acc@32={}".format(result_evalset['acc32']))
    # LOGGER.info("acc@64={}".format(result_evalset['acc64']))
    
    if args.save_predictions:
        output_file = os.path.join(args.output_dir,"predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
