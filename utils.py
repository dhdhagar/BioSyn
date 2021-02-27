import csv
import json
import math
import numpy as np
import pdb
from tqdm import tqdm
import faiss
import nmslib

from IPython import embed


def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)


def predict_topk(biosyn,
                 eval_dictionary,
                 eval_queries,
                 topk,
                 score_mode='hybrid',
                 type_given=False):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    encoder = biosyn.get_dense_encoder()
    tokenizer = biosyn.get_dense_tokenizer()
    sparse_encoder = biosyn.get_sparse_encoder()
    sparse_weight = biosyn.get_sparse_weight().item() # must be scalar value

    # useful if we're conditioning on types
    all_indv_types = [x for t in eval_dictionary[:,1] for x in t.split('|')]
    unique_types = np.unique(all_indv_types).tolist()
    v_check_type = np.vectorize(check_label)
    inv_idx = {t : v_check_type(eval_dictionary[:,1], t).nonzero()[0] 
                    for t in unique_types}

    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:,0], show_progress=True)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:,0], show_progress=True)

    # build the sparse index
    if not type_given:
        sparse_index = nmslib.init(
            method='hnsw',
            space='negdotprod_sparse_fast',
            data_type=nmslib.DataType.SPARSE_VECTOR
        )
        sparse_index.addDataPointBatch(dict_sparse_embeds)
        sparse_index.createIndex({'post': 2}, print_progress=False)
    else:
        sparse_index = {}
        for sty, indices in inv_idx.items():
            sparse_index[sty] = nmslib.init(
                method='hnsw',
                space='negdotprod_sparse_fast',
                data_type=nmslib.DataType.SPARSE_VECTOR
            )
            sparse_index[sty].addDataPointBatch(dict_sparse_embeds[indices])
            sparse_index[sty].createIndex({'post': 2}, print_progress=False)

    # build the dense index
    d = dict_dense_embeds.shape[1]
    if not type_given:
        nembeds = dict_dense_embeds.shape[0]
        if nembeds < 10000: # if the number of embeddings is small, don't approximate
            dense_index = faiss.IndexFlatIP(d)
            dense_index.add(dict_dense_embeds)
        else:
            nlist = int(math.floor(math.sqrt(nembeds))) # number of quantized cells
            nprobe = int(math.floor(math.sqrt(nlist))) # number of the quantized cells to probe
            quantizer = faiss.IndexFlatIP(d)
            dense_index = faiss.IndexIVFFlat(
                quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            dense_index.train(dict_dense_embeds)
            dense_index.add(dict_dense_embeds)
            dense_index.nprobe = nprobe
    else:
        dense_index = {}
        for sty, indices in inv_idx.items():
            sty_dict_dense_embeds = dict_dense_embeds[indices]
            nembeds = sty_dict_dense_embeds.shape[0]
            if nembeds < 10000: # if the number of embeddings is small, don't approximate
                dense_index[sty] = faiss.IndexFlatIP(d)
                dense_index[sty].add(sty_dict_dense_embeds)
            else:
                nlist = int(math.floor(math.sqrt(nembeds))) # number of quantized cells
                nprobe = int(math.floor(math.sqrt(nlist))) # number of the quantized cells to probe
                quantizer = faiss.IndexFlatIP(d)
                dense_index[sty] = faiss.IndexIVFFlat(
                    quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
                )
                dense_index[sty].train(sty_dict_dense_embeds)
                dense_index[sty].add(sty_dict_dense_embeds)
                dense_index[sty].nprobe = nprobe

    # respond to mention queries
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        golden_sty = eval_query[2].replace("+","|")
        pmid = eval_query[3]
        start_char = eval_query[4]
        end_char = eval_query[5]

        dict_mentions = []
        for mention in mentions:

            mention_sparse_embeds = biosyn.embed_sparse(names=np.array([mention]))
            mention_dense_embeds = biosyn.embed_dense(names=np.array([mention]))

            # search the sparse index
            if not type_given:
                sparse_nn = sparse_index.knnQueryBatch(
                    mention_sparse_embeds, k=topk, num_threads=20
                )
            else:
                sparse_nn = sparse_index[golden_sty].knnQueryBatch(
                    mention_sparse_embeds, k=topk, num_threads=20
                )
            sparse_idxs, _ = zip(*sparse_nn)
            s_candidate_idxs = np.asarray(sparse_idxs)
            if type_given:
                # reverse mask index mapping
                s_candidate_idxs = inv_idx[golden_sty][s_candidate_idxs]
            s_candidate_idxs = s_candidate_idxs.astype(np.int64)

            # search the dense index
            if not type_given:
                _, d_candidate_idxs = dense_index.search(
                    mention_dense_embeds, topk
                )
            else:
                _, d_candidate_idxs = dense_index[golden_sty].search(
                    mention_dense_embeds, topk
                )
                # reverse mask index mapping
                d_candidate_idxs = inv_idx[golden_sty][d_candidate_idxs]
            d_candidate_idxs = d_candidate_idxs.astype(np.int64)

            # get the reduced candidate set
            reduced_candidate_idxs = np.unique(
                np.hstack(
                    [s_candidate_idxs.reshape(-1,),
                     d_candidate_idxs.reshape(-1,)]
                )
            )

            # get score matrix
            sparse_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sparse_embeds, 
                dict_embeds=dict_sparse_embeds[reduced_candidate_idxs, :]
            ).todense()
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dict_dense_embeds[reduced_candidate_idxs, :]
            )

            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            else:
                raise NotImplementedError()

            # take care of getting the best indices
            candidate_idxs = biosyn.retrieve_candidate(
                score_matrix=score_matrix, 
                topk=topk
            )
            candidate_idxs = reduced_candidate_idxs[candidate_idxs]

            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'sty':np_candidate[1],
                    'cui':np_candidate[2],
                    'label':check_label(np_candidate[2], golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'pmid':pmid,
                'start_char':start_char,
                'end_char':end_char,
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result

def evaluate(biosyn,
             eval_dictionary,
             eval_queries,
             topk,
             score_mode='hybrid',
             type_given=False):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : str
        hybrid, dense, sparse
    type_given : bool
        whether or not to restrict entity set to ones with gold type

    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(
        biosyn, eval_dictionary, eval_queries, topk, score_mode, type_given
    )
    result = evaluate_topk_acc(result)
    
    return result
