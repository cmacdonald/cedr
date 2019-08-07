import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data
from collections import defaultdict



SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}

aggregation = 'first'

def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
    LR = 0.001
    BERT_LR = 2e-5
    MAX_EPOCH = 100
    PATIENCE = 20 #how many epochs to wait for validation improvement

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    top_valid_score_epoch = None
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    for epoch in range(MAX_EPOCH):
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}', flush=True)
        valid_score = validate(model, dataset, valid_run, qrelf, epoch, model_out_dir)
        print(f'validation epoch={epoch} score={valid_score}', flush=True)
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            model.save(os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            break


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    BATCH_SIZE = 16
    BATCHES_PER_EPOCH = 32
    GRAD_ACC_SIZE = 2
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, qrelf, epoch, model_out_dir):
    VALIDATION_METRIC = 'ndcg'
    runf = os.path.join(model_out_dir, f'{epoch}.run')
    run_model(model, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def score_docsMZ(model, MZdatafame):
    scores=[]
    with torch.no_grad(), tqdm(total=MZdatafame.shape[0], ncols=80, desc="scoring", leave=False) as pbar:
        model.eval()
        BATCH_SIZE = 16
        for records in data.iter_valid_recordsMZ(model, MZdatafame, BATCH_SIZE):
            allScores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            thelist= allScores.flatten().tolist()
            scores.extend( thelist )
            pbar.update(len(thelist))
    return scores
    

def run_model(model, dataset, run, runf, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = defaultdict(lambda: defaultdict(int))
    #a defauldict where the default values are defaultdicts, whose default values are 0, qid->did->score
    print(aggregation)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            if aggregation == 'first':
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    if not did in rerank_run[qid]:
                        rerank_run[qid][did] = score.item()
            elif aggregation == 'sum':
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    rerank_run[qid][did] += score.item()
            elif aggregation == 'max':
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    #should be 0 if the document hasnt been seen before
                    if score.item() > rerank_run[qid][did]:
                        rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    print(rerank_run[64527]["D414820"])
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def run_rest(model, dataset, run, desc='valid',aggregation):
    aggregation =aggregation
    BATCH_SIZE = 16
    rerank_run = defaultdict(lambda: defaultdict(int))
    #a defauldict where the default values are defaultdicts, whose default values are 0, qid->did->score
#     print(aggregation)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            if aggregation == 'first':
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    if not did in rerank_run[qid]:
                        rerank_run[qid][did] = score.item()
            elif aggregation == 'sum':
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    rerank_run[qid][did] += score.item()
            elif aggregation == 'max':
                for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                    #should be 0 if the document hasnt been seen before
                    if score.item() > rerank_run[qid][did]:
                        rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
#     print(rerank_run[64527]["D414820"])
    scores = []
    for qid in rerank_run:
        scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
        for i, (did, score) in enumerate(scores):
            scores.extend( score)
    return scores


def trec_eval(qrelf, runf, metric):
    trec_eval_f = 'bin/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])


def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()
    model = MODEL_MAP[args.model]().cuda()
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, dataset, train_pairs, qrels, valid_run, args.qrels.name, args.model_out_dir)


if __name__ == '__main__':
    main_cli()
