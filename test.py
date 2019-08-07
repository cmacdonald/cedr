import train
import argparse
import data
import os
import pandas as pd
import signal
import traceback
from collections import defaultdict
from tqdm import tqdm
import more_itertools
import re


def slidingWindow(sequence, winSize, step):
    return [x for x in list(more_itertools.windowed(sequence,n=winSize, step=step)) if x[-1] is not None]

def applyPassaging(df, passageLength, passageStride):
    newRows=[]
    labelCount=defaultdict(int)
    re.compile("\\s+")
    currentQid=None
    rank=0
    with tqdm('passsaging', total=len(df), ncols=80, desc='passaging', leave=False) as pbar:
        for index, row in df.iterrows():
            pbar.update(1)
            qid = row['id_left']
            if currentQid is None or currentQid != qid:
                rank=0
                currentQid = qid
            rank+=1
            if (rank > MAXRANK):
                continue
            toks = re.split("\s+", row['text_right'])
            len_d = len(toks)
            if len_d < passageLength:
                newRow = row.drop(labels=['title'])
                newRow['text_right'] = str(row['title']) + ' '.join(toks)
                labelCount[row['label']] += 1
                newRows.append(newRow)
            else:
                passageCount=0
                for passage in slidingWindow(toks, passageLength, passageStride):
                    newRow = row.drop(labels=['title'])
                    newRow['text_right'] = str(row['title']) + ' ' + ' '.join(passage)
                    labelCount[row['label']] += 1
                    newRows.append(newRow)
                    passageCount+=1
                #print(row["id_right"] + " " + str(passageCount))
            
    print(labelCount)
    newDF = pd.DataFrame(newRows)
    newDF['text_left'].fillna('',inplace=True)
    newDF['text_right'].fillna('',inplace=True)
    newDF['id_left'].fillna('',inplace=True)
    newDF.reset_index(inplace=True,drop=True)
    return newDF


def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    # parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--passage', choices=["first", "max", "sum"], default="first")
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    args = parser.parse_args()

    model = train.MODEL_MAP[args.model]().cuda()

    print("Reading testing file %s" % args.datafiles, flush=True)
    testTable = pd.read_csv(args.datafiles[0], sep='\t', header=0, error_bad_lines = False, index_col=False)
    print(trainTable.columns.values)
    
    if args.passage == "max" or args.passage == "sum" or args.passage == "first":
      testTable = applyPassaging(testTable, 150, 75)


    train.aggregation = args.passage

    os.makedirs(args.model_out_dir, exist_ok=True)
    signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))

    docs={}
    queries={}

    for index, row in testTable.iterrows():
        queries[row['id_left']] = row['text_left']
        docs[row['id_right']] = row['text_right']
    

    dataset=(queries, docs)
    
    test_run={}
    for index, row in testTable.iterrows():
        test_run.setdefault(row['id_left'], {})[row['id_right']] = float(1)

    if args.model_weights is not None:
        model.load(args.model_weights)
    #import pdb; pdb.set_trace()

    runf = os.path.join(model_out_dir, f'test.run')

    train.run_model(model, dataset, run, runf, desc='rerank')
# (model, dataset, train_pairs, qrels, valid_run, qrelsFile, saveDirectory)


if __name__ == '__main__':
    main_cli()
