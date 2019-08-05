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

MAXRANK=200

def main(trainTable, validTable, qrelsFile, modelName, bertWeights, saveDirectory):
    docs={}
    queries={}

    for index, row in trainTable.iterrows():
        queries[row['id_left']] = row['text_left']
        docs[row['id_right']] = row['text_right']
    for index, row in validTable.iterrows():
        queries[row['id_left']] = row['text_left']
        docs[row['id_right']] = row['text_right']

    dataset=(queries, docs)
    valid_run={}
    for index, row in validTable.iterrows():
        valid_run.setdefault(row['id_left'], {})[row['id_right']] = float(1)

    train_pairs={}
    for index, row in trainTable.iterrows():
        train_pairs.setdefault(row['id_left'], {})[row['id_right']] = 1

    qrels={}
    for index, row in trainTable.iterrows():
        qrels.setdefault(row['id_left'], {})[row['id_right']] = int(row['label'])
    for index, row in validTable.iterrows():
        qrels.setdefault(row['id_left'], {})[row['id_right']] = int(row['label'])

    model = train.MODEL_MAP[modelName]().cuda()

    if bertWeights is not None:
        model.load(bertWeights)
    #import pdb; pdb.set_trace()
    train.main(model, dataset, train_pairs, qrels, valid_run, qrelsFile, saveDirectory)

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
    parser = argparse.ArgumentParser('CEDR model training and validation from a single dataframe')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--trainTSV', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--validTSV', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--passage', choices=["first", "max", "sum"], default="first")
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()

    print("Reading training file %s" % args.trainTSV[0], flush=True)
    trainTable = pd.read_csv(args.trainTSV[0], sep='\t', header=0, error_bad_lines = False, index_col=False)
    print(trainTable.columns.values)
    print("Reading validation file %s" % args.validTSV[0], flush=True)
    validTable = pd.read_csv(args.validTSV[0], sep='\t', header=0, error_bad_lines = False, index_col=False)
    if args.passage == "max" or args.passage == "sum":
      trainTable = applyPassaging(trainTable, 150, 75)
      validTable = applyPassaging(validTable, 150, 75)
    train.aggregation = args.passage
    os.makedirs(args.model_out_dir, exist_ok=True)
    signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))
    main(trainTable, validTable, args.qrels.name, args.model, args.initial_bert_weights, args.model_out_dir)



if __name__ == '__main__':
    main_cli()
