import train
import argparse
import data
import os
import pandas as pd

def main(trainTable, validTable, qrelsFile, modelName, bertWeights, saveDirectory):
    docs={}
    queries={}
    for index, row in trainTable.iterrows():
        #print(type(row))
        #print(row)
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

    train.main(model, dataset, train_pairs, qrels, valid_run, qrelsFile, saveDirectory)

def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation from a single dataframe')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--trainTSV', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--validTSV', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()

    print("Reading training file %s" % args.trainTSV[0])
    trainTable = pd.read_csv(args.trainTSV[0], sep='\t', header=0)
    print(trainTable.columns.values)
    print("Reading validation file %s" % args.validTSV[0])
    validTable = pd.read_csv(args.validTSV[0], sep='\t', header=0)
    
    os.makedirs(args.model_out_dir, exist_ok=True)

    main(trainTable, validTable, args.qrels, args.model, args.initial_bert_weights, args.model_out_dir)

if __name__ == '__main__':
    main_cli()
