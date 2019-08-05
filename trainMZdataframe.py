import train
import argparse
import data
import os
import pandas as pd
import signal
import traceback

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

def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable."""

    # Verify the inputs
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1

    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]

def applyPassaging(df, passageLength, passageStride):
    newRows=[]
    for index, row in df.iterrows():
        toks = row['text_right'].split(r"\s+")
        len_d = len(toks)
        if len_d < passageLength:
            newRow = row.drop(labels=['titles'])
            newRow['text_right'] = row['titles'] + ' '.join(toks)
            newRows.append(newRow)
        for passage in slidingWindow(toks, passageLength, passageStride):
            newRow = row.drop(labels=['titles'])
            newRow['text_right'] = row['titles'] + ' ' + ' '.join(passage)
            newRows.append(newRow)
    return pd.DataFrame(newRows)


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
    trainTable = pd.read_csv(args.trainTSV[0], sep='\t', header=0)
    print(trainTable.columns.values)
    print("Reading validation file %s" % args.validTSV[0], flush=True)
    validTable = pd.read_csv(args.validTSV[0], sep='\t', header=0)
    if (args.passage == "max" or args.passage == "sum"):
      applyPassaging(trainTable, 150, 75)
      applyPassaging(validTable, 150, 75)
    train.aggregation = args.passage
    os.makedirs(args.model_out_dir, exist_ok=True)
    signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))
    main(trainTable, validTable, args.qrels.name, args.model, args.initial_bert_weights, args.model_out_dir)



if __name__ == '__main__':
    main_cli()
