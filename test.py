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
import trainMZdataframe



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
      testTable = trainMZdataframe.applyPassaging(testTable, 150, 75)


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

    train.run_model(model, dataset, test_run, runf, desc='rerank')
# (model, dataset, train_pairs, qrels, valid_run, qrelsFile, saveDirectory)


if __name__ == '__main__':
    main_cli()
