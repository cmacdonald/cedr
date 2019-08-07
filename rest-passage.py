from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pandas as pd
import sys
import pickle
import re 
import numpy as np
import data
import argparse
import train
import trainMZdataframe
import threading
import more_itertools
from collections import defaultdict

GPU=True
passageLength = 150
passageStride = 75

#This script make an HTTP REST endpoint that exposes a POST method that can score documents:
#the post expected a JSON object in the following format:
#{ qid: 'query1', query:'information retrieval', docnos : ['d1', 'd2'], 'docs':['keith proposed information retrieval', 'not matching doc']}


#e.g. curl -d '{ "qid": "query1", "query":"information retrieval", "docnos" : ["d1", "d2"], "docs":["keith proposed information retrieval", "not matching doc"]}'  -H "Content-Type: application/json" -X POST http://localhost:3422/

class CEDR(Resource):
    

    def __init__(self, CEDRmodel, lock, score):
        self.model = CEDRmodel
        self.requestCount = 0
        self.lock = lock
        self.score = score
    
    def get(self):
        return "Invalid request - only POST supported, you called get. " + str(self.requestCount) + " requests served thus far", 400

    # def slidingWindow(self, sequence, winSize, step):
    #     return [x for x in list(more_itertools.windowed(sequence,n=winSize, step=step)) if x[-1] is not None]

    # def applyPassaging(self, dataframe, passageLength, passageStride):
    #     newRows=[]
    #     for i, r in dataframe.iterrows():
    #         toks = re.split(r"\s+", r['text_right'])
    #         len_d = len(toks)
    #         if len_d < passageLength:
    #             newRow = r.drop(labels=["title"])
    #             newRow["text_right"] = str(r['title']) + ' ' + ' '.join(toks)
    #             newRows.append(newRow)
    #         else:
    #             for passage in self.slidingWindow(toks, passageLength, passageStride):
    #                 newRow = r.drop(labels=["title"])
    #                 newRow["text_right"] = str(r['title']) + ' ' + ' '.join(passage)
    #                 newRows.append(newRow)
    #     new_df = pd.DataFrame(newRows)
    #     new_df['text_left'].fillna('',inplace=True)
    #     new_df['text_right'].fillna('',inplace=True)
    #     new_df['id_left'].fillna('',inplace=True)
    #     #print(new_df['text_right'],new_df['id_left'])
    #     new_df.reset_index(inplace=True,drop=True)
    #     return new_df


#     def passage_to_docs(self,passageDF, docDF,scores):
#         rerank_run = defaultdict(int)
#         if self.score == 'first':
#             for did, score in zip(passageDF['id_right'], scores):
#                 if not did in rerank_run:
#                     rerank_run[did]=score
#         elif self.score == 'sum':
#             #should be 0 if the document hasnt been seen before
#             for did, score in zip(passageDF['id_right'], scores):
#                 rerank_run[did] += score
#         elif self.score == 'max':
#             for did, score in zip(passageDF['id_right'], scores):
#                 #should be 0 if the document hasnt been seen before
#                 currentScore = rerank_run[did]
#                 if score > currentScore:
#                     rerank_run[did] = score
#         return [rerank_run[row["id_right"]] for i, row in docDF.iterrows()] 

    def post(self):
        self.requestCount+=1
        
        parser = reqparse.RequestParser()
        parser.add_argument("qid", location='json')
        parser.add_argument("query", location='json')
        parser.add_argument("docnos", location='json', type=list)
        parser.add_argument("docs", location='json', type=list)
        parser.add_argument("titles", location = 'json', type = list)
        args = parser.parse_args()

        print("Processing request %d, query %s, getting scores for %d documents" % (self.requestCount, args['query'], len(args['docs'])) )
        
        # Data now in args['name']. e.g., args['DOC'].
        df = pd.DataFrame({
            'text_left': [args['query']]*len(args['docs']),
            'text_right': args['docs'],
            'id_left': [args['qid']]*len(args['docs']),
            'id_right': args['docnos'],
            'title': args['titles']
        })



        df['text_left'].fillna('',inplace=True)
        df['text_right'].fillna('',inplace=True)
        df['id_left'].fillna('',inplace=True)
        df['title'].fillna('',inplace=True)

        # saparating docs into passages, saving the number of passage,
        # to integrate the scores from passages back into docs. 

        #print('len(id_right)',len(np.unique(np.array(df['id_right']))))
        # print('len_df',len(df))
        passage = trainMZdataframe.applyPassaging(df, passageLength, passageStride, 0, False)
        pd.set_option('display.width', 1000)
        pd.options.display.max_colwidth = 200
        print('len_passage',len(passage))
        #print(passage)
        #for i, r in passage.iterrows():
        #    print(r["text_right"])
        
        docs={}
        queries={}

        for index, row in passage.iterrows():
            queries[row['id_left']] = row['text_left']
            docs[row['id_right']] = row['text_right']
        
        dataset=(queries, docs)

        test_run=defaultdict(dict)
        for index, row in passage.iterrows():
            test_run[row['id_left']][row['id_right']] = float(1)
        
        try:
            self.lock.acquire()
            rerank_run = train.score_model(self.model, dataset, test_run, desc='rerank', passageAgg = self.score)
#             scores = train.score_docsMZ(self.model, passage)
        finally:
            self.lock.release() 

        scores=[]
        #print(rerank_run)
        for i, row in df.iterrows():
            did=row["id_right"]
            scores.append(rerank_run[row["id_left"]][did])


        print(scores) 
        # print('len_scores',len(scores))
#         scores = self.passage_to_docs(passage,df,scores)
        #scores = scores.flatten()
        #scores = [s.tolist() for s in scores]
        # print('len_scores',len(scores))
        response = jsonify(scores)
        response.status_code = 200
        return response

    def put(self):
        return "Invalid request - only POST supported, you called put", 400

    def delete(self):
        return "Invalid request - only POST supported, you called delete", 400




def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking REST API')
    parser.add_argument('--model', choices=train.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--port', default=5678)
    parser.add_argument('--passage', choices=["first", "max", "sum"], default="first")

    data.GPU = GPU
    
    args = parser.parse_args()
    model = train.MODEL_MAP[args.model]()
    if GPU:
        model = model.cuda()
    if args.model_weights is not None:
        model.load(args.model_weights.name, cpu=not GPU)
    #train.run_model(model, dataset, run, args.out_path.name, desc='rerank')

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    api = Api(app)
    api.add_resource(CEDR, "/", resource_class_kwargs={'CEDRmodel':model, 'lock':threading.Lock(), 'score' : args.passage})
    app.run(host='0.0.0.0',port=args.port,debug=True)

if __name__ == '__main__':
    main_cli()
