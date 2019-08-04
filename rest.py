from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pandas as pd
import sys
import pickle

import data
import argparse
import train
import threading

GPU=True

#This script make an HTTP REST endpoint that exposes a POST method that can score documents:
#the post expected a JSON object in the following format:
#{ qid: 'query1', query:'information retrieval', docnos : ['d1', 'd2'], 'docs':['keith proposed information retrieval', 'not matching doc']}


#e.g. curl -d '{ "qid": "query1", "query":"information retrieval", "docnos" : ["d1", "d2"], "docs":["keith proposed information retrieval", "not matching doc"]}'  -H "Content-Type: application/json" -X POST http://localhost:3422/

class CEDR(Resource):
    

    def __init__(self, CEDRmodel, lock):
        self.model = CEDRmodel
        self.requestCount = 0
        self.lock = lock
    
    def get(self):
        return "Invalid request - only POST supported, you called get. " + str(self.requestCount) + " requests served thus far", 400

    def post(self):
        self.requestCount+=1
        
        parser = reqparse.RequestParser()
        parser.add_argument("qid", location='json')
        parser.add_argument("query", location='json')
        parser.add_argument("docnos", location='json', type=list)
        parser.add_argument("docs", location='json', type=list)
        args = parser.parse_args()

        print("Processing request %d, query %s, getting scores for %d documents" % (self.requestCount, args['query'], len(args['docs'])) )
        
        # Data now in args['name']. e.g., args['DOC'].
        df = pd.DataFrame({
            'text_left': [args['query']]*len(args['docs']),
            'text_right': args['docs'],
            'id_left': [args['qid']]*len(args['docs']),
            'id_right': args['docnos']
        })
        self.lock.acquire()
        scores = train.score_docsMZ(self.model, df)
        self.lock.release() 
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
    api.add_resource(CEDR, "/", resource_class_kwargs={'CEDRmodel':model, 'lock':threading.Lock()})
    app.run(host='0.0.0.0',port=args.port,debug=True)

if __name__ == '__main__':
    main_cli()
