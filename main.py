import base64
import time
from typing import Optional
import re
import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI
import pdb
from backbones import get_model
import os
from elasticsearch import Elasticsearch, RequestError, ElasticsearchException
import inference as inf
from starlette.requests import Request
from starlette.responses import Response
from elasticsearch import AsyncElasticsearch

# es = Elasticsearch(HOST="localhost", PORT=9200)
es = AsyncElasticsearch(["http://elasticsearch:9200"])

from pydantic import BaseModel

class Image(BaseModel):
    base64: str

class PersonImage(BaseModel):
    base64Image: str
    name: str
    id: str


# es = AsyncElasticsearch()
app = FastAPI()

async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        # you probably want some kind of logging here
        return Response("Internal server error", status_code=500)

app.middleware('http')(catch_exceptions_middleware)

face_dict = {
    "id1": "./encoded/1npy",
    "id2": "./encoded/2.npy",
    "id3": "./encoded/3.npy",
    "id4": "./encoded/4.npy",
    "id5": "./encoded/5.npy",
    "id6": "./encoded/6.npy",
    "id7": "./encoded/7.npy",
    "id8": "./encoded/8.npy",
    "id9": "./encoded/9.npy",    
    "id10": "./encoded/10.npy",
}

def listdir_fullpath(d):
    print([os.path.join(d, f) for f in os.listdir(d) if not f.endswith('.DS_Store')])
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith('.DS_Store')]

class RecognizeModel():
    def __init__(self, name):
        self.net = get_model(name, fp16=True)
        self.net.load_state_dict(torch.load("./backbone.pth", map_location=torch.device('cpu')))
        self.net.eval()

    @torch.no_grad()
    def inference(self, img):
        if img is None:
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        else:
            # img = cv2.imread(img)
            img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        self.feat = self.net(img).numpy()
        # print(feat.shape)
        # return self.feat



    def compare_faces(self):
        

        lst_paths = listdir_fullpath("./encoded/8")

        
        # # results = [self.compare_face(des_emb_path) for des_emb_path in lst_paths] #p.map(self.compare_face, lst_paths)
        
        # results = p.map(self.compare_face, lst_paths)

        # # p.map(self.compare_face, [1, 2, 3, 4])
        # p.close()
        # p.join()
        # pdb.set_trace()

#         start = time.time()
# # n_jobs is the number of parallel jobs
#         results = Parallel(n_jobs=2)(delayed(self.compare_face)(face_emb) for face_emb in lst_paths)
#         end = time.time()
#         print(end-start)
        # return dict(zip(lst_paths, results))
        results = {}
        for face_path in listdir_fullpath("./encoded/8"):
            results[face_path] = self.compare_face(face_path)

        return results
        # return dict(zip(lst_paths, results))
        # pass
    # def compare_face(self, des_emb_path):
    #     des_emb = np.load(des_emb_path) 
    #     # return des_emb
    #     return dst.findEuclideanDistance(dst.l2_normalize(self.feat), dst.l2_normalize(des_emb))





def decode_base64(base64Str):
    # encoded_image = 'data:image/jpeg;base64,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'
    if "," in base64Str:
        _, data = base64Str.split(',', 1)
    else: 
        data = base64Str
    #print('header:', header)
    #print('  data:', data[:20])

    image_data = base64.b64decode(data)
    #print('result:', image_data[:20])
    np_array = np.frombuffer(image_data, np.uint8)
    #print(' array:', np_array[:2])
    try:
        image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    except cv2.error as e:
        print("loi cv2 ne")
    return image

model = RecognizeModel(name= "r50")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/recognize")
async def recognize(base64Image: Image):
    retries=0

    img = decode_base64(base64Image.base64)
    model.inference(img)
    query = {
        "size": 5,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0",
                    # "source": "l2norm(params.queryVector, 'title_vector')", #euclidean distance
                    "params": {
                        "queryVector": list(model.feat[0]) #target_embedding
                    }
                }
            }
        }
    }
    # print(es.get(index="face_recognition", id="MI0090"))
    # try:
    res = await es.search(index="face_recognition", body=query, ignore_unavailable=True)#, ignore=[400, 401, 403, 404, 409])
    # res = es.get(index="face_recognition", id="MI0090")
    # except ElasticsearchException as es1:
        # return {"Hello": "1"}
        # continue
        # print(es1.status_code)
    print(res["hits"]["hits"][0]["_source"]["name"])
    return {
        "name": "{}".format(res["hits"]["hits"][0]["_source"]["name"]),
        "id": "{}".format(res["hits"]["hits"][0]["_id"])
    }

    # return {"Hello": "1"}

@app.post("/add_new_face")
def recognize(person: PersonImage):
    img = decode_base64(person.base64Image)
    model.inference(img)
    doc = {"title_vector": model.feat[0], "name": person.name}
    es.create("face_recognition", id=person.id, body=doc)
