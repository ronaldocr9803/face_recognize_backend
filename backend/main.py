import base64
import io
import os
import pdb
import re
import time
from typing import Optional

import cv2
import numpy as np
import torch
from elasticsearch import (AsyncElasticsearch, Elasticsearch,
                           ElasticsearchException, RequestError)
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image as im
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

import inference as inf
from backbones import get_model
from demo import draw_face, get_face
from detection.retinaface import RetinaNetDetector
from fastapi import File, Form, UploadFile
import utils as ut

# es = Elasticsearch(HOST="localhost", PORT=9200)
es = AsyncElasticsearch(["http://elasticsearch:9200"])


class Image(BaseModel):
    base64: str

class PersonImage(BaseModel):
    base64Image: str
    name: str
    id: str


# es = AsyncElasticsearch()
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# async def catch_exceptions_middleware(request: Request, call_next):
#     try:
#         return await call_next(request)
#     except Exception:
#         # you probably want some kind of logging here
#         return Response("Internal server error", status_code=500)

# app.middleware('http')(catch_exceptions_middleware)

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
        

        lst_paths = ut.listdir_fullpath("./encoded/8")

        
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
        for face_path in ut.listdir_fullpath("./encoded/8"):
            results[face_path] = self.compare_face(face_path)

        return results
        # return dict(zip(lst_paths, results))
        # pass
    # def compare_face(self, des_emb_path):
    #     des_emb = np.load(des_emb_path) 
    #     # return des_emb
    #     return dst.findEuclideanDistance(dst.l2_normalize(self.feat), dst.l2_normalize(des_emb))






model = RecognizeModel(name= "r50")
detector = RetinaNetDetector()

def generate_frames():
    camera=cv2.VideoCapture(0)
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        # print(frame)
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break

@app.get("/test")
async def read_root():
    return {"Hello": "World"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

    return {"Hello": "World"}

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})

@app.get("/video")
async def demo_video():
    
    # print(generate_frames(camera))
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")
def demoYield():
    i = 0
    
    while True:            
        ## read the camera frame
        
        yield(i)
        i = i +1
        if i == 50:
            break
def demoCamera():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # if not ret:
        #     break
        yield frame
        

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    generator = generate_frames()#demoYield()
    while True:
        
        payload = next(generator)
        await websocket.send_text(payload)
        try:
            # Wait for any message from the client
            data=await websocket.receive_text()
            print(data)
            # Send message to the client
            generator = demoYield()
            payload = next(generator)
            await websocket.send_text(str(payload))
            print("Sending")
            # print(data)
        except Exception as e:
            print('error:', e)
            break
    print('Bye..')

@app.post("/recognize")
async def recognize(base64Image: Image):
    retries=0

    img = ut.decode_base64(base64Image.base64)
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
    img = ut.decode_base64(person.base64Image)
    model.inference(img)
    doc = {"title_vector": model.feat[0], "name": person.name}
    es.create("face_recognition", id=person.id, body=doc)


@app.post("/detect_recognize")
async def detectAndRecognize(person: Image):
    img = ut.decode_base64(person.base64Image)
    det_faces = detector.predict(img)
    results = []
    for face_info in det_faces[0]:
        face_img, face_crd = get_face(img, face_info)
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
        result = {
            "name": "{}".format(res["hits"]["hits"][0]["_source"]["name"]),
            "id": "{}".format(res["hits"]["hits"][0]["_id"])
        }
        results.append(result)
    return results


@app.post("/web/detect_recognize")
async def detectAndRecognize(image: UploadFile = File(...)):
    img = ut.load_image_into_numpy_array(await image.read())
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    print(img.shape)
    # import pdb; pdb.set_trace
    det_faces = detector.predict(img)
    results = []
    for face_info in det_faces[0]:
        face_img, face_crd = get_face(img, face_info)
        result = {
            "crd": list(face_crd),
        }
        cv2.imwrite("thangld.png", face_img)
        results.append(result)
    return results
