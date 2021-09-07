from elasticsearch import Elasticsearch
import argparse
import os
from tqdm import tqdm
from backbones import get_model
import torch
import inference as inf


# es = Elasticsearch("http://0.0.0.0:9200")#, timeout=30, max_retries=10, retry_on_timeout=True)
es = Elasticsearch("http://elasticsearch:9200")

def create_index(model):
    mapping = {
        "mappings": {
            "properties": {
                "title_vector":{
                    "type": "dense_vector",
                    "dims": 512
                },
                "name": {"type": "text"}
            }
        }
    }
    es.indices.create(index="face_recognition", body=mapping)
    index = 0
    for image_path in tqdm(listdir_fullpath("./face_database")):
        embedding = inf.inference(model, image_path)

        full_name = image_path.split("/")[-1].split(".")[0]
        id_emp = full_name.split("_")[-1] 
        name = full_name.split("_")[0]
        # print(name + " " + id_emp)
        doc = {"title_vector": embedding[0], "name": name}
        es.create("face_recognition", id=id_emp, body=doc)
        index = index + 1


# print(es.indices.get_alias("*"))

 

def loadModel(args):
    net = get_model(args.network, fp16=True)
    net.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
    net.eval()
    return net

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith('.DS_Store')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='./backbone.pth')
    args = parser.parse_args(args=[])
    model = loadModel(args)
    if not es.indices.exists(index="face_recognition"):
        create_index(model)

    target_embedding = inf.inference(model, "./10.png")[0]
    query = {
        "size": 5,
        "query": {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'title_vector')",
                # "source": "l2norm(params.queryVector, 'title_vector')", #euclidean distance
                "params": {
                    "queryVector": target_embedding
                }
            }
        }
    }}
    res = es.search(index="face_recognition", body=query)
    print(res["hits"]["hits"][0]["_source"]["name"])


    
