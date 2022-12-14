from typing import Union
from torch_geometric.data import Data
import torch 

from fastapi import FastAPI
from model.prediction import create_embedding
import requests as r 
from config import DATABASE_IP

model_app = FastAPI()

@model_app.get("/get_embedding/{client_id}")
def get_embedding(client_id):
    graph, new_client_id = get_neighbourhood(client_id)
    embeddings = create_embedding(graph)
    embedding = embeddings[new_client_id,:].tolist()
    return embedding

def get_neighbourhood(client_id):
    #TODO: write a request to database
    message = "write code here"
    # message.text should be a list of 2 element's lists that stores a pairs of source and destination nodes
    edges = eval(message.text)
    
    rename_edges = list(set([int(client_id)]+[edge[0]  for edge in edges]))
    mapping = dict(zip(rename_edges, range(len(rename_edges))))
    edges = [[mapping[edge[0]], mapping[edge[1]]] for edge in edges]
    edges_torch = torch.Tensor(edges).to(torch.long).T
    
    x = torch.rand(len(rename_edges),100, dtype = torch.float)
    
    data = Data(x = x,edge_index = edges_torch)
    return data, mapping[int(client_id)]


    
