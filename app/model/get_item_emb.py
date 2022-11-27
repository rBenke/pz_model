import pickle as p
from tqdm import tqdm
import numpy as np 
from torch_geometric.data import Data
import torch
from model import RecGCN
def save_all_emb(model, loader):
    model.eval()
    embeddings = []
    for _ in range(100):
        data = generate_data()
        embeddings.append(model.encode_item(data.x, data.adj_t))
    embeddings = torch.cat(embeddings)
    embeddings = embeddings.detach().numpy()
    with open("/app/item_embeddings.pickle", "wb") as file:
        p.dump(embeddings, file)

def generate_data():
    x = numpy.rand(1_000,100)
    edge_index = numpy.randint(0,999,(2,10_000)) 
    data = Data(x = x, edge_index = edge_index)
    return data

if __name__=="__main__":
    model = RecGCN()
    model.load_state_dict(torch.load("/app/model/saved_model"))
    save_all_emb(model, loader)
