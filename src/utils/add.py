import re
import os
import json
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Create the elastic instance
elastic = Elasticsearch(
    "http://localhost:9200",
    request_timeout=1000000
)


for file in ["p2-1.filtered", "p4.filtered", "p5.filtered"]:
    model = "DPR"

    embeddings = np.memmap(
        os.path.join("data/encode", model, "wenshu", file, "text_embeddings.mmp"),
        dtype=np.float32,
        mode="r"
    ).reshape(-1, 768)

    def gendata():
        with open(f"../../../Data/wenshu/{file}.txt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                case = json.loads(line.strip())
                del case["crawl_time"]
                del case["legal_base"]
                del case["tf_content"]
                del case["id"]
                case["vector"] = embeddings[i].tolist()
                yield case

    for x in tqdm(gendata(), desc="Indexing", total=embeddings.shape[0]):
        elastic.index(index="wenshu", document=x)
