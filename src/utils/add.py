import re
import os
import json
import subprocess
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


if __name__ == "__main__":
    # Create the elastic instance
    elastic = Elasticsearch(
        "http://localhost:9200",
        request_timeout=1000000
    )

    for file in ["p2-1.filtered", "p4.filtered", "p5.filtered"]:
        model = "DPR"
        file_path = f"../../../Data/wenshu/{file}.txt"

        embeddings = np.memmap(
            os.path.join("data/encode", model, "wenshu", file, "text_embeddings.mmp"),
            dtype=np.float32,
            mode="r"
        ).reshape(-1, 768)


        def gendata():
            with open(file_path, encoding="utf-8") as f:
                j = 0
                for i, line in enumerate(f):
                    case = json.loads(line.strip())
                    del case["id"]
                    del case["crawl_time"]
                    del case["legal_base"]

                    if "tf_content" in case:
                        case["vector"] = embeddings[j].tolist()
                        del case["tf_content"]
                        j += 1

                    yield case

        error_log_path = f"../../../Data/wenshu/{file}.failed.txt"
        error_log_file = open(error_log_path, "a+")

        text_num = int(subprocess.check_output(["wc", "-l", file_path]).decode("utf-8").split()[0])
        for i, x in enumerate(tqdm(gendata(), desc="Indexing", total=text_num, ncols=100)):
            try:
                elastic.index(index="wenshu", document=x, id=f"{file.split('.')[0]}_{i}")
            except:
                print("i")
                error_log_file.write(json.dumps(x, ensure_ascii=False) + "\n")

