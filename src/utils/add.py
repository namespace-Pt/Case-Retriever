import re
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="p2-1.txt")
    parser.add_argument("--index", default="case")
    parser.add_argument("--model", default="DPR")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    # Create the elastic instance
    elastic = Elasticsearch(
        "http://localhost:9200",
        request_timeout=1000000
    )

    file = args.file
    model = args.model
    index = args.index

    name, extension = file.split(".")
    file_path = f"../../../Data/wenshu/{name}.filtered.{extension}"

    embeddings = np.memmap(
        os.path.join("data/encode", model, "wenshu", name, "text_embeddings.mmp"),
        dtype=np.float32,
        mode="r"
    ).reshape(-1, 768)

    error_log_path = f"../../../Data/wenshu/{file}.index-error.txt"
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    with open(error_log_path, "a+") as error_log_file:
        def gendata():
            with open(file_path, encoding="utf-8") as f:
                j = 0
                for i, line in enumerate(f):
                    if i < args.start:
                        continue
                    elif i >= args.end and args.end > 0:
                        break

                    case = json.loads(line.strip())
                    case["collection"] = file.split('.')[0]
                    case["id"] = i

                    new_legal_base = []
                    try:
                        for x in case["legal_base"]:
                            if "法规名称" in x:
                                values = iter(x.values())
                                fagui = next(values)
                                for fatiao in next(values):
                                    new_legal_base.append(fagui + next(iter(fatiao.values())))
                            elif "tkx" in x:
                                new_legal_base.append(x["fgmc"] + x["tkx"])
                            else:
                                raise ValueError
                        case["legal_base"] = new_legal_base
                    except:
                        new_case = case.copy()
                        error_log_file.write("[Error in collecting legal_base]  " + json.dumps(new_case, ensure_ascii=False) + "\n")

                    case["legal_base"] = new_legal_base

                    del case["crawl_time"]
                    del case["doc_id"]
                    del case["browse_count"]

                    if case["basics_text"]:
                        case["vector"] = embeddings[j].tolist()
                        j += 1

                    yield case

        # text_num = int(subprocess.check_output(["wc", "-l", file_path]).decode("utf-8").split()[0])
        for i, x in enumerate(tqdm(gendata(), desc=f"Indexing {file}", ncols=100)):
            try:
                elastic.index(index=index, document=x)
            except:
                if "vector" in x:
                    del x["vector"]
                error_log_file.write("[Error in indexing]  " + json.dumps(x, ensure_ascii=False) + "\n")

