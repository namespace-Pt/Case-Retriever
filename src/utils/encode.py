import re
import os
import json
import torch
import argparse
import subprocess
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel



class GenericPLMEncoder(torch.nn.Module):
    def __init__(self, plm, tokenizer, device="cpu", pooling_method="cls", metric="cos"):
        super().__init__()
        self.plm = plm
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.device = device
        self.metric = metric

        self.output_dim = self.plm.config.hidden_size
        self.max_length = self.plm.config.max_position_embeddings


    @torch.no_grad()
    def _encode(self, x, max_length=None):
        """ encode a list of text into tensors
        """
        if not max_length:
            max_length = self.max_length
        inputs = self.tokenizer(x, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")

        if self.device != "cpu":
            for k, v in inputs.items():
                inputs[k] = v.to(self.device, non_blocking=True)

        if self.pooling_method == "cls":
            embedding = self.plm(**inputs)[0][:, 0]

        if self.metric == "cos":
            embedding = F.normalize(embedding, dim=-1)

        return embedding


    def encode_text(self, text_path, save_dir, text_num, batch_size=200):
        """ encode the text in text path into dense vectors by plm
        """
        print(f"encoding {text_path} and saving at {save_dir}...")

        all_text_num = int(subprocess.check_output(["wc", "-l", text_path]).decode("utf-8").split()[0])

        os.makedirs(save_dir, exist_ok=True)
        text_embeddings = np.memmap(
            os.path.join(save_dir, "text_embeddings.mmp"),
            dtype=np.float32,
            mode="w+",
            shape=(text_num, self.output_dim)
        )

        with open(text_path, encoding="utf-8") as f:
            batch_text = []
            j = 0
            for i, line in enumerate(tqdm(f, total=all_text_num, ncols=100, desc="Encoding Text")):
                case = json.loads(line.strip())
                if "tf_content" not in case:
                    continue

                if j % batch_size == 0:
                    if j > 0:
                        embedding = self._encode(batch_text).cpu().numpy()
                        text_embeddings[j - batch_size: j] = embedding
                    batch_text = []
                batch_text.append(case["tf_content"])
                j += 1

            if len(batch_text):
                text_embeddings[-len(batch_text):] = self._encode(batch_text).cpu().numpy()

        return text_embeddings


    def encode_single_query(self, query, max_length=None):
        """ encode a single query
        """
        embedding = self._encode(query, max_length=max_length)
        if self.metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding.squeeze(0).tolist()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DPR")
    parser.add_argument("--data", default="wenshu")
    parser.add_argument("--file", default="p4.filtered.txt")
    parser.add_argument("--device", type=lambda x: int(x) if x != "cpu" else "cpu", default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--pooling_method", choices=["cls"], default="cls")
    parser.add_argument("--metric", choices=["l2", "cos"], default="cos")
    args = parser.parse_args()


    plm = AutoModel.from_pretrained(os.path.join("data", "model", args.model)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("data", "model", args.model))

    model = GenericPLMEncoder(plm=plm, tokenizer=tokenizer, device=args.device, pooling_method=args.pooling_method, metric=args.metric)

    text_path = f"../../../Data/{args.data}/{args.file}"
    name = ".".join(args.file.split(".")[:-1])

    stat_path = f"../../../Data/{args.data}/{args.file}.stat"
    stat = json.load(open(stat_path))
    valid_tf_content_count = stat["valid_tf_content_count"]

    save_dir = os.path.join("data", "encode", args.model, args.data, name)
    model.encode_text(text_path, save_dir, batch_size=args.batch_size, text_num=valid_tf_content_count)
