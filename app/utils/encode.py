import os
import json
import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel



class GenericPLMEncoder(torch.nn.Module):
    def __init__(self, plm, tokenizer, device="cpu", pooling_method="cls"):
        super().__init__()
        self.plm = plm
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.device = device

        self.output_dim = self.plm.config.hidden_size
        self.max_length = self.plm.config.max_position_embeddings


    @torch.no_grad()
    def _encode(self, x):
        """ encode a list of text into tensors
        """
        inputs = self.tokenizer(x, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device, non_blocking=True)

        if self.pooling_method == "cls":
            return self.plm(**inputs)[0][:, 0]


    def encode_text(self, text_path, save_dir, batch_size=200):
        """ encode the text in text path into dense vectors by plm
        """
        text_num = int(subprocess.check_output(["wc", "-l", text_path]).decode("utf-8").split()[0])
        os.makedirs(save_dir, exist_ok=True)
        text_embeddings = np.memmap(
            os.path.join(save_dir, "text_embeddings.mmp"),
            dtype=np.float32,
            mode="w+",
            shape=(text_num, self.output_dim)
        )

        with open(text_path, encoding="utf-8") as f:
            batch_text = []
            for i, line in enumerate(tqdm(f, total=text_num, ncols=100, desc="Encoding Text")):
                case = json.loads(line.strip())
                if i % batch_size == 0:
                    if i > 0:
                        embedding = self._encode(batch_text).cpu().numpy()
                        text_embeddings[i - batch_size: i] = embedding
                    batch_text = []
                batch_text.append(" ".join([case["title"] if "title" in case else "", case["content"] if "content" in case else ""]))

        return text_embeddings


    def encode_single_query(self, query):
        """ encode a single query
        """
        embedding = self._encode([query]).squeeze(0).tolist()
        return embedding



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DPR")
    parser.add_argument("--data", default="wenshu")
    parser.add_argument("--device", type=lambda x: int(x) if x != "cpu" else "cpu", default=0)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--pooling_method", choices=["cls"], default="cls")
    args = parser.parse_args()


    plm = AutoModel.from_pretrained(os.path.join("data", "model", args.model)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("data", "model", args.model))

    model = GenericPLMEncoder(plm=plm, tokenizer=tokenizer, device=args.device, pooling_method=args.pooling_method)

    text_path = f"/home/peitian_zhang/Data/{args.data}/wenshu1.json"
    save_dir = os.path.join("data", "encode", args.model, args.data)
    model.encode_text(text_path, save_dir, batch_size=args.batch_size)