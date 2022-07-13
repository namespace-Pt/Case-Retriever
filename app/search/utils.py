def dpr_encode(query, model, tokenizer):
    embedding = model(**tokenizer(query, return_tensors="pt"))[0][0, 0]
    return embedding.tolist()