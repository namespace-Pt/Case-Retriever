def dpr_encode(query, model, tokenizer):
    embedding = model(**tokenizer(query, return_tensors="pt", max_length=512, truncation=True))[0][0, 0]
    return embedding.tolist()