import json
from urllib.parse import quote
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer
from utils.encode import GenericPLMEncoder


elastic = Elasticsearch(
    "http://localhost:9200"
)
default_index = "wenshu"
facet_size = 10

plm_dir = "data/model/DPR"
plm = AutoModel.from_pretrained(plm_dir)
tokenizer = AutoTokenizer.from_pretrained(plm_dir)
model = GenericPLMEncoder(plm=plm, tokenizer=tokenizer)



def bm25_search(query, facets, from_, size):
    """
    returning the processed hits, each element of which is a single document
    """
    resp = elastic.search(
        index=default_index,
        source=False,
        size=size if size is not None else 20,
        from_=from_ if from_ is not None else 0,
        query={
            "combined_fields": {
                "query": query,
                "fields": ["case_name", "content"]
            }
        },
        fields=["case_name", "content", {"field": "publish_date", "format": "year_month_day"}, "court_name", "cause"],
        # set color
        highlight={
            "fields": {
                "case_name": {
                    "pre_tags" : ["<strong>"],
                    "post_tags": ["</strong>"],
                    "number_of_fragments": 1,
                },
                "content": {
                    "pre_tags" : ["<strong>"],
                    "post_tags": ["</strong>"],
                    "number_of_fragments": 1,
                }
            }
        },
        aggs={
            "agg-terms-court_name": {
                # this terms means find unique terms to build buckets
                "terms": {
                    "field": "court_name",
                    "size": facet_size
                }
            },
            "agg-terms-case_type": {
                "terms": {
                    "field": "case_type",
                    "size": facet_size
                }
            },
            "agg-terms-cause": {
                "terms": {
                    "field": "cause",
                    "size": facet_size
                }
            }
        },
        post_filter={
            "bool": {
                "filter": [facet for facet in facets]
            }
        } if facets is not None else None
    )

    hits = resp["hits"]["hits"]
    aggregations = resp["aggregations"]
    # get total hits
    total = resp["hits"]["total"]["value"]

    processed_hits = []
    # merge _source and hightlight when the highlight field is missing
    for hit in hits:
        fields = hit["fields"]
        # add [0] because elastic returns list by default
        new_hit = {
            "case_name": fields["case_name"][0][:100],
            "content": fields["content"][0][:500] if "content" in fields else "EMPTY CONTENT!",
            "court_name": fields["court_name"],
            "publish_date": fields["publish_date"],
        }
        # FIXME: _id is not by the field
        new_hit["_id"] = hit["_id"]

        # first assign the highlighted text
        for k, v in hit["highlight"].items():
            new_hit[k] = v[0]

        # set highlight with blue color
        processed_hits.append(new_hit)

    return {
        "hits": processed_hits,
        "aggregations": aggregations,
        "total": total
    }


def knn_search(query, facets, from_, size):
    hits = elastic.knn_search(
        index=default_index,
        source=False,
        fields=["case_name", "content", {"field": "publish_date", "format": "year_month_day"}, "court_name", "cause"],
        knn={
            "field": "vector",
            "query_vector": model.encode_single_query(query),
            "k": from_ + size,
            # this is necessary
            "num_candidates": 100
        }
    )["hits"]["hits"][from_:]

    processed_hits = []

    for hit in hits:
        fields = hit["fields"]
        # add [0] because elastic returns list by default
        new_hit = {
            "case_name": fields["case_name"][0][:100],
            "content": fields["content"][0][:500] if "content" in fields else "EMPTY CONTENT!",
            "court_name": fields["court_name"],
            "publish_date": fields["publish_date"],
        }
        # FIXME: _id is not by the field
        new_hit["_id"] = hit["_id"]
        # set highlight with blue color
        processed_hits.append(new_hit)
    return {
        "hits": processed_hits,
        "total": size
    }


def main(request):
    if request.method == "GET":
        context = {}
        return render(request, "search/index.html", context)

    elif request.method == "POST":
        data = json.loads(request.body)

        query = data["query"]
        backbone = data["backbone"]
        # use get to return None if the post doesn't contain facets
        facets = data.get("facets")
        from_ = data.get("from")
        size = data.get("size")

        if backbone == "关键词查询":
            resp = bm25_search(query, facets, from_, size)

        elif backbone == "类案查询":
            resp = knn_search(query, facets, from_, size)

        return JsonResponse(data=resp)


def detail(request, id):
    hit = elastic.get(index=default_index, id=id)
    return render(request, "search/detail.html", hit["_source"])


def download(request, id):
    hit = elastic.get(index=default_index, id=id)["_source"]
    # essential to quote the chinese so that the file name can be properly parsed
    filename = quote(f"{hit['case_id']}.txt")
    content = f"{hit['case_name']}\n{hit['content']}"
    response = HttpResponse(content, content_type="text/plain")
    # use string concatenation here instead of formatting
    response["Content-Disposition"] = "attachment; filename=" + filename
    return response


def debug(request):
    if request.method == "GET":
        return render(request, "search/debug.html")
    elif request.method == "POST":
        return HttpResponse("fuck it")

