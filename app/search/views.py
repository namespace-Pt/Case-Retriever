import json
import numpy as np
from tqdm import tqdm
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer
from utils.encode import GenericPLMEncoder


elastic = Elasticsearch(
    "http://localhost:9200"
)
default_index = "wenshu"
search_size = 100
facet_size = 10

plm_dir = "data/model/DPR"
plm = AutoModel.from_pretrained(plm_dir)
tokenizer = AutoTokenizer.from_pretrained(plm_dir)
model = GenericPLMEncoder(plm=plm, tokenizer=tokenizer)



def bm25_search(query, facets):
    """
    returning the processed hits, each element of which is a single document
    """
    resp = elastic.search(
        index=default_index,
        source=False,
        size=search_size,
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
        "aggregations": aggregations
    }


def knn_search(**kargs):
    hits = elastic.knn_search(**kargs)["hits"]["hits"]
    processed_hits = []

    for hit in hits:
        fields = hit["fields"]
        # add [0] because elastic returns list by default
        new_hit = {
            "title": fields["title"][0][:100],
            "content": fields["content"][0][:500] if "content" in fields else "EMPTY CONTENT!",
            "court": fields["court"],
            "date": fields["judge_date"]
        }
        # FIXME: _id is not by the field
        new_hit["_id"] = hit["_id"]
        # set highlight with blue color
        processed_hits.append(new_hit)
    return {
        "hits": processed_hits
    }


def main(request):
    if request.method == "GET":
        context = {}
        return render(request, "search/index.html", context)

    elif request.method == "POST":
        data = json.loads(request.body)

        query = data["query"]
        backbone = data["backbone"].lower()
        # use get to return None if the post doesn't contain facets
        facets = data.get("facets")

        if backbone == "bm25":
            resp = bm25_search(query, facets)

        elif backbone == "dpr":
            resp = knn_search(
                index=default_index,
                fields=["title", "content", {"field": "judge_date", "format": "year_month_day"}, "court"],
                knn={
                    "field": "vector",
                    "query_vector": model.encode_single_query(query),
                    "k": 10,
                    # this is necessary
                    "num_candidates": 100
                }
            )

        return JsonResponse(data=resp)



def detail(request, id):
    hit = elastic.get(index=default_index, id=id)
    return render(request, "search/detail.html", hit["_source"])


def debug(request):
    if request.method == "GET":
        return render(request, "search/debug.html")
    elif request.method == "POST":
        return HttpResponse("fuck it")