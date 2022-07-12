import json
from tqdm import tqdm
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from elasticsearch import Elasticsearch


elastic = Elasticsearch(
    "http://localhost:9200"
)
default_index = "lecard"


def elastic_search(**kargs):
    """
    returning the processed hits, each element of which is a single document
    """
    hits = elastic.search(
        **kargs
    )["hits"]["hits"]

    processed_hits = []
    # merge _source and hightlight when the highlight field is missing
    for hit in hits:
        new_hit = {"_id": hit["_id"]}

        # first assign the highlighted text
        for k, v in hit["highlight"].items():
            new_hit[k] = v[0],

        # next assign the missing text
        if "title" not in new_hit:
            new_hit["title"] = hit["_source"]["title"]
        if "abstract" not in new_hit:
            new_hit["abstract"] = hit["_source"]["abstract"][:500]

        # set highlight with blue color
        processed_hits.append(new_hit)

    return processed_hits


def main(request):
    if request.method == "GET":
        context = {}
        return render(request, "search/index.html", context)

    elif request.method == "POST":
        query = request.POST["query"]
        backbone = request.POST["backbone"]

        hits = elastic_search(
            index=default_index,
            query={
                "combined_fields": {
                    "query": query,
                    "fields": ["title", "abstract"]
                }
            },
            # set color
            highlight={
                "fields": {
                    "title": {
                        "pre_tags" : ["<strong>"],
                        "post_tags": ["</strong>"],
                        "number_of_fragments": 2,
                    },
                    "abstract": {
                        "pre_tags" : ["<strong>"],
                        "post_tags": ["</strong>"],
                        "number_of_fragments": 2,
                    }
                }
            }
        )
        return JsonResponse(data={"hits": hits})


def detail(request, id):
    hit = elastic.get(index=default_index, id=id)
    return render(request, "search/detail.html", hit["_source"])


def debug(request):
    if request.method == "GET":
        return render(request, "search/debug.html")
    elif request.method == "POST":
        return HttpResponse("fuck it")