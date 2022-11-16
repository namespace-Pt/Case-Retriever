import json
import networkx as nx
from urllib.parse import quote
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from collections import defaultdict
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer
from utils.encode import GenericPLMEncoder
from utils.filter import filter_text
from utils.static import field_name_chinese
from utils.explain import get_explanation_features


elastic = Elasticsearch(
    "http://localhost:9200",
    request_timeout=1000000
)
default_index = "case"
facet_size = 10

plm_dir = "data/model/DPR"
plm = AutoModel.from_pretrained(plm_dir)
tokenizer = AutoTokenizer.from_pretrained(plm_dir)
model = GenericPLMEncoder(plm=plm, tokenizer=tokenizer)



def search(query, backbone, search_field, facets, from_, size):
    """
    returning the processed hits, each element of which is a single document
    """
    query_dict = {}
    if backbone == "关键词查询":
        field = "content" if search_field == "全文" else "basics_text"

        # if the query is wrapped in a quote, then find exact match of the entire sentence
        if query[0] in "\'\"“" and query[-1] in "\'\"”":
            # search the exact match
            query_dict["query"] = {
                "match_phrase" : {
                    field: query
                }
            }
        # search the query appearance in several fields
        else:
            query_dict["query"] = {
                "match": {
                    field: query
                }
            }

        # add highlight over searched fields
        query_dict["highlight"] = {
            "fields": {
                field: {
                    "pre_tags" : ["<strong>"],
                    "post_tags": ["</strong>"],
                    "number_of_fragments": 1,
                }
            }
        }

    elif backbone == "类案查询":
        query, _ = filter_text(query)

        query_dict["knn"] = {
            "field": "vector",
            "query_vector": model.encode_single_query(query.replace("\n", "")),
            "k": 1000,
            # this is necessary
            "num_candidates": 1000
        }

    # add fields that can be inspected in resp["hits"]["fields"]
    query_dict["fields"] = ["case_name", "content", "basics_text", "case_id", "court_name", {"field": "publish_date", "format": "year_month_day"}]

    # add aggregations
    query_dict["aggs"] = {
        "sample": {
            "sampler": {
                "shard_size": 10000
            },
            "aggs":{
                "agg-terms-court_name": {
                    # this terms means find unique terms to build buckets
                    "terms": {
                        "field": "court_name",
                        "size": facet_size
                    }
                },
                "agg-terms-court_province": {
                    # this terms means find unique terms to build buckets
                    "terms": {
                        "field": "court_province",
                        "size": facet_size
                    }
                },
                "agg-terms-court_city": {
                    # this terms means find unique terms to build buckets
                    "terms": {
                        "field": "court_city",
                        "size": facet_size
                    }
                },
                "agg-terms-court_region": {
                    # this terms means find unique terms to build buckets
                    "terms": {
                        "field": "court_region",
                        "size": facet_size
                    }
                },
                "agg-terms-court_district": {
                    # this terms means find unique terms to build buckets
                    "terms": {
                        "field": "court_district",
                        "size": facet_size
                    }
                },
                "agg-terms-pub_prosecution_org":{
                    "terms": {
                        "field": "pub_prosecution_org",
                        "size": facet_size
                    }
                },
                "agg-terms-trial_round": {
                    "terms": {
                        "field": "trial_round",
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
                },
                "agg-terms-legal_base": {
                    "terms": {
                        "field": "legal_base",
                        "size": facet_size
                    }
                },
            }
        }
    }

    # add post filter for facets
    query_dict["post_filter"] = {
        "bool": {
            "filter": [facet for facet in facets]
        }
    } if facets is not None else None

    resp = elastic.search(
        index=default_index,
        source=False,
        size=size if size is not None else 20,
        from_=from_ if from_ is not None else 0,
        # here we insert the query string, it's either bm25 search or vector search
        **query_dict
    )

    hits = resp["hits"]["hits"]
    # use the sampler output
    aggregations = resp["aggregations"]["sample"]
    print(aggregations["doc_count"])
    # get total hits
    total = resp["hits"]["total"]["value"]
    took = round(resp["took"] / 1000, 2)

    processed_hits = []
    # merge _source and hightlight when the highlight field is missing
    for hit in hits:
        fields = hit["fields"]
        # this should be avoided when indexing
        if "basics_text" not in fields:
            fields["basics_text"] = [""]

        # generate content based on search field
        if search_field == "全文":
            content = fields["content"][0][:500] if "content" in fields else "MISSING CONTENT!"
        elif search_field == "仅案情":
            content = fields["basics_text"][0][:500] if "basics_text" in fields else "MISSING CONTENT!"

        # add [0] because elastic returns list by default
        new_hit = {
            "case_name": fields["case_name"][0][:100] if "case_name" in fields else "MISSING CONTENT!",
            "content": content,
            "court_name": fields["court_name"][0] if "court_name" in fields else "",
            "publish_date": fields["publish_date"][0] if "publish_date" in fields else "",
            "case_id": fields["case_id"][0] if fields.get("case_id") else "",
        }

        new_hit["_id"] = hit["_id"]
        # this basics_text is used for searching similar cases
        if fields["basics_text"][0]:
            new_hit["pseudo_basics_text"] = fields["basics_text"][0]
        else:
            new_hit["pseudo_basics_text"] = fields["content"][0].replace("\n", "")

        if "highlight" in hit:
            # assign the highlighted text to overwrite the original text
            for k, v in hit["highlight"].items():
                # override basics_text because it has been merged to content
                if k == "basics_text":
                    k = "content"
                new_hit[k] = v[0]

        # set highlight with blue color
        processed_hits.append(new_hit)

    # reorder the aggregations so that it displays in order
    sorted_aggregations = {}
    for agg in query_dict["aggs"]:
        if agg in aggregations:
            sorted_aggregations[agg] = aggregations[agg]

    return {
        "hits": processed_hits,
        "aggregations": sorted_aggregations,
        "total": total,
        "took": took
    }


def main(request):
    if request.method == "GET":
        context = {
            "field_name_chinese": json.dumps(field_name_chinese, ensure_ascii=False)
        }
        return render(request, "vue/index.html", context)

    elif request.method == "POST":
        data = json.loads(request.body)

        query = data["query"]
        backbone = data["backbone"]
        search_field = data["field"]
        # use get to return None if the post doesn't contain facets
        facets = data.get("facets")
        from_ = data.get("from")
        size = data.get("size")

        resp = search(query, backbone, search_field, facets, from_, size)

        return JsonResponse(resp)


def detail(request, id):
    hit = elastic.get(index=default_index, id=id)
    source = hit["_source"]

    new_source = {}
    for k, v in field_name_chinese.items():
        if source.get(k):
            # field value and chinese name
            new_source[k] = (source[k], v)

    return render(request, "vue/detail.html", {"source": new_source})


def download(request, id):
    hit = elastic.get(index=default_index, id=id)["_source"]
    # essential to quote the chinese so that the file name can be properly parsed
    filename = quote(f"{hit['case_id']}.txt")
    content = f"{hit['case_name']}\n{hit['content']}"
    response = HttpResponse(content, content_type="text/plain")
    # use string concatenation here instead of formatting
    response["Content-Disposition"] = "attachment; filename=" + filename
    return response


def explain(request):
    if request.method == "POST":
        data = json.loads(request.body)

        query = data["query"]
        candidate = data["candidate"]
        candidate_id = data["candidate_id"]
        candidate_source = elastic.get(index=default_index, id=candidate_id)["_source"]

        query_sents, candidate_sents, matched_pairs = get_explanation_features(query, candidate)

        g = nx.Graph()
        query_sents = query_sents[0]
        candidate_sents = candidate_sents[0]
        offset = len(query_sents)

        g.add_nodes_from(range(offset))
        g.add_nodes_from(range(offset, offset + len(candidate_sents), 1))

        for edge in matched_pairs:
            g.add_edge(edge[0], edge[1] + offset)

        # maximum connected component
        S = sorted([g.subgraph(c).copy() for c in nx.connected_components(g)], key=len, reverse=True)
        connected_components = []
        for s in S:
            query_node = []
            candidate_node = []
            for n in s.nodes:
                if n < offset:
                    query_node.append(n)
                else:
                    candidate_node.append(n - offset)
            if len(query_node) == 0 or len(candidate_node) == 0:
                break
            connected_components.append((query_node, candidate_node))

        return render(
            request,
            "vue/explain.html",
            {
                "candidate_source": candidate_source,
                # use json dump; ensure_ascii=False so that chinese characters are valid
                "query_sents": json.dumps(query_sents, ensure_ascii=False),
                "candidate_sents": json.dumps(candidate_sents, ensure_ascii=False),
                "connected_components": json.dumps(connected_components, ensure_ascii=False),
            }
        )


def debug(request):
    if request.method == "GET":
        return render(request, "vue/debug.html")
    elif request.method == "POST":
        return HttpResponse("fuck it")

