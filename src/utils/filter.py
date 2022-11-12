import os
import re
import argparse
import json
from tqdm import tqdm

"""
确认以下基本事实：
本院认定如下事实，
本院认定下列事实，
本院认定本案的事实如下：

本院对本案事实认定如下，
本案事实认定如下
对本案主要事实作如下认定：

经审理查明，
经审理认定，
经本院审理查明，
经审理本院认定，
经审理，本院查明：
本院根据上述认证查明：


本院经审理，认定事实如下，
"""

patterns = [
    (re.compile("(认定|查明|查证|确认|证实).?(本案|如下|以下|下列)?.?(法律|案件|主要|基本)?事实(如下)?.(.*)(本院|合议庭评议)(经审查)?(认为|依照)"), 5),
    (re.compile("(认定|查明|查证|确认|证实).?(本案|如下|以下|下列)?.?(法律|案件|主要|基本)?事实(如下)?.(.*)(上述事实|如不服本判决)"), 5),
    (re.compile("(审理|事实)(本院|作如下)?(查明|认定)(如下)?.(.*)(本院|合议庭评议)(经审查)?(认为|依照)"), 5),
    (re.compile("(审理|事实)(本院|作如下)?(查明|认定)(如下)?.(.*)(上述事实|如不服本判决)"), 5),
    (re.compile("本院(根据上述认证)?查明.(.*)(本院|合议庭评议)(经审查)?(认为|依照)"), 2),
    (re.compile("本院(根据上述认证)?查明.(.*)(上述事实|如不服本判决)"), 2),
    (re.compile("(事实和理由|具体事实|事实确认|案件事实)(如下)?.(.*)(本院|合议庭评议)(经审查)?(认为|依照)"), 3),
    (re.compile("(事实和理由|具体事实|事实确认|案件事实)(如下)?.(.*)(上述事实|如不服本判决)"), 3),
    (re.compile("(公诉机关|检察院)指控.(.*)(公诉机关|本院)(经审查)?(认为|依照)"), 2),
    (re.compile("(公诉机关|检察院)指控.(.*)(上述事实|如不服本判决)"), 2),
    # (re.compile("(原告).*(诉称|提出诉讼请求).(.*)(本院|合议庭评议).*(经审查)?(认为|依照)"), 2),
]


def filter_text(text):
    """ extract basic text
    """
    text = text.replace("\n", "")

    for pattern in patterns:
        basic_text = re.search(pattern[0], text)
        if basic_text:
            return basic_text.group(pattern[1]), 1

    return text, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="wenshu")
    parser.add_argument("--file", default="p2-1.txt")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    name, extension = args.file.split(".")
    tf_name = ".".join([name, "filtered", extension])

    # filter text
    skipped = 0
    print(f"filtering {args.file}, the output will be saved at {tf_name}...")

    if args.debug:
        error_log_path = f"../../../Data/{args.data}/{name}.filter-error.txt"
        if os.path.exists(error_log_path):
            os.remove(error_log_path)
        h = open(error_log_path, "a+", encoding="utf-8")

    with open(f"../../../Data/{args.data}/{args.file}", encoding="utf-8") as f, \
        open(f"../../../Data/{args.data}/{tf_name}", "w", encoding="utf-8") as g:
        for i, line in enumerate(tqdm(f)):
            if i < args.start:
                continue
            elif i >= args.end and args.end > 0:
                break

            fields = json.loads(line.strip())

            if fields["basics_text"]:
                content = fields["basics_text"].replace("\n", "")
                content, _ = filter_text(content)
                is_valid = 1
            elif fields["content"]:
                content = fields["content"].replace("\n", "")
                content, is_valid = filter_text(content)
            else:
                is_valid = 0

            skipped += 1 - is_valid

            # write filter content to a new field
            if is_valid:
                fields["basics_text"] = content
            else:
                if fields["case_name"]:
                    clean_title = fields["case_name"].replace(" ", "")
                else:
                    clean_title = ""

                if args.debug:
                    if fields["content"] and "裁定" not in clean_title and "决定" not in clean_title:
                        h.write(fields["content"])

                    if skipped == 10000:
                        break
                    continue

            g.write(json.dumps(fields, ensure_ascii=False) + "\n")

    print(f"{i + 1 - skipped} documents are kept, {skipped} documents are skipped!")

    json.dump({"valid_tf_content_count": i + 1 - skipped}, open(f"../../../Data/{args.data}/{tf_name}.stat", "w", encoding="utf-8"))

    if args.debug:
        h.close()