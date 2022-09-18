import re
import argparse
import json
from tqdm import tqdm

patterns = [
    (re.compile("(本院)?(经审理)?(认定|查明)(本案)?(事实如下)?.(.*)(本院|合议庭评议).*认为"), 6),
    (re.compile("(本院)?(确认)(以下|如下)(法律|案件)?(事实).(.*)(本院|合议庭评议).*认为"), 6),
    (re.compile("(事实和理由|具体事实)(如下)?.(.*)本院.*认为"), 3),
    (re.compile("公诉机关指控.(.*)公诉机关认为"), 1),
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

    with open(f"../../../Data/{args.data}/{args.file}", encoding="utf-8") as f, open(f"../../../Data/{args.data}/{tf_name}", "w", encoding="utf-8") as g:
        for i, line in enumerate(tqdm(f)):
            if i < args.start:
                continue
            elif i >= args.end and args.end > 0:
                break

            fields = json.loads(line.strip())

            if fields["basics_text"]:
                content = fields["basics_text"].replace("\n", "")
                is_valid = 1
            elif fields["content"]:
                content = fields["content"].replace("\n", "")
                content, is_valid = filter_text(content)
            else:
                is_valid = 0

            skipped += 1 - is_valid

            # write filter content to a new field
            if is_valid:
                fields["tf_content"] = content
            else:
                if args.debug:
                    print(fields["content"])
                    if skipped == 50:
                        break
                    continue

            g.write(json.dumps(fields, ensure_ascii=False) + "\n")

    print(f"{i + 1 - skipped} documents are kept, {skipped} documents are skipped!")

    json.dump({"valid_tf_content_count": i + 1 - skipped}, open(f"../../../Data/{args.data}/{tf_name}.stat", "w", encoding="utf-8"))
