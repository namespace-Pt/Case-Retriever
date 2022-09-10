import argparse
import json
from tqdm import tqdm


def filter_text(text):
    text = text.replace("/n", "")

    for x in ("指控：", "指控，", "查明：", "查明，", "诉称：", "诉称，", "请求：", "理由：", "认定：", "认定:", "认定，", "查认为，"):
        idx = text.find(x)
        if idx != -1:
            break

    if idx != -1:
        text = text[idx:]
        return text, 1
    else:
        text = text
        return text, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="wenshu")
    parser.add_argument("--file", default="p4.txt")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)

    args = parser.parse_args()

    name, extension = args.file.split(".")
    filtered_name = ".".join([name, "filtered", extension])

    # filter text
    skipped = 0
    print(f"filtering {args.file}, the output will be save at {filtered_name}...")

    with open(f"../../../Data/{args.data}/{args.file}", encoding="utf-8") as f, open(f"../../../Data/{args.data}/{filtered_name}", "w", encoding="utf-8") as g:
        for i, line in enumerate(tqdm(f)):
            if i < args.start:
                continue
            elif i > args.end and args.end > 0:
                break

            fields = json.loads(line.strip())

            if fields["basics_text"]:
                content = fields["basics_text"].replace("/n", "")
                filtered = 1
            else:
                content = fields["content"].replace("/n", "")
                if not content:
                    skipped += 1
                    continue
                content, filtered = filter_text(content)

            skipped += 1 - filtered
            # write filter content to a new field
            fields["tf_content"] = content
            g.write(json.dumps(fields, ensure_ascii=False) + "\n")

    print(f"{i + 1 - skipped} documents are filtered, {skipped} documents are skipped!")