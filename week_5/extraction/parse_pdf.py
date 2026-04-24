import re
import json
from pathlib import Path

IN_FILE = Path(r"\constitution_data\constitution.txt")
OUT_FILE = Path("ingested/constitution_kg.json")


INSTITUTIONS = [
    "President", "Vice-President", "Prime Minister", "Council of Ministers",
    "Federal Parliament", "House of Representatives", "National Assembly",
    "State Assembly", "Supreme Court", "High Courts", "District Courts",
    "Election Commission", "Attorney General", "Commission for the Investigation of Abuse of Authority",
    "Nepal Police", "Armed Police Force", "National Investigation Department",
    "Comptroller and Auditor General", "Public Service Commission",
    "Constitutional Council", "Local Government Institutions"
]

PART_PAT = re.compile(r'^\s*Part\s*-\s*(\d+)\s*$', re.IGNORECASE)
ARTICLE_PAT = re.compile(r'^\s*(\d+)\.\s*')
SCHEDULE_PAT = re.compile(r'^\s*Schedule\s*-\s*(\d+)\s*$', re.IGNORECASE)
RELATES_TO_ARTICLE_PAT = re.compile(r'Article\s+(\d+)', re.IGNORECASE)


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_title(full_text):
    if ':' in full_text:
        return normalize(full_text.split(':', 1)[0])
    else:
        return normalize(full_text.split('.')[0])

def split_clauses(text):
    # Split clauses that start with (number) anywhere after colon or period
    pattern = re.compile(r'(?<=[\.:])\s*\((\d+)\)\s*')
    matches = list(pattern.finditer(text))
    clauses = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        clause_text = text[start:end].strip()
        if clause_text:
            clauses.append({
                "number": int(match.group(1)),
                "text": normalize(clause_text)
            })
    return clauses

def extract_relates_to(text, current_article_id):
    edges = []
    for match in RELATES_TO_ARTICLE_PAT.finditer(text):
        edges.append({"from": current_article_id, "to": f"article:{match.group(1)}", "relation": "RELATES_TO"})
    return edges

def extract_institutions(text):
    return [inst for inst in INSTITUTIONS if inst in text]

def extract_tags(title, full_text):
    keywords = ["Freedom", "Rights", "Citizenship", "Judiciary", "Federalism", 
                "Political Parties", "Elections", "Parliament", "President", "Local Government"]
    tags = set()
    text = f"{title} {full_text}".lower()
    for kw in keywords:
        if kw.lower() in text:
            tags.add(kw)
    return list(tags)

# Main parser
def parse_constitution():
    lines = IN_FILE.read_text(encoding="utf-8").splitlines()
    parts = []
    schedules = []
    i = 0
    in_schedule_section = False

    while i < len(lines):
        line = lines[i].strip()
        if SCHEDULE_PAT.match(line):
            in_schedule_section = True

        if not in_schedule_section:
            part_match = PART_PAT.match(line)
            if part_match:
                part_no = int(part_match.group(1))
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                part_title = normalize(lines[j]) if j < len(lines) else ""

                k = j + 1
                part_lines = []
                while k < len(lines):
                    if PART_PAT.match(lines[k].strip()) or SCHEDULE_PAT.match(lines[k].strip()):
                        break
                    part_lines.append(lines[k])
                    k += 1

                articles = []
                current_article = None
                for l in part_lines:
                    art_match = ARTICLE_PAT.match(l)
                    if art_match:
                        if current_article:
                            full_text = normalize("\n".join(current_article["lines"]))
                            current_article["text"] = full_text
                            current_article["title"] = extract_title(full_text)
                            current_article["clauses"] = split_clauses(full_text)
                            current_article["relates_to"] = extract_relates_to(full_text, current_article["id"])
                            current_article["institutions"] = extract_institutions(full_text)
                            current_article["governs"] = current_article["institutions"]
                            current_article["tags"] = extract_tags(current_article["title"], full_text)
                            if part_no == 3:
                                current_article["rights"] = [current_article["title"]]
                            if part_no == 4:
                                current_article["directive_principles"] = [current_article["title"]]
                            del current_article["lines"]
                            articles.append(current_article)

                        art_no = int(art_match.group(1))
                        rest_of_line = l[art_match.end():].strip()
                        current_article = {
                            "id": f"article:{art_no}",
                            "number": art_no,
                            "lines": [rest_of_line] if rest_of_line else []
                        }
                    elif current_article:
                        current_article["lines"].append(l)

                if current_article:
                    full_text = normalize("\n".join(current_article["lines"]))
                    current_article["text"] = full_text
                    current_article["title"] = extract_title(full_text)
                    current_article["clauses"] = split_clauses(full_text)
                    current_article["relates_to"] = extract_relates_to(full_text, current_article["id"])
                    current_article["institutions"] = extract_institutions(full_text)
                    current_article["governs"] = current_article["institutions"]
                    current_article["tags"] = extract_tags(current_article["title"], full_text)
                    if part_no == 3:
                        current_article["rights"] = [current_article["title"]]
                    if part_no == 4:
                        current_article["directive_principles"] = [current_article["title"]]
                    del current_article["lines"]
                    articles.append(current_article)

                parts.append({
                    "id": f"part:{part_no}",
                    "number": part_no,
                    "title": part_title,
                    "articles": articles
                })
                i = k
            else:
                i += 1
        else:  # schedules
            sched_match = SCHEDULE_PAT.match(line)
            if sched_match:
                sched_no = int(sched_match.group(1))
                k = i + 1
                sched_lines = []
                while k < len(lines):
                    if SCHEDULE_PAT.match(lines[k].strip()):
                        break
                    sched_lines.append(lines[k])
                    k += 1
                sched_text = normalize("\n".join(sched_lines))
                articles_ref = [f"article:{n}" for n in RELATES_TO_ARTICLE_PAT.findall(sched_text)]
                institutions_ref = extract_institutions(sched_text)
                schedules.append({
                    "id": f"schedule:{sched_no}",
                    "number": sched_no,
                    "text": sched_text,
                    "refers_to_articles": articles_ref,
                    "institutions": institutions_ref
                })
                i = k
            else:
                i += 1

    return {"parts": parts, "schedules": schedules}


# Run parser
if __name__ == "__main__":
    data = parse_constitution()
    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Detected {len(data['parts'])} Parts and {len(data['schedules'])} Schedules")
