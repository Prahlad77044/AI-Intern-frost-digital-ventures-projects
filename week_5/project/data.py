import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from config import cat2idx, EMBED_MODEL

# -------------------------
# CATEGORY MAPPING
# -------------------------
def map_category(article_number):
    try:
        n = int(''.join(filter(str.isdigit, str(article_number))))
    except ValueError:
        return "Other"

    if 16 <= n < 47:
        return "Rights"
    elif 47 <= n < 101:
        return "Governance"
    elif 126 <= n < 142:
        return "Judiciary"
    elif (5 <= n < 16) or (142 <= n < 200):
        return "Federalism"
    return "Other"

# -------------------------
# FETCH FROM NEO4J
# -------------------------
def fetch_graph(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        nodes = session.run("""
            MATCH (n:Article)
            RETURN id(n) AS neo4j_id,
                   n.id AS article_id,
                   n.text AS text,
                   n.title AS title
        """).data()

        edges = session.run("""
            MATCH (a:Article)-[:RELATES_TO]->(b:Article)
            RETURN id(a) AS source_id, id(b) AS target_id
        """).data()

    driver.close()
    return nodes, edges

# -------------------------
# BUILD GRAPH
# -------------------------
def build_graph(nodes, edges):
    embedder = SentenceTransformer(EMBED_MODEL)

    texts, node_map, labels = [], {}, {}
    for i, n in enumerate(nodes):
        aid = (n.get("article_id") or f"article:{i}").lower()
        text = n.get("text") or n.get("title") or ""
        node_map[n["neo4j_id"]] = i
        texts.append(text)
        labels[i] = cat2idx[map_category(aid)]

    x = torch.tensor(
        embedder.encode(texts, normalize_embeddings=True),
        dtype=torch.float
    )

    data = HeteroData()
    data["Article"].x = x
    data["Article"].y = torch.tensor(list(labels.values()))

    edge_index = [
        [node_map[e["source_id"]], node_map[e["target_id"]]]
        for e in edges
        if e["source_id"] in node_map and e["target_id"] in node_map
    ]

    data["Article", "relates_to", "Article"].edge_index = (
        torch.tensor(edge_index).t().contiguous()
    )

    return ToUndirected()(data)

# -------------------------
# SPLITS
# -------------------------
def create_splits(data, train=0.7, val=0.15):
    torch.manual_seed(42)

    n = data["Article"].x.size(0)
    perm = torch.randperm(n)

    n_train = int(train * n)
    n_val = int(val * n)

    data["Article"].train_mask = torch.zeros(n, dtype=torch.bool)
    data["Article"].val_mask = torch.zeros(n, dtype=torch.bool)
    data["Article"].test_mask = torch.zeros(n, dtype=torch.bool)

    data["Article"].train_mask[perm[:n_train]] = True
    data["Article"].val_mask[perm[n_train:n_train+n_val]] = True
    data["Article"].test_mask[perm[n_train+n_val:]] = True

    return data
