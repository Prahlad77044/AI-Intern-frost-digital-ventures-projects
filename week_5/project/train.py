import os, torch
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

from config import *
from data import fetch_graph, build_graph, create_splits
from models import GCN, GraphSAGE

# -------------------------
# TRAIN / EVAL
# -------------------------
def train_epoch(model, data, opt):
    model.train()
    opt.zero_grad()

    out = model(
        data["Article"].x,
        data["Article", "relates_to", "Article"].edge_index
    )

    mask = data["Article"].train_mask
    y = data["Article"].y[mask]

    counts = Counter(y.tolist())
    weights = torch.tensor(
        [1 / counts.get(i, 1) for i in range(out.size(1))]
    )

    loss = F.cross_entropy(out[mask], y, weight=weights)
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(
        data["Article"].x,
        data["Article", "relates_to", "Article"].edge_index
    )
    pred = out.argmax(dim=1)
    y = data["Article"].y[data["Article"][mask]]
    p = pred[data["Article"][mask]]

    return (
        accuracy_score(y, p),
        f1_score(y, p, average="weighted")
    )

# -------------------------
# MAIN
# -------------------------
def main():
    set_seed()

    nodes, edges = fetch_graph(URI, USER, PASSWORD)
    data = create_splits(build_graph(nodes, edges))

    in_c = data["Article"].x.size(1)
    out_c = len(cat2idx)

    for Model in [GCN, GraphSAGE]:
        model = Model(in_c, HIDDEN_DIM, out_c)
        opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)

        best_f1, wait = 0, 0
        for _ in range(EPOCHS):
            train_epoch(model, data, opt)
            _, f1 = evaluate(model, data, "val_mask")

            if f1 > best_f1:
                best_f1 = f1
                best = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait == PATIENCE:
                    break

        model.load_state_dict(best)
        acc, f1 = evaluate(model, data, "test_mask")
        print(f"{Model.__name__}: Acc={acc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    main()
