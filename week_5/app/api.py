"""
FastAPI for Constitutional Article Classification
Uses a pre-trained GCN model with SentenceTransformer embeddings
to predict article categories: Rights, Governance, Judiciary, Federalism, Other.
"""

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --------------------
# Define categories
# --------------------
CATEGORY_MAP = ["Rights", "Governance", "Judiciary", "Federalism", "Other"]
cat2idx = {c: i for i, c in enumerate(CATEGORY_MAP)}
idx2cat = {i: c for i, c in enumerate(CATEGORY_MAP)}

# --------------------
# GCN Model
# --------------------
class GCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.c1 = GCNConv(in_c, hid_c)
        self.c2 = GCNConv(hid_c, hid_c)
        self.c3 = GCNConv(hid_c, out_c)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = F.dropout(x, 0.3, training=self.training)
        return self.c3(x, edge_index)

# --------------------
# Input schema
# --------------------
class ArticleInput(BaseModel):
    text: str

# --------------------
# Initialize FastAPI
# --------------------
app = FastAPI(title="Constitution Article Classification API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load embedding model
# --------------------
embedding_model_name = "all-mpnet-base-v2"  # stronger than MiniLM
embedder = SentenceTransformer(embedding_model_name)

# --------------------
# Load trained GCN model
# --------------------
embedding_size = 768  # all-mpnet-base-v2 output
hidden_channels = 256
num_classes = len(CATEGORY_MAP)

gcn_model = GCN(
    embedding_size,  # in_c
    hidden_channels, # hid_c
    num_classes      # out_c
)
gcn_model.load_state_dict(torch.load(r"C:\Users\Acer\Desktop\projects\week_5\day1_node\saved_models\gcn_article_classifier .pt", map_location=device))
gcn_model.eval()

# --------------------
# Helper function
# --------------------
def map_category(article_number):
    """Map article numbers to simplified categories."""
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
    else:
        return "Other"

# --------------------
# Prediction route
@app.post("/predict")
def predict(article: ArticleInput):
    try:
        # 1️⃣ Generate embedding
        emb = embedder.encode(article.text, convert_to_tensor=True).to(device)  # [768]

        # 2️⃣ Create node feature matrix (1 node)
        x = emb.unsqueeze(0)  # [1, 768]

        # 3️⃣ Dummy self-loop edge
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

        # 4️⃣ Forward pass (unpack x and edge_index)
        with torch.no_grad():
            logits = gcn_model(x, edge_index)      # ✅ pass x, edge_index separately
            probs = F.softmax(logits, dim=1)       # probabilities

            pred_idx = probs.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()
            pred_label = idx2cat[pred_idx]

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
