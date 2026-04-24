import streamlit as st
import json
import re

# ---------------------------------
# Load Constitution KG
# ---------------------------------
@st.cache_data
def load_kg(json_path="constitution_kg.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = {}

    # data = {"parts": [...], "schedules": [...]}
    for part in data.get("parts", []):
        for art in part.get("articles", []):
            articles[art["id"]] = art

    return articles


articles = load_kg()

# ---------------------------------
# Helper Functions
# ---------------------------------
def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def search_articles(query, top_k=5):
    query = query.strip().lower()
    results = []

    # Search by article number
    if query.isdigit():
        art_id = f"article:{int(query)}"
        if art_id in articles:
            return [articles[art_id]]

    # Search by keyword
    for art in articles.values():
        text_blocks = [
            art.get("title", ""),
            art.get("text", "")
        ]

        for c in art.get("clauses", []):
            text_blocks.append(c.get("text", ""))

        combined = " ".join(text_blocks).lower()

        if query in combined:
            results.append(art)

    return results[:top_k]


def get_related_articles(article):
    related_articles = []

    for rel in article.get("relates_to", []):
        if isinstance(rel, dict) and "to" in rel:
            target_id = rel["to"]
        elif isinstance(rel, str):
            target_id = rel
        else:
            continue

        if target_id in articles:
            related_articles.append(articles[target_id])

    return related_articles


# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(
    page_title="Constitution Knowledge Graph",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>📜 Constitution Knowledge Graph Explorer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Search by keyword or article number (e.g., 289)</p>",
    unsafe_allow_html=True
)

query = st.text_input("", placeholder="Enter keyword or article number")
top_k = 5

st.markdown("---")

if query:
    results = search_articles(query, top_k)

    if not results:
        st.warning("No matching articles found.")
    else:
        for art in results:
            st.subheader(f"Article {art['number']} — {art.get('title','')}")

            st.write(art.get("text", ""))

            # Clauses
            if art.get("clauses"):
                st.markdown("**Clauses:**")
                for c in art["clauses"]:
                    st.write(f"({c['number']}) {c['text']}")

            # Institutions
            if art.get("institutions"):
                st.markdown(f"**Institutions Involved:** {', '.join(art['institutions'])}")

            # Tags
            if art.get("tags"):
                st.markdown(f"**Tags:** {', '.join(art['tags'])}")

            # Related Articles (KG edges)
            related = get_related_articles(art)
            if related:
                st.markdown("### 🔗 Related Constitutional Articles")
                for r in related:
                    st.markdown(
                        f"""
                        **Article {r['number']} — {r.get('title','')}**  
                        {r.get('text','')}
                        """
                    )

            st.markdown("---")
