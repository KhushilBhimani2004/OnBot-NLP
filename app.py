import streamlit as st
import json
import re
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------
# ⚙️ Config
# -----------------------------------------------------
st.set_page_config(page_title="Automation Knowledge Hub", layout="wide")
KB_FOLDER = Path("knowledge_base")
KB_FOLDER.mkdir(exist_ok=True)

# -----------------------------------------------------
# 🔧 Model Loading (cached)
# -----------------------------------------------------
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_cross_encoder():
    """Optional reranker model (cached)."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None

model = load_semantic_model()
reranker_model = load_cross_encoder()

# -----------------------------------------------------
# 🧩 Helper Functions
# -----------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and ensure safe string."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def save_bot_to_kb(bot_data: dict):
    """Save bot metadata safely into the knowledge_base folder (no file content)."""
    name = bot_data.get("bot_name") or "UnknownBot"
    safe_name = re.sub(r"[^\w\-_\. ]", "_", name)
    p = KB_FOLDER / f"{safe_name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(bot_data, f, indent=4, ensure_ascii=False)

def load_kb():
    """Load all bot JSONs from knowledge_base folder."""
    bots = []
    for p in KB_FOLDER.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                bots.append(json.load(f))
        except Exception:
            continue
    return bots

def beautify_bot_description(bot_data: dict) -> str:
    """Format bot info nicely for markdown display (no file content)."""
    name = bot_data.get("bot_name", "Unnamed Bot")
    uploader = bot_data.get("user_name", "Unknown")
    upload_date = bot_data.get("upload_date", "Unknown")
    desc = bot_data.get("description", "No description provided.")
    notes = bot_data.get("comments", "")
    functions = bot_data.get("functions_details", "")

    md = f"### 🤖 {name}\n\n"
    md += f"**Uploaded by:** {uploader} | **Upload Date:** {upload_date}\n\n"
    md += f"**Overview:** {desc}\n\n"

    if functions.strip():
        md += f"**Functions / Scripts:** {functions}\n\n"

    if notes.strip():
        md += f"**Notes:** {notes}\n\n"

    return md

def find_matching_bot_names(query: str, kb_bots: list) -> list:
    """Return bots that partially match the search text."""
    q = (query or "").strip().lower()
    return [
        (bot.get("bot_name") or "")
        for bot in kb_bots
        if not q or q in (bot.get("bot_name", "")).lower()
    ]

def get_bot_by_name(name: str, kb_bots: list):
    """Return the bot dict by name."""
    for bot in kb_bots:
        if bot.get("bot_name", "") == name:
            return bot
    return None

def build_bot_text_blocks(bot: dict) -> list:
    """Build searchable text blocks only from description, functions, and notes."""
    blocks = []
    if bot.get("description"):
        blocks.append(("Overview", clean_text(bot["description"])))
    if bot.get("functions_details"):
        blocks.append(("Functions / Scripts", clean_text(bot["functions_details"])))
    if bot.get("comments"):
        blocks.append(("Notes / Comments", clean_text(bot["comments"])))
    return blocks

# -----------------------------------------------------
# 🤖 Semantic Search Function
# -----------------------------------------------------
def semantic_answer_for_bot(
    question: str,
    bot: dict,
    threshold: float = 0.10,   # lowered threshold for better recall
    chunk_size: int = 150,
    top_k: int = 5,
    use_reranker: bool = True,
):
    """Semantic retrieval with optional cross-encoder reranking."""

    def split_into_chunks(text, chunk_size_local=chunk_size):
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size_local])
            for i in range(0, len(words), chunk_size_local)
        ]

    question = clean_text(question)
    blocks = build_bot_text_blocks(bot)
    if not blocks:
        return "No searchable text found for this bot.", 0.0, None

    try:
        q_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    except Exception:
        return "Embedding model failure.", 0.0, None

    candidates = []
    for label, text in blocks:
        if not text.strip():
            continue
        for chunk in split_into_chunks(text):
            try:
                chunk_emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
                sem_score = float(util.cos_sim(q_emb, chunk_emb).item())  # ✅ FIXED HERE
            except Exception:
                sem_score = 0.0
            candidates.append((sem_score, label, chunk.strip()))

    if not candidates:
        return "No relevant section found.", 0.0, None

    # Sort top candidates by similarity
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:max(1, top_k)]

    # Optional cross-encoder reranker
    reranked_best = None
    if use_reranker and reranker_model is not None:
        try:
            pairs = [[question, c[2]] for c in top_candidates]
            rerank_scores = reranker_model.predict(pairs)
            reranked = sorted(
                zip(rerank_scores, top_candidates),
                key=lambda x: x[0],
                reverse=True
            )
            best_rscore, (_, best_label, best_text) = reranked[0]
            raw_score = float(best_rscore)
            reranked_best = (raw_score, best_label, best_text)
        except Exception:
            reranked_best = None

    if reranked_best is not None:
        raw_score, label, text = reranked_best
    else:
        raw_score, label, text = top_candidates[0]

    # ✅ Normalize confidence properly (no negatives)
    if reranked_best is not None:
        norm_conf = max(0.0, min(1.0, raw_score / 5.0))  # CrossEncoder outputs 0–5
    else:
        norm_conf = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))  # cosine sim −1→1

    # ✅ Check threshold on normalized value
    if norm_conf < threshold:
        return f"No relevant information found (confidence {norm_conf:.2f}).", norm_conf, None

    return None, norm_conf, {"label": label, "text": text}

# -----------------------------------------------------
# 🧭 Sidebar Navigation
# -----------------------------------------------------
st.sidebar.title("🧩 Automation Knowledge Hub")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", ["📚 Knowledge Base", "💬 Bot Chat"])
st.sidebar.markdown("---")

# -----------------------------------------------------
# 📚 Knowledge Base Page
# -----------------------------------------------------
if page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base: Upload Bot Information")

    with st.form("bot_upload_form"): 
        st.subheader("Upload Bot Metadata")

        user_name = st.text_input("👤 Your Name")
        bot_name = st.text_input("🤖 Bot Name")
        description = st.text_area("📝 Bot Description / Overview")
        functions_details = st.text_area("⚙️ Functions / Scripts Details")
        comments = st.text_area("💡 Notes / Comments (paths, credentials, etc.)")
        uploaded_file = st.file_uploader("📂 Upload Bot File (txt, pdf, json, etc.)")

        submit = st.form_submit_button("Upload Bot to Knowledge Base")

        if submit:
            if not bot_name.strip():
                st.error("Bot Name is required!")
            else:
                try:
                    bot_data = {
                        "user_name": user_name or "Unknown",
                        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "bot_name": bot_name,
                        "description": description,
                        "functions_details": functions_details,
                        "comments": comments,
                    }
                    save_bot_to_kb(bot_data)
                    st.success(f"✅ Bot '{bot_name}' added to Knowledge Base!")
                    st.markdown(beautify_bot_description(bot_data))
                except Exception as e:
                    st.error(f"Error saving bot: {e}")

    st.markdown("---")
    st.subheader("📖 Current Knowledge Base")
    kb_bots = load_kb()

    if kb_bots:
        for bot in kb_bots:
            st.markdown(beautify_bot_description(bot))
            st.divider()
    else:
        st.info("No bots added yet. Upload one above.")

# -----------------------------------------------------
# 💬 Bot Chat (Semantic NLP)
# -----------------------------------------------------
elif page == "💬 Bot Chat":
    st.title("💬 Bot Chat (Semantic NLP)")

    kb = load_kb()
    if not kb:
        st.warning("No bots found. Please upload JSONs first from 'Knowledge Base'.")
    else:
        search_text = st.text_input("🔍 Search bots by name", placeholder="e.g., GFB")
        matches = find_matching_bot_names(search_text, kb)

        if not matches:
            st.info("No bots match your query.")
        else:
            chosen = matches[0] if len(matches) == 1 else st.selectbox("Select a bot", options=matches)

            if chosen:
                bot = get_bot_by_name(chosen, kb)
                st.markdown(beautify_bot_description(bot))
                st.markdown("---")

                st.subheader("Ask a question about this bot")
                user_q = st.text_input("Your question", placeholder="e.g., What columns does it read?")

                if st.button("Get Answer"):
                    if not user_q.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Thinking..."):
                            err, conf, answer = semantic_answer_for_bot(user_q, bot)
                        if err:
                            st.error(err)
                        elif answer:
                            label, text = answer["label"], answer["text"]
                            if label.lower().startswith("overview"):
                                reply = f"This bot mainly does the following: {text}"
                            elif "function" in label.lower():
                                reply = f"It performs these actions or scripts: {text}"
                            elif "note" in label.lower() or "comment" in label.lower():
                                reply = f"Here’s something important to know: {text}"
                            else:
                                reply = text

                            st.success(f"Top Match (confidence {conf:.2f})")
                            st.markdown(f"{reply}\n\n*(Source: {label})*")
                        else:
                            st.info("No relevant information found.")






# import streamlit as st
# import json
# import re
# from datetime import datetime
# from pathlib import Path
# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline

# # -----------------------------------------------------
# # ⚙️ Config
# # -----------------------------------------------------
# st.set_page_config(page_title="Automation Knowledge Hub", layout="wide")
# KB_FOLDER = Path("knowledge_base")
# KB_FOLDER.mkdir(exist_ok=True)

# # -----------------------------------------------------
# # 🔧 Model Loading (cached)
# # -----------------------------------------------------
# @st.cache_resource
# def load_semantic_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_resource
# def load_cross_encoder():
#     """Optional reranker model (cached)."""
#     try:
#         from sentence_transformers import CrossEncoder
#         return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     except Exception:
#         return None

# @st.cache_resource
# def load_generator():
#     """Load Flan-T5 text generation model (cached)."""
#     return pipeline("text2text-generation", model="google/flan-t5-small")

# model = load_semantic_model()
# reranker_model = load_cross_encoder()
# generator = load_generator()

# # -----------------------------------------------------
# # 🧩 Helper Functions
# # -----------------------------------------------------
# def clean_text(text: str) -> str:
#     if not text:
#         return ""
#     return re.sub(r"\s+", " ", str(text)).strip()

# def save_bot_to_kb(bot_data: dict):
#     name = bot_data.get("bot_name") or "UnknownBot"
#     safe_name = re.sub(r"[^\w\-_\. ]", "_", name)
#     p = KB_FOLDER / f"{safe_name}.json"
#     with open(p, "w", encoding="utf-8") as f:
#         json.dump(bot_data, f, indent=4, ensure_ascii=False)

# def load_kb():
#     bots = []
#     for p in KB_FOLDER.glob("*.json"):
#         try:
#             with open(p, "r", encoding="utf-8") as f:
#                 bots.append(json.load(f))
#         except Exception:
#             continue
#     return bots

# def beautify_bot_description(bot_data: dict) -> str:
#     name = bot_data.get("bot_name", "Unnamed Bot")
#     uploader = bot_data.get("user_name", "Unknown")
#     upload_date = bot_data.get("upload_date", "Unknown")
#     desc = bot_data.get("description", "No description provided.")
#     notes = bot_data.get("comments", "")
#     functions = bot_data.get("functions_details", "")

#     md = f"### 🤖 {name}\n\n"
#     md += f"**Uploaded by:** {uploader} | **Upload Date:** {upload_date}\n\n"
#     md += f"**Overview:** {desc}\n\n"

#     if functions.strip():
#         md += f"**Functions / Scripts:** {functions}\n\n"

#     if notes.strip():
#         md += f"**Notes:** {notes}\n\n"

#     return md

# def find_matching_bot_names(query: str, kb_bots: list) -> list:
#     q = (query or "").strip().lower()
#     return [
#         (bot.get("bot_name") or "")
#         for bot in kb_bots
#         if not q or q in (bot.get("bot_name", "")).lower()
#     ]

# def get_bot_by_name(name: str, kb_bots: list):
#     for bot in kb_bots:
#         if bot.get("bot_name", "") == name:
#             return bot
#     return None

# def build_bot_text_blocks(bot: dict) -> list:
#     blocks = []
#     if bot.get("description"):
#         blocks.append(("Overview", clean_text(bot["description"])))
#     if bot.get("functions_details"):
#         blocks.append(("Functions / Scripts", clean_text(bot["functions_details"])))
#     if bot.get("comments"):
#         blocks.append(("Notes / Comments", clean_text(bot["comments"])))
#     return blocks

# # -----------------------------------------------------
# # 🔍 Semantic Search + Generation
# # -----------------------------------------------------
# def semantic_answer_for_bot(
#     question: str,
#     bot: dict,
#     threshold: float = 0.10,
#     chunk_size: int = 150,
#     top_k: int = 5,
#     use_reranker: bool = True,
# ):
#     def split_into_chunks(text, chunk_size_local=chunk_size):
#         words = text.split()
#         return [
#             " ".join(words[i:i + chunk_size_local])
#             for i in range(0, len(words), chunk_size_local)
#         ]

#     question = clean_text(question)
#     blocks = build_bot_text_blocks(bot)
#     if not blocks:
#         return "No searchable text found for this bot.", 0.0, None

#     try:
#         q_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
#     except Exception:
#         return "Embedding model failure.", 0.0, None

#     candidates = []
#     for label, text in blocks:
#         if not text.strip():
#             continue
#         for chunk in split_into_chunks(text):
#             try:
#                 chunk_emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
#                 sem_score = float(util.cos_sim(q_emb, chunk_emb).item())
#             except Exception:
#                 sem_score = 0.0
#             candidates.append((sem_score, label, chunk.strip()))

#     if not candidates:
#         return "No relevant section found.", 0.0, None

#     candidates.sort(key=lambda x: x[0], reverse=True)
#     top_candidates = candidates[:max(1, top_k)]

#     reranked_best = None
#     if use_reranker and reranker_model is not None:
#         try:
#             pairs = [[question, c[2]] for c in top_candidates]
#             rerank_scores = reranker_model.predict(pairs)
#             reranked = sorted(zip(rerank_scores, top_candidates), key=lambda x: x[0], reverse=True)
#             best_rscore, (_, best_label, best_text) = reranked[0]
#             raw_score = float(best_rscore)
#             reranked_best = (raw_score, best_label, best_text)
#         except Exception:
#             reranked_best = None

#     if reranked_best is not None:
#         raw_score, label, text = reranked_best
#     else:
#         raw_score, label, text = top_candidates[0]

#     if reranked_best is not None:
#         norm_conf = max(0.0, min(1.0, raw_score / 5.0))
#     else:
#         norm_conf = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

#     if norm_conf < threshold:
#         return f"No relevant information found (confidence {norm_conf:.2f}).", norm_conf, None

#     return None, norm_conf, {"label": label, "text": text}

# # -----------------------------------------------------
# # 🧠 Transformer-based Answer Generation
# # -----------------------------------------------------
# def generate_transformer_answer(question: str, context: str) -> str:
#     prompt = f"Answer the following question based only on the given context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
#     result = generator(prompt, max_new_tokens=120, temperature=0.7)
#     return result[0]['generated_text']

# # -----------------------------------------------------
# # 🧭 Sidebar Navigation
# # -----------------------------------------------------
# st.sidebar.title("🧩 Automation Knowledge Hub")
# st.sidebar.markdown("---")
# page = st.sidebar.radio("Navigate to:", ["📚 Knowledge Base", "💬 Bot Chat"])
# st.sidebar.markdown("---")

# # -----------------------------------------------------
# # 📚 Knowledge Base Page
# # -----------------------------------------------------
# if page == "📚 Knowledge Base":
#     st.title("📚 Knowledge Base: Upload Bot Information")

#     with st.form("bot_upload_form"):
#         st.subheader("Upload Bot Metadata")

#         user_name = st.text_input("👤 Your Name")
#         bot_name = st.text_input("🤖 Bot Name")
#         description = st.text_area("📝 Bot Description / Overview")
#         functions_details = st.text_area("⚙️ Functions / Scripts Details")
#         comments = st.text_area("💡 Notes / Comments (paths, credentials, etc.)")
#         uploaded_file = st.file_uploader("📂 Upload Bot File (txt, pdf, json, etc.)")

#         submit = st.form_submit_button("Upload Bot to Knowledge Base")

#         if submit:
#             if not bot_name.strip():
#                 st.error("Bot Name is required!")
#             else:
#                 try:
#                     bot_data = {
#                         "user_name": user_name or "Unknown",
#                         "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         "bot_name": bot_name,
#                         "description": description,
#                         "functions_details": functions_details,
#                         "comments": comments,
#                     }
#                     save_bot_to_kb(bot_data)
#                     st.success(f"✅ Bot '{bot_name}' added to Knowledge Base!")
#                     st.markdown(beautify_bot_description(bot_data))
#                 except Exception as e:
#                     st.error(f"Error saving bot: {e}")

#     st.markdown("---")
#     st.subheader("📖 Current Knowledge Base")
#     kb_bots = load_kb()

#     if kb_bots:
#         for bot in kb_bots:
#             st.markdown(beautify_bot_description(bot))
#             st.divider()
#     else:
#         st.info("No bots added yet. Upload one above.")

# # -----------------------------------------------------
# # 💬 Bot Chat (Semantic + Transformer)
# # -----------------------------------------------------
# elif page == "💬 Bot Chat":
#     st.title("💬 Bot Chat (Semantic + Transformer Intelligence)")

#     kb = load_kb()
#     if not kb:
#         st.warning("No bots found. Please upload JSONs first from 'Knowledge Base'.")
#     else:
#         search_text = st.text_input("🔍 Search bots by name", placeholder="e.g., GFB")
#         matches = find_matching_bot_names(search_text, kb)

#         if not matches:
#             st.info("No bots match your query.")
#         else:
#             chosen = matches[0] if len(matches) == 1 else st.selectbox("Select a bot", options=matches)

#             if chosen:
#                 bot = get_bot_by_name(chosen, kb)
#                 st.markdown(beautify_bot_description(bot))
#                 st.markdown("---")

#                 st.subheader("Ask a question about this bot")
#                 user_q = st.text_input("Your question", placeholder="e.g., What columns does it read?")

#                 if st.button("Get Answer"):
#                     if not user_q.strip():
#                         st.warning("Please enter a question.")
#                     else:
#                         with st.spinner("Thinking..."):
#                             err, conf, answer = semantic_answer_for_bot(user_q, bot)
#                             if err:
#                                 st.error(err)
#                             elif answer:
#                                 label, text = answer["label"], answer["text"]
#                                 final_answer = generate_transformer_answer(user_q, text)
#                                 st.success(f"Confidence {conf:.2f} | Source: {label}")
#                                 st.markdown(final_answer)
#                             else:
#                                 st.info("No relevant information found.")



























# import streamlit as st
# import json
# import re
# from datetime import datetime
# from pathlib import Path
# from sentence_transformers import SentenceTransformer, util

# # -----------------------------------------------------
# # ⚙️ Config
# # -----------------------------------------------------
# st.set_page_config(page_title="Automation Knowledge Hub", layout="wide")
# KB_FOLDER = Path("knowledge_base")
# KB_FOLDER.mkdir(exist_ok=True)

# # -----------------------------------------------------
# # 🔧 Model Loading (cached)
# # -----------------------------------------------------
# @st.cache_resource
# def load_semantic_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_resource
# def load_cross_encoder():
#     """Optional reranker model (cached)."""
#     try:
#         from sentence_transformers import CrossEncoder
#         return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     except Exception:
#         return None

# model = load_semantic_model()
# reranker_model = load_cross_encoder()

# # -----------------------------------------------------
# # 🧩 Helper Functions
# # -----------------------------------------------------
# def clean_text(text: str) -> str:
#     """Normalize whitespace and ensure safe string."""
#     if not text:
#         return ""
#     return re.sub(r"\s+", " ", str(text)).strip()


# def save_bot_to_kb(bot_data: dict):
#     """Save bot metadata safely into the knowledge_base folder (no file content)."""
#     name = bot_data.get("bot_name") or "UnknownBot"
#     safe_name = re.sub(r"[^\w\-_\. ]", "_", name)
#     p = KB_FOLDER / f"{safe_name}.json"
#     with open(p, "w", encoding="utf-8") as f:
#         json.dump(bot_data, f, indent=4, ensure_ascii=False)


# def load_kb():
#     """Load all bot JSONs from knowledge_base folder."""
#     bots = []
#     for p in KB_FOLDER.glob("*.json"):
#         try:
#             with open(p, "r", encoding="utf-8") as f:
#                 bots.append(json.load(f))
#         except Exception:
#             continue
#     return bots


# def beautify_bot_description(bot_data: dict) -> str:
#     """Format bot info nicely for markdown display (no file content)."""
#     name = bot_data.get("bot_name", "Unnamed Bot")
#     uploader = bot_data.get("user_name", "Unknown")
#     upload_date = bot_data.get("upload_date", "Unknown")
#     desc = bot_data.get("description", "No description provided.")
#     notes = bot_data.get("comments", "")
#     functions = bot_data.get("functions_details", "")

#     md = f"### 🤖 {name}\n\n"
#     md += f"**Uploaded by:** {uploader} | **Upload Date:** {upload_date}\n\n"
#     md += f"**Overview:** {desc}\n\n"

#     if functions.strip():
#         md += f"**Functions / Scripts:** {functions}\n\n"

#     if notes.strip():
#         md += f"**Notes:** {notes}\n\n"

#     return md


# def find_matching_bot_names(query: str, kb_bots: list) -> list:
#     """Return bots that partially match the search text."""
#     q = (query or "").strip().lower()
#     return [
#         (bot.get("bot_name") or "")
#         for bot in kb_bots
#         if not q or q in (bot.get("bot_name", "")).lower()
#     ]


# def get_bot_by_name(name: str, kb_bots: list):
#     """Return the bot dict by name."""
#     for bot in kb_bots:
#         if bot.get("bot_name", "") == name:
#             return bot
#     return None


# def build_bot_text_blocks(bot: dict) -> list:
#     """Build searchable text blocks only from description, functions, and notes."""
#     blocks = []
#     if bot.get("description"):
#         blocks.append(("Overview", clean_text(bot["description"])))
#     if bot.get("functions_details"):
#         blocks.append(("Functions / Scripts", clean_text(bot["functions_details"])))
#     if bot.get("comments"):
#         blocks.append(("Notes / Comments", clean_text(bot["comments"])))
#     return blocks


# # -----------------------------------------------------
# # 🤖 Semantic Search Function
# # -----------------------------------------------------
# def semantic_answer_for_bot(
#     question: str,
#     bot: dict,
#     threshold: float = 0.15,   # lowered threshold for better recall
#     chunk_size: int = 150,
#     top_k: int = 5,
#     use_reranker: bool = True,
# ):
#     """Semantic retrieval with optional cross-encoder reranking."""

#     def split_into_chunks(text, chunk_size_local=chunk_size):
#         words = text.split()
#         return [
#             " ".join(words[i:i + chunk_size_local])
#             for i in range(0, len(words), chunk_size_local)
#         ]

#     question = clean_text(question)
#     blocks = build_bot_text_blocks(bot)
#     if not blocks:
#         return "No searchable text found for this bot.", 0.0, None

#     try:
#         q_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
#     except Exception:
#         return "Embedding model failure.", 0.0, None

#     candidates = []
#     for label, text in blocks:
#         if not text.strip():
#             continue
#         for chunk in split_into_chunks(text):
#             try:
#                 chunk_emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
#                 sem_score = float(util.cos_sim(q_emb, chunk_emb))
#             except Exception:
#                 sem_score = 0.0
#             candidates.append((sem_score, label, chunk.strip()))

#     if not candidates:
#         return "No relevant section found.", 0.0, None

#     # Sort top candidates by similarity
#     candidates.sort(key=lambda x: x[0], reverse=True)
#     top_candidates = candidates[:max(1, top_k)]

#     # Optional cross-encoder reranker
#     reranked_best = None
#     if use_reranker and reranker_model is not None:
#         try:
#             pairs = [[question, c[2]] for c in top_candidates]
#             rerank_scores = reranker_model.predict(pairs)
#             reranked = sorted(
#                 zip(rerank_scores, top_candidates),
#                 key=lambda x: x[0],
#                 reverse=True
#             )
#             best_rscore, (_, best_label, best_text) = reranked[0]
#             raw_score = float(best_rscore)
#             reranked_best = (raw_score, best_label, best_text)
#         except Exception:
#             reranked_best = None

#     if reranked_best is not None:
#         raw_score, label, text = reranked_best
#     else:
#         raw_score, label, text = top_candidates[0]

#     # ✅ Normalize confidence properly (no negatives)
#     if reranked_best is not None:
#         norm_conf = max(0.0, min(1.0, raw_score / 5.0))  # CrossEncoder typically outputs 0–5
#     else:
#         norm_conf = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))  # cosine sim −1 to 1 → 0–1

#     # ✅ Check threshold on normalized value, not raw score
#     if norm_conf < threshold:
#         return f"No relevant information found (confidence {norm_conf:.2f}).", norm_conf, None

#     return None, norm_conf, {"label": label, "text": text}

# # -----------------------------------------------------
# # 🧭 Sidebar Navigation
# # -----------------------------------------------------
# st.sidebar.title("🧩 Automation Knowledge Hub")
# st.sidebar.markdown("---")
# page = st.sidebar.radio("Navigate to:", ["📚 Knowledge Base", "💬 Bot Chat"])
# st.sidebar.markdown("---")

# # -----------------------------------------------------
# # 📚 Knowledge Base Page
# # -----------------------------------------------------
# if page == "📚 Knowledge Base":
#     st.title("📚 Knowledge Base: Upload Bot Information")

#     with st.form("bot_upload_form"):
#         st.subheader("Upload Bot Metadata")

#         user_name = st.text_input("👤 Your Name")
#         bot_name = st.text_input("🤖 Bot Name")
#         description = st.text_area("📝 Bot Description / Overview")
#         functions_details = st.text_area("⚙️ Functions / Scripts Details")
#         comments = st.text_area("💡 Notes / Comments (paths, credentials, etc.)")
#         uploaded_file = st.file_uploader("📂 Upload Bot File (txt, pdf, json, etc.)")

#         submit = st.form_submit_button("Upload Bot to Knowledge Base")

#         if submit:
#             if not bot_name.strip():
#                 st.error("Bot Name is required!")
#             else:
#                 try:
#                     bot_data = {
#                         "user_name": user_name or "Unknown",
#                         "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         "bot_name": bot_name,
#                         "description": description,
#                         "functions_details": functions_details,
#                         "comments": comments,
#                     }
#                     save_bot_to_kb(bot_data)
#                     st.success(f"✅ Bot '{bot_name}' added to Knowledge Base!")
#                     st.markdown(beautify_bot_description(bot_data))
#                 except Exception as e:
#                     st.error(f"Error saving bot: {e}")

#     st.markdown("---")
#     st.subheader("📖 Current Knowledge Base")
#     kb_bots = load_kb()

#     if kb_bots:
#         for bot in kb_bots:
#             st.markdown(beautify_bot_description(bot))
#             st.divider()
#     else:
#         st.info("No bots added yet. Upload one above.")

# # -----------------------------------------------------
# # 💬 Bot Chat (Semantic NLP)
# # -----------------------------------------------------
# elif page == "💬 Bot Chat":
#     st.title("💬 Bot Chat (Semantic NLP)")

#     kb = load_kb()
#     if not kb:
#         st.warning("No bots found. Please upload JSONs first from 'Knowledge Base'.")
#     else:
#         search_text = st.text_input("🔍 Search bots by name", placeholder="e.g., GFB")
#         matches = find_matching_bot_names(search_text, kb)

#         if not matches:
#             st.info("No bots match your query.")
#         else:
#             chosen = matches[0] if len(matches) == 1 else st.selectbox("Select a bot", options=matches)

#             if chosen:
#                 bot = get_bot_by_name(chosen, kb)
#                 st.markdown(beautify_bot_description(bot))
#                 st.markdown("---")

#                 st.subheader("Ask a question about this bot")
#                 user_q = st.text_input("Your question", placeholder="e.g., What columns does it read?")

#                 if st.button("Get Answer"):
#                     if not user_q.strip():
#                         st.warning("Please enter a question.")
#                     else:
#                         with st.spinner("Thinking..."):
#                             err, conf, answer = semantic_answer_for_bot(user_q, bot)
#                         if err:
#                             st.error(err)
#                         elif answer:
#                             label, text = answer["label"], answer["text"]
#                             if label.lower().startswith("overview"):
#                                 reply = f"This bot mainly does the following: {text}"
#                             elif "function" in label.lower():
#                                 reply = f"It performs these actions or scripts: {text}"
#                             elif "note" in label.lower() or "comment" in label.lower():
#                                 reply = f"Here’s something important to know: {text}"
#                             else:
#                                 reply = text

#                             st.success(f"Top Match (confidence {conf:.2f})")
#                             st.markdown(f"{reply}\n\n*(Source: {label})*")
#                         else:
#                             st.info("No relevant information found.")

# # working
# import streamlit as st
# import json
# import re
# from datetime import datetime
# from pathlib import Path
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------
# # ⚙️ Config
# # -----------------------------------------------------
# st.set_page_config(page_title="Automation Knowledge Hub", layout="wide")
# KB_FOLDER = Path("knowledge_base")
# KB_FOLDER.mkdir(exist_ok=True)

# # -----------------------------------------------------
# # 🧩 Helper Functions
# # -----------------------------------------------------
# def clean_text(text: str) -> str:
#     """Normalize whitespace and ensure safe string."""
#     if not text:
#         return ""
#     return re.sub(r"\s+", " ", str(text)).strip()


# def save_bot_to_kb(bot_data: dict):
#     """Save bot metadata safely into the knowledge_base folder (no file content)."""
#     name = bot_data.get("bot_name") or "UnknownBot"
#     safe_name = re.sub(r"[^\w\-_\. ]", "_", name)
#     p = KB_FOLDER / f"{safe_name}.json"
#     with open(p, "w", encoding="utf-8") as f:
#         json.dump(bot_data, f, indent=4, ensure_ascii=False)


# def load_kb():
#     """Load all bot JSONs from knowledge_base folder."""
#     bots = []
#     for p in KB_FOLDER.glob("*.json"):
#         try:
#             with open(p, "r", encoding="utf-8") as f:
#                 bots.append(json.load(f))
#         except Exception:
#             continue
#     return bots


# def beautify_bot_description(bot_data: dict) -> str:
#     """Format bot info nicely for markdown display (no file content)."""
#     name = bot_data.get("bot_name", "Unnamed Bot")
#     uploader = bot_data.get("user_name", "Unknown")
#     upload_date = bot_data.get("upload_date", "Unknown")
#     desc = bot_data.get("description", "No description provided.")
#     notes = bot_data.get("comments", "")
#     functions = bot_data.get("functions_details", "")

#     md = f"### 🤖 {name}\n\n"
#     md += f"**Uploaded by:** {uploader} | **Upload Date:** {upload_date}\n\n"
#     md += f"**Overview:** {desc}\n\n"

#     if functions.strip():
#         md += f"**Functions / Scripts:** {functions}\n\n"

#     if notes.strip():
#         md += f"**Notes:** {notes}\n\n"

#     return md


# def find_matching_bot_names(query: str, kb_bots: list) -> list:
#     """Return bots that partially match the search text."""
#     q = (query or "").strip().lower()
#     return [
#         (bot.get("bot_name") or "")
#         for bot in kb_bots
#         if not q or q in (bot.get("bot_name", "")).lower()
#     ]


# def get_bot_by_name(name: str, kb_bots: list):
#     """Return the bot dict by name."""
#     for bot in kb_bots:
#         if bot.get("bot_name", "") == name:
#             return bot
#     return None


# def build_bot_text_blocks(bot: dict) -> list:
#     """Build searchable text blocks only from description, functions, and notes."""
#     blocks = []
#     if bot.get("description"):
#         blocks.append(("Description", clean_text(bot["description"])))
#     if bot.get("functions_details"):
#         blocks.append(("Functions / Scripts", clean_text(bot["functions_details"])))
#     if bot.get("comments"):
#         blocks.append(("Notes / Comments", clean_text(bot["comments"])))
#     return blocks


# def answer_question_for_bot(question: str, bot: dict, top_k: int = 3):
#     """
#     Answer a user question using TF-IDF similarity.
#     Searches ONLY description, functions, and notes (no file content).
#     """
#     question = clean_text(question)
#     blocks = build_bot_text_blocks(bot)
#     if not blocks:
#         return "No searchable text found for this bot.", 0.0, []

#     texts = [b for _, b in blocks]
#     vectorizer = TfidfVectorizer(stop_words="english")

#     try:
#         tf = vectorizer.fit_transform(texts + [question])
#     except ValueError:
#         return "Insufficient text for analysis.", 0.0, []

#     sims = cosine_similarity(tf[-1], tf[:-1]).flatten()
#     sorted_idx = sims.argsort()[::-1]

#     results = []
#     for idx in sorted_idx[:top_k]:
#         label, text = blocks[idx]
#         results.append({"label": label, "text": text, "score": float(sims[idx])})

#     confidence = float(sims[sorted_idx[0]]) if len(sims) > 0 else 0.0
#     return None, confidence, results


# # -----------------------------------------------------
# # 🧭 Sidebar Navigation
# # -----------------------------------------------------
# st.sidebar.title("🧩 Automation Knowledge Hub")
# st.sidebar.markdown("---")
# page = st.sidebar.radio(
#     "Navigate to:", ["📚 Knowledge Base", "💬 Bot Chat"] #, "🧠 File Summarizer"]
# )
# st.sidebar.markdown("---")

# # -----------------------------------------------------
# # 📚 Knowledge Base Page
# # -----------------------------------------------------
# if page == "📚 Knowledge Base":
#     st.title("📚 Knowledge Base: Upload Bot Information")

#     with st.form("bot_upload_form"):
#         st.subheader("Upload Bot Metadata")

#         user_name = st.text_input("👤 Your Name")
#         bot_name = st.text_input("🤖 Bot Name")
#         description = st.text_area("📝 Bot Description / Overview")
#         functions_details = st.text_area("⚙️ Functions / Scripts Details")
#         comments = st.text_area("💡 Notes / Comments (paths, credentials, etc.)")
#         uploaded_file = st.file_uploader("📂 Upload Bot File (txt, pdf, json, etc.)")

#         submit = st.form_submit_button("Upload Bot to Knowledge Base")

#         if submit:
#             if not bot_name.strip():
#                 st.error("Bot Name is required!")
#             else:
#                 try:
#                     # 🧠 Only save metadata — do NOT store or display file content
#                     bot_data = {
#                         "user_name": user_name or "Unknown",
#                         "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         "bot_name": bot_name,
#                         "description": description,
#                         "functions_details": functions_details,
#                         "comments": comments,
#                     }
#                     save_bot_to_kb(bot_data)
#                     st.success(f"✅ Bot '{bot_name}' added to Knowledge Base!")
#                     st.markdown(beautify_bot_description(bot_data))
#                 except Exception as e:
#                     st.error(f"Error saving bot: {e}")

#     st.markdown("---")
#     st.subheader("📖 Current Knowledge Base")
#     kb_bots = load_kb()

#     if kb_bots:
#         for bot in kb_bots:
#             st.markdown(beautify_bot_description(bot))
#             st.divider()
#     else:
#         st.info("No bots added yet. Upload one above.")


# # -----------------------------------------------------
# # 💬 Bot Chat (Offline NLP)
# # -----------------------------------------------------
# elif page == "💬 Bot Chat":
#     st.title("💬 Bot Chat (Offline NLP)")

#     kb = load_kb()
#     if not kb:
#         st.warning("No bots found. Please upload JSONs first from 'Knowledge Base'.")
#     else:
#         search_text = st.text_input("🔍 Search bots by name", placeholder="e.g., GFB")
#         matches = find_matching_bot_names(search_text, kb)

#         if not matches:
#             st.info("No bots match your query.")
#         else:
#             if len(matches) == 1:
#                 chosen = matches[0]
#             else:
#                 chosen = st.selectbox("Select a bot", options=matches)

#             if chosen:
#                 bot = get_bot_by_name(chosen, kb)
#                 st.markdown(beautify_bot_description(bot))
#                 st.markdown("---")

#                 st.subheader("Ask a question about this bot")
#                 user_q = st.text_input(
#                     "Your question", placeholder="e.g., What columns does it read?"
#                 )
#                 if st.button("Get Answer"):
#                     if not user_q.strip():
#                         st.warning("Please enter a question.")
#                     else:
#                         err, conf, answers = answer_question_for_bot(
#                             user_q, bot, top_k=3
#                         )
#                         if err:
#                             st.error(err)
#                         else:
#                             best = max(answers, key=lambda x: x["score"]) if answers else None
#                             if best:
#                                 section = best["label"]
#                                 text = best["text"].strip()

#                                 # 🗣️ Simple rephrasing for a conversational tone
#                                 if section.lower().startswith("overview"):
#                                     reply = f"This automation mainly does the following: {text}"
#                                 elif "function" in section.lower():
#                                     reply = f"The bot uses the following function or script: {text}"
#                                 elif "note" in section.lower() or "comment" in section.lower():
#                                     reply = f"Here’s an important note about this bot: {text}"
#                                 else:
#                                     reply = text

#                                 st.success(f"Top Match (confidence {conf:.2f})")
#                                 st.markdown(f"{reply}\n\n*(Source: {section})*")
#                             else:
#                                 st.info("No relevant information found.")



