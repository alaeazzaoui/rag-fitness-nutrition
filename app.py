import streamlit as st

from src.generation.generator import (
    retrieve_context,
    load_generation_model,
    generate_answer,
    GENERATION_MODELS,
)
from src.hallucination import hallucination_detector as hd

# Règle de négation désactivée (voir l'ablation du notebook)
hd.USE_NEGATION_RULE = False

STRATEGY = "recursive"
TOP_K = 5
REFUSAL_MARKER = "ne dispose pas d'informations suffisantes"

STATUS_ICON = {
    "SUPPORTED": "✅",
    "UNSUPPORTED": "⚠️",
    "CONTRADICTED": "❌",
}

st.set_page_config(
    page_title="NutriMind AI",
    page_icon="🥗",
    layout="wide"
)


@st.cache_resource(show_spinner="Loading the language model...")
def get_model(label):
    return load_generation_model(GENERATION_MODELS[label])


def is_refusal(answer):
    return REFUSAL_MARKER in answer.lower().replace("’", "'")


def render_details(details):
    if not details:
        return

    with st.expander("Sources used"):
        for i, p in enumerate(details["passages"], start=1):
            st.markdown(f"**[S{i}]** {p['source']} (page {p['page']})")
            st.caption(" ".join(p["content"].split())[:400])

    report = details.get("report")
    if report and report["total_claims"] > 0:
        with st.expander(
            f"Faithfulness check: {report['faithfulness_score']}% "
            f"({report['supported']} supported, "
            f"{report['unsupported']} unsupported, "
            f"{report['contradicted']} contradicted)"
        ):
            for claim in report["claims"]:
                icon = STATUS_ICON.get(claim["status"], "")
                st.markdown(f"{icon} **{claim['status']}**: {claim['text']}")
                if claim["evidence"]:
                    st.caption(
                        f"Evidence: {claim['evidence']['source']} "
                        f"(page {claim['evidence']['page']})"
                    )
    elif details.get("refused"):
        st.caption("The assistant declined to answer: the documents do not cover this question.")

    st.caption(f"Generation time: {details['time']:.0f} s")


st.title("🥗 NutriMind AI")
st.subheader("Your intelligent Fitness & Nutrition Assistant")

st.markdown("""
Welcome to **NutriMind AI**!

Ask questions about:
- Sports nutrition
- Fitness and resistance training
- Protein and supplements
- Muscle growth and recovery
""")

st.caption(
    "Answers come only from the document corpus and are not medical advice."
)

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox(
        "Generation model",
        list(GENERATION_MODELS.keys()),
        help="0.5B is faster on CPU, 1.5B gives better answers but is slower."
    )
    check_faithfulness = st.checkbox("Check faithfulness (NLI)", value=True)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_details(message.get("details"))

if prompt := st.chat_input("Ask your nutrition question..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tokenizer, model = get_model(model_label)

        with st.spinner("Searching the documents and writing the answer..."):
            passages = retrieve_context(
                prompt,
                strategy=STRATEGY,
                top_k=TOP_K
            )
            response, gen_time = generate_answer(
                prompt, passages, tokenizer, model
            )

        refused = is_refusal(response)
        report = None

        if check_faithfulness and not refused:
            with st.spinner("Checking the answer against the sources..."):
                report = hd.detect_hallucinations(response, passages)

        details = {
            "passages": passages,
            "report": report,
            "refused": refused,
            "time": gen_time,
        }

        st.markdown(response)
        render_details(details)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "details": details}
    )