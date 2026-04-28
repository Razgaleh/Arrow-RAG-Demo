# Import necessary libraries
import os
import base64
import streamlit as st
from streamlit_theme import st_theme

from rag_engine import (
    RAG_WEBSITE_SOURCES,
    chat_completion,
    get_model,
    list_docs_folder_pdfs,
    local_get_answer_folder_pdf,
    local_get_answer_upload_pdf,
    local_get_answer_url,
    nim_get_answer_folder_pdf,
    nim_get_answer_upload_pdf,
    nim_get_answer_url,
)

working_dir = os.path.dirname(os.path.abspath(__file__))


def _format_bytes(n: int) -> str:
    x = float(max(n, 0))
    i = 0
    units = ("B", "KB", "MB", "GB", "TB")
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    if i == 0:
        return f"{int(x)} {units[i]}"
    return f"{x:.1f} {units[i]}"


MODE_RAG = "RAG"
MODE_CHAT = "Chatbot"


def _init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_reply" not in st.session_state:
        st.session_state.pending_reply = None
    if "_last_app_mode" not in st.session_state:
        st.session_state._last_app_mode = None


def _run_rag_query(
    deployment: str,
    database: str,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    query: str,
    uploaded_file,
) -> str:
    mid = get_model(model_name, deployment)

    if deployment == "Local vLLM":
        if database == "Local PDFs (.pdf)":
            return local_get_answer_folder_pdf(
                mid, temperature, top_p, top_k, query
            )
        if database == "Upload PDFs (.pdf)":
            if uploaded_file is None:
                raise ValueError("UPLOAD_REQUIRED")
            uploaded_file.seek(0)
            bytes_data = uploaded_file.read()
            file_name = uploaded_file.name
            file_path = os.path.join(working_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            return local_get_answer_upload_pdf(
                mid, temperature, top_p, top_k, file_name, query
            )
        if database == "Public Websites (URL)":
            return local_get_answer_url(mid, temperature, top_p, top_k, query)

    if deployment == "NVIDIA NIM":
        if database == "Local PDFs (.pdf)":
            return nim_get_answer_folder_pdf(
                mid, temperature, top_p, top_k, query
            )
        if database == "Upload PDFs (.pdf)":
            if uploaded_file is None:
                raise ValueError("UPLOAD_REQUIRED")
            uploaded_file.seek(0)
            bytes_data = uploaded_file.read()
            file_name = uploaded_file.name
            file_path = os.path.join(working_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            return nim_get_answer_upload_pdf(
                mid, temperature, top_p, top_k, file_name, query
            )
        if database == "Public Websites (URL)":
            return nim_get_answer_url(mid, temperature, top_p, top_k, query)

    raise ValueError("UNKNOWN_CONFIG")


st.set_page_config(
    page_title="Arrow AI Assistant",
    page_icon="./static/img/favicon.png",
    layout="centered",
)


@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_photo_as_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = """
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: scroll;
    }
    </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_photo_as_bg("./static/img/racecars.jpg")

with open("./static/stylesheet/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

lightmode_logo = "./static/img/arrow_worm_black.png"
darkmode_logo = "./static/img/arrow_worm_white.png"
theme = st_theme()

if theme is not None and theme["base"] == "light":
    st.logo(lightmode_logo, icon_image=lightmode_logo)
    st.markdown(
        """
    <style>
        [data-testid='stFileUploaderFileName'] { color: white !important; }
        [data-testid="stFileUploaderDeleteBtn"] { color: white !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] { background-color: rgba(255,255,255,0.12) !important; }
        [data-testid='stException'] {
            background-color: white !important;
            border-radius: 10px !important;
        }
        [data-testid='stException'] p, ul, ol{  color: red !important;}
        [data-testid='stAlert']{
            background-color: white !important;
            border-radius: 10px !important;
        }
        [data-testid='stAlert'] p, ul, ol {  color: black !important;}
    </style>
    """,
        unsafe_allow_html=True,
    )
else:
    st.logo(darkmode_logo, icon_image=darkmode_logo)
    st.markdown(
        """
    <style>
        [data-testid='stException'] {
            background-color: black !important;
            border-radius: 10px !important;
        }
        [data-testid='stException'] p, ul, ol{  color: red !important;}
        [data-testid='stAlert']{
            background-color: black !important;
            border-radius: 10px !important;
        }
        [data-testid='stAlert'] p, ul, ol {  color: white !important;}
    </style>
    """,
        unsafe_allow_html=True,
    )

_init_chat_state()

with st.sidebar:
    app_mode = st.radio(
        "Mode",
        (MODE_RAG, MODE_CHAT),
        help="RAG uses your selected document source. Chat only talks to the LLM with no retrieval.",
    )

    if st.session_state._last_app_mode is not None and st.session_state._last_app_mode != app_mode:
        st.session_state.messages = []
        st.session_state.pending_reply = None
    st.session_state._last_app_mode = app_mode

    deployment = st.selectbox(
        "Select Deployment Option", ("Local vLLM", "NVIDIA NIM")
    )

    database = None
    uploaded_file = None
    if app_mode == MODE_RAG:
        database = st.selectbox(
            "Select RAG Database",
            ("Local PDFs (.pdf)", "Upload PDFs (.pdf)", "Public Websites (URL)"),
        )
        if database == "Upload PDFs (.pdf)":
            uploaded_file = st.file_uploader(
                "Upload a PDF for this session", type=["pdf"]
            )
        elif database == "Local PDFs (.pdf)":
            with st.expander("Indexed PDF documents", expanded=True):
                st.caption(
                    "Retrieval uses text extracted from these files in the docs/ folder. "
                    "Download to open or preview them locally."
                )
                pdfs = list_docs_folder_pdfs()
                if not pdfs:
                    st.info(
                        "No PDFs found in docs/. Add .pdf files to that folder (next to "
                        "the app) and refresh this page."
                    )
                else:
                    for i, entry in enumerate(pdfs):
                        label = f"{entry['name']} — {_format_bytes(entry['size_bytes'])}"
                        with open(entry["path"], "rb") as f:
                            st.download_button(
                                label=label,
                                data=f.read(),
                                file_name=entry["name"],
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"docs_pdf_dl_{i}",
                            )
        elif database == "Public Websites (URL)":
            with st.expander("Indexed web sources", expanded=True):
                st.caption(
                    "Retrieval uses text scraped from these pages. Click to open in your browser."
                )
                for src in RAG_WEBSITE_SOURCES:
                    st.link_button(
                        src["label"],
                        src["url"],
                        use_container_width=True,
                    )
    else:
        st.caption("No document retrieval — conversation uses the model only.")

    model = st.selectbox(
        "Select LLM Model",
        (
            "NVIDIA Nemotron 3 Nano",
            "Meta Llama 3.1",
            "Google Gemma 2",
            "Microsoft Phi 3.5",
        ),
    )

    st.markdown(
        """
    <style>
        [data-testid='stTickBarMin'] {
            display:none !important;
            padding: 0px !important;
            margin: 0px !important;
        }
        [data-testid='stTickBarMax'] {
            display:none !important;
            padding: 0px !important;
            margin: 0px !important;
        }
        [data-testid='stSlider'] {
            margin-top: 20px !important;
        }
        .label-container {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .label {
            padding: 0px;
            margin: -20px 0 0 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    temperature = st.slider("Select LLM's Creativity Level", 0.0, 2.0, 0.4, 0.1)
    temperature_text = str(temperature)
    st.markdown(
        """
    <div class='label-container'>
        <h6 class='label'>Structured</h6>
        <h6 class='label'>Creative</h6>
    </div>
    """,
        unsafe_allow_html=True,
    )

    top_p = st.slider("Select LLM's Top-p Value", 0.0, 1.0, 0.95, 0.01)
    st.markdown(
        """
    <div class='label-container'>
        <h6 class='label'>Narrow</h6>
        <h6 class='label'>Broad</h6>
    </div>
    """,
        unsafe_allow_html=True,
    )

    top_k = st.slider("Select LLM's Top-k Value", 1, 128, 40, 1)
    st.markdown(
        """
    <div class='label-container'>
        <h6 class='label'>Focused</h6>
        <h6 class='label'>Diverse</h6>
    </div>
    """,
        unsafe_allow_html=True,
    )

    db_line = database if app_mode == MODE_RAG else "—"
    st.write(
        "**Mode:** %s  \n**Deployment:** %s  \n**Database:** %s  \n**LLM:** %s  \n**Temperature:** %s  \n**Top-p:** %s  \n**Top-k:** %s"
        % (
            app_mode,
            deployment,
            db_line,
            model,
            temperature_text,
            f"{top_p:.2f}",
            str(top_k),
        )
    )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.pending_reply = None
        st.rerun()

if app_mode == MODE_RAG:
    with st.expander("About this RAG demo", expanded=False):
        st.image("./static/img/rag_diagram.png", width=700)

st.subheader("Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_reply is not None:
    prompt = st.session_state.pending_reply
    st.session_state.pending_reply = None
    with st.chat_message("assistant"):
        if app_mode == MODE_CHAT:
            spin_msg = "Thinking…"
        else:
            spin_msg = "Searching documents…"
        with st.spinner(spin_msg):
            try:
                if app_mode == MODE_CHAT:
                    answer = chat_completion(
                        deployment,
                        model,
                        temperature,
                        top_p,
                        top_k,
                        st.session_state.messages,
                    )
                else:
                    answer = _run_rag_query(
                        deployment,
                        database,
                        model,
                        temperature,
                        top_p,
                        top_k,
                        prompt,
                        uploaded_file,
                    )
            except ValueError as e:
                if str(e) == "UPLOAD_REQUIRED":
                    answer = "Please upload a PDF in the sidebar before chatting."
                else:
                    raise
            except Exception as err:
                answer = f"**Error:** {err}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

chat_placeholder = (
    "Message the model…"
    if app_mode == MODE_CHAT
    else "Message about your documents…"
)

if prompt := st.chat_input(chat_placeholder):
    if not prompt.strip():
        st.warning("Enter a non-empty message.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.pending_reply = prompt.strip()
        st.rerun()
