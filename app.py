import streamlit as st
import time

st.set_page_config(
    page_title="ISLP by Aadi",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f0f0f0;
        background-color: #0d0d0d !important;
    }

    /* HERO BACKGROUND for Home Page — Circuit board / F1 telemetry */
    .stApp {
        background-image: 
            radial-gradient(ellipse at 10% 0%, rgba(230,57,70,0.15) 0%, transparent 60%),
            radial-gradient(ellipse at 90% 100%, rgba(217,154,91,0.1) 0%, transparent 60%),
            linear-gradient(160deg, #0d0d0d 60%, #111111 100%);
        background-attachment: fixed;
    }

    /* Animated circuit-line effect via pseudo border */
    .block-container {
        padding-top: 1.5rem !important;
        border-left: 1px solid rgba(217,154,91,0.15);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }

    /* Envisioned by Aadi: the brand mark */
    .aadi-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 6px;
        color: #d99a5b;
        border-bottom: 2px solid #e63946;
        padding-bottom: 8px;
        margin-bottom: 28px;
        text-align: right;
    }

    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: #d99a5b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #333;
        gap: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: #181818;
        color: #ffffff !important;
        border: 1px solid #d99a5b;
        border-radius: 3px;
        transition: 0.2s all ease-in-out;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: #d99a5b;
        color: #0d0d0d !important;
        border: 1px solid #d99a5b;
        box-shadow: 0 0 20px rgba(217,154,91,0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid rgba(217,154,91,0.2);
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #e63946 !important;
    }

    /* Radio buttons */
    .stRadio > label { color: #e0e0e0 !important; }

    /* Info/warning/error boxes */
    .stAlert { background-color: #161616 !important; border-radius: 3px; }

    /* Code blocks */
    code { font-family: 'JetBrains Mono', monospace; color: #d99a5b; }

    /* Hide Streamlit branding */
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #d99a5b !important;
        background: #161616;
        border: 1px solid #333;
    }

    /* Horizontal rule */
    hr { border: none; border-top: 1px solid #2a2a2a; margin: 2rem 0; }

    /* Data frames */
    .stDataFrame { border: 1px solid #2a2a2a; }
    </style>

    <div class="aadi-header">Envisioned by Aadi</div>
""", unsafe_allow_html=True)

if 'loaded' not in st.session_state:
    with st.spinner('Compiling models...'):
        time.sleep(1.2)
    st.session_state['loaded'] = True
    st.toast('Environment ready.', icon='🏎️')

st.sidebar.markdown("""
<div style='font-weight:800;font-size:1.1rem;color:#fff;letter-spacing:1px;padding-bottom:8px;border-bottom:1px solid #333;'>
ISLP MODULES
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown(" ")
st.sidebar.caption("An Introduction to Statistical Learning — Interactive Edition")
st.sidebar.markdown("**Select a Module** from the menu above to begin.")

# Hero section
st.title("ISLP — Interactive Learning & Visualization")

st.markdown("""
<div style="background: linear-gradient(135deg, #161616 0%, #1a1a1a 100%); border: 1px solid #2a2a2a; border-left: 4px solid #e63946; padding: 1.5rem 2rem; border-radius: 3px; margin-bottom: 2rem;">
<div style="font-size:0.75rem; font-weight:700; letter-spacing:4px; color:#d99a5b; text-transform:uppercase; margin-bottom:0.5rem;">ABOUT THIS TOOL</div>
<div style="color:#e0e0e0; font-size:0.95rem; line-height:1.9;">
This web application serves as a <strong style="color:#fff;">learning and visualization tool</strong> for
<em>An Introduction to Statistical Learning with Applications in Python/R</em> (ISLP),
built for the <strong style="color:#fff;">Applied AI/ML</strong> course under the
<strong style="color:#fff;">Business Data Analytics (BDA)</strong> program.<br><br>

It transforms the dense mathematical theory of Chapters 1–3 into an interactive, hands-on experience.
Every equation is derived step-by-step — from school-level algebra to graduate-level proofs — and paired
with real-time visualizations powered by custom business datasets. The goal is simple: make
statistical learning <em>intuitive</em>, <em>visual</em>, and <em>immediately applicable</em>
to real-world business problems.<br><br>

Each module includes interactive model simulators, 10 real-world case studies spanning industries from
fintech to healthcare, and a 50-question management-style assessment with instant grading and detailed
explanations — designed to bridge the gap between theory and executive decision-making.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #161616 0%, #1a1a1a 100%); border: 1px solid #2a2a2a; border-left: 4px solid #d99a5b; padding: 1.5rem 2rem; border-radius: 3px; margin-bottom: 2rem;">
<div style="font-size:0.75rem; font-weight:700; letter-spacing:4px; color:#d99a5b; text-transform:uppercase; margin-bottom:0.5rem;">MODULE DIRECTORY</div>
<div style="color:#e0e0e0; font-size:0.95rem; line-height:2;">
&nbsp;&nbsp;▸ &nbsp;<strong style="color:#fff;">MODULE 1 — Data Foundations</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dataset exploration, the Y=f(X)+ε paradigm, and the Netflix $1M Prize Case Study.<br>
&nbsp;&nbsp;▸ &nbsp;<strong style="color:#fff;">MODULE 2 — Statistical Learning Engine</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deep-dive: Master Equation, Prediction vs Inference, Parametric vs Non-Parametric, Bias-Variance Tradeoff, Classification. 10 Case Studies. Full mathematical derivations.<br>
&nbsp;&nbsp;▸ &nbsp;<strong style="color:#fff;">MODULE 3 — Linear Regression</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The parametric workhorse: OLS derivation, interactive β-tuning minigame, multivariate inference, and the Zillow Case Study.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("👈 **Select a Module in the sidebar to begin.**")

st.image(
    "https://images.unsplash.com/photo-1542744094-3a31f272c490?q=80&w=2070&auto=format&fit=crop",
    caption="Applied AI/ML for Business Data Analytics — Precision. Data. Insight.",
    use_column_width=True
)
