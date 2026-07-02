import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quiz_engine import render_quiz

st.set_page_config(page_title="Module 2: Statistical Learning", page_icon="🏎️", layout="wide")

plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#0d0d0d', 'figure.facecolor': '#0d0d0d',
    'text.color': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'axes.edgecolor': '#333', 'grid.color': '#222'
})

st.markdown("""
<style>
.stApp {
    background-image:
        radial-gradient(ellipse at 0% 100%, rgba(230,57,70,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 0%, rgba(217,154,91,0.08) 0%, transparent 50%),
        linear-gradient(135deg, #0d0d0d 0%, #111 100%);
    background-attachment: fixed;
}
</style>
<div style="font-family:'Inter',sans-serif;font-weight:800;text-transform:uppercase;font-size:0.78rem;letter-spacing:6px;color:#d99a5b;border-bottom:2px solid #e63946;padding-bottom:8px;margin-bottom:28px;text-align:right;">Envisioned by Aadi</div>
""", unsafe_allow_html=True)

st.title("MODULE 2 — Statistical Learning Engine 🏎️")

import sys
@st.cache_data
def load_datasets():
    if sys.platform == "emscripten":
        d = "/home/pyodide/dataset"
    else:
        d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    files = {
        "RealEstate": "Real_Estate_Dataset.csv",
        "LTV": "Customer_LTV_Dataset.csv",
        "Factory": "Factory_Output_Dataset.csv",
        "Employee": "Employee_Productivity_Dataset.csv",
        "Loan": "Loan_Approval_Dataset.csv",
        "Churn": "Customer_Churn_Dataset.csv",
    }
    
    datasets = {}
    for key, filename in files.items():
        fp = os.path.join(d, filename)
        if os.path.exists(fp):
            datasets[key] = pd.read_csv(fp)
        else:
            return None
    return datasets

ds = load_datasets()

if ds is None:
    st.error("⚠️ Datasets not found! If running locally, run `python data_generator.py` first. On Netlify, ensure `dataset/*.csv` is committed.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab_quiz = st.tabs([
    "1. MASTER EQ & SUPERVISED/UNSUPERVISED",
    "2. PREDICTION vs INFERENCE",
    "3. PARAMETRIC vs NON-PARAMETRIC",
    "4. BIAS-VARIANCE MATH",
    "5. REGRESSION vs CLASSIFICATION",
    "ASSESSMENT — 50 Q's"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1: THE MASTER EQUATION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.header("1. The Master Equation")
    st.latex(r"Y = f(X) + \epsilon")

    st.markdown("#### Complete Derivation: What Does This Actually Mean?")
    with st.expander("📐 STEP-BY-STEP MATH — From School Algebra to the Master Equation"):
        st.markdown("**Step 1 — School-Level Idea:**")
        st.markdown("In school, you learned $y = mx + c$. That is one straight line relating one input to one output. Statistical learning generalizes this.")
        st.latex(r"y = mx + c \quad\text{(school algebra — 1 predictor, 1 response)}")

        st.markdown("**Step 2 — Multiple Predictors:**")
        st.markdown("Now imagine many inputs: house size, location, age. We extend to $p$ predictors:")
        st.latex(r"Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p")

        st.markdown("**Step 3 — Generalization:**")
        st.markdown("But who says the relationship is a straight line? The true relationship could be any shape. We replace the specific linear formula with a generic function $f$:")
        st.latex(r"Y = f(X_1, X_2, \dots, X_p)")

        st.markdown("**Step 4 — Adding Reality (Noise):**")
        st.markdown("In the real world, outcomes aren't deterministic. Two identical houses can sell for different prices (the buyer's mood, market timing). This randomness is $\\epsilon$:")
        st.latex(r"Y = f(X) + \epsilon, \quad E[\epsilon] = 0, \quad \text{Var}(\epsilon) = \sigma^2")

        st.markdown("**Step 5 — The Conditional Expectation:**")
        st.markdown("Taking the expected value of both sides given $X$:")
        st.latex(r"E[Y | X] = E[f(X) + \epsilon | X] = f(X) + E[\epsilon] = f(X)")
        st.markdown("Therefore $f(X) = E[Y|X]$ — the true function is the expected value of $Y$ given $X$.")

    st.markdown("---")
    st.subheader("Supervised vs Unsupervised Learning")
    st.markdown("""
    * **Supervised:** Both $X$ and $Y$ are observed. We estimate $f$ to map inputs to outputs.
    * **Unsupervised:** Only $X$ is observed. No $Y$ to guide learning — we discover structure (clusters, patterns).
    * **Semi-supervised:** Some observations have $Y$, others don't.
    """)

    st.markdown("#### 🕹️ Interactive: K-Means Clustering (Unsupervised)")
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        n_clus = st.slider("Number of clusters:", 2, 6, 3, key="clus_t1")
    with col_c2:
        df_u = ds["Churn"][['Usage_Frequency', 'Price_Sensitivity']]
        km = KMeans(n_clusters=n_clus, random_state=42, n_init=10).fit(df_u)
        fig_c, ax_c = plt.subplots(figsize=(5, 3))
        ax_c.scatter(df_u.iloc[:, 0], df_u.iloc[:, 1], c=km.labels_, cmap='autumn', edgecolor='#0d0d0d', s=40)
        for c in km.cluster_centers_:
            ax_c.scatter(c[0], c[1], marker='X', s=120, color='#e63946', edgecolor='white', linewidths=1.5)
        ax_c.set_xlabel("Usage Frequency"); ax_c.set_ylabel("Price Sensitivity")
        st.pyplot(fig_c)

    st.markdown("---")
    st.subheader("📂 Case Studies")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**1 — Target's Pregnancy Prediction (Supervised)**\nTarget's analysts built a supervised model using shopping patterns ($X$) to predict pregnancy ($Y$). The model was so accurate it sent baby-product coupons to a teenager before her family knew she was pregnant. This sparked a massive debate on data ethics and prediction power.")
    with c2:
        st.info("**2 — Spotify Discover Weekly (Unsupervised)**\nSpotify clusters millions of songs by audio features (tempo, energy, valence) using unsupervised learning — no explicit labels. This powers their Discover Weekly playlist, grouping similar-sounding tracks users might enjoy.")

# ══════════════════════════════════════════════════════════════════
# TAB 2: PREDICTION vs INFERENCE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.header("2. Prediction vs Inference")

    st.markdown("#### Complete Derivation: Reducible vs Irreducible Error")
    with st.expander("📐 STEP-BY-STEP MATH — Error Decomposition"):
        st.markdown("**Step 1 — Start from the prediction error:**")
        st.latex(r"E[(Y - \hat{Y})^2]")
        st.markdown("**Step 2 — Substitute $Y = f(X) + \\epsilon$ and $\\hat{Y} = \\hat{f}(X)$:**")
        st.latex(r"= E[(f(X) + \epsilon - \hat{f}(X))^2]")
        st.markdown("**Step 3 — Let $A = f(X) - \\hat{f}(X)$ (a constant given fixed $X$ and $\\hat{f}$):**")
        st.latex(r"= E[(A + \epsilon)^2]")
        st.markdown("**Step 4 — Expand the square:**")
        st.latex(r"= E[A^2 + 2A\epsilon + \epsilon^2]")
        st.markdown("**Step 5 — Linearity of expectation:**")
        st.latex(r"= A^2 + 2A \cdot E[\epsilon] + E[\epsilon^2]")
        st.markdown("**Step 6 — Since $E[\\epsilon] = 0$:**")
        st.latex(r"= \underbrace{[f(X) - \hat{f}(X)]^2}_{\text{Reducible Error}} + \underbrace{\text{Var}(\epsilon)}_{\text{Irreducible Error}}")
        st.success("The reducible error shrinks as f̂ improves. The irreducible error is a hard floor — no model can beat it.")

    st.markdown("#### 🕹️ Interactive: Prediction vs Inference Simulator")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prediction Focus — Real Estate**")
        st.caption("We only care about the final predicted price (black box).")
        X_re = ds["RealEstate"][['House_Size_sqft']].values
        y_re = ds["RealEstate"]['Price_USD'].values
        deg = st.slider("Polynomial degree:", 1, 8, 1, key="pred_deg")
        poly = PolynomialFeatures(deg)
        Xp = poly.fit_transform(X_re)
        model_re = LinearRegression().fit(Xp, y_re)
        xs = np.linspace(X_re.min(), X_re.max(), 200).reshape(-1, 1)
        ys = model_re.predict(poly.transform(xs))
        fig_p, ax_p = plt.subplots(figsize=(5, 3))
        ax_p.scatter(X_re, y_re, color='#555', s=15)
        ax_p.plot(xs, ys, color='#e63946', linewidth=2)
        ax_p.set_xlabel("Size"); ax_p.set_ylabel("Price")
        st.pyplot(fig_p)
        st.metric("Train MSE", f"{mean_squared_error(y_re, model_re.predict(Xp)):,.0f}")

    with col2:
        st.markdown("**Inference Focus — Customer LTV**")
        st.caption("We need exact coefficient values to explain each factor's impact.")
        X_ltv = ds["LTV"][['Months_Active', 'Support_Tickets']]
        y_ltv = ds["LTV"]['LTV_USD']
        m_ltv = LinearRegression().fit(X_ltv, y_ltv)
        st.markdown(f"""
        | Predictor | Coefficient (β) | Interpretation |
        |---|---|---|
        | Intercept | {m_ltv.intercept_:,.2f} | Baseline LTV when all predictors = 0 |
        | Months Active | {m_ltv.coef_[0]:,.2f} | Each extra month → +${m_ltv.coef_[0]:,.2f} LTV |
        | Support Tickets | {m_ltv.coef_[1]:,.2f} | Each ticket → ${m_ltv.coef_[1]:,.2f} LTV |
        """)

    st.markdown("---")
    st.subheader("📂 Case Studies")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**3 — Amazon Dynamic Pricing (Prediction)**\nAmazon's algorithms change prices millions of times daily. They don't ask 'why does $9.99 convert better than $10.05?' — they just predict the conversion rate and maximize revenue. Pure black-box prediction.")
    with c2:
        st.info("**4 — McKinsey Marketing Mix Modeling (Inference)**\nConsulting firms build transparent regression models for CMOs, showing exactly: '$1M on TV → $3.2M revenue lift, $1M on Digital → $4.7M lift.' The coefficients ARE the deliverable.")

# ══════════════════════════════════════════════════════════════════
# TAB 3: PARAMETRIC vs NON-PARAMETRIC
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.header("3. Parametric vs Non-Parametric Methods")

    with st.expander("📐 STEP-BY-STEP MATH — Parametric Estimation"):
        st.markdown("**Step 1 — Assume a functional form:**")
        st.latex(r"f(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p")
        st.markdown("**Step 2 — The problem reduces from estimating an arbitrary function to estimating $p+1$ numbers** ($\\beta_0, \\beta_1, \\ldots, \\beta_p$).")
        st.markdown("**Step 3 — Least Squares: minimize the sum of squared residuals:**")
        st.latex(r"\min_{\beta} \sum_{i=1}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}\right)^2")
        st.markdown("**Step 4 — In matrix form (from linear algebra):**")
        st.latex(r"\hat{\beta} = (X^T X)^{-1} X^T Y")
        st.markdown("This closed-form solution gives us the optimal linear coefficients in one calculation.")

    with st.expander("📐 MATH — Non-Parametric (KNN Regression)"):
        st.markdown("**No functional form assumed.** For a query point $x_0$:")
        st.latex(r"\hat{f}(x_0) = \frac{1}{K} \sum_{x_i \in N_K(x_0)} y_i")
        st.markdown("Where $N_K(x_0)$ is the set of K training observations closest to $x_0$ in Euclidean distance:")
        st.latex(r"d(x_i, x_0) = \sqrt{\sum_{j=1}^{p} (x_{ij} - x_{0j})^2}")

    st.markdown("#### 🕹️ Interactive: Side-by-Side Comparison")
    col1, col2 = st.columns(2)
    X_fac = ds["Factory"]['Machine_Temp_C'].values.reshape(-1, 1)
    Y_fac = ds["Factory"]['Output_Units'].values
    xs_f = np.linspace(X_fac.min(), X_fac.max(), 200).reshape(-1, 1)

    with col1:
        st.markdown("**Parametric (Polynomial)**")
        p_deg = st.slider("Polynomial degree:", 1, 10, 1, key="par_deg")
        poly_f = PolynomialFeatures(p_deg)
        Xp_f = poly_f.fit_transform(X_fac)
        m_par = LinearRegression().fit(Xp_f, Y_fac)
        fig_par, ax_par = plt.subplots(figsize=(5, 3))
        ax_par.scatter(X_fac, Y_fac, color='#555', s=15)
        ax_par.plot(xs_f, m_par.predict(poly_f.transform(xs_f)), color='#e63946', linewidth=2)
        ax_par.set_xlabel("Temp"); ax_par.set_ylabel("Output")
        st.pyplot(fig_par)
        st.metric("Parameters", f"{p_deg + 1}")

    with col2:
        st.markdown("**Non-Parametric (KNN)**")
        k_reg = st.slider("K neighbors:", 1, 50, 5, key="knn_reg")
        m_knn = KNeighborsRegressor(n_neighbors=k_reg).fit(X_fac, Y_fac)
        fig_knn, ax_knn = plt.subplots(figsize=(5, 3))
        ax_knn.scatter(X_fac, Y_fac, color='#555', s=15)
        ax_knn.plot(xs_f, m_knn.predict(xs_f), color='#d99a5b', linewidth=2)
        ax_knn.set_xlabel("Temp"); ax_knn.set_ylabel("Output")
        st.pyplot(fig_knn)
        st.metric("Parameters", "None (data-driven)")

    st.markdown("---")
    st.subheader("📂 Case Studies")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**5 — Banking Credit Scoring (Parametric)**\nBanks use Logistic Regression for credit decisions because regulators demand explainability. Each denied applicant must receive a specific reason ('insufficient income', 'high debt-to-income'). Only parametric models can deliver this.")
    with c2:
        st.info("**6 — Kaggle's $1M Competition Winners (Non-Parametric)**\nAlmost every top Kaggle solution uses XGBoost or Random Forests — extremely flexible non-parametric ensembles. When the only metric is prediction accuracy (no inference needed), flexibility wins.")

# ══════════════════════════════════════════════════════════════════
# TAB 4: BIAS-VARIANCE MATH
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.header("4. The Bias-Variance Tradeoff")

    with st.expander("📐 COMPLETE DERIVATION — School Level to Final Equation", expanded=True):
        st.markdown("**Pre-requisite: Variance and Expectation from school stats:**")
        st.latex(r"\text{Var}(Z) = E[Z^2] - (E[Z])^2")
        st.latex(r"\text{Bias}(\hat{f}(x_0)) = E[\hat{f}(x_0)] - f(x_0)")

        st.markdown("**Step 1 — Start with the Expected Test MSE at a point $x_0$:**")
        st.latex(r"E\left[(y_0 - \hat{f}(x_0))^2\right]")

        st.markdown("**Step 2 — Substitute $y_0 = f(x_0) + \\epsilon$:**")
        st.latex(r"= E\left[(f(x_0) + \epsilon - \hat{f}(x_0))^2\right]")

        st.markdown("**Step 3 — Let $\\mu = E[\\hat{f}(x_0)]$. Add and subtract $\\mu$ inside:**")
        st.latex(r"= E\left[(f(x_0) - \mu + \mu - \hat{f}(x_0) + \epsilon)^2\right]")

        st.markdown("**Step 4 — Group: let $A = (f(x_0) - \\mu)$ (a constant — this is the Bias), and $B = (\\mu - \\hat{f}(x_0))$:**")
        st.latex(r"= E\left[(A - B + \epsilon)^2\right]")

        st.markdown("**Step 5 — Expand the trinomial square:**")
        st.latex(r"= E[A^2] - 2E[AB] + E[B^2] + 2E[A\epsilon] - 2E[B\epsilon] + E[\epsilon^2]")

        st.markdown("**Step 6 — Simplify each term:**")
        st.markdown("""
        * $E[A^2] = A^2 = (f(x_0) - \\mu)^2 = [\\text{Bias}]^2$ (A is a constant)
        * $E[B^2] = E[(\\hat{f}(x_0) - \\mu)^2] = \\text{Var}(\\hat{f}(x_0))$
        * $E[AB] = A \\cdot E[B] = A \\cdot (\\mu - E[\\hat{f}(x_0)]) = A \\cdot 0 = 0$ (since $\\mu = E[\\hat{f}]$)
        * $E[A\\epsilon] = A \\cdot E[\\epsilon] = 0$
        * $E[B\\epsilon] = E[B] \\cdot E[\\epsilon] = 0$ (independence + $E[\\epsilon]=0$)
        * $E[\\epsilon^2] = \\text{Var}(\\epsilon) = \\sigma^2$
        """)

        st.markdown("**Step 7 — Final result:**")
        st.latex(r"\boxed{E\left[(y_0 - \hat{f}(x_0))^2\right] = \text{Var}(\hat{f}(x_0)) + [\text{Bias}(\hat{f}(x_0))]^2 + \sigma^2}")

        st.success("This is equation (2.7) from the ISLP textbook. Every term is non-negative, and σ² sets an absolute floor.")

    st.markdown("#### 🕹️ Interactive: The U-Curve Simulator")
    col1, col2 = st.columns([1, 2])
    with col1:
        noise_level = st.slider("Irreducible error (σ²):", 0.5, 5.0, 2.0, 0.5, key="noise")
        bias_strength = st.slider("Bias decay rate:", 1.0, 20.0, 10.0, 1.0, key="bias_s")
        var_strength = st.slider("Variance growth rate:", 0.1, 2.0, 0.5, 0.1, key="var_s")
    with col2:
        flex = np.linspace(1, 10, 200)
        b2 = bias_strength / flex
        v = var_strength * flex**1.5
        total = b2 + v + noise_level

        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        ax4.fill_between(flex, 0, noise_level, alpha=0.15, color='#666', label="σ² floor")
        ax4.plot(flex, b2, color='#e63946', linewidth=2, label="Bias²")
        ax4.plot(flex, v, color='#d99a5b', linewidth=2, label="Variance")
        ax4.plot(flex, total, color='#ffffff', linewidth=2.5, label="Test MSE")
        ax4.axhline(noise_level, color='#555', linestyle='--', linewidth=0.8)
        opt_idx = np.argmin(total)
        ax4.axvline(flex[opt_idx], color='#4CAF50', linestyle=':', linewidth=1.5, label=f"Optimum ≈ {flex[opt_idx]:.1f}")
        ax4.set_xlabel("Model Flexibility"); ax4.set_ylabel("Error")
        ax4.legend(fontsize=8); ax4.set_ylim(0, max(total)*1.1)
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("📂 Case Studies")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**7 — The 2010 Flash Crash (High Variance)**\nAlgorithmic trading bots used hyper-flexible models that memorized random order-book noise. When a large sell order appeared, the bots interpreted normal noise as a crash signal, triggering cascading automated selling that wiped trillions in minutes.")
    with c2:
        st.info("**8 — AlphaGo (Bias-Variance Balance)**\nDeepMind combined high-variance deep neural networks with high-bias Monte Carlo Tree Search. The MCTS provided structural constraints (bias), while the neural net provided pattern recognition (flexibility). This balance defeated the world Go champion.")

# ══════════════════════════════════════════════════════════════════
# TAB 5: REGRESSION vs CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.header("5. Regression vs Classification")
    st.markdown("""
    * **Regression:** $Y$ is continuous (price, temperature, revenue).
    * **Classification:** $Y$ is categorical (spam/not-spam, approved/denied).
    """)

    with st.expander("📐 MATH — The Bayes Classifier & KNN Classification"):
        st.markdown("**The Bayes Classifier** assigns observation $x_0$ to the class $j$ with the highest posterior probability:")
        st.latex(r"\text{Classify to class } j^* = \arg\max_{j} \; Pr(Y = j \mid X = x_0)")
        st.markdown("The **Bayes Error Rate** is the lowest achievable error (analogous to $\\sigma^2$ in regression):")
        st.latex(r"1 - E\left[\max_j Pr(Y = j \mid X)\right]")
        st.markdown("Since we don't know the true conditional distribution, **KNN** estimates it:")
        st.latex(r"Pr(Y = j \mid X = x_0) \approx \frac{1}{K} \sum_{i \in N_K(x_0)} \mathbb{1}(y_i = j)")
        st.markdown("$\\mathbb{1}(y_i = j)$ is the indicator function: 1 if $y_i = j$, else 0. This is literally a majority vote among K neighbors.")

    st.markdown("#### 🕹️ Interactive: KNN Decision Boundary")
    col1, col2 = st.columns([1, 2])
    with col1:
        k_cls = st.slider("K (voters):", 1, 60, 5, key="knn_cls")
        dataset_choice = st.radio("Dataset:", ["Customer Churn", "Loan Approval"], key="cls_ds")

    with col2:
        if dataset_choice == "Customer Churn":
            Xc = ds["Churn"][['Usage_Frequency', 'Price_Sensitivity']].values
            yc = ds["Churn"]['Churn'].values
            xlab, ylab = "Usage Frequency", "Price Sensitivity"
        else:
            Xc = ds["Loan"][['Credit_Score', 'Income_USD']].values
            yc = ds["Loan"]['Approved'].values
            xlab, ylab = "Credit Score", "Income ($)"

        knn_c = KNeighborsClassifier(n_neighbors=k_cls).fit(Xc, yc)
        xx, yy = np.meshgrid(np.linspace(Xc[:,0].min(), Xc[:,0].max(), 120),
                             np.linspace(Xc[:,1].min(), Xc[:,1].max(), 120))
        Z = knn_c.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        ax5.contourf(xx, yy, Z, alpha=0.25, cmap='autumn')
        ax5.scatter(Xc[:,0], Xc[:,1], c=yc, cmap='autumn', edgecolor='#0d0d0d', s=25)
        ax5.set_xlabel(xlab); ax5.set_ylabel(ylab)
        st.pyplot(fig5)

    if k_cls == 1:
        st.warning("⚠️ K=1: Maximum variance — boundary memorizes every point.")
    elif k_cls > 40:
        st.warning("⚠️ K too high: Boundary is overly smooth — high bias.")
    else:
        st.success("✅ Balanced K — good bias-variance tradeoff.")

    st.markdown("---")
    st.subheader("📂 Case Studies")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**9 — Medical Imaging: Tumor Classification**\nRadiologists use SVMs and CNNs to classify tumors as malignant vs benign from MRI scans. False negatives (missing cancer) are penalized 10× more than false positives, requiring asymmetric loss functions.")
    with c2:
        st.info("**10 — Gmail Spam Filter (Naive Bayes)**\nGmail's spam filter calculates $Pr(\\text{Spam} | \\text{words in email})$ using Naive Bayes — a parametric probabilistic classifier. It processes billions of emails daily with remarkable accuracy.")

# ══════════════════════════════════════════════════════════════════
# QUIZ TAB — 50 QUESTIONS
# ══════════════════════════════════════════════════════════════════
with tab_quiz:
    st.header("Executive Assessment — Module 2 (50 Questions)")
    st.caption("Submit to see correct answers and full explanations.")

    ch2_q = [
        {"q":"In Y = f(X) + ε, f(X) is formally defined as:","options":["The sample mean of Y","E[Y | X] — the conditional expectation of Y given X","The variance of Y","The model's prediction"],"answer":"E[Y | X] — the conditional expectation of Y given X","explanation":"f(X) = E[Y|X] because it captures the systematic part of Y explained by X, averaging out ε."},
        {"q":"ε has E[ε] = 0. This means:","options":["ε is always zero","On average, the noise cancels out","There is no noise","The model is perfect"],"answer":"On average, the noise cancels out","explanation":"Individual ε values can be positive or negative, but their expected value (long-run average) is zero."},
        {"q":"Var(ε) = σ² represents:","options":["The model's complexity","The irreducible error floor","The training MSE","The number of features"],"answer":"The irreducible error floor","explanation":"σ² is the variance of the noise — the minimum achievable test MSE regardless of model quality."},
        {"q":"In the error decomposition, the 'reducible error' is:","options":["Var(ε)","[f(X) − f̂(X)]²","Bias²","Variance of f̂"],"answer":"[f(X) − f̂(X)]²","explanation":"This term shrinks as our estimate f̂ improves — it's the gap we can 'reduce' with better modeling."},
        {"q":"An unmeasured variable (like a customer's mood) contributes to:","options":["f(X)","ε","β₀","The training set"],"answer":"ε","explanation":"Unmeasured variables that affect Y but aren't in X become part of the irreducible error ε."},
        {"q":"In supervised learning, we always have:","options":["Only X","Only Y","Both X and Y","Neither X nor Y"],"answer":"Both X and Y","explanation":"Supervised learning requires paired observations (xᵢ, yᵢ) to learn the X→Y mapping."},
        {"q":"K-Means clustering is:","options":["Supervised","Unsupervised","Semi-supervised","Reinforcement learning"],"answer":"Unsupervised","explanation":"K-Means finds groups in X-space without any Y labels."},
        {"q":"Target predicted a teen's pregnancy using:","options":["Unsupervised clustering","Supervised learning on shopping patterns","Random guessing","Linear algebra"],"answer":"Supervised learning on shopping patterns","explanation":"They had labeled data (known pregnant customers + shopping history) and built a supervised predictive model."},
        {"q":"In prediction, f̂ is treated as:","options":["A transparent equation","A black box","A clustering algorithm","An unsupervised method"],"answer":"A black box","explanation":"For prediction, we don't care about f̂'s internal form — only its output accuracy matters."},
        {"q":"In inference, we need to know:","options":["Only the predicted Y","The exact form and coefficients of f̂","The training MSE only","The number of clusters"],"answer":"The exact form and coefficients of f̂","explanation":"Inference requires understanding which predictors matter and how much each contributes."},
        {"q":"Amazon changes prices millions of times daily. This is:","options":["Inference","Prediction","Clustering","Semi-supervised"],"answer":"Prediction","explanation":"Amazon's algorithm predicts conversion rates to maximize revenue — pure black-box prediction."},
        {"q":"McKinsey building a model to show a CMO exactly how much ROI TV ads generate is:","options":["Prediction","Inference","Unsupervised","Classification"],"answer":"Inference","explanation":"The coefficients (how much $1 of TV spend → revenue) ARE the deliverable."},
        {"q":"Parametric methods assume:","options":["No structure for f","f has a specific functional form (e.g., linear)","The data has no noise","K neighbors are needed"],"answer":"f has a specific functional form (e.g., linear)","explanation":"Parametric = assume a shape, then estimate a fixed number of parameters."},
        {"q":"The parametric assumption f(X) = β₀ + β₁X₁ + ... + βₚXₚ reduces estimation to finding:","options":["An infinite-dimensional function","p + 1 numbers","Zero parameters","The exact noise values"],"answer":"p + 1 numbers","explanation":"The p coefficients plus the intercept — a finite, manageable set of parameters."},
        {"q":"The least squares solution in matrix form is:","options":["β̂ = X⁻¹Y","β̂ = (XᵀX)⁻¹XᵀY","β̂ = YᵀX","β̂ = X + Y"],"answer":"β̂ = (XᵀX)⁻¹XᵀY","explanation":"This closed-form solution minimizes the sum of squared residuals analytically."},
        {"q":"Non-parametric methods' main advantage is:","options":["They need less data","They can fit a wider range of shapes for f","They are always more interpretable","They have no variance"],"answer":"They can fit a wider range of shapes for f","explanation":"By not assuming a form for f, non-parametric methods adapt to any underlying pattern."},
        {"q":"Non-parametric methods' main disadvantage is:","options":["They can only fit lines","They need far more data for accurate estimates","They have zero flexibility","They always overfit"],"answer":"They need far more data for accurate estimates","explanation":"Without parametric constraints, the model needs many observations to estimate f well."},
        {"q":"KNN regression predicts f̂(x₀) by:","options":["Fitting a global line","Averaging the Y values of the K nearest training points","Using gradient descent","Running K-Means"],"answer":"Averaging the Y values of the K nearest training points","explanation":"f̂(x₀) = (1/K)Σyᵢ for xᵢ in the K nearest neighbors of x₀."},
        {"q":"Banks use parametric models for credit scoring because:","options":["They're more accurate","Regulators require explainability","They're faster to train","Non-parametric models don't exist"],"answer":"Regulators require explainability","explanation":"Each denied applicant must receive a specific, traceable reason — only parametric models provide this."},
        {"q":"Kaggle competition winners typically use:","options":["Simple linear regression","Non-parametric ensembles (XGBoost, Random Forest)","K-Means clustering","Manual data entry"],"answer":"Non-parametric ensembles (XGBoost, Random Forest)","explanation":"When the only metric is prediction accuracy, maximum flexibility (non-parametric) wins."},
        {"q":"MSE stands for:","options":["Maximum Squared Estimate","Mean Squared Error","Minimum Standard Error","Model Selection Efficiency"],"answer":"Mean Squared Error","explanation":"MSE = (1/n)Σ(yᵢ − f̂(xᵢ))² — the average squared prediction error."},
        {"q":"Training MSE generally _____ as flexibility increases.","options":["Increases","Decreases monotonically","Stays the same","Oscillates"],"answer":"Decreases monotonically","explanation":"More flexible models fit training data more closely, always reducing training MSE."},
        {"q":"Test MSE generally follows a _____ shape as flexibility increases.","options":["Flat line","Monotone decrease","U-shape","Exponential growth"],"answer":"U-shape","explanation":"Test MSE first drops (less bias) then rises (more variance) — the classic U-curve."},
        {"q":"Overfitting means the model has:","options":["Low training MSE, high test MSE","High training MSE, low test MSE","Low training MSE, low test MSE","High training MSE, high test MSE"],"answer":"Low training MSE, high test MSE","explanation":"The model memorized training noise (low train error) but fails on new data (high test error)."},
        {"q":"In the Bias-Variance decomposition, Bias(f̂(x₀)) = :","options":["E[f̂(x₀)] − f(x₀)","Var(f̂(x₀))","σ²","f̂(x₀) − y₀"],"answer":"E[f̂(x₀)] − f(x₀)","explanation":"Bias measures how far the average prediction (across many training sets) is from the truth."},
        {"q":"Variance of f̂(x₀) measures:","options":["How far the prediction is from the truth on average","How much f̂(x₀) would change across different training sets","The irreducible noise","The model's interpretability"],"answer":"How much f̂(x₀) would change across different training sets","explanation":"Variance captures the sensitivity of the model to the specific training data used."},
        {"q":"A straight line fit to curved data has:","options":["High variance","High bias","Low bias","Zero error"],"answer":"High bias","explanation":"The rigid linear assumption is wrong for curved data — this systematic error is bias."},
        {"q":"A KNN model with K=1 has:","options":["High bias, low variance","Low bias, high variance","Both high","Both low"],"answer":"Low bias, high variance","explanation":"K=1 exactly interpolates training data (low bias) but is extremely sensitive to individual points (high variance)."},
        {"q":"The Bias² + Variance tradeoff implies:","options":["You can minimize both simultaneously","Decreasing one generally increases the other","Both always increase together","They are independent"],"answer":"Decreasing one generally increases the other","explanation":"This is the fundamental tradeoff — more flexibility reduces bias but increases variance, and vice versa."},
        {"q":"Cross-validation estimates:","options":["Training MSE","Test MSE from training data","Irreducible error exactly","The true f"],"answer":"Test MSE from training data","explanation":"CV partitions training data to simulate the train/test split, estimating out-of-sample error."},
        {"q":"The 2010 Flash Crash was caused by models with:","options":["High bias (too rigid)","High variance (memorized noise)","Perfect calibration","Too few parameters"],"answer":"High variance (memorized noise)","explanation":"The trading algorithms were so flexible they interpreted normal noise as crash signals."},
        {"q":"AlphaGo balanced bias and variance by combining:","options":["Two linear models","Deep neural nets (flexible) + Monte Carlo Tree Search (constrained)","Only KNN","Only logistic regression"],"answer":"Deep neural nets (flexible) + Monte Carlo Tree Search (constrained)","explanation":"MCTS provided structural constraints (bias) while neural nets provided pattern recognition (variance)."},
        {"q":"In classification, the Bayes Classifier assigns x₀ to:","options":["The class with lowest probability","The class with highest Pr(Y=j|X=x₀)","A random class","The nearest training point"],"answer":"The class with highest Pr(Y=j|X=x₀)","explanation":"The Bayes Classifier is the theoretically optimal rule: pick the most probable class."},
        {"q":"The Bayes Error Rate is analogous to _____ in regression.","options":["Bias","Variance","Var(ε) / Irreducible error","Training MSE"],"answer":"Var(ε) / Irreducible error","explanation":"It's the lowest achievable error rate — the classification equivalent of σ²."},
        {"q":"KNN classification estimates Pr(Y=j|X=x₀) by:","options":["Fitting a logistic curve","Counting the proportion of class j among K nearest neighbors","Using gradient descent","Minimizing MSE"],"answer":"Counting the proportion of class j among K nearest neighbors","explanation":"Pr(Y=j|X=x₀) ≈ (1/K)Σ𝟙(yᵢ=j) for the K nearest points — a majority vote."},
        {"q":"The indicator function 𝟙(yᵢ = j) returns:","options":["The probability of class j","1 if yᵢ equals j, 0 otherwise","The distance to x₀","The variance"],"answer":"1 if yᵢ equals j, 0 otherwise","explanation":"It's a binary flag: 1 when the condition is true, 0 when false."},
        {"q":"Gmail's spam filter uses:","options":["KNN","Naive Bayes","K-Means","Linear Regression"],"answer":"Naive Bayes","explanation":"Naive Bayes calculates Pr(Spam|words) using Bayes' theorem with independence assumptions."},
        {"q":"Medical imaging classifiers penalize false negatives heavily because:","options":["Missing cancer is far worse than a false alarm","False positives are more expensive","The data is balanced","Doctors prefer high bias"],"answer":"Missing cancer is far worse than a false alarm","explanation":"A false negative means sending a cancer patient home untreated — the cost is catastrophic."},
        {"q":"A quantitative response leads to _____ problems.","options":["Classification","Regression","Clustering","Unsupervised"],"answer":"Regression","explanation":"Continuous/numerical responses are modeled with regression methods."},
        {"q":"A qualitative (categorical) response leads to _____ problems.","options":["Regression","Classification","Dimensionality reduction","Feature engineering"],"answer":"Classification","explanation":"Categorical responses (Yes/No, A/B/C) are modeled with classification methods."},
        {"q":"Logistic regression, despite its name, is primarily used for:","options":["Regression","Classification","Clustering","Feature selection"],"answer":"Classification","explanation":"It estimates class probabilities and assigns categorical labels, making it a classifier."},
        {"q":"The flexibility-interpretability tradeoff means:","options":["More flexible models are easier to interpret","More flexible models are harder to interpret","All models are equally interpretable","Flexibility doesn't affect interpretability"],"answer":"More flexible models are harder to interpret","explanation":"As model complexity grows, the internal logic becomes opaque — the core tradeoff."},
        {"q":"The Lasso makes linear regression MORE interpretable by:","options":["Adding more features","Setting some coefficients to exactly zero","Increasing polynomial degree","Using KNN instead"],"answer":"Setting some coefficients to exactly zero","explanation":"The L1 penalty forces irrelevant coefficients to zero, performing automatic variable selection."},
        {"q":"Degrees of freedom in a model measures:","options":["Data quality","The effective number of parameters / flexibility","The number of observations","The irreducible error"],"answer":"The effective number of parameters / flexibility","explanation":"More degrees of freedom = more flexibility = higher capacity to fit complex patterns."},
        {"q":"In Figure 2.6 (ISLP), the thin-plate spline with zero training error is:","options":["The optimal model","Overfitting — too wiggly","Underfitting — too rigid","The Bayes optimal"],"answer":"Overfitting — too wiggly","explanation":"Zero training error means it passes through every point, capturing noise instead of the true smooth f."},
        {"q":"For the Employee Productivity dataset (step function), which model would work best?","options":["Linear regression","A decision tree or non-parametric method","A constant mean model","Logistic regression"],"answer":"A decision tree or non-parametric method","explanation":"Step functions are inherently non-linear — parametric linear models would miss the jumps entirely."},
        {"q":"If E[f̂(x₀)] = f(x₀) exactly, the model has:","options":["Zero variance","Zero bias","Zero irreducible error","Perfect test MSE"],"answer":"Zero bias","explanation":"Bias = E[f̂(x₀)] − f(x₀) = 0 when the expected prediction equals the truth."},
        {"q":"To minimize Expected Test MSE, we need:","options":["Zero bias only","Zero variance only","Simultaneously low bias AND low variance","Zero irreducible error"],"answer":"Simultaneously low bias AND low variance","explanation":"Test MSE = Bias² + Variance + σ². We can't control σ², so we minimize the sum of the first two."},
        {"q":"The optimal model flexibility is at the:","options":["Maximum of the U-curve","Minimum of the U-curve","Point where training MSE is zero","Point where variance is zero"],"answer":"Minimum of the U-curve","explanation":"The bottom of the U-curve represents the best bias-variance balance and lowest test MSE."},
        {"q":"If we could collect ALL possible training sets and average our predictions, the remaining error would be:","options":["Zero","Bias² + σ²","σ² only","Variance only"],"answer":"Bias² + σ²","explanation":"Averaging over training sets eliminates variance, leaving only the systematic bias and irreducible noise."},
    ]

    render_quiz("ch2", ch2_q)
