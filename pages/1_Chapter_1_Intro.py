import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quiz_engine import render_quiz

st.set_page_config(page_title="Module 1: Intro", page_icon="📊", layout="wide")

plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#0d0d0d', 'figure.facecolor': '#0d0d0d',
    'text.color': '#e0e0e0', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0',
    'axes.edgecolor': '#333', 'grid.color': '#222'
})

# Creative page-level background
st.markdown("""
<style>
.stApp {
    background-image:
        radial-gradient(ellipse at 50% -20%, rgba(217,154,91,0.12) 0%, transparent 55%),
        linear-gradient(180deg, #0d0d0d 0%, #111 100%);
    background-attachment: fixed;
}
</style>
<div style="font-family:'Inter',sans-serif;font-weight:800;text-transform:uppercase;font-size:0.78rem;letter-spacing:6px;color:#d99a5b;border-bottom:2px solid #e63946;padding-bottom:8px;margin-bottom:28px;text-align:right;">Envisioned by Aadi</div>
""", unsafe_allow_html=True)

st.title("MODULE 1 — Data Foundations 📊")

tab_learn, tab_quiz = st.tabs(["LEARN", "ASSESSMENT — 50 QUESTIONS"])

# ─────────────────────────── LEARN TAB ───────────────────────────
with tab_learn:
    st.markdown("""
### Why Statistical Learning?
Think about how Netflix recommends movies, or how TikTok knows your next scroll.
At its core, **Statistical Learning** uses data to:

1. **Predict** an unknown outcome.
2. **Infer** the relationship between variables.
    """)

    import sys
    @st.cache_data
    def load_data():
        if sys.platform == "emscripten":
            fp = "/home/pyodide/dataset/Employee_Productivity_Dataset.csv"
        else:
            fp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "Employee_Productivity_Dataset.csv"))
        return pd.read_csv(fp) if os.path.exists(fp) else None

    df = load_data()
    if df is not None:
        with st.expander("VIEW RAW DATA MATRIX"):
            st.dataframe(df.head(15))
        col = st.selectbox("Select a feature:", df.select_dtypes(include=['number']).columns)
        if col:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            df[col].hist(bins=20, ax=ax, color="#d99a5b", edgecolor="#0d0d0d")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
### 📂 Case Study: The Netflix $1 M Prize
In 2006 Netflix offered **$1,000,000** for a 10 % accuracy improvement on movie ratings prediction.

* **Paradigm:** Pure **Prediction** — the algorithm is a black box; Netflix doesn't need to know *why* you liked Inception.
* **Winner:** "BellKor's Pragmatic Chaos" ensembled hundreds of models, sacrificing interpretability for raw accuracy.
* **Business takeaway:** Maximum prediction accuracy often trades off against model interpretability — a core theme of Chapter 2.
    """)

# ─────────────────────────── QUIZ TAB ────────────────────────────
with tab_quiz:
    st.header("Executive Assessment — Module 1 (50 Questions)")
    st.caption("After submitting, correct answers and explanations are shown for every question.")

    ch1_questions = [
        {"q":"In Y = f(X) + ε, what does f(X) represent?","options":["Random noise","The true systematic relationship between X and Y","The model's prediction","The training data"],"answer":"The true systematic relationship between X and Y","explanation":"f(X) is the fixed, unknown function that maps predictors to the response — the signal we are trying to estimate."},
        {"q":"ε (epsilon) in the master equation represents:","options":["The model's weights","Irreducible random error with mean zero","The learning rate","The number of features"],"answer":"Irreducible random error with mean zero","explanation":"ε captures unmeasured variables and inherent randomness. By definition E[ε]=0 and it cannot be reduced no matter how good the model."},
        {"q":"A CEO wants to know exactly how much each $1 of TV ad spend boosts revenue. This is:","options":["Prediction","Inference","Classification","Clustering"],"answer":"Inference","explanation":"Inference seeks to understand the specific relationship between predictors and response, not just predict the outcome."},
        {"q":"Netflix recommending a movie to a user is an example of:","options":["Inference","Prediction","Unsupervised Learning","Parametric Modeling"],"answer":"Prediction","explanation":"Netflix treats f̂ as a black box — the only goal is an accurate predicted rating, not understanding why."},
        {"q":"Grouping customers by purchasing behavior without any labels is:","options":["Supervised Learning","Unsupervised Learning","Linear Regression","Deep Learning"],"answer":"Unsupervised Learning","explanation":"With no Y variable to guide the model, this is unsupervised — the algorithm must discover structure on its own."},
        {"q":"Which term in Y = f(X) + ε can NEVER be reduced?","options":["f(X)","X","ε","β₀"],"answer":"ε","explanation":"ε is the irreducible error — it represents randomness that exists regardless of model quality."},
        {"q":"If you need regulators to understand exactly why a loan was denied, you prioritize:","options":["Prediction accuracy","Model interpretability (Inference)","A neural network","Maximizing flexibility"],"answer":"Model interpretability (Inference)","explanation":"Regulatory compliance demands explainability — you must be able to trace the decision back to specific predictors."},
        {"q":"Predicting whether a stock goes UP or DOWN is a:","options":["Regression problem","Classification problem","Clustering problem","Inference problem"],"answer":"Classification problem","explanation":"The response variable is categorical (UP/DOWN), making this classification."},
        {"q":"Predicting the exact price of a house in dollars is a:","options":["Regression problem","Classification problem","Clustering problem","Parametric constraint"],"answer":"Regression problem","explanation":"The response is a continuous numerical value, so this is regression."},
        {"q":"The Netflix Prize winners prioritized:","options":["Inference over Prediction","Prediction over Inference","Linear regression only","Ignoring the data"],"answer":"Prediction over Inference","explanation":"The competition's sole metric was prediction accuracy (RMSE), so interpretability was irrelevant."},
        {"q":"Which approach assumes f(X) has a specific mathematical shape (e.g. a line)?","options":["Non-parametric","Parametric","Unsupervised","K-Nearest Neighbors"],"answer":"Parametric","explanation":"Parametric methods assume a functional form (e.g. linear), reducing estimation to finding a fixed number of parameters."},
        {"q":"The ultimate goal of statistical learning in business is:","options":["Drawing charts","Memorizing training data","Accurately estimating f(X) to make decisions on unseen data","Eliminating ε"],"answer":"Accurately estimating f(X) to make decisions on unseen data","explanation":"We want f̂ that generalizes to new data for actionable decision-making; we cannot eliminate ε."},
        {"q":"Training data is used to:","options":["Evaluate model performance","Teach the model how to estimate f","Test the model on unseen examples","Reduce irreducible error"],"answer":"Teach the model how to estimate f","explanation":"Training data is the set of observations {(x₁,y₁),...,(xₙ,yₙ)} used to fit (train) the model."},
        {"q":"Test data is used to:","options":["Fit the model parameters","Evaluate prediction performance on unseen observations","Increase the training set size","Eliminate bias"],"answer":"Evaluate prediction performance on unseen observations","explanation":"Test data measures how well the model generalizes beyond the data it was trained on."},
        {"q":"A hospital classifying tumors as malignant vs benign uses:","options":["Regression","Classification","Clustering","Dimensionality reduction"],"answer":"Classification","explanation":"The response is categorical (malignant/benign), making this a classification task."},
        {"q":"A model that memorizes training data but fails on new data is:","options":["Underfitting","Overfitting","Perfectly calibrated","Parametric"],"answer":"Overfitting","explanation":"Overfitting means the model captures noise in training data rather than the true signal, leading to poor generalization."},
        {"q":"An overfit model has:","options":["High bias, low variance","Low bias, high variance","Low bias, low variance","High bias, high variance"],"answer":"Low bias, high variance","explanation":"Overfitting = too flexible, so bias is low but the model is highly sensitive to the specific training set (high variance)."},
        {"q":"An underfit model has:","options":["High bias, low variance","Low bias, high variance","Low bias, low variance","High bias, high variance"],"answer":"High bias, low variance","explanation":"Underfitting = too rigid, so the model makes strong wrong assumptions (high bias) but is stable (low variance)."},
        {"q":"If f(X) is perfectly estimated, the remaining error is:","options":["Zero","Var(ε)","Bias²","MSE"],"answer":"Var(ε)","explanation":"Even a perfect estimate of f still can't predict ε, so Var(ε) remains as the irreducible floor."},
        {"q":"Which of these is a quantitative (numerical) variable?","options":["Customer ZIP code category","Loan approval (Yes/No)","House price in dollars","Brand name (A/B/C)"],"answer":"House price in dollars","explanation":"House price takes continuous numerical values, while the others are categorical."},
        {"q":"Which of these is a qualitative (categorical) variable?","options":["Temperature in °C","Stock price","Customer segment (Gold/Silver/Bronze)","Revenue in $"],"answer":"Customer segment (Gold/Silver/Bronze)","explanation":"Customer segment takes discrete category values, not continuous numbers."},
        {"q":"Semi-supervised learning is when:","options":["All observations have labels","No observations have labels","Some observations have labels and some don't","The model is unsupervised"],"answer":"Some observations have labels and some don't","explanation":"Semi-supervised learning uses both labeled (m) and unlabeled (n-m) observations for training."},
        {"q":"In the equation E(Y − Ŷ)² = [f(X)−f̂(X)]² + Var(ε), the first term is:","options":["Irreducible error","Reducible error","Variance of ε","The Bayes error"],"answer":"Reducible error","explanation":"[f(X)−f̂(X)]² is the gap between the true function and our estimate — we can shrink this by improving f̂."},
        {"q":"A 'black box' model means:","options":["It's painted black","We can't interpret how it makes predictions","It has high bias","It's unsupervised"],"answer":"We can't interpret how it makes predictions","explanation":"A black box model produces accurate outputs but its internal decision logic is opaque."},
        {"q":"Which scenario is purely a Prediction problem?","options":["Understanding how rain affects truck delays","Predicting tomorrow's stock price","Identifying which ad medium drives the most sales","Explaining why customers churn"],"answer":"Predicting tomorrow's stock price","explanation":"We only care about the predicted number, not understanding the underlying causal mechanism."},
        {"q":"Which scenario is purely an Inference problem?","options":["Predicting Netflix ratings","Understanding how 1 extra year of education affects salary","Classifying spam emails","Clustering market segments"],"answer":"Understanding how 1 extra year of education affects salary","explanation":"Inference demands understanding the precise form and magnitude of the X→Y relationship."},
        {"q":"Linear regression is:","options":["Non-parametric","Parametric","Unsupervised","A classification method"],"answer":"Parametric","explanation":"It assumes f(X) = β₀ + β₁X₁ + ... + βₚXₚ — a fixed functional form with parameters to estimate."},
        {"q":"K-Nearest Neighbors is:","options":["Parametric","Non-parametric","Only for regression","Only for binary data"],"answer":"Non-parametric","explanation":"KNN makes no assumption about f's shape — it simply looks at the K closest training points."},
        {"q":"As model flexibility increases, training MSE generally:","options":["Increases","Decreases","Stays the same","Oscillates"],"answer":"Decreases","explanation":"More flexible models can fit training data more closely, always reducing training MSE (potentially to zero)."},
        {"q":"As model flexibility increases, test MSE generally:","options":["Always decreases","Always increases","First decreases then increases (U-shape)","Stays flat"],"answer":"First decreases then increases (U-shape)","explanation":"Initially, added flexibility captures true signal (lower error), but eventually it captures noise (higher error)."},
        {"q":"The p(p-1)/2 formula tells us:","options":["The number of training samples needed","The number of distinct scatterplots for p variables","The degrees of freedom","The irreducible error"],"answer":"The number of distinct scatterplots for p variables","explanation":"With p variables, we can make p(p-1)/2 unique pairwise scatterplots — visual inspection becomes infeasible for large p."},
        {"q":"Cluster analysis is an example of:","options":["Supervised learning","Unsupervised learning","Semi-supervised learning","Reinforcement learning"],"answer":"Unsupervised learning","explanation":"Clustering finds groups in data without a labeled response variable."},
        {"q":"In the Advertising dataset, asking 'Which media generates the biggest boost in sales?' is:","options":["Prediction","Inference","Classification","Clustering"],"answer":"Inference","explanation":"We want to understand the magnitude and direction of each predictor's effect — that's inference."},
        {"q":"A model with 0 training error is likely:","options":["Perfect","Overfitting","Underfitting","Irreducible"],"answer":"Overfitting","explanation":"Zero training error usually means the model memorized every data point, including noise."},
        {"q":"The 'No Free Lunch' theorem says:","options":["Linear regression is always best","No single method dominates on all datasets","Neural networks always win","We should always use KNN"],"answer":"No single method dominates on all datasets","explanation":"Different datasets have different underlying structures, so no one-size-fits-all method exists."},
        {"q":"Logistic regression, despite its name, is used for:","options":["Regression","Classification","Clustering","Dimensionality reduction"],"answer":"Classification","explanation":"Logistic regression estimates class probabilities for a categorical Y (typically binary)."},
        {"q":"What does 'degrees of freedom' measure?","options":["Data quality","Model flexibility/complexity","Number of training samples","Irreducible error magnitude"],"answer":"Model flexibility/complexity","explanation":"Degrees of freedom summarizes how many parameters/dimensions a model can freely adjust."},
        {"q":"A CEO says 'I don't care how the model works, just give me accurate prices.' This is:","options":["Inference","Prediction","Clustering","Unsupervised"],"answer":"Prediction","explanation":"The CEO treats the model as a black box — only the output accuracy matters."},
        {"q":"A scientist says 'I need to know exactly which gene drives this disease.' This is:","options":["Prediction","Inference","Clustering","Semi-supervised"],"answer":"Inference","explanation":"The scientist needs to understand the precise role of each predictor, not just predict outcomes."},
        {"q":"If X₁ = age and X₂ = income, f(X) maps these to:","options":["A random number","The systematic expected value of Y","The irreducible error","Another set of predictors"],"answer":"The systematic expected value of Y","explanation":"f(X) = E(Y|X) captures how predictors systematically relate to the response."},
        {"q":"In the Income dataset example, the vertical black lines in Figure 2.2 represent:","options":["f(X)","The predicted values","The error terms ε","The regression coefficients"],"answer":"The error terms ε","explanation":"These lines show the distance between actual observations and the true f curve — that's ε."},
        {"q":"When we say a model 'generalizes well', we mean:","options":["It has low training MSE","It has low test MSE on unseen data","It memorizes training data","It has zero parameters"],"answer":"It has low test MSE on unseen data","explanation":"Good generalization means the model performs well on data it has never seen during training."},
        {"q":"Which is more interpretable?","options":["A neural network with 10M parameters","A linear regression with 3 coefficients","A random forest with 500 trees","A KNN model with K=1"],"answer":"A linear regression with 3 coefficients","explanation":"Fewer parameters and a fixed functional form make it easy to trace each predictor's contribution."},
        {"q":"The Lasso is more restrictive than plain linear regression because:","options":["It adds more parameters","It sets some coefficients to exactly zero","It uses neural networks","It ignores the data"],"answer":"It sets some coefficients to exactly zero","explanation":"The Lasso penalty forces some βⱼ = 0, effectively performing variable selection and increasing interpretability."},
        {"q":"GAMs extend linear models by:","options":["Adding more linear terms","Allowing certain non-linear relationships between each predictor and Y","Removing all predictors","Switching to classification"],"answer":"Allowing certain non-linear relationships between each predictor and Y","explanation":"Generalized Additive Models replace β₁X₁ with f₁(X₁), allowing curves while maintaining additive structure."},
        {"q":"Bagging and Boosting are examples of:","options":["Highly interpretable methods","Highly flexible methods","Parametric methods","Unsupervised methods"],"answer":"Highly flexible methods","explanation":"These ensemble methods combine many learners, achieving high flexibility but sacrificing interpretability."},
        {"q":"In real estate, predicting if a house is under- or over-valued is:","options":["Prediction","Inference","Both prediction and inference","Unsupervised"],"answer":"Prediction","explanation":"We just want an accurate price estimate to compare against the asking price."},
        {"q":"In real estate, understanding how river proximity affects value is:","options":["Prediction","Inference","Classification","Clustering"],"answer":"Inference","explanation":"We want to quantify the specific effect of one predictor (river proximity) on the response (price)."},
        {"q":"Cross-validation is a method for:","options":["Generating training data","Estimating test MSE using only training data","Eliminating ε","Increasing model flexibility"],"answer":"Estimating test MSE using only training data","explanation":"Cross-validation partitions training data to simulate the process of testing on unseen observations."},
        {"q":"The thin-plate spline in Figure 2.6 has zero training error. This is:","options":["A good thing — perfect fit","Bad — it's overfitting the noise","Irreducible error at work","A sign of high bias"],"answer":"Bad — it's overfitting the noise","explanation":"Zero training error means the surface passes through every point, capturing noise rather than the true smooth f."},
    ]

    render_quiz("ch1", ch1_questions)
