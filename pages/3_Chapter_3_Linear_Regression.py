import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quiz_engine import render_quiz

st.set_page_config(page_title="Module 3: Linear Regression", page_icon="📈", layout="wide")

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
        radial-gradient(ellipse at 80% 20%, rgba(230,57,70,0.08) 0%, transparent 50%),
        linear-gradient(180deg, #0d0d0d 0%, #0f0f0f 100%);
    background-attachment: fixed;
}
</style>
<div style="font-family:'Inter',sans-serif;font-weight:800;text-transform:uppercase;font-size:0.78rem;letter-spacing:6px;color:#d99a5b;border-bottom:2px solid #e63946;padding-bottom:8px;margin-bottom:28px;text-align:right;">Envisioned by Aadi</div>
""", unsafe_allow_html=True)

st.title("MODULE 3 — Linear Regression 📈")

tab_learn, tab_quiz = st.tabs(["LEARN", "ASSESSMENT — 50 Q's"])

@st.cache_data
def load_data():
    d = r"C:\Users\aadis\Downloads\dataset"
    return {
        "RealEstate": pd.read_csv(os.path.join(d, "Real_Estate_Dataset.csv")),
        "LTV": pd.read_csv(os.path.join(d, "Customer_LTV_Dataset.csv")),
        "Factory": pd.read_csv(os.path.join(d, "Factory_Output_Dataset.csv")),
    }

ds = load_data()

with tab_learn:
    st.header("The Fundamental Parametric Model")
    st.latex(r"Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon")

    with st.expander("📐 COMPLETE DERIVATION — From School to Least Squares"):
        st.markdown("**Step 1 — School Level ($y = mx + c$):**")
        st.markdown("You learned that a line has a slope $m$ and intercept $c$. In statistics, we use $\\beta_1$ and $\\beta_0$:")
        st.latex(r"Y = \beta_0 + \beta_1 X")

        st.markdown("**Step 2 — Define the residual for one point:**")
        st.latex(r"e_i = y_i - \hat{y}_i = y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i)")

        st.markdown("**Step 3 — We want to minimize the sum of squared residuals (RSS):**")
        st.latex(r"RSS = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2")

        st.markdown("**Step 4 — Take partial derivatives and set to zero (from calculus):**")
        st.latex(r"\frac{\partial \text{RSS}}{\partial \hat{\beta}_0} = -2 \sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0")
        st.latex(r"\frac{\partial \text{RSS}}{\partial \hat{\beta}_1} = -2 \sum_{i=1}^{n} x_i(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0")

        st.markdown("**Step 5 — Solve the two equations:**")
        st.markdown("From the first equation:")
        st.latex(r"\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}")
        st.markdown("Substituting into the second equation:")
        st.latex(r"\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}")

        st.markdown("**Step 6 — In matrix form (multiple regression):**")
        st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}")
        st.success("This is the closed-form least squares solution. No iteration, no gradient descent — pure linear algebra.")

    st.markdown("---")
    st.markdown("### 🕹️ MINIGAME: Be the Algorithm")
    st.write("Manually tune β₀ and β₁ to minimize MSE on the Real Estate dataset.")

    col1, col2 = st.columns([1, 2])
    with col1:
        beta_0 = st.slider("Intercept (β̂₀)", -200000, 200000, 0, step=5000)
        beta_1 = st.slider("Slope (β̂₁)", -200, 500, 0, step=5)

        X = ds["RealEstate"]['House_Size_sqft'].values
        Y = ds["RealEstate"]['Price_USD'].values
        preds = beta_0 + beta_1 * X
        mse_val = np.mean((Y - preds)**2)

        # Compute optimal for comparison
        m_opt = LinearRegression().fit(X.reshape(-1,1), Y)
        opt_mse = mean_squared_error(Y, m_opt.predict(X.reshape(-1,1)))

        st.metric("Your MSE", f"{mse_val:,.0f}")
        st.metric("Optimal MSE", f"{opt_mse:,.0f}")
        gap = ((mse_val - opt_mse) / opt_mse) * 100
        if gap < 5:
            st.success(f"🟢 {gap:.1f}% above optimal — excellent!")
        elif gap < 30:
            st.warning(f"🟡 {gap:.1f}% above optimal — keep tuning.")
        else:
            st.error(f"🔴 {gap:.1f}% above optimal — far off.")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, Y, color='#555', s=15)
        x_line = np.array([X.min(), X.max()])
        ax.plot(x_line, beta_0 + beta_1 * x_line, color='#e63946', linewidth=2.5, label="Your guess")
        ax.plot(x_line, m_opt.predict(x_line.reshape(-1,1)), color='#4CAF50', linewidth=1.5, linestyle='--', label="Optimal")
        ax.set_xlabel("House Size (sqft)"); ax.set_ylabel("Price ($)")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.markdown("---")

    st.markdown("### Multivariate: Real Estate with 2 Predictors")
    X_multi = ds["RealEstate"][['Distance_to_City_Miles', 'House_Size_sqft']]
    y_multi = ds["RealEstate"]['Price_USD']
    m_multi = LinearRegression().fit(X_multi, y_multi)
    st.markdown(f"""
    | Predictor | Coefficient (β) | Business Interpretation |
    |---|---|---|
    | Intercept | {m_multi.intercept_:,.0f} | Base price when distance=0 and size=0 |
    | Distance to City | {m_multi.coef_[0]:,.0f} | Each extra mile → ${m_multi.coef_[0]:,.0f} on price |
    | House Size | {m_multi.coef_[1]:,.0f} | Each extra sqft → +${m_multi.coef_[1]:,.0f} on price |
    """)

    st.markdown("---")
    st.markdown("""
### 📂 Case Study: Zillow's Zestimate
* **The Model:** Zillow initially used massive Multivariate Linear Regression.
* **Inference Power:** They could tell users exactly how much adding a bathroom (+$15K) or a bedroom (+$8K) would affect home value — because each β is directly interpretable.
* **The Limitation:** Adding a 5th bathroom ≠ adding a 2nd bathroom (diminishing returns). This non-linear reality broke the rigid linear assumption, forcing Zillow to adopt Random Forests.
* **The Lesson:** Linear regression is invaluable for inference but limited for complex non-linear prediction.
    """)

with tab_quiz:
    st.header("Executive Assessment — Module 3 (50 Questions)")
    st.caption("Submit to see correct answers and full explanations.")

    ch3_q = [
        {"q":"In Y = β₀ + β₁X + ε, β₁ represents:","options":["The intercept","The expected change in Y for a 1-unit increase in X","The irreducible error","The training MSE"],"answer":"The expected change in Y for a 1-unit increase in X","explanation":"β₁ is the slope — it quantifies the marginal effect of X on Y, holding all else constant."},
        {"q":"β₀ (the intercept) represents:","options":["The slope","The expected value of Y when all predictors are zero","The error term","The number of features"],"answer":"The expected value of Y when all predictors are zero","explanation":"β₀ is the Y-intercept — the predicted Y when X₁=X₂=...=Xₚ=0."},
        {"q":"RSS stands for:","options":["Random Sample Selection","Residual Sum of Squares","Regression Standard Score","Root Sum of Squares"],"answer":"Residual Sum of Squares","explanation":"RSS = Σ(yᵢ − ŷᵢ)² — the total squared prediction error that least squares minimizes."},
        {"q":"The least squares method minimizes:","options":["The absolute errors","The squared residuals (RSS)","The number of parameters","The variance of ε"],"answer":"The squared residuals (RSS)","explanation":"Ordinary Least Squares finds β̂ that minimizes Σ(yᵢ − β₀ − β₁xᵢ)²."},
        {"q":"The formula β̂₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)² comes from:","options":["Guessing","Setting the partial derivative of RSS w.r.t. β₁ to zero","Neural network backpropagation","Random sampling"],"answer":"Setting the partial derivative of RSS w.r.t. β₁ to zero","explanation":"Calculus optimization: ∂RSS/∂β₁ = 0 yields the closed-form solution for the slope."},
        {"q":"β̂₀ = ȳ − β̂₁x̄ means:","options":["The regression line passes through (x̄, ȳ)","The intercept equals the slope","There is no error","X and Y are independent"],"answer":"The regression line passes through (x̄, ȳ)","explanation":"This formula guarantees the fitted line goes through the point of means."},
        {"q":"In matrix notation, the OLS solution is:","options":["β̂ = XY","β̂ = (XᵀX)⁻¹XᵀY","β̂ = X⁻¹Y","β̂ = YᵀX"],"answer":"β̂ = (XᵀX)⁻¹XᵀY","explanation":"This matrix formula generalizes the scalar formulas to handle p predictors simultaneously."},
        {"q":"Linear regression is a _____ method.","options":["Non-parametric","Parametric","Unsupervised","Classification"],"answer":"Parametric","explanation":"It assumes the specific form f(X) = β₀ + β₁X₁ + ... + βₚXₚ — a fixed parametric structure."},
        {"q":"The biggest advantage of linear regression for business is:","options":["Maximum prediction accuracy","High interpretability — each β tells a clear story","It works on categorical Y","It has zero bias"],"answer":"High interpretability — each β tells a clear story","explanation":"Each coefficient directly quantifies the effect of its predictor, making results actionable for stakeholders."},
        {"q":"The biggest limitation of linear regression is:","options":["It's too interpretable","It assumes a linear relationship (high bias if reality is non-linear)","It has too many parameters","It requires unsupervised data"],"answer":"It assumes a linear relationship (high bias if reality is non-linear)","explanation":"If the true f is curved, a straight line will systematically miss the pattern."},
        {"q":"If β₁ = 0, this means:","options":["X has a massive effect on Y","X has no linear association with Y","The model is overfitting","Y is always zero"],"answer":"X has no linear association with Y","explanation":"A zero slope means changes in X produce no expected change in Y (linearly)."},
        {"q":"If β₁ = 200 for House Size (sqft) → Price ($):","options":["Each sqft reduces price by $200","Each sqft increases price by $200","The house is worth $200 total","Size doesn't matter"],"answer":"Each sqft increases price by $200","explanation":"β₁=200 means a 1-unit (1 sqft) increase in X is associated with a $200 increase in Y."},
        {"q":"Zillow switched from linear regression to Random Forests because:","options":["Linear regression was too slow","Linear regression couldn't capture diminishing returns (non-linear effects)","Random Forests are more interpretable","They ran out of data"],"answer":"Linear regression couldn't capture diminishing returns (non-linear effects)","explanation":"The 5th bathroom doesn't add as much value as the 2nd — a non-linear reality that breaks linearity."},
        {"q":"A residual eᵢ is defined as:","options":["yᵢ − ŷᵢ","ŷᵢ − yᵢ","yᵢ + ŷᵢ","β₀ + β₁xᵢ"],"answer":"yᵢ − ŷᵢ","explanation":"The residual is the difference between what we observed and what we predicted."},
        {"q":"If all residuals are zero, the model:","options":["Is optimal","Has perfectly interpolated the training data (likely overfitting)","Has high bias","Is underfitting"],"answer":"Has perfectly interpolated the training data (likely overfitting)","explanation":"Zero residuals means the model passes through every training point — capturing noise."},
        {"q":"R² (R-squared) measures:","options":["The irreducible error","The proportion of variance in Y explained by the model","The number of features","The slope"],"answer":"The proportion of variance in Y explained by the model","explanation":"R² ranges from 0 to 1; higher means the model explains more of Y's variability."},
        {"q":"An R² of 0.95 means:","options":["The model explains 95% of Y's variance","The model is 95% accurate on every prediction","There is 95% irreducible error","5% of features are significant"],"answer":"The model explains 95% of Y's variance","explanation":"95% of the total variation in Y is captured by the regression; 5% remains unexplained."},
        {"q":"Adding more predictors to linear regression will always:","options":["Decrease training RSS","Increase training RSS","Have no effect","Increase test MSE"],"answer":"Decrease training RSS","explanation":"More predictors = more flexibility = training RSS can only stay the same or decrease."},
        {"q":"Adding irrelevant predictors risks:","options":["Improving inference","Overfitting (higher test MSE despite lower training RSS)","Reducing variance","Eliminating ε"],"answer":"Overfitting (higher test MSE despite lower training RSS)","explanation":"Irrelevant predictors add noise to the model, increasing variance without reducing bias."},
        {"q":"Multiple regression extends simple regression by:","options":["Using more training data","Using multiple X predictors simultaneously","Switching to classification","Removing the intercept"],"answer":"Using multiple X predictors simultaneously","explanation":"Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ uses p predictors instead of just one."},
        {"q":"Multicollinearity occurs when:","options":["Predictors are uncorrelated","Two or more predictors are highly correlated with each other","Y has no variance","The model has zero bias"],"answer":"Two or more predictors are highly correlated with each other","explanation":"High correlation between predictors inflates coefficient variance and makes interpretation unreliable."},
        {"q":"The F-statistic in regression tests:","options":["Whether β₀ = 0","Whether at least one predictor has a non-zero coefficient","Whether the data is normal","Whether ε = 0"],"answer":"Whether at least one predictor has a non-zero coefficient","explanation":"H₀: β₁=β₂=...=βₚ=0. A significant F-statistic means at least one predictor matters."},
        {"q":"A p-value < 0.05 for β₁ suggests:","options":["β₁ is exactly zero","There is statistically significant evidence that β₁ ≠ 0","The model is overfitting","ε is large"],"answer":"There is statistically significant evidence that β₁ ≠ 0","explanation":"At the 5% significance level, we reject the null hypothesis that X₁ has no effect on Y."},
        {"q":"Confidence intervals for β₁ represent:","options":["The range where Y always falls","The range of plausible values for the true β₁","The training MSE range","The irreducible error bounds"],"answer":"The range of plausible values for the true β₁","explanation":"A 95% CI means we're 95% confident the true population β₁ lies within this interval."},
        {"q":"RSE (Residual Standard Error) estimates:","options":["The true f","σ — the standard deviation of ε","The number of predictors","The intercept"],"answer":"σ — the standard deviation of ε","explanation":"RSE = √(RSS/(n-p-1)) estimates the typical size of prediction errors."},
        {"q":"In the minigame, minimizing MSE manually is equivalent to:","options":["Random guessing","Performing least squares by hand","Overfitting","Unsupervised clustering"],"answer":"Performing least squares by hand","explanation":"You're searching for the β₀ and β₁ that minimize the average squared error — exactly what OLS does."},
        {"q":"A prediction interval is _____ a confidence interval.","options":["Narrower than","The same width as","Wider than","Unrelated to"],"answer":"Wider than","explanation":"Prediction intervals account for both parameter uncertainty AND irreducible error, making them wider."},
        {"q":"The difference between MSE and RSS is:","options":["They are identical","MSE = RSS/n (MSE is the average)","RSS = MSE²","MSE is for classification only"],"answer":"MSE = RSS/n (MSE is the average)","explanation":"MSE normalizes RSS by the number of observations to give a per-observation error measure."},
        {"q":"Polynomial regression (degree 3) is still technically linear regression because:","options":["It uses a straight line","It's linear in the PARAMETERS β (not in X)","It's non-parametric","It has no intercept"],"answer":"It's linear in the PARAMETERS β (not in X)","explanation":"Y = β₀ + β₁X + β₂X² + β₃X³ is non-linear in X but linear in the β's — still OLS-solvable."},
        {"q":"An interaction term X₁·X₂ captures:","options":["The sum of two effects","The synergy/joint effect of two predictors","The intercept","The residual"],"answer":"The synergy/joint effect of two predictors","explanation":"β₃(X₁·X₂) allows the effect of X₁ on Y to depend on the level of X₂."},
        {"q":"Leverage measures:","options":["How unusual a point's Y value is","How far a point's X values are from the center of the X-space","The model's bias","The irreducible error"],"answer":"How far a point's X values are from the center of the X-space","explanation":"High-leverage points have extreme X values and can disproportionately influence the fitted line."},
        {"q":"An outlier in regression is a point with:","options":["A large residual (unusual Y given X)","A small residual","Perfect prediction","Average X values"],"answer":"A large residual (unusual Y given X)","explanation":"Outliers have Y values far from what the model predicts for their X values."},
        {"q":"Cook's distance combines:","options":["Bias and variance","Leverage and residual magnitude","Training and test MSE","β₀ and β₁"],"answer":"Leverage and residual magnitude","explanation":"Cook's D identifies points that are both high-leverage and have large residuals — maximum influence."},
        {"q":"If Distance_to_City β = −5000, a house 10 miles further costs:","options":["$50,000 more","$50,000 less","$5,000 more","The same"],"answer":"$50,000 less","explanation":"β = −5000 × 10 miles = −$50,000. Negative coefficient + positive distance = price decrease."},
        {"q":"Adjusted R² penalizes for:","options":["Large sample sizes","Adding too many predictors (model complexity)","High R²","Low bias"],"answer":"Adding too many predictors (model complexity)","explanation":"Unlike R², Adjusted R² can decrease when adding useless predictors, penalizing unnecessary complexity."},
        {"q":"VIF (Variance Inflation Factor) detects:","options":["Overfitting","Multicollinearity among predictors","Underfitting","Outliers"],"answer":"Multicollinearity among predictors","explanation":"VIF > 5 or 10 suggests a predictor is highly correlated with others, inflating coefficient variance."},
        {"q":"Standardizing predictors (z-scoring) allows:","options":["Removing ε","Comparing coefficient magnitudes across different-scaled predictors","Eliminating multicollinearity","Perfect prediction"],"answer":"Comparing coefficient magnitudes across different-scaled predictors","explanation":"After standardization, a larger |β| means a stronger effect, regardless of original units."},
        {"q":"Forward selection starts with:","options":["All predictors in the model","No predictors, adding one at a time","Random predictors","Only the intercept and all interactions"],"answer":"No predictors, adding one at a time","explanation":"Forward selection begins with the null model and greedily adds the most significant predictor at each step."},
        {"q":"Backward elimination starts with:","options":["No predictors","All predictors, removing one at a time","Random selection","Only interactions"],"answer":"All predictors, removing one at a time","explanation":"Backward elimination begins with all predictors and removes the least significant one at each step."},
        {"q":"The residual plot should ideally show:","options":["A clear pattern (curved/fanned)","Random scatter with no pattern","All positive residuals","A perfect line"],"answer":"Random scatter with no pattern","explanation":"Patterns in residual plots indicate the model is missing systematic structure (non-linearity, heteroscedasticity)."},
        {"q":"Heteroscedasticity means:","options":["Constant variance of residuals","Non-constant variance of residuals (fan shape)","Zero residuals","Perfect prediction"],"answer":"Non-constant variance of residuals (fan shape)","explanation":"When residual spread changes with X, OLS estimates are still unbiased but standard errors are wrong."},
        {"q":"A dummy variable encodes:","options":["Numerical features as categories","Categorical features as 0/1 numerical values","The intercept","The slope"],"answer":"Categorical features as 0/1 numerical values","explanation":"Dummy variables convert qualitative predictors (e.g., City=A vs B) into binary indicators for regression."},
        {"q":"With k categories, we need _____ dummy variables.","options":["k","k − 1","k + 1","2k"],"answer":"k − 1","explanation":"One category becomes the reference level (captured by β₀); k−1 dummies capture deviations from it."},
        {"q":"The assumption E[ε] = 0 in linear regression means:","options":["There is no noise","On average, the model's errors cancel out","Every prediction is exact","ε is always positive"],"answer":"On average, the model's errors cancel out","explanation":"Individual errors can be positive or negative, but their expected value is zero."},
        {"q":"Linear regression assumes ε values are:","options":["Correlated across observations","Independent and identically distributed (i.i.d.)","Always positive","Equal to β₁"],"answer":"Independent and identically distributed (i.i.d.)","explanation":"The standard assumption: errors are independent of each other and share the same distribution."},
        {"q":"If the true relationship is Y = 5 + 3X², a simple linear regression will:","options":["Perfectly capture the relationship","Underfit due to high bias","Overfit due to high variance","Produce zero MSE"],"answer":"Underfit due to high bias","explanation":"A straight line cannot capture the X² curvature — the rigid assumption creates systematic error (bias)."},
        {"q":"The key takeaway from the Zillow case study is:","options":["Always use neural networks","Linear regression is worthless","Linear regression excels at inference but breaks down on non-linear prediction problems","Non-parametric models are always better"],"answer":"Linear regression excels at inference but breaks down on non-linear prediction problems","explanation":"Zillow used linear regression for interpretable insights but needed flexible models for complex pricing patterns."},
        {"q":"In business, a negative β₁ for 'Support Tickets → LTV' means:","options":["More tickets increase LTV","Each additional ticket is associated with lower customer lifetime value","Tickets have no effect","The model is wrong"],"answer":"Each additional ticket is associated with lower customer lifetime value","explanation":"A negative coefficient means a 1-unit increase in the predictor is associated with a decrease in Y."},
        {"q":"The 'best' linear regression model balances:","options":["Maximum R² regardless of complexity","Goodness of fit with model simplicity (parsimony)","Zero training error","The most predictors possible"],"answer":"Goodness of fit with model simplicity (parsimony)","explanation":"Overly complex models overfit; the best model captures the signal with the fewest necessary predictors."},
    ]

    render_quiz("ch3", ch3_q)
