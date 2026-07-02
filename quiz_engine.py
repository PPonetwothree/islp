import streamlit as st

def render_quiz(quiz_id: str, questions: list):
    """
    Renders an interactive quiz with instant grading and explanations.
    
    questions: list of dicts, each with:
        - "q": question text
        - "options": list of 4 strings
        - "answer": the correct option string
        - "explanation": why it's correct
    """
    score = 0
    total = len(questions)
    
    with st.form(f"quiz_{quiz_id}"):
        user_answers = {}
        for i, item in enumerate(questions):
            st.markdown(f"**Q{i+1}. {item['q']}**")
            user_answers[i] = st.radio(
                "Select:", item["options"], key=f"{quiz_id}_{i}", label_visibility="collapsed"
            )
            st.markdown("---")
        
        submitted = st.form_submit_button("SUBMIT ASSESSMENT", type="primary")
    
    if submitted:
        for i, item in enumerate(questions):
            is_correct = user_answers[i] == item["answer"]
            if is_correct:
                score += 1
                st.markdown(f"**Q{i+1}. ✅ CORRECT** — {item['q']}")
            else:
                st.markdown(f"**Q{i+1}. ❌ INCORRECT** — {item['q']}")
                st.markdown(f"&emsp;Your answer: *{user_answers[i]}*")
                st.markdown(f"&emsp;Correct answer: **{item['answer']}**")
            st.caption(f"💡 {item['explanation']}")
            st.markdown("---")
        
        pct = int(score / total * 100)
        if pct >= 90:
            st.success(f"🏆 **Score: {score}/{total} ({pct}%)** — Outstanding.")
            st.balloons()
        elif pct >= 70:
            st.warning(f"📊 **Score: {score}/{total} ({pct}%)** — Solid foundation, review the misses above.")
        else:
            st.error(f"📉 **Score: {score}/{total} ({pct}%)** — Re-read the module content and retry.")
