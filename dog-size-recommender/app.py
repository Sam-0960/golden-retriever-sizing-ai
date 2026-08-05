"""
app.py
------
Streamlit web app. Running `streamlit run app.py` opens this as a local
website in your browser (http://localhost:8501) -- that IS the "website"
deliverable; no separate HTML/Flask build needed.

Flow: image -> predict_breed() -> (age/sex/weight) -> estimate_measurements()
(only for fields the user didn't measure directly) -> recommend_size().
"""

import os
import tempfile

import streamlit as st
from PIL import Image

from src.data_loader import load_growth_reference, load_adult_reference, load_brand_chart, load_growth_curve
from src.predict_breed import predict_breed
from src.measurement_estimator import estimate_measurements
from src.size_recommender import recommend_size
from src.utils import validate_dog_input

st.set_page_config(page_title="Dog Size Recommender", page_icon="🐶", layout="centered")

st.markdown("""
<style>
.stMetric { background-color: rgba(120,180,120,0.08); padding: 10px; border-radius: 8px; }
.small-note { color: #888; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

st.title("🐶 Dog Clothing & Harness Size Recommender")
st.caption("Golden Retriever & Labrador Retriever — sized against the real Supertails chart")

with st.sidebar:
    st.header("1. Photo")
    uploaded_image = st.file_uploader("Upload a clear photo of the dog", type=["jpg", "jpeg", "png"])

    st.header("2. Basic details")
    product_type = st.radio("Looking for", ["clothing", "harness"])
    sex = st.radio("Sex", ["male", "female"])
    age_months = st.number_input("Age (months)", min_value=0, max_value=240, value=18)
    weight_kg = st.number_input("Weight (kg)", min_value=0.1, max_value=80.0, value=28.0, step=0.5)

    st.header("3. Optional: real measurements")
    st.caption("If you have a tape measure, enter these for the most accurate result. Leave at 0 to auto-estimate from breed/age/weight instead.")
    neck_cm_input = st.number_input("Neck girth (cm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    chest_cm_input = st.number_input("Chest girth (cm)", min_value=0.0, max_value=150.0, value=0.0, step=0.5)
    back_cm_input = st.number_input("Back length (cm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)

    run_button = st.button("Get size recommendation", type="primary", use_container_width=True)

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded photo", width=300)
else:
    st.info("👈 Upload a photo and fill in details in the sidebar to get started.")

if run_button:
    if uploaded_image is None:
        st.error("Please upload a photo of the dog first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img = Image.open(uploaded_image).convert("RGB")
            img.save(tmp.name)
            tmp_path = tmp.name

        with st.spinner("Analyzing photo and estimating size..."):
            breed_result = predict_breed(tmp_path)

            if breed_result["is_placeholder"]:
                st.warning(
                    "⚠️ No trained breed classifier found -- using a placeholder 50/50 "
                    "prediction. Train the model: `python -m training.train_breed_classifier`"
                )

            predicted_breed = breed_result["predicted_breed"]
            breed_confidence = breed_result["confidence"]

            if breed_result["needs_manual_confirmation"]:
                st.info(f"Breed confidence is low ({breed_confidence*100:.0f}%). Please confirm manually:")
                predicted_breed = st.selectbox(
                    "Confirm breed", ["Golden Retriever", "Labrador Retriever"],
                    index=["Golden Retriever", "Labrador Retriever"].index(predicted_breed),
                )

            try:
                validate_dog_input(predicted_breed, age_months, weight_kg, sex, product_type)
            except ValueError as e:
                st.error(str(e))
                os.unlink(tmp_path)
                st.stop()

            growth_ref = load_growth_reference()
            growth_curve = load_growth_curve()
            adult_ref = load_adult_reference()
            brand_chart = load_brand_chart()

            user_neck = neck_cm_input if neck_cm_input > 0 else None
            user_chest = chest_cm_input if chest_cm_input > 0 else None
            user_back = back_cm_input if back_cm_input > 0 else None

            measurement_source = {}
            final_neck, final_chest, final_back = user_neck, user_chest, user_back
            weight_implausible, expected_weight = False, None

            need_estimate = user_neck is None or user_chest is None or user_back is None
            if need_estimate:
                est = estimate_measurements(
                    predicted_breed, sex, age_months, weight_kg,
                    adult_ref, growth_curve, growth_ref,
                )
                weight_implausible = est.get("weight_looks_implausible", False)
                expected_weight = est.get("expected_weight_kg")

                if est["insufficient_data"] and user_chest is None:
                    st.error(est["message"])
                    os.unlink(tmp_path)
                    st.stop()

                for key, user_val, est_val in [
                    ("neck", user_neck, est["neck_cm"]),
                    ("chest", user_chest, est["chest_cm"]),
                    ("back", user_back, est["back_length_cm"]),
                ]:
                    if key == "neck":
                        final_neck = user_val if user_val is not None else est_val
                        measurement_source["neck"] = "actual" if user_val is not None else "estimated"
                    elif key == "chest":
                        final_chest = user_val if user_val is not None else est_val
                        measurement_source["chest"] = "actual" if user_val is not None else "estimated"
                    else:
                        final_back = user_val if user_val is not None else est_val
                        measurement_source["back"] = "actual" if user_val is not None else "estimated"

                if est["growth_fraction"] is not None:
                    st.caption(f"📈 Growth stage: ~{est['growth_fraction']*100:.0f}% of adult size at this age (dogs reach full size around 24 months).")
            else:
                measurement_source = {"neck": "actual", "chest": "actual", "back": "actual"}

            if weight_implausible:
                st.warning(
                    f"⚠️ The entered weight ({weight_kg}kg) looks unusual for a {age_months}-month-old "
                    f"{predicted_breed} (expected roughly {expected_weight}kg). This will lower confidence."
                )

            user_input = {
                "breed": predicted_breed, "sex": sex, "age_months": age_months, "weight_kg": weight_kg,
                "product_type": product_type, "breed_confidence": breed_confidence,
                "neck_cm": final_neck, "chest_cm": final_chest, "back_length_cm": final_back,
                "measurement_source": measurement_source,
                "weight_looks_implausible": weight_implausible, "expected_weight_kg": expected_weight,
            }

            result = recommend_size(user_input, brand_chart, growth_ref, adult_ref)

        os.unlink(tmp_path)

        st.subheader("📋 Result")
        conf_color = {"High": "green", "Medium": "orange", "Low": "red"}[result["confidence_label"]]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted breed", predicted_breed)
            st.metric("Breed confidence", f"{breed_confidence*100:.1f}%")
        with col2:
            st.metric("Recommended size", result["recommended_size"] or "N/A")
            st.markdown(f"**Confidence:** :{conf_color}[{result['confidence_score']}/100 — {result['confidence_label']}]")

        st.write("**Measurements used:**")
        m_cols = st.columns(3)
        labels = {"neck": "Neck", "chest": "Chest", "back": "Back length"}
        for i, (key, m) in enumerate(result["measurements_used"].items()):
            with m_cols[i]:
                if m["value"] is not None:
                    icon = "📏" if m["source"] == "actual" else "🧮"
                    st.metric(f"{icon} {labels[key]}", f"{m['value']:.1f} cm", m["source"])
                else:
                    st.metric(labels[key], "N/A")

        st.write(f"**Reason:** {result['reason']}")

        for w in result["warnings"]:
            st.warning(w)

        if result["next_best_size"]:
            st.caption(f"Next best size: {result['next_best_size']}")

        with st.expander("See all size candidates (debug view)"):
            st.write(result["candidates"])

        st.markdown('<p class="small-note">🧮 = estimated from breed/age/weight growth model, not measured. 📏 = you entered this directly.</p>', unsafe_allow_html=True)
