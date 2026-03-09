import streamlit as st
import joblib

# Cargar modelo y mapa de ODS
modelo = joblib.load("modelo_ods.pkl")
ods_map = joblib.load("ods_map.pkl")

st.title("Clasificación de textos según ODS")

st.write(
    "Ingresa un texto en español y el modelo lo clasificará según el Objetivo de Desarrollo Sostenible (ODS) más relacionado."
)

texto_usuario = st.text_area("Pega aquí el texto a clasificar", height=200)

if st.button("Clasificar texto"):
    if not texto_usuario.strip():
        st.warning("Por favor ingresa un texto antes de clasificar.")
    else:
        # El pipeline ya incluye TfidfVectorizer, SVD y LogisticRegression
        pred = modelo.predict([texto_usuario])[0]
        probas = modelo.predict_proba([texto_usuario])[0]
        prob_ods = max(probas)

        nombre_ods = ods_map.get(int(pred), f"ODS {pred}")

        st.subheader("Resultado de la clasificación")
        st.markdown(f"**ODS predicho:** {nombre_ods} (código {pred})")
        st.markdown(f"**Probabilidad estimada:** {prob_ods:.2%}")
