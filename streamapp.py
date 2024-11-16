import streamlit as st
st.title('Hola, streamlit')
st.write('esta es mi primera aplicacion iterectiva')

nombre = st.text_input('Cual es tu nombre')

if nombre:
    st.write(f('Hola, papi {nombre}'))