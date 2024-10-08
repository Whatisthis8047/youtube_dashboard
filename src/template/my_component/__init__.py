import streamlit.components.v1 as components
import streamlit as st

_my_component = components.declare_component(
    "my_component",
    url = "http://localhost:3001"
)

return_value = _my_component(name='Streamlit', greeting='Hey!!!!!')
st.write(return_value)