---
title: Streamlit에서 footer 및 burger 아이콘 제거하는 법
author: oksjjj
date: 2025-04-05 09:30:00 +0900
categories: [Streamlit]
tags: [Streamlit, footer, burger icon, remove]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/streamlit_tip.png
  alt: Streamlit Tip
  src: "https://oksjjj.github.io/streamlit_tip.png"
---

app.py 파일 하단에 아래 코드를 추가하면  
  
streamlit footer와 burger 아이콘을 제거할 수 있음  

```python
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
            
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
```