"""
Reusable headline card for the sentiment feed.
"""
import streamlit as st

def headline_card(text: str, score: float, magnitude: float, source: str, category: str):
    """
    Renders a news headline with sentiment badges and magnitude bars.
    """
    if score > 0.2:
        badge_color = "#00C853"
        icon = "<div style='display:inline-block;width:10px;height:10px;border-radius:50%;background-color:#00C853;margin-right:5px;'></div>"
    elif score < -0.2:
        badge_color = "#D32F2F"
        icon = "<div style='display:inline-block;width:10px;height:10px;border-radius:50%;background-color:#D32F2F;margin-right:5px;'></div>"
    else:
        badge_color = "#F9A825"
        icon = "<div style='display:inline-block;width:10px;height:10px;border-radius:50%;background-color:#F9A825;margin-right:5px;'></div>"
        
    mag_pct = min(100, max(0, int(magnitude * 100)))
    
    html = f'''
    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid {badge_color};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div style="font-size: 1.1rem; font-weight: 500; color: #E0E0E0; max-width: 80%;">{text}</div>
            <div style="background-color: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 4px; font-weight: bold; color: {badge_color}; display: flex; align-items: center;">
                {icon} {score:+.2f}
            </div>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.85rem; color: #a0aab2; margin-bottom: 8px;">
            <div><span style="border: 1px solid #555; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">{source}</span> <span style="background-color: #333; padding: 2px 6px; border-radius: 3px;">{category}</span></div>
            <div>Magnitude: {magnitude:.2f}</div>
        </div>
        <div style="width: 100%; background-color: #333; height: 4px; border-radius: 2px;">
            <div style="width: {mag_pct}%; background-color: {badge_color}; height: 100%; border-radius: 2px;"></div>
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)
