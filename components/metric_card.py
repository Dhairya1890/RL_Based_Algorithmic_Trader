"""
Reusable metric card component with tooltip support.
"""
import streamlit as st

def metric_card(label: str, value: str, info: str, delta: str = None,
                delta_color: str = "normal", good_direction: str = "up"):
    """
    Renders a metric with an ⓘ tooltip and optional delta vs benchmark.
    good_direction: 'up' = green when positive, 'down' = green when negative (for drawdown)
    """
    # Create a unique HTML/CSS block for the metric to show a tooltip
    tooltip_html = f'''
    <div style="display: flex; flex-direction: column; padding: 10px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 0.9rem; color: #a0aab2;">{label}</span>
            <span title="{info}" style="cursor: help; color: #a0aab2;">ⓘ</span>
        </div>
        <div style="font-size: 1.8rem; font-weight: 600; margin-top: 5px; margin-bottom: 5px;">{value}</div>
    '''
    
    if delta:
        # Simple color logic for delta
        is_positive = not delta.startswith("-")
        
        if delta_color == "off":
            color = "#a0aab2"
        elif good_direction == "up":
            color = "#00C853" if is_positive else "#D32F2F"
        else: # good_direction == "down"
            color = "#D32F2F" if is_positive else "#00C853"
            
        arrow = "↑" if is_positive else "↓"
        delta_val = delta.lstrip("-")
        tooltip_html += f'<div style="font-size: 0.9rem; color: {color}; font-weight: 500;">{arrow} {delta_val}</div>'
        
    tooltip_html += '</div>'
    
    st.markdown(tooltip_html, unsafe_allow_html=True)
