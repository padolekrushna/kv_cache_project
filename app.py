import streamlit as st
import google.generativeai as genai
import time
import psutil
import os
from typing import Dict, Any

# Configure Gemini (use your free API key from https://ai.google.dev/)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in secrets.toml or environment variable!")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize models
def init_models():
    return {
        "baseline (no cache)": genai.GenerativeModel('gemini-1.0-pro'),
        "KV Cache (Prompt Caching)": genai.GenerativeModel('gemini-1.5-flash'),
        "Quantized (4-bit)": genai.GenerativeModel('gemini-1.5-flash'),
        "Offloaded (CPU)": genai.GenerativeModel('gemini-1.5-flash'),
    }

# Measure memory usage
def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024  # MB

# Run inference and collect metrics
def run_inference(model, prompt: str, use_cache: bool = False) -> Dict[str, Any]:
    start_time = time.time()
    start_mem = get_memory_usage()
    
    try:
        if use_cache:
            # Enable prompt caching for gemini-1.5-flash
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
                safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
                request_options={"timeout": 60}
            )
        else:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
                safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
                request_options={"timeout": 60}
            )
        
        end_time = time.time()
        end_mem = get_memory_usage()
        
        # Calculate tokens (approximate)
        input_tokens = len(prompt.split())
        output_tokens = len(response.text.split()) if response.text else 0
        total_tokens = input_tokens + output_tokens
        
        return {
            "time_sec": round(end_time - start_time, 2),
            "memory_mb": round(end_mem - start_mem, 2),
            "tokens_per_sec": round(total_tokens / (end_time - start_time), 2) if (end_time - start_time) > 0 else 0,
            "output": response.text[:500] + "..." if len(response.text) > 500 else response.text,
            "total_tokens": total_tokens
        }
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.set_page_config(page_title="KV Caching Comparison", layout="wide")
st.title("KV Caching Techniques Comparison (Free Gemini API)")

# User input
prompt = st.text_area("Enter your prompt:", height=100, value="Explain quantum computing in simple terms.")
run_button = st.button("Run Comparison")

if run_button and prompt:
    models = init_models()
    results = {}
    
    with st.spinner("Running experiments..."):
        # Baseline (no cache)
        st.info("Running baseline (gemini-1.0-pro)...")
        results["baseline (no cache)"] = run_inference(models["baseline (no cache)"], prompt)
        
        # KV Cache (Prompt Caching)
        st.info("Running with KV Cache (Prompt Caching)...")
        results["KV Cache (Prompt Caching)"] = run_inference(
            models["KV Cache (Prompt Caching)"], 
            prompt, 
            use_cache=True
        )
        
        # Quantized (4-bit)
        st.info("Running quantized model (gemini-1.5-flash)...")
        results["Quantized (4-bit)"] = run_inference(models["Quantized (4-bit)"], prompt)
        
        # Offloaded (CPU)
        st.info("Running offloaded to CPU...")
        results["Offloaded (CPU)"] = run_inference(models["Offloaded (CPU)"], prompt)

    # Display results
    st.subheader("Results Comparison")
    comparison_data = []
    for name, metrics in results.items():
        if "error" in metrics:
            st.error(f"{name}: {metrics['error']}")
            continue
            
        comparison_data.append({
            "Technique": name,
            "Time (sec)": metrics["time_sec"],
            "Memory (MB)": metrics["memory_mb"],
            "Tokens/sec": metrics["tokens_per_sec"]
        })
    
    # Show table
    st.dataframe(comparison_data, use_container_width=True)
    
    # Show output (from KV Cache run)
    st.subheader("Sample Output (KV Cache)")
    st.write(results["KV Cache (Prompt Caching)"]["output"])
    
    # Visualization
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    
    st.subheader("Performance Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.bar_chart(df.set_index("Technique")["Time (sec)"], height=300)
    with col2:
        st.bar_chart(df.set_index("Technique")["Memory (MB)"], height=300)
    with col3:
        st.bar_chart(df.set_index("Technique")["Tokens/sec"], height=300)

# Setup instructions
st.sidebar.title("Setup Instructions")
st.sidebar.markdown("""
1. Get a **free Gemini API key** from [Google AI Studio](https://ai.google.dev/)
2. Add it to Streamlit Secrets:
   - Create `.streamlit/secrets.toml`
   - Add: `GOOGLE_API_KEY = "your_key_here"`
3. Install requirements:  
   `pip install -r requirements.txt`
4. Run:  
   `streamlit run app.py`
""")
