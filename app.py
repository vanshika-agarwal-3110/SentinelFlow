import streamlit as st
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt

# Set dark theme friendly layout configuration
st.set_page_config(
    page_title="SentinelFlow Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling to add card blocks and clean font layouts
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2430;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .control-card {
        background-color: #1a1c24;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #2d313f;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦺 SentinelFlow: Crowd Density Monitoring Suite")
st.caption("Real-Time Spatial Deep Learning & Behavioral Analytics Pipeline")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("🕹️ System Configuration")
source_type = st.sidebar.radio(
    "Select Video Input Channel:",
    ["Project Benchmark Video", "Upload Custom Video stream", "Live Hardware Webcam Feed"]
)

video_to_run = "background_video_people_walking_720p.mp4"
if source_type == "Upload Custom Video stream":
    uploaded_file = st.sidebar.file_uploader("Upload Target Video (MP4/AVI)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        video_to_run = f"uploaded_{uploaded_file.name}"
        with open(video_to_run, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success("✅ Buffer stream synced.")
elif source_type == "Live Hardware Webcam Feed":
    video_to_run = "0"

# --- Main Dashboard ---
col_run, col_results = st.columns([1, 2])

with col_run:
    st.subheader("🚀 Operational Controls")
    
    # Wrap tracking panel in a card structure
    st.markdown(f"""
        <div class="control-card">
            <h4>Active Target Pipeline</h4>
            <p style='color:#a3a8b4;'>System is currently mapped to route operations through the <b>{source_type}</b> node.</p>
        </div>
        <br>
    """, unsafe_allow_html = True)
    
    if st.button("▶️ Launch Core Detection Engine", type="primary", use_container_width=True):
        with st.spinner("Spawning YOLOv8 instance framework... (Press ESC in view matrix window to kill process)"):
            os.environ["SENTINEL_SOURCE"] = video_to_run
            subprocess.run(["python", "main.py"], capture_output=True, text=True)
            st.toast("Core session logging pipeline completed.", icon="✅")

with col_results:
    st.subheader("📊 Dynamic Analytics Control Deck")
    
    if os.path.exists("crowd_data.csv") and os.path.getsize("crowd_data.csv") > 50:
        df = pd.read_csv("crowd_data.csv")
        
        # Calculate key metric states
        latest_frames = df.tail(3)
        total_current_count = int(latest_frames["Count"].sum())
        max_seen = int(df["Count"].max())
        critical_alerts = int(len(df[df["Status"] == "DANGER"]))

        # Beautifully formatted metric blocks
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><p style="color:#8892b0;margin:0;">Active Crowd Volume</p><h2 style="margin:0;color:#fff;">{total_current_count}</h2></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card" style="border-left-color:#ff4b4b;"><p style="color:#8892b0;margin:0;">System Peak Volume</p><h2 style="margin:0;color:#fff;">{max_seen}</h2></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card" style="border-left-color:#ffb86c;"><p style="color:#8892b0;margin:0;">Risk Threshold Breaches</p><h2 style="margin:0;color:#fff;">{critical_alerts}</h2></div>', unsafe_allow_html=True)

        st.markdown("<br>### 📈 Real-Time Spatial Spatial Trends", unsafe_allow_html=True)
        
        # Upgraded Dark-Mode Integrated Matplotlib Chart
        plt.style.use('dark_background')  # Adapts instantly to the web wrapper!
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        # Use cleaner color palettes
        zone_colors = {"Left Area": "#5dade2", "Central Area": "#ec7063", "Right Area": "#58d68d"}
        
        for zone in df["Zone"].unique():
            zone_data = df[df["Zone"] == zone].tail(40)
            ax.plot(
                range(len(zone_data)), 
                zone_data["Count"], 
                label=zone, 
                color=zone_colors.get(zone, "#ffffff"), 
                linewidth=2,
                marker='o',
                markersize=4
            )
        
        ax.set_ylabel("Personnel Metrics", color="#8892b0", fontsize=10)
        ax.set_xlabel("Time Progression Series (Frames Indexed)", color="#8892b0", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Transparent background matching the dashboard background flawlessly
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        ax.legend(facecolor='#1a1c24', edgecolor='none')
        st.pyplot(fig)
        plt.close(fig)
        
        # Summary Area Split
        st.markdown("---")
        if st.button("📝 Compile Narrative Executive AI Summary Report", use_container_width=True):
            with st.spinner("Processing deep safety heuristics reports..."):
                subprocess.run(["python", "analyze.py"])
                if os.path.exists("reports.txt"):
                    with open("reports.txt", "r") as f:
                        st.text_area("Generated System Narrative Output", f.read(), height=250)
    else:
        st.info("📊 Run the analytical camera tracking matrix from the panel controls to generate visual trend layouts.")