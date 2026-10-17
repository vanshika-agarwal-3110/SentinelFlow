import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ultralytics import YOLO

from core import COLUMNS, ZONES, overall, process_frame
from analyze import (
    prepare_data,
    compute_zone_stats,
    generate_alerts,
    generate_visualizations,
    generate_report,
)

st.set_page_config(
    page_title="SentinelFlow | Crowd Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SentinelFlow | Crowd Monitoring")
st.caption(
    "YOLO person detection · Three-zone monitoring · "
    "Prototype crowd risk assessment"
)

st.info(
    "Counts represent YOLO detections, not guaranteed people counts. "
    "Risk thresholds are configurable demonstration rules, "
    "not validated crowd-safety limits."
)



@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(weights):
    return YOLO(weights)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "records": [],
    "last_frame": None,
    "run_state": "idle",
    "frame_number": 0,
    "source_frames": 0,
    "capture": None,
    "temp_path": None,
    "run_settings": {},
    "last_counts": [0, 0, 0],
    "last_statuses": ["SAFE", "SAFE", "SAFE"],
    "finish_reason": "",
    "run_error": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def release_capture():
    """Release capture and temporary uploaded video."""
    cap = st.session_state.capture

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    st.session_state.capture = None

    temp_path = st.session_state.temp_path

    if temp_path:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

    st.session_state.temp_path = None


def finish_run(reason):
    """Finish without deleting collected detection data."""
    release_capture()

    st.session_state.run_state = "finished"
    st.session_state.finish_reason = reason


def reset_run():
    """Reset the current run without reloading the YOLO model."""

    release_capture()

    st.session_state.records = []
    st.session_state.last_frame = None
    st.session_state.run_state = "idle"
    st.session_state.frame_number = 0
    st.session_state.source_frames = 0
    st.session_state.run_settings = {}
    st.session_state.last_counts = [0, 0, 0]
    st.session_state.last_statuses = ["SAFE", "SAFE", "SAFE"]
    st.session_state.finish_reason = ""
    st.session_state.run_error = ""

# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------

with st.sidebar:

    st.header("System Configuration")

    source_kind = st.radio(
        "Input source",
        [
            "Benchmark video",
            "Upload video",
            "Local server webcam",
        ],
    )

    uploaded = None

    if source_kind == "Upload video":
        uploaded = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov"],
        )

    if source_kind == "Local server webcam":
        st.warning(
            "Webcam mode accesses the machine running "
            "Streamlit, not the visitor's browser webcam "
            "on cloud hosting."
        )

    weights = st.selectbox(
        "YOLO weights",
        ["yolov8n.pt", "yolov8s.pt"],
        index=0,
    )

    confidence = st.slider(
        "Detection confidence",
        0.10,
        0.90,
        0.25,
        0.05,
    )

    safe_max = st.number_input(
        "SAFE maximum (people per zone)",
        min_value=0,
        value=9,
        step=1,
    )

    moderate_max = st.number_input(
        "MODERATE maximum",
        min_value=1,
        value=19,
        step=1,
    )

    max_frames = st.number_input(
        "Maximum frames per run",
        min_value=1,
        max_value=10000,
        value=150,
        step=50,
    )

    frame_stride = st.number_input(
        "Process every Nth source frame",
        min_value=1,
        max_value=30,
        value=1,
    )

    show_conf = st.checkbox(
        "Show confidence labels",
        value=True,
    )

    use_ai = st.checkbox(
        "Optional OpenAI narrative",
        value=False,
    )

    st.caption(
        "Settings are captured when you start a run. "
        "Changing them during detection will not alter "
        "the active run."
    )


# --------------------------------------------------
# START DETECTION
# --------------------------------------------------

def start_detection():
    if moderate_max <= safe_max:
        st.session_state.run_error = (
            "MODERATE maximum must be greater "
            "than SAFE maximum."
        )
        return

    # Clear previous results only when starting a new run.
    reset_run()

    temp_path = None
    cap = None

    try:
        if source_kind == "Benchmark video":

            source = (
                "background_video_people_walking_720p.mp4"
            )

            if not Path(source).is_file():
                raise FileNotFoundError(
                    f"Benchmark video not found: {source}"
                )

        elif source_kind == "Upload video":

            if uploaded is None:
                raise ValueError(
                    "Upload a video before starting."
                )

            suffix = Path(uploaded.name).suffix.lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp:
                temp.write(uploaded.getvalue())
                temp_path = temp.name

            source = temp_path

        else:
            source = 0

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(
                "Unable to open the selected video source."
            )

        # Load model before marking the run active.
        load_model(weights)

        st.session_state.capture = cap
        st.session_state.temp_path = temp_path

        st.session_state.run_settings = {
            "weights": weights,
            "confidence": confidence,
            "safe_max": int(safe_max),
            "moderate_max": int(moderate_max),
            "max_frames": int(max_frames),
            "frame_stride": int(frame_stride),
            "show_conf": show_conf,
        }

        st.session_state.run_state = "running"
        st.session_state.run_error = ""

    except Exception as exc:

        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        st.session_state.run_state = "idle"
        st.session_state.run_error = str(exc)


# --------------------------------------------------
# DETECTION PROCESSING
# --------------------------------------------------

def process_next_frame():
    """Process one sampled frame per fragment update."""

    if st.session_state.run_state != "running":
        return

    settings = st.session_state.run_settings
    cap = st.session_state.capture

    if cap is None:
        finish_run("Video capture is unavailable.")
        return

    try:
        # Skip source frames according to the stride.
        while True:
            success, frame = cap.read()

            if not success:
                finish_run("Video ended or camera stopped.")
                return

            st.session_state.source_frames += 1

            source_index = (
                st.session_state.source_frames - 1
            )

            if source_index % settings["frame_stride"] == 0:
                break

        model = load_model(settings["weights"])

        annotated, rows, counts, statuses = process_frame(
            frame,
            model,
            settings["confidence"],
            st.session_state.frame_number,
            settings["safe_max"],
            settings["moderate_max"],
            settings["show_conf"],
        )

        # Save records after EVERY completed frame.
        st.session_state.records.extend(rows)

        st.session_state.frame_number += 1

        st.session_state.last_counts = counts
        st.session_state.last_statuses = statuses

        st.session_state.last_frame = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

        if (
            st.session_state.frame_number
            >= settings["max_frames"]
        ):
            finish_run(
                "Maximum configured frame limit reached."
            )

    except Exception as exc:

        st.session_state.run_error = str(exc)

        finish_run(
            "Processing stopped because of an error."
        )


# --------------------------------------------------
# LIVE CONTROL PANEL
# --------------------------------------------------

st.subheader("Live Detection Feed")

# Fragment reruns allow buttons to respond between frames.
# One frame is processed per automatic refresh.

@st.fragment(run_every="0.5s")
def detection_panel():

    state = st.session_state.run_state

    if state in ("idle", "finished"):

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "▶ Start Detection",
                type="primary",
                use_container_width=True,
            ):
                start_detection()
                st.rerun()

        with c2:
            if st.button(
                "↺ Reset",
                use_container_width=True,
            ):
                reset_run()
                st.rerun()

    elif state == "running":

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button(
                "⏸ Pause",
                use_container_width=True,
            ):
                st.session_state.run_state = "paused"
                st.rerun()

        with c2:
            if st.button(
                "■ Stop & Generate Report",
                type="primary",
                use_container_width=True,
            ):
                finish_run("Stopped manually by user.")
                st.rerun()

        with c3:
            st.button(
                "▶ Processing...",
                disabled=True,
                use_container_width=True,
            )

    elif state == "paused":

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "▶ Resume",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.run_state = "running"
                st.rerun()

        with c2:
            if st.button(
                "■ Stop & Generate Report",
                use_container_width=True,
            ):
                finish_run("Stopped manually while paused.")
                st.rerun()

    # Process one frame per fragment update.
    if st.session_state.run_state == "running":
        process_next_frame()

        if st.session_state.run_state == "finished":
            st.rerun()

    # Show last annotated frame.
    if st.session_state.last_frame is not None:

        st.image(
            st.session_state.last_frame,
            channels="RGB",
            use_container_width=True,
        )

    # Current metrics.
    if st.session_state.frame_number > 0:

        counts = st.session_state.last_counts
        statuses = st.session_state.last_statuses

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Left Area", counts[0])
        m2.metric("Central Area", counts[1])
        m3.metric("Right Area", counts[2])
        m4.metric("Total Detected", sum(counts))

        current_risk = overall(statuses)

        if current_risk == "DANGER":

            st.error(
                "CURRENT FRAME: DANGER — "
                "one or more zones exceeded "
                "the configured prototype threshold."
            )

        elif current_risk == "MODERATE":

            st.warning(
                "CURRENT FRAME: MODERATE — "
                "caution threshold reached."
            )

        else:

            st.success(
                "CURRENT FRAME: SAFE under "
                "the configured prototype rules."
            )

        st.caption(
            " | ".join(
                f"{zone}: {status}"
                for zone, status in zip(ZONES, statuses)
            )
        )

    # Processing progress.
    processed = st.session_state.frame_number

    settings = st.session_state.run_settings

    if settings:

        limit = settings["max_frames"]

        st.progress(
            min(processed / limit, 1.0)
        )

        st.caption(
            f"Processed {processed} / {limit} frames "
            f"(read {st.session_state.source_frames} "
            "source frames)"
        )

    # Run state information.
    state = st.session_state.run_state

    if state == "running":
        st.info("Detection is running.")

    elif state == "paused":
        st.warning(
            f"Detection paused at frame {processed}. "
            "Click Resume or Stop & Generate Report."
        )

    elif state == "finished":

        st.success(
            f"Detection finished after {processed} frames. "
            f"{st.session_state.finish_reason}"
        )

        if processed:
            st.info(
                "Analytics and the executive report "
                "are available below."
            )

    if st.session_state.run_error:
        st.error(st.session_state.run_error)


detection_panel()


# --------------------------------------------------
# HISTORICAL ANALYTICS
# --------------------------------------------------

st.divider()

st.subheader("Historical Analytics — Current Run Only")

state = st.session_state.run_state

if state in ("running", "paused"):

    st.info(
        "Analytics and the final report will appear "
        "when you click Stop & Generate Report, "
        "or when processing finishes automatically."
    )

elif not st.session_state.records:

    st.info(
        "Start detection to collect crowd monitoring data."
    )

else:

    try:

        df = prepare_data(
            pd.DataFrame(
                st.session_state.records,
                columns=COLUMNS,
            )
        )

        if df.empty:
            st.warning("No valid detection data.")
            st.stop()

        latest_frame = df["Frame"].max()

        latest = df[
            df["Frame"] == latest_frame
        ]

        totals = df.groupby("Frame")["Count"].sum()

        current_risk = overall(
            latest["Status"].tolist()
        )

        # ------------------------------------------
        # SUMMARY METRICS
        # ------------------------------------------

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Latest Total",
            int(latest["Count"].sum()),
        )

        c2.metric(
            "Peak Frame Total",
            int(totals.max()),
        )

        c3.metric(
            "Frames Processed",
            int(df["Frame"].nunique()),
        )

        c4.metric(
            "DANGER Zone-Frames",
            int((df["Status"] == "DANGER").sum()),
        )

        c5.metric(
            "Current Risk",
            current_risk,
        )

        st.caption(
            "Counts represent frame-level detections. "
            "They are not unique people counted "
            "across the entire video."
        )

        # ------------------------------------------
        # CURRENT FRAME WARNINGS
        # ------------------------------------------

        st.subheader("Current-Frame Warnings")

        if current_risk == "DANGER":

            danger_zones = [
                f"{row.Zone} ({int(row.Count)})"
                for _, row in latest.iterrows()
                if row.Status == "DANGER"
            ]

            st.error(
                "DANGER: " + ", ".join(danger_zones)
            )

        elif current_risk == "MODERATE":

            moderate_zones = [
                f"{row.Zone} ({int(row.Count)})"
                for _, row in latest.iterrows()
                if row.Status == "MODERATE"
            ]

            st.warning(
                "MODERATE: "
                + ", ".join(moderate_zones)
            )

        else:

            st.success(
                "All zones are SAFE under "
                "the configured prototype rules."
            )

        # ------------------------------------------
        # HISTORICAL WARNINGS
        # ------------------------------------------

        st.subheader(
            "Historical and Statistical Warnings"
        )

        alerts = generate_alerts(df)

        historical_alerts = [
            alert
            for alert in alerts
            if alert.startswith("PERSISTENT")
            or alert.startswith("STATISTICAL")
        ]

        if historical_alerts:

            for alert in historical_alerts:
                st.warning(alert)

        else:

            st.success(
                "No persistent-risk or statistical "
                "anomaly warnings identified."
            )

        # ------------------------------------------
        # VISUALIZATIONS
        # ------------------------------------------

        st.subheader("Crowd Analytics")

        figures = generate_visualizations(df)

        try:

            chart_titles = [
                ("counts", "Crowd Counts Over Frames"),
                ("density", "Density Trends"),
                (
                    "distribution",
                    "Average Crowd Distribution",
                ),
            ]

            for key, title in chart_titles:

                st.markdown(f"**{title}**")

                st.pyplot(
                    figures[key],
                    clear_figure=False,
                )

        finally:

            for fig in figures.values():
                plt.close(fig)

        st.caption(
            "Density is calculated as detected persons "
            "divided by zone pixel area, not persons "
            "per square meter."
        )

        # ------------------------------------------
        # ZONE STATISTICS
        # ------------------------------------------

        st.subheader("Zone Statistics")

        stats = compute_zone_stats(df)

        stats_df = pd.DataFrame.from_dict(
            stats,
            orient="index",
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
        )

        # ------------------------------------------
        # OPTIONAL OPENAI CONFIGURATION
        # ------------------------------------------

        if use_ai:

            try:
                if (
                    "OPENAI_API_KEY" not in os.environ
                    and "OPENAI_API_KEY" in st.secrets
                ):
                    os.environ["OPENAI_API_KEY"] = (
                        st.secrets["OPENAI_API_KEY"]
                    )
            except Exception:
                pass

        # ------------------------------------------
        # EXECUTIVE REPORT
        # ------------------------------------------

        st.subheader("Executive Report")

        report = generate_report(
            df,
            use_ai=use_ai,
        )

        st.text_area(
            "Report Preview",
            report,
            height=350,
        )

        # ------------------------------------------
        # DOWNLOAD BUTTONS
        # ------------------------------------------

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "Download crowd_data.csv",
                data=df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="crowd_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with d2:

            st.download_button(
                "Download reports.txt",
                data=report.encode("utf-8"),
                file_name="reports.txt",
                mime="text/plain",
                use_container_width=True,
            )

    except Exception as exc:

        st.error(
            f"Analytics error: {exc}"
        )
