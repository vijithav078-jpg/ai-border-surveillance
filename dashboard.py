import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Intrusion Monitoring Dashboard", layout="wide")

st.title("🛡️ Intrusion Monitoring Dashboard")

# Check if file exists
if not os.path.exists("intrusion_log.csv"):
    st.warning("No data available yet. Run detection first.")
else:
    try:
        df = pd.read_csv("intrusion_log.csv")

        if df.empty:
            st.warning("CSV is empty. Run detection first.")
        else:
            st.success("Data Loaded Successfully ✅")

            # Show raw data
            st.subheader("📄 Intrusion Logs")
            st.dataframe(df)

            # Count intrusions
            if "Status" in df.columns:
                intrusion_count = df[df["Status"] == "INTRUSION"].shape[0]
            else:
                intrusion_count = len(df)

            st.metric("🚨 Total Intrusions", intrusion_count)

            # Time-based chart (if Time exists)
            if "Time" in df.columns:
                df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
                df = df.dropna()

                df["Hour"] = df["Time"].dt.hour
                chart = df.groupby("Hour").size()

                st.subheader("📊 Intrusions by Hour")
                st.line_chart(chart)

    except Exception as e:
        st.error("Error reading CSV file")
        st.text(str(e))