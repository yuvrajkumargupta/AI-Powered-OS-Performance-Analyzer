import subprocess
import time
import os

def run_all():
    print("🔍 Collecting System Data...")
    monitor_process = subprocess.Popen(["python", "monitor.py"])

    time.sleep(5)  # Allow time for monitor.py to start collecting data

    print("🤖 Running AI Predictions...")
    ai_process = subprocess.Popen(["python", "ai_analysis.py"])

    time.sleep(2)  # Optional: Wait a bit before launching dashboard

    print("📊 Launching Streamlit Dashboard...")
    dashboard_process = subprocess.Popen(["streamlit", "run", "dashboard.py"])

    return monitor_process, ai_process, dashboard_process

if __name__ == "__main__":
    try:
        monitor, ai, dash = run_all()
        print("\n✨ All modules launched successfully!\nPress Ctrl+C to terminate.")
        monitor.wait()
        ai.wait()
        dash.wait()
    except KeyboardInterrupt:
        print("\n🛑 Terminating all processes...")
        monitor.terminate()
        ai.terminate()
        dash.terminate()
        print("✔️ All processes terminated.")
