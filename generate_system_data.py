import psutil
import pandas as pd
import time
from datetime import datetime
import os

filename = "system_data.csv"

# Create the file with headers if it doesn't exist
if not os.path.exists(filename):
    df = pd.DataFrame(columns=["Timestamp", "CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)", "Running Processes"])
    df.to_csv(filename, index=False)

print("Logging system performance data... (Press Ctrl+C to stop)")

try:
    while True:
        # Collect live system stats
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        processes = len(psutil.pids())

        # Append to CSV
        data = pd.DataFrame([[timestamp, cpu, memory, disk, processes]],
                            columns=["Timestamp", "CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)", "Running Processes"])
        data.to_csv(filename, mode='a', header=False, index=False)

        time.sleep(1)  # Delay for 1 second
except KeyboardInterrupt:
    print("\nStopped logging.")
