import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import psutil
import os

# 1. Load and Preprocess the Dataset
def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    print(f"Data Loaded: {data.shape}")
    
    # Inspecting data
    print(data.head())

    # Separate features and labels
    X = data.drop("label", axis=1)  # Assuming 'label' is the target column
    y = data["label"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# 2. Train the Model
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# 3. Evaluate the Model
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# 4. Collect System Metrics (Simulate real-time system metrics)
def collect_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_read = psutil.disk_io_counters().read_bytes
    disk_write = psutil.disk_io_counters().write_bytes
    network_recv = psutil.net_io_counters().bytes_recv
    network_sent = psutil.net_io_counters().bytes_sent

    return np.array([cpu_usage, memory_usage, disk_read, disk_write, network_recv, network_sent]).reshape(1, -1)

# 5. Trigger Isolation and Backup
def trigger_isolation():
    print("Ransomware detected! Isolating the system...")
    os.system("sudo ifconfig eth0 down")  # Disables network (Adjust for your system)
    os.system("killall -9 suspicious_process_name")  # Adjust to terminate suspicious processes

def backup_important_files():
    print("Backing up important files...")
    os.system("rsync -a /important_files /backup_location")  # Backup command, adjust paths

# 6. Real-Time Monitoring and Detection
def monitor_and_detect(clf):
    while True:
        # Collect system metrics
        metrics = collect_system_metrics()

        # Predict using the trained model
        prediction = clf.predict(metrics)
        if prediction[0] == 1:  # Ransomware-like behavior detected
            trigger_isolation()  # Isolate system
            backup_important_files()  # Backup critical files
        else:
            print("System is secure.")
        
        time.sleep(1)  # Sleep for 1 second before checking again

# 7. Main Function to Execute
def main():
    data_path = "path_to_your_ransomware_detection_dataset.csv"  # Replace with your actual dataset path
    
    # Step 1: Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Step 2: Train the Random Forest model
    clf = train_random_forest(X_train, y_train)
    
    # Step 3: Evaluate the model
    evaluate_model(clf, X_test, y_test)
    
    # Step 4: Monitor system and detect ransomware in real-time
    print("Starting real-time ransomware detection...")
    monitor_and_detect(clf)

if __name__ == "__main__":
    main()

#Replace data_path with the actual file path of the Kaggle dataset ("path_to_your_ransomware_detection_dataset.csv").