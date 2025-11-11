# ===============================
# Automated Log Classifier System
# ===============================

# Step 1. Import Libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 2. Load Dataset
# You can replace this with your own logs dataset
data = {
    'log_message': [
       "System boot completed successfully",
"Error reading configuration file",
"User logged in successfully",
"Low memory warning detected",
"Debugging authentication module",
"Connection timeout after 30 seconds",
"Error while accessing database",
"System shutdown initiated",
"Debugging database connection",
"File not found: settings.ini",
"User profile updated successfully",
"Disk space running low",
"Debugging file upload handler",
"Network error: failed to reach server",
"Backup completed successfully",
"Memory usage exceeded threshold",
"Debugging API response time",
"Invalid credentials provided",
"Service restarted automatically",
"Warning: Unstable network detected",
"Debugging encryption module",
"File permission denied",
"System rebooted after crash",
"Battery level low warning",
"Debugging session timeout",
"Error connecting to remote host",
"System update installed successfully",
"CPU temperature high warning",
"Debugging cache management",
"Access denied for user admin",
"Server started on port 8080",
"Application configuration loaded",
"High latency detected in network",
"Debugging payment gateway API",
"Database query failed",
"System idle for 5 minutes",
"Warning: Deprecated API used",
"Debugging system log parser",
"File upload failed due to timeout",
"Process completed successfully",
"Memory leak detected in process",
"Debugging frontend performance",
"Unable to establish secure connection",
"System restored from backup",
"High CPU usage warning",
"Debugging cache invalidation",
"Error while saving user data",
"Scheduled maintenance completed",
"Insufficient permissions for operation",
"User registration successful",
"Warning: High disk utilization",
"Debugging cloud sync feature",
"Unable to connect to database",
"System diagnostics run completed",
"Low bandwidth warning",
"Debugging websocket communication",
"Error initializing service",
"Login session expired",
"System configuration changed",
"Debugging API request handler",
"Network unreachable",
"Backup scheduled successfully",
"File system corruption detected",
"Debugging SSL certificate validation",
"User session timed out",
"Application started successfully",
"Error retrieving file from server",
"Low battery warning",
"Debugging HTTP response codes",
"Cache cleared successfully",
"Database locked by another process",
"User logged out successfully",
"Warning: Slow response from server",
"Debugging UI rendering",
"Error loading module: missing dependency",
"System initialized successfully",
"Power supply fluctuation warning",
"Debugging authentication token",
"File access error: path not found",
"Software update available",
"Debugging notification service",
"Warning: Application running slowly",
"Error writing to log file",
"User password changed successfully",
"Debugging email sending module",
"Server timeout occurred",
"Database migration failed",
"System booted in safe mode",
"Debugging disk cleanup module",
"Configuration parameter missing",
"Service started successfully",
"Network bandwidth threshold exceeded",
"Debugging I/O performance",
"Error parsing configuration",
"Application closed successfully",
"Debugging backend data sync",
"File size exceeds limit warning",
"Error while processing request",
"System resources optimized",
"Debugging kernel performance"

    ],
    'label': [
      "INFO",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"WARNING",
"ERROR",
"INFO",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"WARNING",
"INFO",
"DEBUG",
"ERROR",
"INFO",
"ERROR",
"DEBUG",
"WARNING",
"INFO",
"ERROR",
"WARNING",
"DEBUG",
"INFO",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"DEBUG",
"ERROR",
"INFO",
"WARNING",
"DEBUG",
"ERROR",
"INFO",
"DEBUG",
"WARNING",
"ERROR",
"INFO",
"DEBUG"
    ]
}

df = pd.DataFrame(data)
print(df.head())

# Step 3. Text Preprocessing
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text

df['cleaned'] = df['log_message'].apply(clean_text)

# Step 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42)

# Step 5. Feature Extraction
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6. Model Training
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Step 7. Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8. Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9. Save Model & Vectorizer
joblib.dump(model, "log_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n Model and vectorizer saved successfully!")