
import http.client
import json
import numpy as np
import os

# Initialize connection
conn = http.client.HTTPSConnection("www.neuronpedia.org")
headers = { 'X-Api-Key': "bfbdaf32-118d-41a6-81db-ee8ee2e030ed" }

# Feature array


# Generate a random sample of 1000 numbers from 0 to 24000 without replacement
np.random.seed(42)
feats = np.random.choice(np.arange(24001), size=1000, replace=False)


# Create dataset folder if it doesn't exist
dataset_dir = "../dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Logging file to track failed requests
log_file = os.path.join(dataset_dir, "failed_requests.log")

# Check if a log file exists, otherwise create it
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        pass  # Create an empty log file

# Get the list of already processed features (to skip if already downloaded)
processed_features = set(os.listdir(dataset_dir))

# Function to make a request and save the response
def save_feature_data(feature_id):
    try:
        # Skip if file already exists
        file_name = f"{feature_id}.json"
        if file_name in processed_features:
            print(f"Feature {feature_id} already processed. Skipping...")
            return
        # Make request
        conn.request("GET", f"/api/feature/gpt2-small/5-tres-dc/{feature_id}", headers=headers)
        res = conn.getresponse()
        
        # Read response data
        data = res.read()
        
        # Check if response is valid
        if res.status != 200:
            raise Exception(f"Request failed with status code {res.status} for feature {feature_id}")
        
        # Parse response data
        data_dict = json.loads(data.decode("utf-8"))
        
        # Save data to a JSON file
        with open(os.path.join(dataset_dir, file_name), "w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        
        print(f"Feature {feature_id} saved successfully.")
    
    except Exception as e:
        # Log the failure in the log file
        with open(log_file, "a") as log:
            log.write(f"Failed to retrieve feature {feature_id}: {str(e)}\n")
        print(f"Failed to retrieve feature {feature_id}. Logged error.")

# Loop over all feature IDs and request data
for feature_id in feats:
    save_feature_data(feature_id)

# Close the connection
conn.close()
