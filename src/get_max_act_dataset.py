
import http.client
import json
import numpy as np
import os

# Initialize connection
conn = http.client.HTTPSConnection("www.neuronpedia.org")
headers = { 'X-Api-Key': "bfbdaf32-118d-41a6-81db-ee8ee2e030ed" }

# Feature array
feats = np.array([33990, 39052, 33206, 35355, 11046, 40337, 12221, 20298, 25443,
       20084,  2863, 45072, 34437, 31080,  9325, 37974,  4142, 41231,
       14632, 28057, 44333, 42403,  3997, 32129, 17363, 25412, 31646,
       32647, 44893, 26684,   321, 37630, 18380,  8515, 32556, 11911,
        9190, 15501, 29084, 21868,  1343, 29012, 24183, 23024, 18912,
        1401, 49576, 48600, 47988, 27386, 48063, 24045, 20557, 48334,
       21726, 38086, 21032, 12151, 18378, 40012, 41420, 28313, 17898,
       45691, 26097, 36999, 47885, 34220,   763,  5879, 48809,  1230,
       36280, 47575, 37439,  8673, 18947,  8209, 46168, 19671, 12485,
       43617, 10802, 28005, 13442, 30510, 40889,  9123, 38418, 10715,
       25262, 12214, 43511, 36937, 15313, 19006, 18115, 43100, 32760,
       27083])

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
        conn.request("GET", f"/api/feature/gpt2-small/5-att-kk/{feature_id}", headers=headers)
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
