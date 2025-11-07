import os
import json

if __name__ == "__main__":
    print("Hello from the first processing step!")
    
    # Check if running in SageMaker or locally
    if os.path.exists("/opt/ml/processing"):
        # Running in SageMaker
        output_dir = "/opt/ml/processing/output"
        print("Running in SageMaker environment")
    else:
        # Running locally
        output_dir = os.path.dirname(os.path.abspath(__file__))
        print("Running in local environment")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/hello.txt"
    with open(output_file, "w") as f:
        f.write("Hello World from Step 1!\n")
    
    print(f"First step completed successfully!")
    print(f"Output written to: {output_file}")
