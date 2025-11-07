import os

if __name__ == "__main__":
    print("Running hello world script...")
    
    # Check if running in SageMaker or locally
    if os.path.exists("/opt/ml/processing"):
        # Running in SageMaker - use SageMaker output directory
        output_dir = "/opt/ml/processing/output"
    else:
        # Running locally - use current directory
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write hello world to a text file
    output_file = os.path.join(output_dir, "hello_world.txt")
    with open(output_file, "w") as f:
        f.write("Hello World!\n")
    
    print(f"Success! File created at: {output_file}")
