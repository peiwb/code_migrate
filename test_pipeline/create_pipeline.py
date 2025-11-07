import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# Step 3.1: Initialize SageMaker
print("Initializing SageMaker session...")
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()  # Your IAM role
bucket = sagemaker_session.default_bucket()

print(f"✓ Using S3 bucket: {bucket}")
print(f"✓ Using IAM role: {role}")
print(f"✓ Region: {region}")

# Step 3.2: Upload the script to S3
print("\nUploading script to S3...")
s3_client = boto3.client('s3')
s3_client.upload_file(
    'hello_world.py',  # Local file
    bucket,  # S3 bucket
    'pipeline-scripts/hello_world.py'  # S3 key
)
print(f"✓ Script uploaded to s3://{bucket}/pipeline-scripts/hello_world.py")

# Step 3.3: Create a processor
print("\nCreating processor...")
sklearn_processor = SKLearnProcessor(
    framework_version="1.0-1",
    role=role,
    instance_type="ml.t3.medium",  # Small instance for testing
    instance_count=1,
    sagemaker_session=sagemaker_session
)
print("✓ Processor created")

# Step 3.4: Create a processing step
print("\nCreating processing step...")
step_hello_world = ProcessingStep(
    name="HelloWorldStep",
    processor=sklearn_processor,
    code=f"s3://{bucket}/pipeline-scripts/hello_world.py",
    outputs=[
        ProcessingOutput(
            output_name="hello_output",
            source="/opt/ml/processing/output",  # Where script writes in SageMaker
            destination=f"s3://{bucket}/pipeline-output/hello-world"  # Where to save in S3
        )
    ]
)
print("✓ Processing step created")

# Step 3.5: Create the pipeline
print("\nCreating pipeline...")
pipeline_name = "HelloWorldPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    steps=[step_hello_world],
    sagemaker_session=sagemaker_session
)
print(f"✓ Pipeline '{pipeline_name}' created")

# Step 3.6: Deploy the pipeline to SageMaker
print("\nDeploying pipeline to SageMaker...")
pipeline.upsert(role_arn=role)
print(f"✓ Pipeline '{pipeline_name}' deployed successfully!")

# Step 3.7: Start the pipeline execution
print("\n" + "="*60)
print("Starting pipeline execution...")
print("="*60)
execution = pipeline.start()
print(f"✓ Execution started!")
print(f"✓ Execution ARN: {execution.arn}")
print(f"\nYou can monitor the execution in the SageMaker Console:")
print(f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines")

# Step 3.8: Wait for completion (optional)
print("\nWaiting for pipeline to complete (this may take 5-10 minutes)...")
print("(You can press Ctrl+C to stop waiting - the pipeline will continue running)")

try:
    execution.wait()
    status = execution.describe()['PipelineExecutionStatus']
    print(f"\n✓ Pipeline completed with status: {status}")
    
    if status == "Succeeded":
        print(f"\n🎉 SUCCESS! Check the output file at:")
        print(f"s3://{bucket}/pipeline-output/hello-world/hello_world.txt")
        print(f"\nTo download the file, run:")
        print(f"aws s3 cp s3://{bucket}/pipeline-output/hello-world/hello_world.txt .")
    
except KeyboardInterrupt:
    print("\n\nStopped waiting. Pipeline is still running in the background.")
    print(f"Check status at: https://{region}.console.aws.amazon.com/sagemaker/")
