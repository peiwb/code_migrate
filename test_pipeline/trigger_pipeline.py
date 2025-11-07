import boto3
from datetime import datetime

sagemaker_client = boto3.client('sagemaker')
pipeline_name = "HelloWorldPipeline"

print(f"Manually triggering: {pipeline_name}")

response = sagemaker_client.start_pipeline_execution(
    PipelineName=pipeline_name,
    PipelineExecutionDisplayName=f'Manual-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
)

print(f"✓ Pipeline execution started!")
print(f"Execution ARN: {response['PipelineExecutionArn']}")
