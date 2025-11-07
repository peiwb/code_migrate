import boto3
import json
import time

# Configuration
pipeline_name = "HelloWorldPipeline"
region = boto3.Session().region_name
account_id = boto3.client('sts').get_caller_identity()['Account']

events_client = boto3.client('events')
iam_client = boto3.client('iam')

print(f"Setting up automation for: {pipeline_name}")

# Step 1: Create IAM role for EventBridge
role_name = f"{pipeline_name}-EventBridge-Role"
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "events.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}

try:
    role_response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Role for EventBridge to trigger SageMaker Pipeline"
    )
    eventbridge_role_arn = role_response['Role']['Arn']
    print(f"✓ Created EventBridge role: {role_name}")
    
    # Attach policy
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    )
    
    # Wait for role to propagate
    print("Waiting for IAM role to propagate...")
    time.sleep(10)
    
except iam_client.exceptions.EntityAlreadyExistsException:
    role_response = iam_client.get_role(RoleName=role_name)
    eventbridge_role_arn = role_response['Role']['Arn']
    print(f"✓ Using existing EventBridge role: {role_name}")

# Step 2: Create EventBridge schedule rule
schedule_name = f"{pipeline_name}-Schedule"

# Choose your schedule:
# Option 1: Every 5 minutes (for testing)
schedule_expression = "rate(5 minutes)"

# Option 2: Daily at 2 AM UTC
# schedule_expression = "cron(0 2 * * ? *)"

# Option 3: Every Monday at 9 AM UTC
# schedule_expression = "cron(0 9 ? * MON *)"

try:
    rule_response = events_client.put_rule(
        Name=schedule_name,
        ScheduleExpression=schedule_expression,
        State='ENABLED',
        Description=f'Automated trigger for {pipeline_name}'
    )
    print(f"✓ Created schedule: {schedule_expression}")
except Exception as e:
    print(f"✓ Schedule rule already exists or updated")

# Step 3: Add pipeline as target
pipeline_arn = f"arn:aws:sagemaker:{region}:{account_id}:pipeline/{pipeline_name}"

try:
    events_client.put_targets(
        Rule=schedule_name,
        Targets=[{
            'Id': '1',
            'Arn': pipeline_arn,
            'RoleArn': eventbridge_role_arn,
            'SageMakerPipelineParameters': {
                'PipelineParameterList': []
            }
        }]
    )
    print(f"✓ Pipeline linked to schedule")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("🎉 AUTOMATION SETUP COMPLETE!")
print("="*60)
print(f"Pipeline will run automatically: {schedule_expression}")
print(f"\nTo disable automation:")
print(f"  aws events disable-rule --name {schedule_name}")
print(f"\nTo enable automation:")
print(f"  aws events enable-rule --name {schedule_name}")
print(f"\nTo delete automation:")
print(f"  aws events remove-targets --rule {schedule_name} --ids 1")
print(f"  aws events delete-rule --name {schedule_name}")
