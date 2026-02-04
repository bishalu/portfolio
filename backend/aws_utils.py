import os
import boto3
from botocore.client import Config
import pandas as pd
from io import StringIO
from datetime import datetime
import json

def get_aws_credentials():
    # Get AWS credentials from environment variables (should be set properly in shell)
    aws_access_key_id = os.getenv('AWS_ID')
    aws_secret_access_key = os.getenv('AWS_SEC')
    
    # If the credentials are not found in environment variables, check if running in Streamlit Cloud
    if not (aws_access_key_id and aws_secret_access_key):
        # Try standard AWS environment variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # If still missing, check Streamlit secrets
    if not (aws_access_key_id and aws_secret_access_key):
        try:
            from streamlit import secrets as st_secrets  # Import only if needed
            aws_access_key_id = st_secrets.aws_id
            aws_secret_access_key = st_secrets.aws_sec
        except ImportError:
            # Running in Lambda or other environment without Streamlit
            # AWS SDK will use IAM roles or other credential providers
            pass
        except AttributeError:
            # Streamlit secrets don't have AWS credentials
            # AWS SDK will use IAM roles or other credential providers  
            pass

    return aws_access_key_id, aws_secret_access_key

AWS_ID, AWS_SEC = get_aws_credentials()

def create_boto3_client(service_name, region_name='us-east-2', **kwargs):
    """Create boto3 client with proper credential handling."""
    client_kwargs = {
        'service_name': service_name,
        'region_name': region_name,
        **kwargs
    }
    
    # Only add credentials if they exist (otherwise use IAM roles)
    if AWS_ID and AWS_SEC:
        client_kwargs['aws_access_key_id'] = AWS_ID
        client_kwargs['aws_secret_access_key'] = AWS_SEC
    
    return boto3.client(**client_kwargs)

def create_boto3_resource(service_name, region_name='us-east-2', **kwargs):
    """Create boto3 resource with proper credential handling."""
    resource_kwargs = {
        'service_name': service_name,
        'region_name': region_name,
        **kwargs
    }
    
    # Only add credentials if they exist (otherwise use IAM roles)
    if AWS_ID and AWS_SEC:
        resource_kwargs['aws_access_key_id'] = AWS_ID
        resource_kwargs['aws_secret_access_key'] = AWS_SEC
    
    return boto3.resource(**resource_kwargs)

def get_secret(secret_name):
    region_name = "us-east-2"  # Set to the desired AWS region

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        aws_access_key_id=AWS_ID,
        aws_secret_access_key=AWS_SEC,
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        from utils.error_handling import handle_aws_error
        raise handle_aws_error(e, f"get_secret({secret_name})")
    
    # Decrypts secret using the associated KMS CMK.
    # Depending on whether the secret is a string or binary, one of these fields will be populated.
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret)
        return secret_dict
    else:
        # Your logic to handle the binary secret goes here
        # Assuming you don't store API keys as binary, this part may not be necessary
        raise Exception('Binary secret format is not supported.')


def connect_to_s3(timeout=10, client=False):
    """Establish a connection to S3 using the credentials in config.py."""
    # Create a Config object with the desired timeout settings
    config = Config(
        connect_timeout=timeout,
        read_timeout=timeout,
        retries={'max_attempts': 3}  # Example of setting max retry attempts
    )

    if client:
        # Use boto3.client for low-level API access
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ID,
            aws_secret_access_key=AWS_SEC,
            config=config
        )
    else:
        # Use boto3.resource for high-level resource access
        s3 = boto3.resource(
            's3',
            aws_access_key_id=AWS_ID,
            aws_secret_access_key=AWS_SEC,
            config=config
        )

    return s3


def retrieve_csv_files_from_s3(bucket_name, s3):
    """Retrieve all CSV files from an S3 bucket."""
    bucket = s3.Bucket(bucket_name)
    csv_files = [obj.key for obj in bucket.objects.all() if obj.key.endswith('.csv')]
    return csv_files

def load_csv_from_s3(bucket_name, file_key, s3):
    """Load a CSV file from S3 into a Pandas DataFrame."""
    client = s3.meta.client
    csv_obj = client.get_object(Bucket=bucket_name, Key=file_key)
    csv_data = csv_obj['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_data))

def delete_file_from_s3(bucket_name, file_key, s3):
    """Delete a specific file from an S3 bucket."""
    obj = s3.Object(bucket_name, file_key)
    obj.delete()

def cloud_dump(setlist, args=None, dump_all=False):
    # Initialize AWS S3 resources and client
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=AWS_ID,
        aws_secret_access_key=AWS_SEC
    )
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ID,
        aws_secret_access_key=AWS_SEC
    )

    # Determine bucket name and filtering based on the mode
    if dump_all and args is not None:
        bucket_name = 'vibesets'
        data_to_dump = setlist
        # Create filename using args dictionary
        args_str = '_'.join([f"{key}-{value}" for key, value in args.items()])
        file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{args_str}.csv"
    else:
        bucket_name = 'chatgpt-setlist-to-database'
        # Filter the dataframe
        data_to_dump = setlist[(setlist['in_database'] == False) & (setlist['is_hallucination'] == False)]
        file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        if data_to_dump.empty:
            print("No songs to dump.")
            return

    # Convert the DataFrame into a CSV string
    csv_buffer = StringIO()
    data_to_dump.to_csv(csv_buffer, index=False)

    # Check if bucket exists; if not, create it
    if bucket_name not in [bucket['Name'] for bucket in s3_client.list_buckets()['Buckets']]:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"S3 bucket {bucket_name} created.")

    # Put the object (upload the file)
    s3_resource.Object(bucket_name, file_name).put(Body=csv_buffer.getvalue())

    print(f"Dumped to {file_name} in S3 bucket {bucket_name}.")


# ---------------------------------------------------------------------
# V5 SQS and Lambda Functions for Async Job Processing
# ---------------------------------------------------------------------

# Environment-based configuration using centralized detection
from utils.environment import detect_environment, get_sqs_queue_name, VIBESET_ENV

def get_v5_queue_name():
    """Get V5 queue name based on environment."""
    return get_sqs_queue_name()

def get_sqs_client():
    """Get SQS client with configured credentials."""
    return create_boto3_client('sqs')

def get_lambda_client():
    """Get Lambda client with configured credentials."""
    return create_boto3_client('lambda')

# Cache for queue URLs to avoid repeated API calls
_QUEUE_URL_CACHE = {}

def create_v5_sqs_queue():
    """
    Create the main SQS FIFO queue for V5 job processing.
    Safe to run multiple times - only creates if doesn't exist.
    Uses caching to avoid repeated list_queues API calls.
    """
    queue_name = get_v5_queue_name()
    
    # Check cache first
    if queue_name in _QUEUE_URL_CACHE:
        return _QUEUE_URL_CACHE[queue_name]
    
    sqs = get_sqs_client()
    
    try:
        # Check if queue already exists
        queue_prefix = queue_name.split('.')[0]  # Get prefix without .fifo
        response = sqs.list_queues(QueueNamePrefix=queue_prefix)
        if 'QueueUrls' in response and response['QueueUrls']:
            # Find the exact queue (not the DLQ)
            for queue_url in response['QueueUrls']:
                if queue_url.endswith(queue_name):
                    print(f"✓ SQS queue {queue_name} already exists")
                    # Cache the URL for future use
                    _QUEUE_URL_CACHE[queue_name] = queue_url
                    return queue_url
            # If exact match not found, create it
            print(f"⚠️  Found queues with prefix {queue_prefix} but not exact match {queue_name}")
        
        # Create FIFO queue with proper attributes
        queue_response = sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                'FifoQueue': 'true',
                'ContentBasedDeduplication': 'true',
                'VisibilityTimeout': '1200',  # 20 minutes
                'MessageRetentionPeriod': '1209600',  # 14 days
                'ReceiveMessageWaitTimeSeconds': '20'  # Long polling
            }
        )
        
        queue_url = queue_response['QueueUrl']
        print(f"✓ Created SQS queue: {queue_name}")
        # Cache the URL for future use
        _QUEUE_URL_CACHE[queue_name] = queue_url
        return queue_url
        
    except Exception as e:
        print(f"✗ Error creating SQS queue: {str(e)}")
        raise

def send_v5_job_to_queue(job_data):
    """
    Send a V5 job to the SQS queue for processing.
    
    Args:
        job_data: Dict containing job_id and parameters
    """
    sqs = get_sqs_client()
    
    try:
        # Get queue URL
        queue_url = create_v5_sqs_queue()  # Safe to call - won't recreate
        
        # Send message
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(job_data),
            MessageGroupId=f"vibeset-v5-{job_data['job_id']}",  # Unique group per job for parallel processing
            MessageDeduplicationId=f"{job_data['job_id']}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        print(f"✓ Sent job {job_data['job_id']} to queue")
        return response['MessageId']
        
    except Exception as e:
        print(f"✗ Error sending job to queue: {str(e)}")
        raise

def receive_v5_jobs_from_queue(max_messages=1):
    """
    Receive jobs from the V5 SQS queue (used by Lambda worker).
    
    Args:
        max_messages: Maximum number of messages to receive
        
    Returns:
        List of job messages
    """
    sqs = get_sqs_client()
    
    try:
        # Get queue URL
        queue_url = create_v5_sqs_queue()
        
        # Receive messages
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=20
        )
        
        messages = response.get('Messages', [])
        print(f"✓ Received {len(messages)} job(s) from queue")
        return messages
        
    except Exception as e:
        print(f"✗ Error receiving jobs from queue: {str(e)}")
        raise

def delete_v5_job_from_queue(receipt_handle):
    """
    Delete a completed job from the SQS queue.
    
    Args:
        receipt_handle: SQS message receipt handle
    """
    sqs = get_sqs_client()
    
    try:
        queue_url = create_v5_sqs_queue()
        
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        
        print("✓ Deleted completed job from queue")
        
    except Exception as e:
        print(f"✗ Error deleting job from queue: {str(e)}")
        raise

def deploy_v5_lambda_worker():
    """
    Deploy the V5 Lambda worker function with proper configuration.
    """
    import zipfile
    import tempfile
    import shutil
    
    lambda_client = get_lambda_client()
    function_name = 'vibeset-v5-worker'
    
    try:
        # Create deployment package
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy necessary files to temp directory
            files_to_include = [
                'lambda_worker.py',
                'build.py', 
                'utils/',
                'pipeline/',
                'prompts_params.py'
            ]
            
            print("Packaging Lambda function...")
            
            # Copy files
            for file_path in files_to_include:
                if os.path.isfile(file_path):
                    shutil.copy2(file_path, temp_dir)
                elif os.path.isdir(file_path):
                    shutil.copytree(file_path, os.path.join(temp_dir, file_path))
            
            # Create ZIP file
            zip_path = os.path.join(temp_dir, 'deployment.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file != 'deployment.zip':
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, temp_dir)
                            zip_file.write(file_path, arc_name)
            
            # Read ZIP file
            with open(zip_path, 'rb') as zip_file:
                zip_content = zip_file.read()
        
        # Check if function exists
        try:
            lambda_client.get_function(FunctionName=function_name)
            function_exists = True
            print(f"✓ Function {function_name} exists, updating...")
        except lambda_client.exceptions.ResourceNotFoundException:
            function_exists = False
            print(f"Creating new function {function_name}...")
        
        if function_exists:
            # Update existing function
            response = lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            print(f"✓ Updated Lambda function: {function_name}")
        else:
            # Create IAM role first
            role_arn = create_lambda_execution_role()
            
            # Create new function
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_worker.lambda_handler',
                Code={'ZipFile': zip_content},
                Description='V5 Vibeset Build Worker',
                Timeout=900,  # 15 minutes
                MemorySize=1024,
                Environment={
                    'Variables': {
                        'AWS_ID': AWS_ID,
                        'AWS_SEC': AWS_SEC
                    }
                }
            )
            print(f"✓ Created Lambda function: {function_name}")
        
        return response['FunctionArn']
        
    except Exception as e:
        print(f"✗ Lambda deployment failed: {str(e)}")
        raise

def configure_sqs_lambda_trigger():
    """
    Configure SQS to trigger the Lambda function.
    """
    lambda_client = get_lambda_client()
    function_name = 'vibeset-v5-worker'
    queue_url = create_v5_sqs_queue()
    
    try:
        # Get queue ARN
        sqs = get_sqs_client()
        queue_attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn']
        )
        queue_arn = queue_attrs['Attributes']['QueueArn']
        
        # Create event source mapping
        try:
            response = lambda_client.create_event_source_mapping(
                EventSourceArn=queue_arn,
                FunctionName=function_name,
                BatchSize=1,
                MaximumBatchingWindowInSeconds=0
            )
            print(f"✓ Created SQS trigger for Lambda function")
            return response['UUID']
        except lambda_client.exceptions.ResourceConflictException:
            print(f"✓ SQS trigger already exists for Lambda function")
            return None
            
    except Exception as e:
        print(f"✗ SQS trigger configuration failed: {str(e)}")
        raise

def get_account_id():
    """Get AWS account ID."""
    import boto3
    sts_client = boto3.client(
        'sts',
        aws_access_key_id=AWS_ID,
        aws_secret_access_key=AWS_SEC,
        region_name='us-east-2'
    )
    return sts_client.get_caller_identity()['Account']

def create_lambda_execution_role():
    """Create IAM role for Lambda execution."""
    import boto3
    import json
    
    iam_client = boto3.client(
        'iam',
        aws_access_key_id=AWS_ID,
        aws_secret_access_key=AWS_SEC,
        region_name='us-east-2'
    )
    
    role_name = 'vibeset-lambda-execution-role'
    
    # Trust policy for Lambda
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if role exists
        try:
            role = iam_client.get_role(RoleName=role_name)
            print(f"✓ IAM role {role_name} already exists")
            return role['Role']['Arn']
        except iam_client.exceptions.NoSuchEntityException:
            pass
        
        # Create role
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for Vibeset V5 Lambda worker',
            MaxSessionDuration=3600
        )
        
        role_arn = response['Role']['Arn']
        print(f"✓ Created IAM role: {role_name}")
        
        # Attach basic Lambda execution policy
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        
        # Attach SQS access policy
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/AmazonSQSFullAccess'
        )
        
        # Create custom policy for database access
        custom_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "secretsmanager:GetSecretValue",
                        "rds:DescribeDBInstances",
                        "rds-data:*"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        # Create and attach custom policy
        policy_name = 'vibeset-lambda-custom-policy'
        try:
            iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(custom_policy),
                Description='Custom policy for Vibeset Lambda worker'
            )
            print(f"✓ Created custom policy: {policy_name}")
        except iam_client.exceptions.EntityAlreadyExistsException:
            print(f"✓ Custom policy {policy_name} already exists")
        
        # Attach custom policy
        account_id = get_account_id()
        custom_policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn=custom_policy_arn
        )
        
        print(f"✓ Attached policies to role")
        
        # Wait a moment for IAM propagation
        import time
        print("Waiting for IAM propagation...")
        time.sleep(10)
        
        return role_arn
        
    except Exception as e:
        print(f"✗ IAM role creation failed: {str(e)}")
        raise
