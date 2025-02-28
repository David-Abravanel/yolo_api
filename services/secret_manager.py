import json
import boto3
import os

client = boto3.client('secretsmanager', region_name='il-central-1')


def load_secret(secret_name):
    try:
        print(f"Attempting to retrieve secret: {secret_name}")
        response = client.get_secret_value(SecretId=secret_name)
        secret = response['SecretString']
        secret_dict = json.loads(secret)

        for key, value in secret_dict.items():
            # print(key, "=", value)
            os.environ[key] = value
        print(f"Loaded secrets from {secret_name} into environment variables.")
    except Exception as e:
        print(f"Error retrieving secret: {e}")


def load_AWS_env(secret_name):
    load_secret(secret_name)
