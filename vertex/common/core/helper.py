from google.cloud import secretmanager

def get_secret(projec_id: str, secret_name: str) -> str:
    """
    Retrieves the plaintext value of a secret from Google Cloud Secret Manager.

    This function accesses the 'latest' version of the specified secret. It is kept
    outside the Settings class to prevent mandatory GCP authentication during 
    module imports, allowing for easier local development.

    Args:
        project_id (str): The Google Cloud Project ID where the secret is hosted.
        secret_name (str): The name/ID of the secret to retrieve.

    Returns:
        str: The secret value decoded as a UTF-8 string.
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{projec_id}/secrets/{secret_name}/versions/latest" 
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        raise RuntimeError(f"Error obtaining the secret '{secret_name}': {e}")