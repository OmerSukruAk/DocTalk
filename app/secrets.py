"""Azure Key Vault -> environment bridge.

If KEY_VAULT_URL is set, every secret in the vault is exposed to Settings as an environment
variable *before* Settings is built (Key Vault names use dashes: ``LLM-API-KEY`` becomes
``LLM_API_KEY``). Real environment variables always win, so local development and
``docker compose`` keep working unchanged. Authentication uses DefaultAzureCredential, i.e.
the managed identity of the App Service / Container App in Azure and ``az login`` locally,
so there is no credential in the repo or in the image."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("doctalk")


def env_name(secret_name: str) -> str:
    return secret_name.strip().replace("-", "_").upper()


def load_key_vault_into_env() -> int:
    url = os.getenv("KEY_VAULT_URL", "").strip()
    if not url:
        return 0
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    client = SecretClient(vault_url=url, credential=DefaultAzureCredential())
    loaded = 0
    for props in client.list_properties_of_secrets():
        if not props.enabled:
            continue
        name = env_name(props.name)
        if name in os.environ:
            continue
        os.environ[name] = client.get_secret(props.name).value or ""
        loaded += 1
    logger.info("Loaded %d secret(s) from Key Vault", loaded)
    return loaded
