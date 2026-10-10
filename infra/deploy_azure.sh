#!/usr/bin/env bash
# Azure Container Apps deployment for DocTalk.  STATUS: written, NOT executed against a subscription.
# Prereqs: az CLI logged in; an Entra app registration exposing scope "access_as_user"
#          (set ENTRA_TENANT_ID and ENTRA_AUDIENCE=api://<app-id>); a Mongo-compatible DB
#          (Cosmos DB for MongoDB) and a Qdrant endpoint; secrets listed below stored in Key Vault.
set -euo pipefail
: "${RG:=doctalk-rg}" "${LOC:=westeurope}" "${APP:=doctalk-api}" "${ENV:=doctalk-env}"
: "${ACR:?set ACR to a globally unique registry name}" "${KV:?set KV to a globally unique vault name}"
: "${ENTRA_TENANT_ID:?}" "${ENTRA_AUDIENCE:?}" "${MONGO_URI:?}" "${QDRANT_URL:?}"

az group create -n "$RG" -l "$LOC"
az acr create -g "$RG" -n "$ACR" --sku Basic
az acr build -r "$ACR" -t doctalk:latest .                    # add --build-arg WITH_ML=1 for NLI/re-ranker
az keyvault create -g "$RG" -n "$KV" --enable-rbac-authorization true
az containerapp env create -g "$RG" -n "$ENV" -l "$LOC"

az containerapp create -g "$RG" -n "$APP" --environment "$ENV" \
  --image "$ACR.azurecr.io/doctalk:latest" --registry-server "$ACR.azurecr.io" --registry-identity system \
  --system-assigned --target-port 8000 --ingress external --min-replicas 1 --max-replicas 3 \
  --env-vars AUTH_MODE=entra ENTRA_TENANT_ID="$ENTRA_TENANT_ID" ENTRA_AUDIENCE="$ENTRA_AUDIENCE" \
             MONGO_URI="$MONGO_URI" QDRANT_URL="$QDRANT_URL" LOG_TEXT=false \
             KEY_VAULT_URL="https://$KV.vault.azure.net"

# Managed identity may read secrets; the app reads LLM-API-KEY, EMBED-API-KEY ... at start-up (app/secrets.py)
PRINCIPAL=$(az containerapp show -g "$RG" -n "$APP" --query identity.principalId -o tsv)
az role assignment create --assignee "$PRINCIPAL" --role "Key Vault Secrets User" \
  --scope "$(az keyvault show -n "$KV" --query id -o tsv)"
az containerapp ingress show -g "$RG" -n "$APP" --query fqdn -o tsv
echo "Now: az keyvault secret set --vault-name $KV --name LLM-API-KEY --value <key>   (and EMBED-API-KEY), then restart the revision."
echo "Persistent uploads: mount an Azure Files volume at /data/files (az containerapp env storage set ...) for multi-replica use."
