# Deployment Guide

## Configuration file

Every project has a `lumora.toml` file at the repository root. The `build.size` key accepts
`small` (2 GB RAM), `medium` (4 GB RAM) or `large` (8 GB RAM). The default is `small`.

## Ports

Your service must listen on the port in the `PORT` environment variable. Lumora sets it to 8080
by default. Services that do not respond to the health check within 60 seconds are rolled back.

## Rollbacks

Every deployment is kept for 30 days. Run `lumora rollback <deployment-id>` to restore a previous
deployment in under a minute. Rollbacks do not revert database migrations.

## Environment variables

Secrets are stored encrypted with AES-256 and are injected at runtime. Changing a secret
triggers a redeploy. Variables larger than 32 KB are rejected.
