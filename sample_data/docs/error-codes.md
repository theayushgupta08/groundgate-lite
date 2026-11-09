# Error Code Reference

Every Lumora API error returns a JSON body with a `code`, a `message` and a `request_id`.
Quote the `request_id` when you contact support.

## Authentication errors

| Code | HTTP | Meaning | Fix |
|------|------|---------|-----|
| LMR-4010 | 401 | API key missing | Send the key in the `Authorization: Bearer` header |
| LMR-4011 | 401 | API key revoked | Create a new key in Settings > API Keys |
| LMR-4031 | 403 | Key lacks the required scope | Add the scope to the key, or use a key with `deploy:write` |

## Rate limit errors

LMR-4290 means you exceeded the per-minute request limit for your plan. The response includes a
`Retry-After` header in seconds. Back off exponentially and retry; do not retry in a tight loop.

LMR-4291 means you exceeded the monthly compute-hour quota. Requests fail until the quota resets
on the first day of the next billing cycle, or until you buy extra compute hours.

## Deployment errors

LMR-5003 means the build container ran out of memory. Increase the build size to `large` in
`lumora.toml` or reduce the size of your dependencies.

LMR-5070 means the health check failed: your service did not respond on port 8080 within 60
seconds of starting. Make sure your app listens on the port given in the `PORT` environment
variable.
