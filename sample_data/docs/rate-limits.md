# API Rate Limits

Rate limits are applied per API key and reset every minute.

| Plan | Requests per minute | Concurrent deployments | Monthly compute hours |
|------|--------------------|------------------------|-----------------------|
| Starter | 60 | 1 | 100 |
| Growth | 600 | 5 | 1,000 |
| Scale | 3,000 | 20 | 10,000 |
| Enterprise | Custom | Custom | Custom |

## Burst allowance

Growth and Scale keys may burst to twice their per-minute limit for up to 10 seconds. Starter
keys have no burst allowance.

## Raising limits

Scale customers can request a temporary limit increase for launches through the support portal
at least 3 business days in advance. Enterprise limits are set in the order form.
