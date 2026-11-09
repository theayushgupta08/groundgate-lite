# Data Retention and Deletion

## Logs

Application logs are kept for 7 days on Starter, 30 days on Growth and 90 days on Scale.
Enterprise customers can configure retention up to 400 days.

## Backups

Managed Postgres databases are backed up daily. Backups are kept for 7 days on Growth and
35 days on Scale. Starter databases are not backed up.

## Account deletion

When a workspace is deleted, its data is soft-deleted for 14 days so it can be restored by
support. After 14 days all data, including backups, is permanently erased within 30 days.

## Data residency

Workspaces can be created in the Mumbai (ap-south-1), Frankfurt (eu-central-1) or Virginia
(us-east-1) region. Data never leaves the selected region, except for billing records, which are
stored in Virginia.
