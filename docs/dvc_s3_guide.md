# DVC Remote Storage Setup & Commands (AWS S3)

This guide documents the setup and execution workflow for using an Amazon S3 bucket (`s3://om-dvc-bucket`) as the remote storage backend for DVC in **BikeSense AI**.

---

## 1. Prerequisites & AWS Configuration

Ensure that your AWS credentials with proper S3 permissions (PutObject, GetObject, ListBucket) are configured in your local environment.

```powershell
# 1. Install AWS CLI if not present
pip install awscli

# 2. Configure AWS credentials (Access Key, Secret Key, Region)
aws configure
```

---

## 2. DVC Remote Configuration

Add the S3 bucket as a DVC remote and set it as the default target.

```powershell
# Add S3 remote and set as default (-d)
dvc remote add -d myremotes3 s3://om-dvc-bucket
```

This modifies `.dvc/config` (committed to git) without storing your credentials:

```ini
[core]
    remote = myremotes3
['remote "myremotes3"']
    url = s3://om-dvc-bucket
```

---

## 3. Data Pipeline & S3 Sync Workflow

### Step 1: Verify Bucket Access
Verify connectivity and permissions against the bucket:

```powershell
aws s3 ls s3://om-dvc-bucket
```

### Step 2: Reproduce Data Pipelines
Run the ingestion, validation, and feature engineering stages tracked by `dvc.yaml`:

```powershell
dvc repro
```
*Note: If no inputs or code changed, DVC outputs `Data and pipelines are up to date.`*

### Step 3: Push Data Artifacts to S3
Upload all cached data files (`data/raw/`, `data/external/`, `data/processed/`) to the S3 bucket:

```powershell
dvc push
```
Output confirms transfer:
```text
Collecting                                           |18.0 [00:00, 592entry/s]
Pushing
18 files pushed
```

### Step 4: Verify Files in S3 Bucket
List the content-addressed MD5 hash objects stored in the bucket:

```powershell
aws s3 ls s3://om-dvc-bucket/files/md5/ --recursive
```

---

## 4. Pulling Data on Another Machine

For a collaborator, CI/CD pipeline, or clean environment clone:

```powershell
# Clone the repository
git clone <repo-url>
cd bikesense-ai

# Fetch and link all data dependencies from S3
dvc pull
```

---

## 5. Security & Best Practices

- **Never commit AWS credentials to Git:** Keep `*accessKeys*.csv` and `.env` files in `.gitignore`.
- DVC uses the credentials configured in `~/.aws/credentials` or environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), ensuring secrets never enter source control.
