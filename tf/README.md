Follow steps [here](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) to install terraform. Follow the steps for "manual installation".

Follow the below steps outside of Terraform:

1. Create a GCP project and note the project ID, region and zone.
2. Create a service account named "hexz-provisioner" with role "Project -> Editor" for the above project ID and note the service account email address.
3. Create a service account key for the new service account.
4. Download the new service account key in JSON format to some path outside of version control, e.g. ~/secrets/my-gcp-project-service-account-key.json

If you haven't already, install the gcloud CLI following [these instructions](https://cloud.google.com/sdk/docs/install).
When it comes to the `gcloud init` step, set the project ID, region, and zone associated with the project.
Below I'll assume that you said yes to adding `gcloud` to your $PATH.

Create a (gitignored) file named terraform.tfvars in this README's directory with the following contents:

```terraform
project          = "YOUR_PROJECT_ID_GOES_HERE"
credentials_file = "PATH_TO_YOUR_ACCOUNT_KEY_JSON_FILE_GOES_HERE"
```

Register gcloud as a Docker credential helper:

```bash
gcloud auth configure-docker europe-west6-docker.pkg.dev
```

Enable required cloud APIs:

```bash
gcloud services enable artifactregistry.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable redis.googleapis.com
gcloud services enable run.googleapis.com
```

Import the already existing service account that you created above into the terraform state, and then grant
permission to the service account to push to artifactregistry. This saves us a bit of pointing and clicking.
(Icky clicky.)

```bash
terraform import google_service_account.hexz-service-account projects/$project_id/serviceAccounts/$service_account_email
gcloud projects add-iam-policy-binding $project_id \
    --member=serviceAccount:hexz-provisioner@${project_id}.iam.gserviceaccount.com \
    --role=roles/artifactregistry.admin \
    --condition=None
```

Now you can do the normal things:

```bash
terraform init
terraform apply
```
