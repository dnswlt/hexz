# Create a (gitignored) file named terraform.tfvars in this file's directory with the following contents:
# 
# ```
# project          = "YOUR_PROJECT_ID_GOES_HERE"
# credentials_file = "PATH_TO_YOUR_ACCOUNT_KEY_JSON_FILE_GOES_HERE"
# ```


variable "project" {}


variable "credentials_file" {}


variable "region" {
  default = "europe-west6"
}


variable "zone" {
  default = "europe-west6-a"
}
