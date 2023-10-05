# Cheatsheet Downloader
It downloads the picture of cheatsheet downloaded in google Drive.


## Setup
Before running the code make sure to edit the following global variables in `main.py`:
* CREDENTIAL_FILE_PATH: path that contains the credentials of the Google Drive API
* GOOGLEDRIVE_FOLDER_ID_TXT_PATH: the local path of the txt file that contains the ID of the folder that contains the picture of the cheatsheet on Google Drive
    * It is stored in a separate txt for security reason (Since it will be upladed on github)
* TEMP_DIR: local path of the folder to download the file to