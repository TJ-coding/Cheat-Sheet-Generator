import os
import io
from typing import Dict, Any, List, Optional
import pathlib
import time

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import googleapiclient.discovery
from googleapiclient.http import MediaIoBaseDownload

# === PATHS TO EDIT ========================
CREDENTIAL_FILE_PATH: str = '~/Desktop/Desktop_Scripts/Read Ahead Lecture Sheet Generator/credentials.json'
GOOGLEDRIVE_FOLDER_ID_TXT_PATH: str = '~/Documents/git-repo/Cheat-Sheet-Generator/cheatsheet_downloader_proj/CHEATSHEET_FOLDER_ID.txt'
TEMP_DIR: str = '~/Desktop/Desktop_Scripts/Cheat Sheet Temp Folder'
# ==========================================
# Expand the tilde
CREDENTIAL_FILE_PATH = os.path.expanduser(CREDENTIAL_FILE_PATH)
GOOGLEDRIVE_FOLDER_ID_TXT_PATH = os.path.expanduser(GOOGLEDRIVE_FOLDER_ID_TXT_PATH)
TEMP_DIR = os.path.expanduser(TEMP_DIR)


def get_token()->Credentials:
    '''Get Token for Google Drive API.'''
    SCOPES = ['https://www.googleapis.com/auth/drive']
    if os.path.exists('token.json'):
        # Token Retrieved in the Past and is Stored
        creds = Credentials.from_authorized_user_file(
            'token.json', SCOPES)
        if creds.valid:
            return creds
        # Not valid
        if creds.expired and creds.refresh_token:
            # Token is expired and contains Refresh Token
            creds.refresh(Request())
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            return creds
    # Nothing we can do
    # ask to login again (To retrieve user token)
    flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIAL_FILE_PATH, SCOPES)
    creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    return creds # type: ignore

def get_files_in_cheatsheet_folder(DRIVE: googleapiclient.discovery.Resource, 
                                   folder_id)->List[Dict[str, str]]:
    '''Get files names in Google Drive 'Cheat Sheat' Folder'''
    response: Dict[str, Any] = DRIVE.files().list(
    q=f"mimeType='image/jpeg' and parents in '{folder_id}'"
                              ).execute()
    return response['files']

def get_file(files: List[Dict[str, str]])->Dict[str, str]:
    '''
    If there are more than one files, ask the user to select one.
    ==Returned File Format==
    {'kind': 'drive#file',
     'mimeType': 'image/jpeg',
     'id': '1RbdoBFiWXqAb0bkmDwvIrGmfKUb1WeZG',
     'name': 'Test image'}
    '''
    if len(files) ==1:
        # Just select the only file
        return files[0]
    print('Select the file to use')
    for i, file in enumerate(files):
        print(f'{i+1}: {file}')
    input_str: str = input('Type in the number: ')
    selected_file: Optional[Dict[str, str]] = None
    try:
        selected_file = files[int(input_str)-1]
    except:
        pass
    while selected_file is None:
        try:
            selected_file = files[int(input_str)-1]
        except:
            pass
    return selected_file

def download_file(DRIVE: googleapiclient.discovery.Resource, 
                  file_id: str,
                  file_name: str,
                  file_download_dir: str)->str:
    '''Download the selected file from Google Drive.'''
    request = DRIVE.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(F'Download {int(status.progress() * 100)}.')
    file_path:str = f'{file_download_dir}/{file_name}.jpeg'
    with open(file_path, 'wb') as handle:
        handle.write(file.getvalue())
    return file_path

def delete_file(DRIVE: googleapiclient.discovery.Resource,
                file_id: str):
    '''Delete the downloaded file from Google Drive.'''
    DRIVE.files().delete(fileId=file_id).execute()

def open_vscode(img_file_path: str, text_file_path: str):
    '''Open the image and text file with VS Code'''
    with open(text_file_path, 'w') as handle:
        handle.write("")
    print(img_file_path)
    print(text_file_path)
    os.system(f'nautilus "{TEMP_DIR}"')
    os.system(f'code "{text_file_path}" -n')
    os.system(f'code "{img_file_path}"')

if __name__ == '__main__':
    if len(list(pathlib.Path(TEMP_DIR).glob('*'))) > 0:
        print(f'Pleaes empty the temp folder: {TEMP_DIR}.')
        os.system(f'nautilus "{TEMP_DIR}"')
        input()
        raise ValueError(f'Pleaes empty the temp folder: {TEMP_DIR}.')
    
    with open(GOOGLEDRIVE_FOLDER_ID_TXT_PATH, 'r') as handle:
        google_drive_folder_id: str = handle.read()
    creds = get_token()
    DRIVE = build('drive', 'v3', credentials=creds)
    files: List[Dict[str, str]] = get_files_in_cheatsheet_folder(DRIVE, google_drive_folder_id)
    if len(files)==0:
        print('There was no files in the specified Google Drive folder.')
        input()
        raise ValueError('There was no files in the specified Google Drive folder.')
    file: Dict[str, str] = get_file(files)
    img_file_path: str = download_file(DRIVE, file['id'], file['name'], file_download_dir=TEMP_DIR)
    delete_file(DRIVE, file['id'])
    text_file_path: str = f'{TEMP_DIR}/{file["name"]}.txt'
    time.sleep(1.5)
    open_vscode(img_file_path, text_file_path)