import os
import re
from typing import Tuple, List, Optional
import pathlib

from insert_image import insert_images_into_drawio

from code2drawio import main as code2drawio

# === PATHS TO EDIT ========================
TEMP_DIR: str = '~/Desktop/Desktop_Scripts/Cheat Sheet Temp Folder'
MINDMAP_DIR='~/Documents/Mind Maps/Generated Mindmaps'
# ==========================================
# Expand the tilde (~)
TEMP_DIR = os.path.expanduser(TEMP_DIR)
MINDMAP_DIR = os.path.expanduser(MINDMAP_DIR)

def get_files_in_temp_folder()-> List[str]:
    file_paths_in_temp_dir: List[pathlib.Path] = list(pathlib.Path(TEMP_DIR).glob('*'))
    file_str_paths_in_temp_dir: List[str] = [str(path) for path in file_paths_in_temp_dir]
    return file_str_paths_in_temp_dir

def get_image_file_paths()->List[str]:
    image_paths_in_temp_dir: List[pathlib.Path] = list(pathlib.Path(TEMP_DIR).glob('*.jpeg'))
    image_paths_in_temp_dir+= list(pathlib.Path(TEMP_DIR).glob('*.JPEG'))
    image_paths_in_temp_dir+= list(pathlib.Path(TEMP_DIR).glob('*.jpg'))
    image_paths_in_temp_dir+= list(pathlib.Path(TEMP_DIR).glob('*.JPG'))
    image_paths_in_temp_dir+= list(pathlib.Path(TEMP_DIR).glob('*.png'))
    image_paths_in_temp_dir+= list(pathlib.Path(TEMP_DIR).glob('*.PNG'))
    return [str(path) for path in image_paths_in_temp_dir]

def get_code_file_(file_paths: List[str])-> Tuple[str, str]:
    '''Returns (code_file_path, code_file_name) of the file that contains the code.'''
    txt_file_names: List[Optional[re.Match]] = [re.search('([^/]+).txt', str_path) for str_path in file_paths]
    txt_file_names = list(filter(lambda d: d is not None, txt_file_names))
    # Check if more than one .txt file exist in the temp folder
    if len(txt_file_names)>1:
        print(f'There are more than one .txt files in the temp folder: {TEMP_DIR}')
        os.system(f'nautilus "{TEMP_DIR}"')
        input()
        raise ValueError('There are more than one .txt files in the temp folder.')
    if len(txt_file_names) == 0:
        print(f'There are no .txt files in the temp folder: {TEMP_DIR}')
        os.system(f'nautilus "{TEMP_DIR}"')
        input()
        raise ValueError('There are no .txt files in the temp folder.')
    code_file_path: str = txt_file_names[0].string
    code_file_name: str = txt_file_names[0].group(1)    # File name without the .txt
    return code_file_path, code_file_name

def make_project_folder(project_folder_path: str):
    '''Make a folder to put the drawio file e.t.c in.'''
    if os.path.isdir(project_folder_path):  # Check folder with same name do not already exist
        print(f'A folder with the same name: {code_file_name} already exists in the mindmap_dir: {MINDMAP_DIR}')
        os.system(f'nautilus "{MINDMAP_DIR}"')
        input()
        raise ValueError(f'A folder with the same name: {code_file_name} already exists in the mindmap_dir: {MINDMAP_DIR}')
    os.mkdir(project_folder_path)
    
def file_path2file_name(file_path: str) -> str:
    return file_path.split('/')[-1]
    
def move_temp_files_to_project_folder(temp_files_path: List[str], project_folder_path: str):
    '''Copy contents of the temp folder to the project folder.'''
    for file_path in temp_files_path:
        os.rename(file_path, f'{project_folder_path}/{file_path2file_name(file_path)}')

def open_drawio(drawio_file_path: str):
    os.system(f'drawio "{drawio_file_path}"')
    
if __name__ == '__main__':
    file_paths: List[str] = get_files_in_temp_folder()
    code_file_path, code_file_name = get_code_file_(file_paths)
    with open(code_file_path, 'r') as handle:
        code: str = handle.read()
    project_folder_path: str = f"{MINDMAP_DIR}/{code_file_name}"
    make_project_folder(project_folder_path)
    try:
        code2drawio.render_code(code, f'{code_file_name}.drawio', project_folder_path)
    except BaseException as e:  # Print Error
        print('Error occured whilst parsing the code.')
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")        
        input()
    image_paths: List[str] = get_image_file_paths()
    insert_images_into_drawio(f'{project_folder_path}/{code_file_name}.drawio', image_paths)
    move_temp_files_to_project_folder(file_paths, project_folder_path)
    open_drawio(f'{project_folder_path}/{code_file_name}.drawio')