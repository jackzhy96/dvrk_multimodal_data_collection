from typing import Union
from pathlib import Path
import shutil


def convert_pathlib_type(folder:Union[Path,str])->Path:
    '''
    Converts a pathlib Path to a Path object.
    folder: your selected path
    output: your selected path in Pathlib Path object
    '''
    if isinstance(folder, Path):
        new_path = folder
    elif isinstance(folder, str):
        new_path = Path(folder)
    else:
        raise TypeError('Folder must be of type Path or str')
    return new_path


def clear_folder(folder: Union[Path,str])->None:
    '''
    Clear the folder contents.
    folder: path of folder to clear
    '''
    new_path = convert_pathlib_type(folder)
    input(f'You are about to remove all files in {new_path}, press ENTER to continue')
    for item_obj in new_path.iterdir():
        if item_obj.is_dir():
            shutil.rmtree(item_obj, ignore_errors=True)
            print(f'Removing subfolder: {item_obj}')
        else:
            item_obj.unlink(missing_ok=True)
            print(f'Removing file: {item_obj}')

def create_folder(folder: Union[Path,str])->None:
    '''
    Create a new folder.
    folder: path of new folder
    '''
    new_path = convert_pathlib_type(folder)
    new_path.mkdir(parents=True, exist_ok=True)
    print(f'Created new path: {new_path}')


if __name__ == '__main__':
    pass