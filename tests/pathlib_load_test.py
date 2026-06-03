import os
import sys
from pathlib import Path
dynamic_path = os.path.abspath(__file__+"/../../")
print(dynamic_path)
sys.path.append(dynamic_path)


if __name__ == '__main__':
    # if not os.path.exists(img_folder):
    #     os.makedirs(img_folder)
    test = 1
    print(Path.cwd())