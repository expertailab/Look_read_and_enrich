import requests
from tqdm import tqdm
import math
import zipfile
import os

def download_and_unzip(file,output_path):
    url = "https://zenodo.org/record/3258126/files/"+file
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0 
    with open(file, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', desc = file, leave = True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
    f.close
    print("Unzipping... this process might take several minutes to complete.")
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall(output_path)
    zip_ref.close()
    os.remove(file)
    print("Done")

download_and_unzip("jsons.zip",".")
download_and_unzip("saved.zip",".")
download_and_unzip("scigraph.zip","./images/")
download_and_unzip("semscholar.zip","./images/")
download_and_unzip("tqa.zip","./images/")
