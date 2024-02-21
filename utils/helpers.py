from typing import Any, Dict, List
import os
import json


def merge_dicts(dicts:List[Dict[str, Any]]):
    """
    Merges a list of dictionaries into one dictionary
    """
    merged_dict = []
    for d in dicts:
        merged_dict.extend(d)
    return merged_dict

def get_files_from_path(path: str, extension: str = None) -> List[str]:
    """
    Returns a list of files from a given path
    """
    
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if extension is not None:
                if extension in file:
                    files.append(os.path.join(r, file))
            else:
                files.append(os.path.join(r, file))
    return files

def get_dict_from_json(path:str):
    """
    Returns a dictionary from a json file
    """
    
    with open(path, 'r') as f:
        return json.load(f)

def get_dicts_from_jsons(paths:List[str]):
    """
    Returns a list of dictionaries from a list of json files
    """
    
    dicts = []
    for path in paths:
        dicts.append(get_dict_from_json(path))
    return dicts

def save_to_json(data:List[Dict[str,any]], path:str):
    """
    Saves a list of dictionaries to a json file
    """
    
    with open(path, 'w') as f:
        json.dump(data, f)


def load_files_from_path_merge_and_save(path:str, save_path:str,extension:str=None):
    """
    Loads a list of dictionaries from a list of json files, merges them and saves them to a json file
    """
    
    dicts = get_dicts_from_jsons(get_files_from_path(path,extension=extension))
    merged_dict = merge_dicts(dicts)
    print(merged_dict)
    data_dict = {}
    for entry in merged_dict:
        for model, model_data in entry.items():
            if model not in data_dict:
                data_dict[model] = {}
            for band, method_data in model_data.items():
                if band not in data_dict[model]:
                    data_dict[model][band] = {}
                for method, metrics in method_data[0].items():
                    data_dict[model][band][method] = metrics
    save_to_json(data_dict, save_path)




if __name__=="__main__":
    load_path = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results"
    save_path = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results_merged.json"
    load_files_from_path_merge_and_save(load_path,save_path=save_path,extension=".json")