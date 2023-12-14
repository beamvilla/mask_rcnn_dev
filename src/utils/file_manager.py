import json
from typing import Dict, Any


def load_json_file(file_path: str) -> Dict[object, object]:
    with open(file_path, "r") as f:
        return json.load(f)
    

def print_prettier_json(json_object: Dict[object, object]) -> None:
    print(json.dumps(json_object, indent=4, ensure_ascii=False))


def save_json_output(data: Dict[Any, Any], output_path: str) -> None:
    output_obj = json.dumps(data, ensure_ascii=False)

    with open(output_path, "w") as f:
        f.write(output_obj)
