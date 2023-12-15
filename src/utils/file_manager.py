import json
from typing import Dict, Any, List


def load_json_file(file_path: str) -> Dict[object, object]:
    with open(file_path, "r") as f:
        return json.load(f)
    

def print_prettier_json(json_object: Dict[object, object]) -> None:
    print(json.dumps(json_object, indent=4, ensure_ascii=False))


def save_json_output(data: Dict[Any, Any], output_path: str) -> None:
    output_obj = json.dumps(data, ensure_ascii=False)

    with open(output_path, "w") as f:
        f.write(output_obj)

def write_text_lines_file(lines: List[str], output_path: str) -> None:
    with open(output_path, "w") as file:
        file.writelines(lines)
        file.close()