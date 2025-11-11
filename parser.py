import json
import os
from datetime import datetime

def parse_bot_file(file_path: str) -> dict:
    """
    Parses Automation Anywhere / UiPath / Pega bot JSON files.
    Extracts metadata like bot name, actions, dependencies, etc.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}

    bot_name = os.path.splitext(os.path.basename(file_path))[0]

    # Try to extract key fields heuristically
    actions = []
    dependencies = []

    def recursive_search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "action" in k.lower() or "command" in k.lower():
                    actions.append(str(v))
                if "dependency" in k.lower():
                    dependencies.append(str(v))
                recursive_search(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_search(item)

    recursive_search(data)

    return {
        "bot_name": bot_name,
        "actions": list(set(actions))[:10],
        "dependencies": list(set(dependencies))[:10],
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
