import os
import json


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File does not exist.")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def process_data(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, list):
            total = 0
            for item in value:
                if isinstance(item, (int, float)):
                    total += item
            result[key] = total / len(value) if value else 0
        elif isinstance(value, dict):
            result[key] = len(value.keys())
        else:
            result[key] = value
    return result


def unsafe_function(user_input):
    # ❌ Intentional security flaw
    return eval(user_input)


def save_results(data, out_file):
    try:
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {out_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    file_path = "data.json"
    try:
        data = load_data(file_path)
        processed = process_data(data)
        save_results(processed, "output.json")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
