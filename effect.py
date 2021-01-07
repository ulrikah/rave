import json
from types import SimpleNamespace


def parse_json_effect(effect_json_path: str):
    try:
        with open(PATH, 'r') as file:
            data = file.read()
            fx = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
            return fx
    except:
        print("Unable to parse effect", effect_json_path)


if __name__ == "__main__":
    PATH = "effects/distortion.json"
    fx = parse_json_effect(PATH)
    [print(p.name, type(p.mapping.min_value), p.mapping.max_value)
     for p in fx.parameters]
