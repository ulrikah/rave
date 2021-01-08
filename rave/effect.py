import json
from types import SimpleNamespace


def parse_json_effect(effect_json_path: str):
    try:
        with open(effect_json_path, 'r') as file:
            data = file.read()
            fx = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
            return fx
    except json.decoder.JSONDecodeError as error:
        print("Unable to parse effect", effect_json_path)
        raise error


def apply_effect(sound, effect):
    """
    Applies an effect to a sound source by generating a CSound orchestra

    Args:
        sound: path to a .wav file
        effect: an object representation of an effect
    Return:
        path to the generated .wav file
    """
    return


def main():
    PATH = "rave/effects/distortion.json"
    fx = parse_json_effect(PATH)
    [print(p.name, type(p.mapping.min_value), p.mapping.max_value)
     for p in fx.parameters]


if __name__ == "__main__":
    main()
