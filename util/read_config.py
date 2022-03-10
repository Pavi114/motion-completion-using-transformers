from pathlib import Path
import yaml

def read_config(config_name='default') -> object:
    config = Path(f'./config/{config_name}.yml')

    if not config.exists():
        raise Exception("ConfigNotExistsException")

    with config.open() as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    