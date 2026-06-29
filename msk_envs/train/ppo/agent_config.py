import json
from dataclasses import dataclass


@dataclass
class AgentConfig:
    backbone_dims: tuple = (1024, 1024)
    activation_fn: str = "elu"
    fixed_sigma: bool = True

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_json_file(cls, file_path):
        with open(file_path, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def to_dict(self):
        return self.__dict__
