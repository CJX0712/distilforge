"""统一错误码 E100~E500。

E100 配置 / E200 数据 / E300 模型 / E400 流水线 / E500 依赖。
"""


class DistilForgeError(Exception):
    code = "E000"

    def __init__(self, msg: str = ""):
        super().__init__(f"[{self.code}] {msg}")
        self.msg = msg


class ConfigError(DistilForgeError):
    code = "E100"


class DataError(DistilForgeError):
    code = "E200"


class ModelError(DistilForgeError):
    code = "E300"


class PipelineError(DistilForgeError):
    code = "E400"


class DependencyError(DistilForgeError):
    code = "E500"
