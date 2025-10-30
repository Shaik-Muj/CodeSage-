"""Local model loader (placeholder for HF models like CodeT5/StarCoder)."""


def load_model_placeholder(name: str = "codet5-base"):
    return {"model_name": name, "status": "loaded-placeholder"}
