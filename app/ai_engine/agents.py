"""Multi-agent orchestration placeholders."""


class AgentManager:
    def __init__(self):
        self.agents = []

    def register(self, agent):
        self.agents.append(agent)

    def run_all(self, payload):
        return [getattr(a, 'run', lambda p: None)(payload) for a in self.agents]
