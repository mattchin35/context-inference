from ..agents import agents
from ..task import MDP


def test_Qlearning():
    pass


def test_ForgettingQlearning():
    pass


def test_HMM():
    params = MDP.TaskParams(blocks=1000)
    agent = agents.select_agent(agent_name='HMM', params=params)


def test_Logistic():
    pass