from openenv.core import create_fastapi_app
from server.environment import PromptOptimizerEnvironment
from models import PromptOptimizerAction, PromptOptimizerObservation

app = create_fastapi_app(
    env=PromptOptimizerEnvironment,
    action_cls=PromptOptimizerAction,
    observation_cls=PromptOptimizerObservation,
)
