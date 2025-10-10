import abc
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Generic, TypeVar

import pydantic


class Message(pydantic.BaseModel, abc.ABC):
    """Abstract base class representing messages exchanged between agents.

    Environment implementers should create a subclass of this and add
    environment-specific information.

    Agent implementers may not directly access this class as they mainly use
    Observation.format_as_prompt_text() to create prompts. For more detailed
    message processing, they will need to access other elements in this class.
    Since it inherits from pydantic.BaseModel, data can be transformed and
    processed as a dict using methods like model_dump().

    Attributes:
        time: The time step when the message was sent.
        src_agent_id: The ID of the source agent (sender).
        dst_agent_id: The ID of the destination agent (receiver). None indicates
            a broadcast message or an unspecified destination.
    """

    time: int = pydantic.Field(
        description="The time step when the message was sent.",
    )
    src_agent_id: int | None = pydantic.Field(
        description="ID of the source agent (sender). None means a message from environment (e.g., role description)."
    )
    dst_agent_id: int | None = pydantic.Field(
        description="ID of the destination agent (receiver). None means broadcast or unspecified destination."
    )


TMessage = TypeVar("TMessage", bound=Message)

TParameters = TypeVar("TParameters", bound=pydantic.BaseModel)


class ToolResponse(pydantic.BaseModel, abc.ABC):
    """Abstract base class representing responses from tool executions.

    Environment implementers should create a subclass of this for each tool,
    adding tool-specific response fields and implementing format_as_prompt_text.

    Agent implementers primarily interact with the formatted text responses via
    format_as_prompt_text, which converts tool responses into a standardized
    text format suitable for inclusion in prompts.

    Since it inherits from pydantic.BaseModel, responses can be transformed and
    processed as dictionaries using methods like model_dump().

    Typical implementations should include:
    - A parameters field containing the original request parameters
    - Result fields specific to the tool's output
    - A format_as_prompt_text method that returns a human-readable representation
    """

    @abc.abstractmethod
    def format_as_prompt_text(self) -> str:
        """Format tool response as readable text.

        This method converts the structured tool response into a standardized
        text format that can be easily included in prompts or displayed to users.

        Returns:
            A human-readable string representation of the tool response.
        """
        raise NotImplementedError()


class Tool(pydantic.BaseModel, Generic[TParameters]):
    """Class representing a function tool that agents can use.

    Tools provide a standardized interface for agents to access external
    functionalities, services, or data. They are defined by a name, description,
    parameter specifications, and the actual function implementation.

    Environment implementers create tools and make them available to agents
    by including them in observations. Tools are typically registered with
    LLM APIs to enable function calling capabilities.

    Agent implementers use tools by parsing their descriptions and parameters,
    and invoking them with appropriate arguments based on the context and task.

    Note:
        Environment implementers are encouraged to design tools to be side-effect free
        (i.e., not modifying environment variables). Tools are called at the agent's
        discretion within Agent.step, which may be evaluated concurrently. Therefore,
        tools must be carefully designed to prevent race conditions and ensure
        reproducibility. The simplest approach is to use tools only for information
        retrieval. Actions that affect the environment should be returned as the
        result of Agent.step rather than implemented as tools.

    Attributes:
        name: The unique identifier for the tool, used by agents and LLMs to reference it.
        description: A human-readable description of what the tool does and when to use it.
        parameters_type: A Pydantic model class specifying the required and optional parameters.
        fun: The callable function that implements the tool's functionality, taking
            parameters of the specified type and returning a ToolResponse.
    """

    name: str = pydantic.Field(
        description="The unique identifier for the tool, used by agents and LLMs to reference it."
    )
    description: str = pydantic.Field(
        description="A human-readable description of what the tool does and when to use it."
    )
    parameters_type: type[TParameters] = pydantic.Field(
        description="A Pydantic model class specifying the required and optional parameters."
    )
    fun: Callable[[TParameters], ToolResponse] = pydantic.Field(
        description="The callable function that implements the tool's functionality."
    )


class Observation(pydantic.BaseModel, abc.ABC, Generic[TMessage]):
    """Abstract base class representing observations received by agents.

    Environment implementers should create a subclass of this and add
    environment-specific information.

    Agent implementers mainly use Observation.format_as_prompt_text() to create
    prompts. For more detailed observation processing, they will need to access
    other elements in this class. Since it inherits from pydantic.BaseModel,
    data can be transformed and processed as a dict using methods like
    model_dump().

    Attributes:
        agent_id: The ID of the agent receiving this observation.
        messages: List of messages received by the agent.
        reward: Optional reward signal associated with this observation.
        response_type: Optional type specification for expected response format.
    """

    agent_id: int = pydantic.Field(
        description="The ID of the agent receiving this observation.",
    )
    messages: list[TMessage] = pydantic.Field(
        description="List of messages received by the agent.",
    )
    reward: float | None = pydantic.Field(
        default=None,
        description="Optional reward signal associated with this observation.",
    )
    response_type: type[pydantic.BaseModel] | None = pydantic.Field(
        default=None,
        description=(
            "Optional type specification for expected response format. "
            "When specified, the agent is expected to return an object of this type, "
            "a string (which is expected to be parseable into this type), or None. "
            "When it is None, the agent is expected to return a string (free form) or None."
        ),
    )
    tools: list[Tool] = pydantic.Field(
        default=[],
        description="List of tools that the agent can use to generate responses.",
    )

    @abc.abstractmethod
    def format_as_prompt_text(self) -> str:
        """Format observation as a text prompt.

        Environment implementers must override this method to provide
        a way to convert observations into text prompts.

        Agent implementers can either use this method to create text prompts
        for LLMs, or they can implement their own custom method to create
        prompts from the observation.

        Returns:
            A string representation of the observation as a prompt.
        """
        raise NotImplementedError()

    def format_as_prompt_payload(self) -> list[dict]:
        """Format observation as a payload for LLM API calls.

        Creates a standardized payload structure containing the text prompt
        that can be used in API requests to language models.

        For text-only environments, no modification is needed. However,
        for multimodal environments (e.g., where images or audio are passed
        to agents), environment implementers should override this method
        to provide the appropriate payload structure.

        Returns:
            A list of dictionaries representing the prompt payload.
        """
        return [
            {
                "type": "text",
                "text": self.format_as_prompt_text(),
            }
        ]


TResult = TypeVar("TResult", bound=pydantic.BaseModel)


class Environment(abc.ABC, Generic[TResult]):
    """Abstract base class representing a simulation environment.

    Environment implementers must override all abstract methods to create
    concrete simulation environments with specific mechanics, rules, and
    interaction patterns.
    """

    @abc.abstractmethod
    def num_agents(self) -> int:
        """Returns the number of agents in the environment.

        Returns:
            The total number of agents in this environment.
        """
        raise NotImplementedError()

    def get_default_agent_configs(self) -> list[dict] | None:
        """Returns the default configurations for each agent.

        This is an optional method so it is without @abc.abstractmethod.

        Returns:
            A list of dictionaries, each containing the default configuration for an agent,
            or None, suggesting there is not such default configs.
        """
        return None

    @abc.abstractmethod
    def done(self) -> bool:
        """Checks if the environment has terminated.

        Returns:
            True if the environment has terminated, False otherwise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def reset(self) -> dict[int, Observation]:
        """Resets the environment to its initial state.

        Returns:
            A dictionary mapping agent IDs to their initial observations.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def step(
        self,
        responses: dict[int, str | pydantic.BaseModel | None],
    ) -> dict[int, Observation]:
        """Advances the simulation by one time step.

        Args:
            responses: A dictionary mapping agent IDs to their responses.
                Responses can be strings (for text-based agents), Pydantic models
                (for structured responses), or None (for agents that don't respond).

        Returns:
            A dictionary mapping agent IDs to their new observations.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_result(self) -> TResult:
        """Returns the result of the simulation.

        Returns:
            The result of the simulation as an instance of TResult.
        """
        raise NotImplementedError()


TAggregatedResult = TypeVar("TAggregatedResult", bound=pydantic.BaseModel)


class Task(abc.ABC, Generic[TResult, TAggregatedResult]):
    @abc.abstractmethod
    async def iterate_environments(self) -> AsyncIterator[Environment[TResult]]:
        """Iterates over environments in the task.

        Returns:
            An asynchronous iterator over environments in the task.
        """
        raise NotImplementedError()

        # We need to yield None here to satisfy the AsyncIterator protocol.
        # https://github.com/python/mypy/issues/17128
        # https://mypy.readthedocs.io/en/stable/more_types.html#asynchronous-iterators
        yield None  # type: ignore

    @abc.abstractmethod
    def aggregate_results(self, results: Sequence[TResult]) -> TAggregatedResult:
        """Aggregates results from multiple environments.

        Args:
            results: A sequence of results from environments.

        Returns:
            The aggregated result.
        """
        raise NotImplementedError()


class BaseMemory(abc.ABC):
    @abc.abstractmethod
    def add_record(self, messages: list[dict[str, str]]) -> None:
        pass

    @abc.abstractmethod
    def retrieve(self, query: str | None = None) -> str:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass


class Agent(abc.ABC):
    """Abstract base class representing an agent in the simulation.

    Agent implementers must override the step method to define the agent's
    behavior and decision-making process.
    """

    @abc.abstractmethod
    async def step(self, observation: Observation) -> str | pydantic.BaseModel | None:
        """Processes an observation and returns a response.

        This method defines the agent's decision-making process. It receives
        an observation from the environment, processes it, and returns a
        response that represents the agent's action or communication.

        The step method can return responses in three forms:
        - A string (for text-based responses): Simple text responses can be returned
          directly as strings.
        - A Pydantic model (for structured responses): When observation.response_type
          is specified, agents can return a response of that type. This may involve
          using the structured output feature of LLM APIs, or parsing text outputs
          into structured data on the agent side.
        - None: If the agent chooses not to take any action or respond in this step.

        Args:
            observation: The observation received from the environment, containing
                messages from other agents and possibly reward signals.

        Returns:
            A string, Pydantic model (of the type specified in observation.response_type),
            or None as described above.
        """
        raise NotImplementedError()

    def update_config(self, kwargs_as_dict: dict) -> None:
        """Optionally update config for this agent."""
        pass
