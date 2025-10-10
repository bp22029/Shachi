import logging

from sotopia.generation_utils.output_parsers import OutputParser, OutputType, PydanticOutputParser
from sotopia.messages import ActionType, AgentAction
from sotopia.utils import format_docstring

log = logging.getLogger(__name__)


def _agenerate__pre(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: float = 0.7,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> OutputType:
    """Generate text using LiteLLM instead of Langchain."""
    # Format template with input values
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()

    # Process template
    template = format_docstring(template)

    # Replace template variables
    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))

    return template


def _agenerate_action__pre(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: float = 0.7,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
):
    if script_like:
        # model as playwright
        template = """
            Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
            You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
            You should try your best to achieve {agent}'s goal in a way that align with their character traits.
            Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
            {history}.
            The script has proceeded to Turn #{turn_number}. Current available action types are
            {action_list}.
            Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.

            Please only generate a JSON string including the action type and the argument.
            Your action should follow the given format:
            {format_instructions}
        """
    else:
        # Normal case, model as agent
        template = """
            Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
            You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
            Note that {agent}'s goal is only visible to you.
            You should try your best to achieve {agent}'s goal in a way that align with their character traits.
            Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
            {history}.
            You are at Turn #{turn_number}. Your available action types are
            {action_list}.
            Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

            Please only generate a JSON string including the action type and the argument.
            Your action should follow the given format:
            {format_instructions}
        """

    return _agenerate__pre(
        model_name=model_name,
        template=template,
        input_values=dict(
            agent=agent,
            turn_number=str(turn_number),
            history=history,
            action_list=" ".join(action_types),
        ),
        output_parser=PydanticOutputParser(pydantic_object=AgentAction),
        temperature=temperature,
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


def _agenerate__post(result: str, output_parser: OutputParser[OutputType]) -> OutputType:
    try:
        parsed_result = output_parser.parse(result)
    except Exception:
        assert False, "TODO(shachi): handle bad output (Refer to `agenerate`)"

    log.info(f"Generated result: {parsed_result}")
    return parsed_result
