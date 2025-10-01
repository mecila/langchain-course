from typing import Union

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI

from callbackhandler import AgentCallbackHandler

load_dotenv()


@tool
def check_length(text: str) -> int:
    """returns the length of the input text and returns it as an integer"""
    print(f"Input text: {text}")
    text = text.strip("'\n").strip("'")
    """strips away the non alphanumeric characters from the text just in case"""
    return len(text)


def find_tool_name(tool_name: str, tools: list[Tool]):
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("hello react langchain")
    tools = [check_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
    """
    intermediate_steps = []

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation:"],
        model_name="gpt-4",
        callbacks=[AgentCallbackHandler()],
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "what is the length of DOG ",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(f"Agent step value: {agent_step}")

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_name(tool_name, tools)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(f"Final output: {agent_step.return_values}")
