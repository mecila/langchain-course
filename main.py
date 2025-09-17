from dotenv import load_dotenv
import langchain
langchain.debug = True

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from langchain_tavily import TavilySearch
from schemas import AgentResponse



tools = [TavilySearch()]
llm = ChatOpenAI(model_name="gpt-4")
react_prompt = hub.pull("hwchase17/react")



output_parser=PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tool_names","input", "agent_scratchpad" ],
).partial(format_instructions= output_parser.get_format_instructions())

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extracted_output=RunnableLambda(lambda x: x["output"])
parse_output=RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor|extracted_output|parse_output


def main():
    response = chain.invoke(
        input={
            "input": "search 3 job postings for an ai engineer using langchain in the bay area of linkedin and list their details",
        }
    )

    print(response)



    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
