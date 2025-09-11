from dotenv import load_dotenv
from langchain import hub
from langchain.agents  import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

tools=[TavilySearch()]
llm=ChatOpenAI(model_name="gpt-4")
react_prompt=hub.pull("hwchase17/react")
agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
chain=agent_executor

def main():
    response = chain.invoke(
        input={"input":"search 3 job postings for an ai engineer using langflow in the bay area of linkedin and list their details",}
        )
    print(response)

    print("Hello from langchain-course!")
   

if __name__ == "__main__":
    main()
