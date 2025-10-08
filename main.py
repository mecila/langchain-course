from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from callbackhandler import AgentCallbackHandler

load_dotenv()


@tool
def check_length(text: str) -> int:
    """Returns the length of the input text and returns it as an integer"""
    print(f"🔧 Tool Input: {text}")
    # Clean the text by stripping quotes and newlines
    cleaned_text = text.strip("'\n").strip("'")
    length = len(cleaned_text)
    print(f"📏 Text length: {length}")
    return length


def create_tool_bound_agent():
    """Create a modern LangChain agent using tool.bind"""

    # Define available tools
    tools = [check_length]

    # Create LLM with callbacks
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        callbacks=[AgentCallbackHandler()],
    )

    # Bind tools to LLM - this enables automatic tool calling
    llm_with_tools = llm.bind_tools(tools)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant with access to tools.
        
When you need to calculate text length, use the check_length tool.
Always think step by step and explain your reasoning.
Provide clear, helpful answers to user questions.""",
            ),
            ("user", "{input}"),
        ]
    )

    # Create the agent chain
    agent = prompt | llm_with_tools

    return agent, tools


def execute_agent_with_tools(agent, tools, query: str, max_iterations: int = 3):
    """Execute agent with automatic tool calling"""

    print(f"🚀 Query: {query}")
    print("=" * 50)

    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 Iteration {iteration}:")

        # Get agent response
        result = agent.invoke({"input": query})
        print(f"🤖 Agent: {result.content}")

        # Check if agent wants to use tools
        if result.tool_calls:
            print(f"\n🔧 Agent requested {len(result.tool_calls)} tool call(s):")

            tool_results = []

            # Execute each tool call
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"  📞 Calling: {tool_name}({tool_args})")

                # Execute the tool
                if tool_name == "check_length":
                    tool_result = check_length.invoke(tool_args)
                    tool_results.append(
                        f"The length of '{tool_args['text']}' is {tool_result}"
                    )
                    print(f"  ✅ Result: {tool_result}")

            # Update query with tool results for next iteration
            if tool_results:
                query = f"Based on these tool results: {'; '.join(tool_results)}. Please provide a complete final answer to the original question."
        else:
            # No more tool calls needed
            print("✨ Agent provided final answer without tools")
            break

    print(f"\n🏁 Completed in {iteration} iterations")
    return result


if __name__ == "__main__":
    print("🎯 Hello React LangChain with Tool Binding!")
    print("Using modern LangChain tool.bind approach")

    # Create agent with bound tools
    agent, tools = create_tool_bound_agent()

    # Test queries
    test_queries = [
        "What is the length of DOG?",
        "What is the length of 'Hello World'?",
        "How many characters are in the word 'Python'?",
    ]

    # Execute each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"🧪 TEST {i}")
        print("=" * 60)

        try:
            execute_agent_with_tools(agent, tools, query)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 All tests completed!")
