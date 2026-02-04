from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from typing import Literal
from langchain_mistralai import ChatMistralAI

key = "Enter your Api key here"

model = ChatMistralAI(
    model="mistral-large-2512",
    mistral_api_key= key,  # Direct key
    temperature=0.7
)

class ChatBot(TypedDict):
  messages : Annotated[list[BaseMessage],add_messages]

class Sentiment_Analysis(BaseModel):
  sentiment : Annotated[Literal["Approved","Disapproved"],Field(description= "Analyse the sentiment of the user message and analyse that the user is giving approval to the command or disapproved the command")]

model_with_structure = model.with_structured_output(Sentiment_Analysis)

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


def purchase_stock(symbol : str, quantity : str) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol} yes or No")

    if decision == "Approved" :
       return {
          "status" : "success",
          "message" : f"Purchase order place for {quantity} share of {symbol}", # it is a command that is given to the model
          "symbol" : symbol,
          "quantity" : quantity
          }
    else: 
       return {
        "status" : "cancelled",
        "message" : f"Order for {quantity} share of {symbol} is cancelled by human", # it is a command that is given to the model
          "symbol" : symbol,
        "symbol" : symbol,
        "quantity" : quantity 

       }

tools = [get_stock_price, purchase_stock]
model_with_tools = model.bind_tools(tools)

graph = StateGraph(ChatBot)

def chat_node(State : ChatBot):
   message = State["messages"]
   response = model_with_tools.invoke(message)
   return {"messages": response}

tool_node = ToolNode(tools)

graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools","chat_node")

checkpointer = MemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    while True :
        thread_id = 2
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break
        initial_state = {"messages" : [HumanMessage(content = user_input)]}
        result = chatbot.invoke(initial_state,config= {"configurable": {"thread_id": thread_id}})
        interrupts = result.get("__interrupt__",[])
        if interrupts:
            message_send_to_human = interrupts[0].value
            print(f"HITL : {message_send_to_human}")
            user_decision = input("Your decision :")
            user_input = model_with_structure.invoke(user_decision)
            result1 = user_input.model_dump()
            result = chatbot.invoke(
                Command(resume = result1["sentiment"]),
                config ={"configurable": {"thread_id": thread_id}}) 
        message = result["messages"]
        last_message = message[-1]
        print(f"Bot: {last_message.content}\n")