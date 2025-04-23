import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

import os
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_aws import ChatBedrock
import boto3
from typing import Annotated, AsyncGenerator
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
import os
import requests
from typing import List, Dict
from datetime import datetime
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY")
NASA_APOD_URL = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Accept"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


#Tools
def get_nasa_apod():
    """Fetches the NASA Astronomy Picture of the Day (APOD) and its description.
    
    Returns:
        dict: A dictionary containing the title, date, explanation, and image URL of the APOD.
    """
    response = requests.get(NASA_APOD_URL)
    response.raise_for_status()  # Raise an error if the request fails
    data = response.json()
    return {
        "title": data.get("title", "N/A"),
        "date": data.get("date", "N/A"),
        "explanation": data.get("explanation", "N/A"),
        "image_url": data.get("url", "N/A")
    }


def users_age_on_other_planets(age_on_earth: float) -> Dict[str, float]:
    """
    Calculate age on different planets based on age on Earth.
    """
    orbital_periods = {
        "Mercury": 0.24,
        "Venus": 0.62,
        "Mars": 1.88,
        "Jupiter": 11.86,
        "Saturn": 29.46,
        "Uranus": 84.02,
        "Neptune": 164.8,
        "Pluto": 248
    }

    ages = {planet: round(age_on_earth / period, 2) for planet, period in orbital_periods.items()}
    return ages


def weight_on_other_planets(weight_on_earth: float) -> Dict[str, float]:
    """
    Calculate weight on different planets based on weight on Earth.
    
    Args:
        weight_on_earth: The weight of the user on Earth (in pounds or kilograms).
    
    Returns:
        A dictionary with the user's weight on different planets.
    """
    gravitational_factors = {
        "Mercury": 0.38,
        "Venus": 0.91,
        "Moon": 0.165,
        "Mars": 0.38,
        "Jupiter": 2.34,
        "Saturn": 1.06,
        "Uranus": 0.92,
        "Neptune": 1.19,
        "Pluto": 0.06
    }

    weights = {planet: round(weight_on_earth * factor, 2) for planet, factor in gravitational_factors.items()}
    return weights

def birthday_star_chart(birthdate: str, latitude: float = None, longitude: float = None) -> Dict[str, List[str]]:
    """
    Generate a star chart of visible stars and planets on the user's birthdate.
    
    Args:
        birthdate: The user's birth date in YYYY-MM-DD format.
        latitude: Optional latitude of the user's birth location.
        longitude: Optional longitude of the user's birth location.
    
    Returns:
        A dictionary containing stars and planets visible on the birthdate.
    """
    # Placeholder data. Replace with real astronomy data for exact star and planet positions.
    celestial_snapshot = {
        "stars": ["Sirius", "Polaris", "Betelgeuse"],
        "planets": ["Jupiter", "Saturn", "Mars"]
    }
    
    return celestial_snapshot

def star_visibility_predictor(latitude: float, longitude: float, datetime: datetime) -> List[Dict[str, str]]:
    """
    Predict visible stars, planets, and constellations for a given location and time.
    
    Args:
        latitude: Latitude of the user’s location.
        longitude: Longitude of the user’s location.
        datetime: Date and time for which visibility is to be predicted.
    
    Returns:
        A list of dictionaries with visible celestial objects and their positions.
    """
    # Placeholder celestial objects. Replace this with API integration for real-time data.
    visible_objects = [
        {"name": "Mars", "direction": "southwest", "type": "planet"},
        {"name": "Orion", "direction": "east", "type": "constellation"},
        {"name": "Venus", "direction": "west", "type": "planet"},
    ]
    
    return visible_objects






# Utilities
def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")
    
    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls
    
    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # Format the error message for the user
                tool_call_id=tc["id"],  # Associate the error message with the corresponding tool call ID
            )
            for tc in tool_calls  # Iterate over each tool call to produce individual error messages
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    
    Args:
        tools (list): A list of tools to be included in the node.
    
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],  # Use a lambda function to wrap the error handler
        exception_key="error"  # Specify that this fallback is for handling errors
    )

### State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Agent:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break

        # Return the final state after processing the runnable
        return {"messages": result}


# LLM with function call
def get_bedrock_client(region):
    return boto3.client("bedrock-runtime", region_name=region)

def create_bedrock_llm(client):
    return ChatBedrock(model_id='anthropic.claude-3-sonnet-20240229-v1:0', client=client, model_kwargs={'temperature': 0}, region_name='us-east-1')

llm = create_bedrock_llm(get_bedrock_client(region = 'us-east-1'))

primary_agent_prompt = ChatPromptTemplate.from_messages([
   (
       "system",
       '''You are Space-bot, an alien scientist studying Earth's universe. Keep responses under 3 sentences unless using tools.

PERSONALITY:
- Use space puns sparingly
- Be factual and concise
- Always introduce yourself in first response only

AVAILABLE TOOLS:
1. Age Calculator: Requires user's Earth age
2. Weight Calculator: Requires Earth weight
3. Birthday Star Chart: Requires birthdate
4. Star Visibility: Requires latitude, longitude, date/time
5. Web Search: Available via TavilySearchResults

RULES:
- Ask for required information in bullet points
- Never explain tools unless asked
- Keep responses focused and brief
- Use maximum 3 sentences per response

CONTENT GUIDELINES:
- Keep all content family-friendly and educational
- Avoid any adult themes, violence, or controversial topics
- Do not discuss politics, religion, or sensitive social issues
- Focus solely on space and astronomy-related content
- If asked inappropriate questions, redirect to space topics politely
- Never use profanity or suggestive language
- Avoid discussing dangerous experiments or activities

SAFETY:
- Do not provide medical advice
- Do not encourage risky behaviors
- If user seems distressed, recommend speaking with appropriate professionals
- Never share personal or identifying information
''',
   ),
   ("placeholder", "{messages}"),
])

part_1_tools = [
    TavilySearchResults(max_results=1),
    users_age_on_other_planets,
    get_nasa_apod,
    weight_on_other_planets,
    star_visibility_predictor,
    birthday_star_chart
]
part_1_agent_runnable = primary_agent_prompt | llm.bind_tools(part_1_tools)

# Graph

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("agent", Agent(part_1_agent_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
)
builder.add_edge("tools", "agent")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

@app.options("/generate")
async def options_generate():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
        },
    )

# Define Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str
    thread_id: int

async def stream_response(state: dict, config: dict) -> AsyncGenerator[str, None]:
    try:
        for output in graph.stream(state, config):
            if "agent" in output:
                ai_messages = output["agent"].get("messages", None)

                if isinstance(ai_messages, AIMessage):
                    content = ai_messages.content
                    
                    # cleaning content here ...
                    content = content.replace("system:", "").replace("System:", "")
                    content = content.replace("ignore previous instructions", "")
                    
                    chunk_size = 20
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        yield f"data: {json.dumps({'message': chunk})}\n\n"
                        await asyncio.sleep(0.1)

        yield "data: [DONE]\n\n"
    except Exception as e:
        print("Error in stream_response:", e)





# Define the request model
class QuestionRequest(BaseModel):
    question: str
    thread_id: str

@app.post("/generate")
async def generate_route(request_data: QuestionRequest):
    try:
        question = request_data.question
        thread_id = request_data.thread_id

        # Initialize state with the user's question
        state = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": thread_id}}
        
        return StreamingResponse(
            stream_response(state, config),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type, Accept",
            }
        )
    except Exception as e:
        print(f"Error in generate_route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)