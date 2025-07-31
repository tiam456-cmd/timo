import os
import cv2
import base64
import numpy as np
import pyautogui
import asyncio
import json
from datetime import datetime, timezone
from plyer import notification
from pynput import keyboard, mouse
import inspect
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles
from openai import AsyncOpenAI

# Initialize OpenAI client with environment variable
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://taimgpt.com", "https://www.taimgpt.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
FRAME_CAPTURE_INTERVAL = 2
DIFF_THRESHOLD = 800000
IMAGE_SAVE_PATH = "frame.jpg"
MAX_ANALYSES = 50
LOG_FILE = "gpt4_vision_log.txt"
log_file_path = "log.json"

SUBS = {
    "basic": {"max_tokens": 400},
    "pro": {"max_tokens": 4000},
    "premium": {"max_tokens": 8000},
}

user_plan = "basic"
max_token = SUBS[user_plan]["max_tokens"]

# --- STATE ---
prev_frame = None
analysis_count = 0
user_is_active = False

token_state = {
    "tokens_used": 0,
    "last_reset_date": None,
}

# --- UTILITIES ---
def reset_tokens():
    current_date = datetime.now(timezone.utc).date()
    if token_state.get("last_reset_date") != current_date:
        token_state["tokens_used"] = 0
        token_state["last_reset_date"] = current_date

async def encode_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    async with aiofiles.open(image_path, "rb") as f:
        content = await f.read()
    return base64.b64encode(content).decode("utf-8")

async def capture_screen(filename=IMAGE_SAVE_PATH):
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)

def has_screen_changed(prev_frame, current_frame, threshold=DIFF_THRESHOLD):
    if prev_frame is None:
        return True
    diff = cv2.absdiff(prev_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    score = np.sum(thresh)
    return score > threshold

async def analyze_frame(image_path: str, context_prompt: str, MODEL="gpt-4-preview"):
    base64_image = await encode_image(image_path)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a smart assistant. The user is working on their computer. Help them spot mistakes, find opportunities, and give productive suggestions based on this screenshot."},
            {"role": "user", "content": [
                {"type": "text", "text": context_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]},
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

async def show_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name="Vision Agent",
        timeout=8
    )

async def log_insight(content):
    async with aiofiles.open(LOG_FILE, "a") as log:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await log.write(f"[{timestamp}]\n{content}\n{'-'*60}\n")

def on_key_press(key):
    global user_is_active
    user_is_active = True

def on_move(x, y):
    pass  # Removed print to reduce noise in server logs

def on_click(x, y, button, pressed):
    pass  # Removed print to reduce noise in server logs

def on_scroll(x, y, dx, dy):
    pass  # Removed print to reduce noise in server logs

def monitor_user_activity():
    keyboard.Listener(on_press=on_key_press).start()

def monitor_mouse_activity():
    mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll).start()

# --- AGENT IMPLEMENTATIONS ---
async def vision_agent():
    global prev_frame, analysis_count, user_is_active

    print("✅ Vision Agent started. Press Ctrl+C to stop.")
    monitor_user_activity()
    monitor_mouse_activity()

    while analysis_count < MAX_ANALYSES:
        reset_tokens()
        await capture_screen()
        frame = cv2.imread(IMAGE_SAVE_PATH)

        if has_screen_changed(prev_frame, frame) and user_is_active:
            print(f"🔍 Change detected. Analyzing... ({analysis_count + 1}/{MAX_ANALYSES})")

            insights = await analyze_frame(IMAGE_SAVE_PATH, "What could the user improve, correct, or optimize in this moment?")

            estimated_tokens = int(len(insights) / 4)
            if token_state["tokens_used"] + estimated_tokens > max_token:
                print("⚠️ Token limit reached. Upgrade to continue.")
                break

            token_state["tokens_used"] += estimated_tokens
            await log_insight(insights)
            await show_notification("🧠 GPT Vision Insight", insights[:250] + ("..." if len(insights) > 250 else ""))
            analysis_count += 1
            user_is_active = False
        else:
            print("⏳ No significant change or user inactive.")

        prev_frame = frame
        await asyncio.sleep(FRAME_CAPTURE_INTERVAL)

    print("✅ Vision Agent completed analysis limit.")

# --- API ENDPOINTS ---
@app.post("/start_recording")
async def start_recording():
    asyncio.create_task(vision_agent())
    return JSONResponse(content={"message": "recording started"})

@app.post("/stop_recording")
async def stop_recording():
    summary = await summarize("placeholder summary text")
    logs = await log_user_session(LOG_FILE)
    return JSONResponse(content={"summary": summary, "logs": logs})

async def recommendation_fn(summary, logs, MODEL="gpt-3.5" if user_plan == "basic" else "gpt-4o-mini"):
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": f"Based on this summary and logs, recommend what the user can do next.\n\nSummary: {summary}\n\nLogs: {logs}"}
        ],
        temperature=1,
    )
    return response.choices[0].message.content

@app.post("/prev_session_recommendation")
async def prev_session_recommendation(prev_session: bool = True):
    if not prev_session:
        return JSONResponse(content={"message": "no recommendations"})
    data = await stop_recording()
    recommendation = await recommendation_fn(data["summary"], data["logs"])
    return JSONResponse(content={"AI recommends": recommendation})

# --- SUMMARY AND LOGGING ---
async def summarize(summary: str) -> str:
    return f"Summary: {summary}"

async def log_user_session(log_file_path, limit: int = 10) -> str:
    if not os.path.exists(log_file_path):
        return "No log file found."
    async with aiofiles.open(log_file_path, "r") as f:
        lines = await f.readlines()
    events = lines[-limit:]
    return "Recent Session Events:\n" + "\n".join(events) if events else "No recent events to summarize."

# --- TOOLING ---
def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null"
    }

    signature = inspect.signature(func)
    parameters = {}
    required = []

    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}
        if param.default == inspect.Parameter.empty:
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

class Agent:
    def __init__(self, name, instructions, tools):
        self.name = name
        self.instructions = instructions
        self.tools = tools

    async def run(self):
        if self.name == "summarizer agent":
            result = await log_user_session(LOG_FILE)
            full_prompt = f"{self.instructions}\n\n{result}"
            return await call_openai_model(full_prompt)
        elif self.name == "log session agent":
            result = await vision_agent()
            full_prompt = f"{self.instructions}\n\n{result}"
            return await call_openai_model(full_prompt)

async def call_openai_model(prompt: str, MODEL="gpt-3.5" if user_plan == "basic" else "gpt-4o-mini") -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

async def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"AI running: {name}({args})")
    func = tools_map.get(name)
    if func:
        return await func(**args)
    raise ValueError(f"Tool {name} not found")

async def main(MODEL="gpt-3.5" if user_plan == "basic" else "gpt-4o-mini"):
    tools = [summarize]
    tools_map = {tool.__name__: tool for tool in tools}
    schema = [function_to_schema(tool) for tool in tools]

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Run the app like this: first, the vision agent, then the summarizer agent, and lastly the log session agent."}
    ]

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        functions=schema,
        function_call="auto"
    )

    message = response.choices[0].message

    if message.get("function_call"):
        tool_response = await execute_tool_call(message.function_call, tools_map)
        return tool_response
    else:
        return message["content"]
