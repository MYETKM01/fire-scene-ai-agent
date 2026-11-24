import os
import json
import base64
import logging
from io import BytesIO
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(
    page_title="Fire Scene AI Agent",
    page_icon="🔥",
    layout="wide"
)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")

if not API_KEY:
    st.error("GROQ_API_KEY is missing. Please configure your .env file.")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# Knowledge Base
# ---------------------------
FIRE_KNOWLEDGE = [
    {
        "topic": "smoke",
        "content": (
            "Smoke can reduce visibility and may contain toxic gases. "
            "Avoid entering smoke-filled areas without appropriate training "
            "and protective equipment."
        )
    },
    {
        "topic": "evacuation",
        "content": (
            "When a fire occurs, prioritize evacuation. Follow designated "
            "escape routes, avoid elevators, and do not re-enter a burning building."
        )
    },
    {
        "topic": "fire",
        "content": (
            "If a fire is uncontrolled, evacuate the area and contact local "
            "emergency services. Do not attempt to fight a fire beyond your training."
        )
    },
    {
        "topic": "visibility",
        "content": (
            "Image enhancement can improve visual visibility for analysis, "
            "but enhanced images may introduce artifacts and should not be "
            "treated as ground-truth evidence."
        )
    }
]

# ---------------------------
# Tools
# ---------------------------
def enhance_fire_image(image_bytes):
    exe_path = os.getenv("THESIS_ENHANCER_EXE")

    if not exe_path:
        raise RuntimeError("THESIS_ENHANCER_EXE is not configured in .env")
    # print("🔥 EXE PATH INSIDE FUNCTION:", exe_path)
    # print("🔥 FILE EXISTS:", os.path.exists(exe_path))
    exe_path = Path(exe_path)
    
    if not exe_path.exists():
        raise FileNotFoundError(f"EXE does not exist: {exe_path}")

    # print("🔥 EXE PATH INSIDE FUNCTION:", exe_path)
    # print("🔥 FILE EXISTS:", os.path.exists(exe_path))
    # MSYS2 UCRT64 runtime DLL path
    msys_bin = r"C:\msys64\ucrt64\bin"

    env = os.environ.copy()
    env["PATH"] = msys_bin + os.pathsep + env.get("PATH", "")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        input_path = temp_dir / "input.jpeg"
        output_path = temp_dir / "enhanced.jpeg"

        # Convert uploaded bytes into a standard JPEG file
        Image.open(BytesIO(image_bytes)).convert("RGB").save(input_path, "JPEG")

        command = [
            str(exe_path),
            "--image",
            str(input_path),
            str(output_path)
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(exe_path.parent)
            )

        if process.returncode != 0:
            raise RuntimeError(
                f"C++ exited with code {process.returncode}"
            )

        if not output_path.exists():
            raise RuntimeError(
                f"C++ did not create output file: {output_path}"
            )

        enhanced = Image.open(output_path).convert("RGB")
        return np.array(enhanced)
    
def search_fire_knowledge(query):
    query = query.lower()

    results = []

    for item in FIRE_KNOWLEDGE:
        if item["topic"] in query or any(
            word in item["content"].lower()
            for word in query.split()
        ):
            results.append(item["content"])

    if not results:
        results = [
            "No directly relevant knowledge-base entry was found. "
            "Use caution and consult official fire safety guidance."
        ]

    return "\n".join(results)


def calculate_image_metrics(image_bytes):
    """Return basic, non-diagnostic image measurements."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    gradient = cv2.Laplacian(gray, cv2.CV_64F)
    average_gradient = float(np.mean(np.abs(gradient)))

    return {
        "image_width": int(image.width),
        "image_height": int(image.height),
        "entropy_bits": round(entropy, 4),
        "average_gradient": round(average_gradient, 3)
    }


# ---------------------------
# Agent Tools
# ---------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enhance_fire_image",
            "description": (
                "Run the original C++ fire-scene image enhancement algorithm "
                "implemented in the master's thesis. Use this tool when the uploaded "
                "fire-scene image is affected by smoke or poor visibility and "
                "enhancement is useful."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_fire_knowledge",
            "description": (
                "Search the local fire safety knowledge base "
                "for relevant information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Fire safety topic to search."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_image_metrics",
            "description": (
                "Calculate image resolution, grayscale entropy, and average Laplacian magnitude."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def execute_tool(tool_name, arguments, image_bytes):
    logging.info(
        "Tool called: %s | Arguments: %s",
        tool_name,
        arguments
    )

    if tool_name == "search_fire_knowledge":
        return search_fire_knowledge(
            arguments.get("query", "")
        )

    elif tool_name == "calculate_image_metrics":
        return calculate_image_metrics(image_bytes)

    return {"error": "Unknown tool"}


# ---------------------------
# AI Agent
# ---------------------------
def run_agent(user_question, image_bytes):
    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": """
You are Fire Scene AI Agent, an AI assistant designed to
support fire-scene image analysis and information workflows.

You can:
1. Analyze fire-scene images.
2. Decide when to use image enhancement.
3. Search the fire safety knowledge base.
4. Retrieve basic image metrics.
5. Generate a structured analysis report.

IMPORTANT SAFETY RULES:
- You are NOT an emergency responder.
- Never claim that an image alone can determine whether
  a building is safe.
- Do not give dangerous firefighting instructions.
- Prioritize evacuation and professional emergency services.
- Clearly state uncertainty and limitations.
- Image enhancement may introduce artifacts.
- Never treat image enhancement as proof of hidden objects.

When appropriate, use your available tools before producing
your final response.

Return your final report using these sections:

## Scene Overview
## Visual Observations
## Image Enhancement
## Potential Hazards
## Safety Recommendations
## Limitations

Keep the report practical and clearly communicate uncertainty.
"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/jpeg;base64,"
                            + image_base64
                        )
                    }
                }
            ]
        }
    ]

    enhanced_image = None
    tool_results = []

    for iteration in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=500
        )

        assistant_message = response.choices[0].message

        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        if not assistant_message.tool_calls:
            return (
                assistant_message.content,
                enhanced_image,
                tool_results
            )

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            try:
                if function_name == "enhance_fire_image":
                    enhanced_image = enhance_fire_image(image_bytes)
                    result = {
                        "status": "success",
                        "message": "Original C++ thesis enhancement algorithm executed.",
                        "output_image_available": enhanced_image is not None
                    }
                else:
                    result = execute_tool(
                        function_name,
                        arguments,
                        image_bytes
                    )

                tool_results.append({
                    "tool": function_name,
                    "result": result
                })

                tool_result_content = json.dumps(
                    result,
                    ensure_ascii=False
                )

            except Exception as e:
                logging.exception("Tool execution failed")

                failure_result = {
                    "status": "error",
                    "error": str(e)
                }
                tool_results.append({
                    "tool": function_name,
                    "result": failure_result
                })
                tool_result_content = json.dumps(
                    failure_result,
                    ensure_ascii=False
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result_content
            })

    return (
        "Agent reached the maximum tool execution limit.",
        enhanced_image,
        tool_results
    )


# ---------------------------
# User Interface
# ---------------------------
st.title("🔥 Fire Scene AI Agent")

st.caption(
    "AI-powered fire scene image analysis, "
    "image enhancement, knowledge retrieval, and report generation."
)

st.warning(
    "This prototype is for research and demonstration only. "
    "It must not be used as a substitute for emergency services "
    "or professional fire safety assessment."
)

uploaded_file = st.file_uploader(
    "Upload a fire scene image",
    type=["jpg", "jpeg", "png"]
)

user_question = st.text_area(
    "What would you like the AI Agent to do?",
    value=(
        "Analyze this fire scene. Improve the image visibility "
        "if necessary, identify visible hazards, and generate "
        "a structured report."
    )
)

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    current_image_key = str(hash(image_bytes))
    if st.session_state.get("image_key") != current_image_key:
        st.session_state["image_key"] = current_image_key
        st.session_state["enhanced_image"] = None

    original_image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            original_image,
            use_container_width=True
        )

    with col2:
        st.subheader("AI Enhanced Image")
        if st.session_state.get("enhanced_image") is None:
            st.caption("Run the Agent to generate the enhanced image.")
        else:
            st.image(
                st.session_state["enhanced_image"],
                caption="Enhanced Image - Original C++ Thesis Algorithm",
                use_container_width=True
            )

    if st.button(
        "🚀 Run Fire Scene AI Agent",
        type="primary"
    ):
        with st.spinner(
            "AI Agent is analyzing the image and executing workflow..."
        ):
            report, enhanced_image, tool_results = run_agent(
                user_question,
                image_bytes
            )

        st.success("Agent execution completed!")

        st.session_state["enhanced_image"] = enhanced_image
        if enhanced_image is None:
            st.warning("C++ image enhancement did not produce an output.")

        st.subheader("📋 AI Analysis Report")
        st.markdown(report)

        st.subheader("⚙️ Agent Workflow Execution")

        for item in tool_results:
            with st.expander(
                f"Tool: {item['tool']}"
            ):
                st.json(item["result"])

        st.subheader("📊 Execution Summary")

        st.write({
            "Tools Executed": len(tool_results),
            "Execution Status": "Completed"
        })

        logging.info(
            "Agent execution completed. Tools: %s",
            len(tool_results)
        )

else:
    st.info(
        "Upload a fire-scene image to start the AI Agent."
    )