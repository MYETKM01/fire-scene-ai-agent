# Architecture and workflow

The application uses a tool-calling loop. The model receives the uploaded image and the user's request, then decides whether to call one or more tools. Python executes the selected tool and sends the result back to the model. The model can then produce a final report.


## Components

- **UI:** Streamlit image upload, prompt, image comparison, report, and tool results.
- **Agent:** Vision-capable model accessed through the Groq API's OpenAI-compatible endpoint.
- **Tool dispatcher:** Python routes tool calls.
- **C++ enhancer:** Existing thesis program launched as a command-line process. Python writes a temporary JPEG, calls the executable, checks its exit code and output file, and reads the result.
- **Knowledge retrieval:** Small static list with simple keyword matching.
- **Image metrics:** Width/height, grayscale entropy, and average Laplacian magnitude.
- **Logging:** Python logging to a local log file.

## Data flow

1. Uploaded image bytes are encoded for the model's image input.
2. Model response may contain tool calls.
3. Python executes each requested tool and captures results/errors.
4. Tool results are appended to the conversation.
5. The model writes the final report.

## Failure handling

- Missing executable configuration raises a clear configuration error.
- Missing executable file raises a file-not-found error.
- C++ process has a timeout.
- Non-zero process exit code or missing output file is treated as failure.
- Tool exceptions are logged and returned to the agent as error information.
