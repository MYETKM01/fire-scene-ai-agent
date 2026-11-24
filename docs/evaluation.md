# Evaluation plan

The current prototype was manually checked by uploading an image, triggering the agent, verifying the C++ program was invoked, and confirming that an enhanced image and report were returned. This is a smoke test, not a formal model evaluation.


## Suggested evaluation before claiming quality

- Tool-call success rate across a fixed set of images and prompts.
- C++ enhancement process success rate and runtime.
- Image metrics before and after enhancement: entropy and average Laplacian magnitude (interpret cautiously; neither alone means perceptual quality is better).
- Human review of scene descriptions for factual accuracy and hallucinations.
- Safety review of recommendations, especially evacuation-first language and avoidance of unsupported firefighting instructions.
- Error cases: unreadable images, missing executable, timeout, unsupported image, API errors, and tool-call loops.

No formal accuracy, safety, or production-readiness result is claimed in this repository.
