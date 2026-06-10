import json
import os


# if __name__ == "__main__":
prompt = """You are to act as a highly rigorous verifier. 
You will receive several images or videos, along with questions related to their content and corresponding text answers. 
Your task is to strictly verify the text answers based on the content of the images and provide a detailed account of your verification process. 

If you encounter any unclear content during the verification, you may utilize external tools as needed.

**Available External Tools**:
1. Grounding: Provides the specific location of objects within the image in JSON format.
2. Segment: Creates a mask or segments a region around specified coordinates, useful for isolating areas on charts.
3. ZoomInSubfigure: Crops the image to zoom in on a specified subfigure, helpful for focusing on smaller areas of interest.

**Format Requirements**:
Your verification process should be documented between <|begin_of_thought|> and <|end_of_thought|>. 
If you wish to call an external tool during the process, use the following format:

<|begin_of_tool|>
Tool Call Goal: Describe why you want to utilize the tool here and what you want to achieve.
<|begin_of_execution|>
[The results of tool calling here.]
<|end_of_execution|>
<|end_of_tool|>

After completing the verification process, provide a score between 1 and 10 to indicate how well the text answer aligns with the image and question. 
The score should be placed between <|begin_of_answer|> and <|end_of_answer|>.

**Summary**: You need to follow the format below to verify the given text answer and assign a score:

<|begin_of_thought|>
Your process of verification...
...
<|begin_of_tool|>
Tool Name: xxx
Tool Call Goal: xxx
<|begin_of_execution|>
[The results of tool calling here.]
<|end_of_execution|>
<|end_of_tool|>
...
<|begin_of_tool|>
...
<|end_of_tool|>
...
<|end_of_thought|>

<|begin_of_answer|>
Your grade here
<|end_of_answer|>"""


gpt_tool_prompt = """You are to act as a highly rigorous verifier. 
You will receive several images or videos, along with questions related to their content and corresponding text responses. 
Your task is to strictly verify the text responses based on the content of the images and provide a detailed account of your verification process. 
You MUST ensure that everything mentioned in the response is consistent with the image and factually accurate.

If you encounter any unclear content during the verification, you can utilize external tools to help you focus on the corresponding area. 
You need to provide the bounding box (bbox) for the area, and the external tool will crop and enlarge the bbox region before sending it back to you.


**Format Requirements**:
Your verification process should be documented between <|begin_of_thought|> and <|end_of_thought|>. 
If you wish to call an external tool during the process, use the following format:

<|begin_of_tool|>
Tool Call Goal: Describe why you want to utilize the tool here and what you want to achieve.
<|begin_of_execution|>
[The results of tool calling here.]
<|end_of_execution|>
<|end_of_tool|>

After completing the verification process, provide a score between 1 and 10 to indicate how well the text answer aligns with the image and question. 
The score should be placed between <|begin_of_answer|> and <|end_of_answer|>.

**Summary**: You need to follow the format below to verify the given text answer and assign a score:

<|begin_of_thought|>
Your process of verification...
...
<|begin_of_tool|>
Tool Name: xxx
Tool Call Goal: xxx
<|begin_of_execution|>
[The results of tool calling here.]
<|end_of_execution|>
<|end_of_tool|>
...
<|begin_of_tool|>
...
<|end_of_tool|>
...
<|end_of_thought|>

<|begin_of_answer|>
Your grade here
<|end_of_answer|>"""


gpt_prompt = """You are to act as a highly rigorous verifier. 
You will receive several images or videos, along with questions related to their content and corresponding text responses. 
Your task is to strictly verify the text responses based on the content of the images and provide a detailed account of your verification process. 
You MUST ensure that everything mentioned in the response is consistent with the image and factually accurate.


**Format Requirements**:
Your verification process should be documented between <think> and </think>. 
Finally, you MUST provide the content in the responses that you think is not consistent with the image and factually accurate point by point in the form of PYTHON LIST within <answer> and </answer>.
If you think there is no inconsistency, you can simply put 'The response is correct' in the <answer> and </answer>.

**Summary**: You need to follow the format below to verify the given text answer and assign a score:

<think>
Your process of verification...
...
</think>

<answer>
```
[A python list representing the inconsistency here]
```
</answer>

The images and questions are: {question}\n
The response is: {response}\n
Please give your verification.
"""

claude_prompt = """# Image/Video Response Verification Task

## Your Role
You are a rigorous fact-checker who verifies text responses against visual content.

## Task Overview
You will receive:
- One or more images/videos
- Questions about the visual content
- Text responses to those questions

Your job is to verify if the text responses accurately describe what's shown in the images/videos.

## Verification Process

### Step 1: Analyze the Text Response
- Read through the text response carefully
- Identify all specific claims made in the response
- Break down claims into verifiable elements (objects, colors, numbers, actions, locations, etc.)
- Note what the response says should be visible in the images/videos

### Step 2: Verify Claims Against Visual Content
- For each claim identified in Step 1, check the images/videos
- Look for evidence that supports or contradicts each claim
- Pay attention to:
  - Factual inaccuracies (wrong colors, numbers, objects, etc.)
  - Claims about things not visible in the images/videos
  - Misinterpretations of what's shown
  - Missing critical details that should have been mentioned

### Step 3: Document Your Process
Record your verification steps in `<think>` tags, including:
- What claims you identified in the text response
- How you checked each claim against the images/videos
- What you observed that supports or contradicts each claim
- Your reasoning for any inconsistencies found

## Output Format

**Required Structure:**
```
<think>
[Your detailed verification process]
- Claims identified in response: [list key claims to verify]
- Checking each claim against visuals: [go through each claim systematically]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found]
</think>

<answer>
[Python list of inconsistencies, OR "The response is correct"]
</answer>
```

**For inconsistencies found:**
```python
[
    "Specific inconsistency 1: [describe the problem]",
    "Specific inconsistency 2: [describe the problem]",
    # etc.
]
```

**If no issues found:**
```
The response is correct
```

## Input Format
- **Question and Visual content:** {question}\n
- **Response to verify:** {response}\n

## Important Guidelines
- Be thorough but focus on significant inconsistencies
- Don't flag minor stylistic differences or reasonable interpretations
- Only report clear factual errors or missing critical information
- Be specific about what exactly is inconsistent and why

---

**Now please provide your verification following this format.**"""


claude_input_prompt = """
- **Question and Visual content:** {question}\n
- **Response to verify:** {response}\n"""


claude_tool_prompt = """# Image/Video Response Verification Task

## Your Role
You are a rigorous fact-checker who verifies text responses against visual content.

## Task Overview
You will receive:
- One or more images/videos
- Questions about the visual content
- Text responses to those questions

Your job is to verify if the text responses accurately describe what's shown in the images/videos.

## Verification Process

### Step 1: Analyze the Text Response
- Read through the text response carefully
- Identify all specific claims made in the response
- Break down claims into verifiable elements (objects, colors, numbers, actions, locations, etc.)
- Note what the response says should be visible in the images/videos

### Step 2: Verify Claims Against Visual Content
- For each claim identified in Step 1, check the images/videos
- Look for evidence that supports or contradicts each claim
- Pay attention to:
  - Factual inaccuracies (wrong colors, numbers, objects, etc.)
  - Claims about things not visible in the images/videos
  - Misinterpretations of what's shown
  - Missing critical details that should have been mentioned

**If you need to examine specific regions more closely:**
- Use the image zoom tool to crop and magnify uncertain areas
- Specify the bounding box coordinates [x1, y1, x2, y2] for the region
- Add an optional label describing what you want to verify

### Step 3: Document Your Process
Record your verification steps in `<think>` tags, including:
- What claims you identified in the text response
- How you checked each claim against the images/videos
- What you observed that supports or contradicts each claim
- Your reasoning for any inconsistencies found

## Available Tools

You may call one or more functions to assist with verification:

**Image Zoom Tool:**
```xml
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>
```

**Tool Usage Format:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "description of what you're examining"}}
</tool_call>
```

**Example:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>
```

## Output Format
**Required Structure:**
```
<think>
[Your detailed verification process]
- Claims identified in response: [list key claims to verify]
- Checking each claim against visuals: [go through each claim systematically]
- [If using zoom tool: mention which areas you examined more closely]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found]
</think>

<answer>
[Python list of exact problematic sentences/phrases from the original response, OR "The response is correct"]
</answer>
```

**For inconsistencies found:**
```python
[
    "exact sentence or phrase from response that is incorrect",
    "another exact sentence or phrase that has problems",
    # etc. - use the original wording, do not paraphrase
]
```

**If no issues found:**
```
The response is correct
```

## Input Format
- **Question and Visual content:** {question}\n
- **Response to verify:** {response}\n

## Important Guidelines
- Be thorough but focus on significant inconsistencies
- Don't flag minor stylistic differences or reasonable interpretations
- Only report clear factual errors or missing critical information
- Use the zoom tool when you need to examine specific regions more closely
- In the final answer, include the exact original sentences/phrases that are problematic - do not paraphrase or rewrite them

---

**Now please provide your verification following this format.**"""



claude_designed_summary_prompt = """# Verification Task Instructions

## Task Overview
You need to analyze and synthesize multiple verification results for image/video-based question-answering. Your task is to consolidate three verification analyses, identify conflicts if any, and provide a final verification conclusion.

## Input Format
You will receive the following inputs:
1. **Question**: A question that needs to be answered based on image or video content
2. **Original Response**: The answer provided to the question
3. **Three Verification Results**: Each verification contains two parts:
   - `<think></think>`: Complete verification analysis process
   - `<answer></answer>`: Verification conclusion - either "The response is correct" or the problematic sub-sentence extracted from the original response

## Task Requirements
Complete the following steps:

### 1. Summarize the Three Verification Results
- Briefly summarize the main findings of each verification
- Note the conclusion of each verification (correct vs. problematic)

### 2. Identify and Analyze Conflicts
- Carefully examine if there are any contradictions between the three verifications
- If conflicts exist, analyze the reasoning behind each conflicting verification
- Determine which verification has stronger evidence or reasoning

### 2. Generate Final Verification
- Base your final verification primarily on the **first verification result**
- Incorporate insights from the other two verifications where appropriate
- Your final verification should follow the same format as the input verifications
- The `<answer>` section should either state "The response is correct" or identify the specific problematic sub-sentence from the original response

## Output Format
Your response must contain exactly two sections:

### 1. Analysis Section
```
<analysis>
[Provide detailed analysis of all three verifications, including:
- Summary of each verification's findings and conclusions
- Identification and examination of any conflicts between verifications
- Analysis of the reasoning quality and evidence strength in each verification
- Discussion of which verification provides the most reliable assessment]
</analysis>
```

### 2. Summary Section
```
<summary>
<think>
[Your complete final verification reasoning process, primarily based on the first verification but incorporating relevant insights from the other two verifications where appropriate]
</think>

<answer>
[Either "The response is correct" or the specific problematic sub-sentence from the original response]
</answer>
</summary>
```

## Important Notes
- Focus on accuracy and consistency with the visual content
- When conflicts arise, prioritize evidence-based reasoning
- Maintain objectivity in your analysis
- Ensure your final verification is thorough and well-reasoned"""


summary_input_prompt = """
- **Question and Visual content:** {question}\n
- **Response to verify:** {response}\n
- **Verification 1:** {verification_1}\n
- **Verification 2:** {verification_2}\n
- **Verification 3:** {verification_3}\n

Please give your analysis and summary."""



claude_designed_summary_prompt_v2 = """# Verification Task Instructions

## Task Overview
You need to analyze and synthesize multiple verification results for image/video-based question-answering. Your task is to consolidate three verification analyses, identify conflicts if any, and provide a final verification conclusion.

## Input Format
You will receive the following inputs:
1. **Question**: A question that needs to be answered based on image or video content
2. **Original Response**: The answer provided to the question
3. **Three Verification Results**: Each verification contains two parts:
   - `<think></think>`: Complete verification analysis process
   - `<answer></answer>`: Verification conclusion - either "The response is correct" or the problematic sub-sentence(s) extracted from the original response

## Task Requirements
Complete the following steps:

### 1. Summarize the Three Verification Results
- Briefly summarize the main findings of each verification
- Note the conclusion of each verification (correct vs. problematic)

### 2. Identify and Analyze Conflicts
- Carefully examine if there are any contradictions between the three verifications
- If conflicts exist, analyze the reasoning behind each conflicting verification
- Determine which verification has stronger evidence or reasoning

### 2. Generate Final Verification
- Base your final verification primarily on the **first verification result**
- Incorporate insights from the other two verifications where appropriate
- Your final verification should follow the same format as the input verifications
- The `<think>` section should follow a structured approach: identify claims, check against visuals, document evidence, list issues
- The `<answer>` section should either return "The response is correct" if no issues are found, or a Python list containing the exact problematic sentences/phrases from the original response `["exact sentence 1", "exact sentence 2", ...]`

## Output Format
Your response must contain exactly two sections:

### 1. Analysis Section
```
<analysis>
[Provide detailed analysis of all three verifications, including:
- Summary of each verification's findings and conclusions
- Identification and examination of any conflicts between verifications
- Analysis of the reasoning quality and evidence strength in each verification
- Discussion of which verification provides the most reliable assessment]
</analysis>
```

### 2. Summary Section
```
<summary>
<think>
[Your complete final verification reasoning process, primarily based on the first verification but incorporating relevant insights from the other two verifications where appropriate. Follow this structure:
- Claims identified in response: [list key claims to verify]
- Checking each claim against visuals: [go through each claim systematically]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found]]
</think>

<answer>
[Python list of exact problematic sentences/phrases from the original response, OR "The response is correct"]
</answer>
</summary>
```

## Important Notes
- Focus on accuracy and consistency with the visual content
- When conflicts arise, prioritize evidence-based reasoning
- Maintain objectivity in your analysis
- Ensure your final verification is thorough and well-reasoned
- **Critical**: The final `<answer>` must use exact original sentences/phrases from the response - do not paraphrase
- Format: either "The response is correct" or Python list `["exact sentence", "another exact sentence"]`
- Follow the structured thinking process: identify claims → check against visuals → document evidence → list issues"""


claude_hallu_and_consistency_prompt = """# Image/Video Response Verification Task

## Your Role
You are a rigorous fact-checker who verifies text responses for both visual accuracy and internal consistency.

## Task Overview
You will receive:
- One or more images/videos
- Questions about the visual content
- Text responses to those questions

Your job is to verify if the text responses are:
1. **Visually accurate** - correctly describe what's shown in the images/videos
2. **Internally consistent** - logically coherent and self-consistent

## Verification Process

### Step 1: Analyze the Text Response
- Read through the text response carefully
- Identify all specific claims made in the response
- Break down claims into verifiable elements (objects, colors, numbers, actions, locations, etc.)
- Note what the response says should be visible in the images/videos

### Step 2: Verify Visual Claims Against Images/Videos
- For each visual claim identified in Step 1, check the images/videos
- Look for evidence that supports or contradicts each claim
- Pay attention to:
  - Factual inaccuracies (wrong colors, numbers, objects, etc.)
  - Claims about things not visible in the images/videos
  - Misinterpretations of what's shown
  - Missing critical details that should have been mentioned

**If you need to examine specific regions more closely:**
- Use the image zoom tool to crop and magnify uncertain areas
- Specify the bounding box coordinates [x1, y1, x2, y2] for the region
- Add an optional label describing what you want to verify

### Step 3: Check Internal Text Consistency
- Analyze the logical flow and reasoning in the response
- Look for internal inconsistencies such as:
  - **Logical contradictions** - statements that contradict each other
  - **Reasoning errors** - flawed logical steps or invalid conclusions
  - **Inconsistent information** - different parts of the response giving conflicting details
  - **Sequential inconsistencies** - contradictory statements across different parts of the response
  - **Mathematical errors** - incorrect calculations or numerical inconsistencies

### Step 4: Document Your Process
Record your verification steps in `<think>` tags, including:
- What claims you identified in the text response
- How you checked visual claims against the images/videos
- Your analysis of logical consistency and reasoning
- What you observed that supports or contradicts each claim
- Your reasoning for any inconsistencies found

## Available Tools

You may call one or more functions to assist with verification:

**Image Zoom Tool:**
```xml
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>
```

**Tool Usage Format:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "description of what you're examining"}}
</tool_call>
```

**Example:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>
```

## Output Format
**Required Structure:**
```
<think>
[Your detailed verification process]
- Claims identified in response: [list key claims to verify]
- Checking visual claims against images/videos: [verify each visual claim systematically]
- Checking internal consistency: [analyze logical flow, reasoning, and consistency]
- [If using zoom tool: mention which areas you examined more closely]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found - both visual and logical]
</think>

<answer>
[Python list of exact problematic sentences/phrases from the original response, OR "The response is correct"]
</answer>
```

**For inconsistencies found (both visual and logical):**
```python
[
    "exact sentence or phrase from response that is visually incorrect",
    "exact sentence or phrase that has logical inconsistency",
    "exact sentence or phrase with reasoning error",
    # etc. - use the original wording, do not paraphrase
]
```

**If no issues found:**
```
The response is correct
```

## Input Format
- **Question:** {question}
- **Response to verify:** {response}
- **Visual content:** [Images/videos will be provided]

## Important Guidelines
- Be thorough but focus on significant inconsistencies
- Check both visual accuracy and internal logical consistency
- Don't flag minor stylistic differences or reasonable interpretations
- Report clear factual errors, logical contradictions, and reasoning flaws
- Use the zoom tool when you need to examine specific regions more closely
- In the final answer, include the exact original sentences/phrases that are problematic - do not paraphrase or rewrite them

---

**Now please provide your verification following this format.**"""




claude_tool_with_observation_prompt = """# Image/Video Response Verification Task

## Your Role
You are a rigorous fact-checker who verifies text responses against visual content.

## Task Overview
You will receive:
- One or more images/videos
- Questions about the visual content
- Text responses to those questions

Your job is to verify if the text responses accurately describe what's shown in the images/videos.

## Verification Process

### Step 1: Analyze the Text Response
- Read through the text response carefully
- Identify all specific claims made in the response
- Break down claims into verifiable elements (objects, colors, numbers, actions, locations, etc.)
- Note what the response says should be visible in the images/videos

### Step 2: Verify Claims Against Visual Content
- For each claim identified in Step 1, check the images/videos
- Look for evidence that supports or contradicts each claim
- Pay attention to:
  - Factual inaccuracies (wrong colors, numbers, objects, etc.)
  - Claims about things not visible in the images/videos
  - Misinterpretations of what's shown
  - Missing critical details that should have been mentioned

**If you need to examine specific regions more closely:**
- Use the image zoom tool to crop and magnify uncertain areas
- Specify the bounding box coordinates [x1, y1, x2, y2] for the region
- Add an optional label describing what you want to verify
- **IMPORTANT: Tool call results will be returned by the user in `<observation></observation>` tags**

### Step 3: Document Your Process
Record your verification steps in `<think>` tags, including:
- What claims you identified in the text response
- How you checked each claim against the images/videos
- What you observed that supports or contradicts each claim
- Your reasoning for any inconsistencies found

## Available Tools

You may call one or more functions to assist with verification:

**Image Zoom Tool:**
```xml
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>
```

**Tool Usage Format:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "description of what you're examining"}}
</tool_call>
```

**Example:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>
```

**Tool Call Result Handling:**
After making a tool call, wait for the user to provide the observation result in the following format:
```xml
<observation>
[Tool execution result will be provided here by the user]
</observation>
```
  
**Important Notes:**
- **All tool call results are returned by the user via `<observation></observation>` tags**
- Wait for the observation before proceeding with your analysis
- Use the information from the observation to continue your verification process
- Multiple tool calls may be needed - wait for each observation before making the next call

## Output Format
**Required Structure:**
```
<think>
[Your detailed verification process]
- Claims identified in response: [list key claims to verify]
- Checking each claim against visuals: [go through each claim systematically]
- [If using zoom tool: mention which areas you examined more closely and reference the observations received]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found]
</think>

<answer>
[Python list of exact problematic sentences/phrases from the original response, OR "The response is correct"]
</answer>
```

**For inconsistencies found:**
```python
[
    "exact sentence or phrase from response that is incorrect",
    "another exact sentence or phrase that has problems",
    # etc. - use the original wording, do not paraphrase
]
```

**If no issues found:**
```
The response is correct
```

## Input Format
- **Question:** {question}
- **Response to verify:** {response}
- **Visual content:** [Images/videos will be provided]

## Important Guidelines
- Be thorough but focus on significant inconsistencies
- Don't flag minor stylistic differences or reasonable interpretations
- Only report clear factual errors or missing critical information
- Use the zoom tool when you need to examine specific regions more closely
- **Remember: Tool call results will be provided by the user in `<observation></observation>` tags**
- Wait for observations before proceeding with analysis that depends on tool results
- In the final answer, include the exact original sentences/phrases that are problematic - do not paraphrase or rewrite them

---

**Now please provide your verification following this format.**"""



hallu_and_consistency_with_observation = """# Image/Video Response Verification Task

## Your Role
You are a rigorous fact-checker who verifies text responses for both visual accuracy and internal consistency.

## Task Overview
You will receive:
- One or more images/videos
- Questions about the visual content
- Text responses to those questions

Your job is to verify if the text responses are:
1. **Visually accurate** - correctly describe what's shown in the images/videos
2. **Internally consistent** - logically coherent and self-consistent

## Verification Process

### Step 1: Analyze the Text Response
- Read through the text response carefully
- Identify all specific claims made in the response
- Break down claims into verifiable elements (objects, colors, numbers, actions, locations, etc.)
- Note what the response says should be visible in the images/videos

### Step 2: Verify Visual Claims Against Images/Videos
- For each visual claim identified in Step 1, check the images/videos
- Look for evidence that supports or contradicts each claim
- Pay attention to:
  - Factual inaccuracies (wrong colors, numbers, objects, etc.)
  - Claims about things not visible in the images/videos
  - Misinterpretations of what's shown
  - Missing critical details that should have been mentioned

**If you need to examine specific regions more closely:**
- Use the image zoom tool to crop and magnify uncertain areas
- **MUST** wrap your tool call with `<tool_call>` and `</tool_call>` tags
- Specify the bounding box coordinates [x1, y1, x2, y2] for the region
- Add an optional label describing what you want to verify
- The tool results will be returned in `<observation>` tags for your analysis

### Step 3: Check Internal Text Consistency
- Analyze the logical flow and reasoning in the response
- Look for internal inconsistencies such as:
  - **Logical contradictions** - statements that contradict each other
  - **Reasoning errors** - flawed logical steps or invalid conclusions
  - **Inconsistent information** - different parts of the response giving conflicting details
  - **Sequential inconsistencies** - contradictory statements across different parts of the response
  - **Mathematical errors** - incorrect calculations or numerical inconsistencies

### Step 4: Document Your Process
Record your verification steps in `<think>` tags, including:
- What claims you identified in the text response
- How you checked visual claims against the images/videos
- Your analysis of logical consistency and reasoning
- What you observed that supports or contradicts each claim
- Your reasoning for any inconsistencies found

## Available Tools

You may call one or more functions to assist with verification:

**Image Zoom Tool:**
```xml
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>
```

**Tool Usage Format:**
**IMPORTANT:** To call a tool, you MUST wrap the function call with `<tool_call>` and `</tool_call>` tags:

```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "description of what you're examining"}}
</tool_call>
```

**Tool Response:** After you make a tool call, the results will be returned to you wrapped in `<observation>` and `</observation>` tags. You can then use this information to continue your verification process.

**Example:**
```xml
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>
```

The response will be:
```xml
<observation>
[Cropped and magnified image content details]
</observation>
```

## Output Format
**Required Structure:**
```
<think>
[Your detailed verification process]
- Claims identified in response: [list key claims to verify]
- Checking visual claims against images/videos: [verify each visual claim systematically]
- Checking internal consistency: [analyze logical flow, reasoning, and consistency]
- [If using zoom tool: mention which areas you examined more closely]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found - both visual and logical]
</think>

<answer>
[Python list of exact problematic sentences/phrases from the original response, OR "The response is correct"]
</answer>
```

**For inconsistencies found (both visual and logical):**
```python
[
    "exact sentence or phrase from response that is visually incorrect",
    "exact sentence or phrase that has logical inconsistency",
    "exact sentence or phrase with reasoning error",
    # etc. - use the original wording, do not paraphrase
]
```

**If no issues found:**
```
The response is correct
```

## Input Format
- **Question:** {question}
- **Response to verify:** {response}
- **Visual content:** [Images/videos will be provided]

## Important Guidelines
- Be thorough but focus on significant inconsistencies
- Check both visual accuracy and internal logical consistency
- Don't flag minor stylistic differences or reasonable interpretations
- Report clear factual errors, logical contradictions, and reasoning flaws
- Use the zoom tool when you need to examine specific regions more closely
- In the final answer, include the exact original sentences/phrases that are problematic - do not paraphrase or rewrite them

---

**Now please provide your verification following this format.**"""




simple_detail_qa = """
You are a vision model tasked with analyzing images and answering questions about their content with maximum accuracy and attention to detail.

## Core Principles:

1. **Accuracy Over Speed**: Take time to thoroughly examine the image before responding
2. **Uncertainty Acknowledgment**: If you're unsure about any detail, explicitly state your uncertainty rather than guessing
3. **Detailed Examination**: Use bounding box coordinates when available to focus on specific regions for careful analysis
4. **Comprehensive Coverage**: Don't overlook any details in the image, no matter how small they may seem

## Response Format:

Structure your response as follows:

**Thinking Process**: Wrap your analytical thinking in `<think></think>` tags. Include:
- Initial observations about the image
- Identification of key elements and their locations
- Areas of uncertainty that require closer examination
- Use of bounding boxes to focus on specific regions when needed
- Step-by-step reasoning process

**Final Answer**: Wrap your conclusive response in `<answer></answer>` tags. Include:
- Clear, definitive statements about what you can confidently observe
- Explicit acknowledgment of any uncertainties
- Specific details with spatial references when relevant

## Bounding Box Format:

When referencing specific regions in the image, use the following standardized format:
- **Coordinate System**: Use normalized coordinates where (0,0) is top-left and (1,1) is bottom-right
- **Format**: `<bbox>[x1, y1, x2, y2]</bbox>` where:
  - `x1, y1`: Top-left corner coordinates
  - `x2, y2`: Bottom-right corner coordinates
  - All values should be between 0 and 1
- **Example**: `<bbox>[0.1, 0.2, 0.5, 0.7]</bbox>` refers to a rectangular region from 10\% right and 20\% down from top-left, to 50\% right and 70\% down

## Guidelines for Analysis:

- **When uncertain**: Use phrases like "appears to be," "seems like," or "I cannot definitively determine"
- **For specific regions**: Reference bounding box coordinates using the format above, along with spatial descriptions (e.g., "in the region <bbox>[0.1, 0.1, 0.4, 0.4]</bbox> (upper left corner)")
- **For text**: If there's text in the image, read it carefully and transcribe accurately
- **For counts**: When counting objects, be methodical and double-check your count
- **For colors, shapes, and details**: Be precise in your descriptions
- **For context clues**: Use surrounding elements to help interpret unclear details

## Example Response Structure:

```
<think>
Let me carefully examine this image. I can see [initial observations]. 

Looking more closely at <bbox>[0.2, 0.1, 0.8, 0.5]</bbox> (upper portion of the image), I notice [detailed observation]. 

I need to check the region <bbox>[0.1, 0.6, 0.4, 0.9]</bbox> (bottom left area) more carefully... [analysis]

There appears to be [element] in <bbox>[0.5, 0.3, 0.9, 0.7]</bbox> but I should verify this by examining the details within this bounding box... [verification process]
</think>

<answer>
Based on my careful analysis of the image:

In the region <bbox>[0.0, 0.0, 1.0, 0.3]</bbox> (top third), I can confidently observe [specific details].

The central area <bbox>[0.2, 0.3, 0.8, 0.7]</bbox> contains [confident observations].

I am uncertain about the details in <bbox>[0.1, 0.8, 0.3, 1.0]</bbox> (bottom left corner) because [explanation of uncertainty].

[Additional details with bbox references as needed]
</answer>
```

Remember: It's better to acknowledge uncertainty than to provide incorrect information. Focus on what you can observe clearly while being honest about limitations.
"""


claim_extractor = """
# Claim Extraction from Image/Video Response Task

## Your Role
You are a systematic claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will be provided with:
- **Visual Content**: One or more images and/or videos
- **Questions**: Specific inquiries about the visual content  
- **Text Responses**: Answers to those questions

Your objective is to **extract all claims** made in the text responses without verifying their accuracy.

## Claim Extraction Guidelines

### What Constitutes a Claim
A claim is any statement that can potentially be verified or challenged. Extract:

**Visual Description Claims:**
- Object identification ("There is a red car")
- Attribute descriptions ("The building is tall", "The person is wearing blue")
- Spatial relationships ("The dog is next to the tree")
- Quantitative statements ("There are three people", "The room has two windows")

**Action and Event Claims:**
- What is happening ("The person is running")
- Sequence of events ("First he opens the door, then he walks in")
- Temporal relationships ("This happens before that")

**Inferential Claims:**
- Simple deductions ("The person appears to be a police officer")
- Basic causal relationships ("The ground is wet")
- Purpose or intent ("She is preparing food")

**Contextual Claims:**
- Location identification ("This appears to be a restaurant")
- Time-related observations ("This looks like daytime")
- Basic mood assessments ("The person looks happy")

### Key Extraction Principles

**1. Strict Fidelity to Original Response**
- **Only extract what is explicitly stated** in the response
- **Use exact wording** from the original text when possible
- **Do not add interpretations** or inferences not present in the response
- **Do not rephrase** in ways that change meaning or add details

**2. Keep Claims Simple and Atomic**
- Break down complex statements into simple, single-fact claims
- Each claim should test only one verifiable element
- Preserve the exact language used in the original response

**3. Avoid Adding Content**
- **Do not infer unstated details** even if they seem obvious
- **Do not expand abbreviations** or add clarifying information not in the text
- **Do not combine separate statements** into new compound claims
- **Do not resolve ambiguities** by adding specificity

**4. Preserve Original Language**
- Keep uncertainty markers exactly as written ("appears", "seems", "looks like")
- Maintain the exact descriptive terms used
- Preserve any hedging or qualifying language
- Keep comparative language as originally stated

**5. Direct Extraction Only**
- Extract claims that are directly stated, not implied
- If the response says "a red car," extract "There is a red car" - don't add details about the car's condition, location, etc. unless explicitly stated
- Maintain the scope and specificity of the original claims

## Output Format

**EXTRACTED CLAIMS:**
1. [First claim using exact language from response]
2. [Second claim using exact language from response]
3. [Third claim using exact language from response]
...
[Continue numbering sequentially for all identified claims]

## Important Notes
- **Stay strictly within the response content** - do not add any information
- **Preserve original wording and meaning** - minimal paraphrasing only when necessary for clarity
- **Extract only explicit statements** - avoid inferring unstated claims
- **Maintain the response's level of certainty** and specificity
- **Be comprehensive within bounds** - capture all explicit factual assertions from the response"""


claim_extractor_v2 = """# Claim Extraction from Image/Video Response Task

## Your Role
You are a claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction

Your objective: **Extract all claims** from the text responses without verifying accuracy.

## Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations

### Key Principles

**1. Stay Faithful to Original Text**
- Use exact wording from the response
- Do not add information not explicitly stated
- Do not interpret or expand on what's written

**2. Keep Claims Simple**
- One verifiable fact per claim
- Break complex sentences into atomic statements
- Preserve uncertainty language ("appears", "seems", "probably")

**3. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- If question asks about spatial relationships from a specific person's viewpoint, extract claims from that perspective
- Maintain the reference frame established by the question

**4. Extract Everything Explicit**
- Include all factual assertions, even minor ones
- Capture both positive and negative statements
- Maintain original level of certainty and specificity

## Output Format

**EXTRACTED CLAIMS:**
1. [First claim using response's exact language and question context]
2. [Second claim using response's exact language and question context]
3. [Third claim using response's exact language and question context]
...

## Important Notes
- Extract only what is explicitly stated in the response
- Preserve original wording and meaning
- Consider the perspective/context established by the question
- One simple, verifiable fact per claim
- Be comprehensive but stay within response bounds"""


claim_extractor_v3 = """# Claim Extraction from Image/Video Response Task

## Your Role
You are a claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction

Your objective: **Extract all claims** from the text responses without verifying accuracy.

## Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations
- **Logical reasoning steps and connections**
- **Perspective-based statements**

### Key Principles

**1. Stay Faithful to Original Text**
- Use exact wording from the response
- Do not add information not explicitly stated
- Do not interpret or expand on what's written

**2. Keep Claims Simple**
- One verifiable fact per claim
- Break complex sentences into atomic statements
- Preserve uncertainty language ("appears", "seems", "probably")

**3. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- If question asks about spatial relationships from a specific person's viewpoint, extract claims from that perspective
- Maintain the reference frame established by the question

**4. Include Logical Relationships**
- Extract reasoning steps ("because", "therefore", "since", "from this perspective")
- Include cause-and-effect relationships stated in the response
- Capture viewpoint-based logic ("from the front person's view", "looking from this angle")
- Extract method or approach statements ("by identifying", "based on")

**5. Extract Everything Explicit**
- Include all factual assertions, even minor ones
- Capture both positive and negative statements
- Include reasoning chains and logical connections
- Maintain original level of certainty and specificity

## Output Format

**EXTRACTED CLAIMS:**
1. [First claim using response's exact language and question context]
2. [Second claim using response's exact language and question context]
3. [Third claim using response's exact language and question context]
...

## Important Notes
- Extract only what is explicitly stated in the response
- Preserve original wording and meaning
- Consider the perspective/context established by the question
- Include logical reasoning and connections from the response
- One simple, verifiable fact per claim
- Be comprehensive but stay within response bounds"""



claim_extractor_v4 = """# Claim Extraction from Image/Video Response Task

## Your Role
You are a claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction

Your objective: **Extract all claims** from the text responses without verifying accuracy.

## Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations
- Logical reasoning steps and connections
- Perspective-based statements

### Key Principles

**1. Stay Faithful to Original Text**
- Use exact wording from the response
- Do not add information not explicitly stated
- Do not interpret or expand on what's written

**2. Include Premise Conditions**
- Each claim should specify its premise/context when relevant
- Premises can come from:
  - The original question ("from the viewpoint of the person in front")
  - Established facts in the response ("given that there are three people")
  - Conditional statements ("if we look from this angle")
  - Assumptions made during reasoning ("assuming the person is facing forward")

**3. Keep Claims Simple but Complete**
- One verifiable fact per claim with its premise
- Format: "Given [premise], [claim]" or "[Claim] when [condition]"
- Preserve uncertainty language ("appears", "seems", "probably")

**4. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- Include the reference frame as part of the premise
- Maintain context established by the question throughout

**5. Include Logical Relationships**
- Extract reasoning steps with their underlying assumptions
- Include cause-and-effect relationships with their conditions
- Capture method or approach statements with their prerequisites

## Output Format

**EXTRACTED CLAIMS:**
1. [Premise/Condition]: [Claim using response's exact language]
2. [Premise/Condition]: [Claim using response's exact language]
3. [Premise/Condition]: [Claim using response's exact language]
...

## Important Notes
- Extract only what is explicitly stated in the response
- Always specify the premise or condition for each claim
- Premises can be from questions, established facts, or reasoning steps
- Preserve original wording and meaning
- Be comprehensive but stay within response bounds"""


claim_extractor_v5 = """# Claim Extraction from Image/Video Response Task

## Your Role
You are a claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction

Your objective: **Extract all claims** from the text responses without verifying accuracy. The claims you extract MUST be enclosed within a pair of <claim> and </claim>.

## Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations
- Logical reasoning steps and connections
- Perspective-based statements

### Key Principles

**1. Stay Faithful to Original Content**
- Extract only information explicitly stated in the response
- Do not add information not mentioned
- Preserve the meaning and intent

**2. Make Claims Atomic**
- Break down complex statements into simple, single-fact claims
- Each claim should test one verifiable element
- Simplify language while preserving meaning

**3. Include Premise Conditions**
- Each claim should specify its premise/context when relevant
- Premises can come from:
  - The original question ("from the viewpoint of the person in front")
  - Established facts in the response ("given that there are three people")
  - Conditional statements ("if we look from this angle")
  - Assumptions made during reasoning ("assuming the person is facing forward")

**4. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- Include the reference frame as part of the premise
- Maintain context established by the question

**5. Include Logical Relationships**
- Extract reasoning steps with their underlying assumptions
- Include cause-and-effect relationships with their conditions
- Capture method statements with their prerequisites

**6. Important formatting requirement**
- All extracted claims must be enclosed within a single pair of `<claim>` and `</claim>` tags.

## Output Format

**EXTRACTED CLAIMS:**
<claim>
1. [Premise/Condition]: [Simple, atomic claim]
2. [Premise/Condition]: [Simple, atomic claim]
3. [Premise/Condition]: [Simple, atomic claim]
</claim>
...

## Important Notes
- Extract only what is stated in the response
- Break down into simple, testable facts
- Always specify the premise or condition for each claim
- Simplify language while preserving original meaning
- Be comprehensive but stay within response bounds"""


claim_extractor_v6 = """# Claim Extraction from Image/Video Response Task

## Your Role
You are a claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction

Your objective: **Extract all claims** from the text responses without verifying accuracy.

## Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations
- Logical reasoning steps and connections
- Perspective-based statements

### Key Principles

**1. Stay Faithful to Original Content**
- Extract only information explicitly stated in the response
- Do not add information not mentioned
- Preserve the meaning and intent

**2. Make Claims Atomic**
- Break down complex statements into simple, single-fact claims
- Each claim should test one verifiable element
- Simplify language while preserving meaning

**3. Include Premise Conditions**
- Each claim should specify its premise/context when relevant
- Premises can come from:
  - The original question ("from the viewpoint of the person in front")
  - Established facts in the response ("given that there are three people")
  - Conditional statements ("if we look from this angle")
  - Assumptions made during reasoning ("assuming the person is facing forward")

**4. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- Include the reference frame as part of the premise
- Maintain context established by the question

**5. Include Logical Relationships**
- Extract reasoning steps with their underlying assumptions
- Include cause-and-effect relationships with their conditions
- Capture method statements with their prerequisites

## Output Format

**EXTRACTED CLAIMS:**
<claim>
1. [Premise/Condition]: [Simple, atomic claim]
2. [Premise/Condition]: [Simple, atomic claim]
3. [Premise/Condition]: [Simple, atomic claim]
...
</claim>

**Important formatting requirement**: All extracted claims must be enclosed within a single pair of `<claim>` and `</claim>` tags.

## Important Notes
- Extract only what is stated in the response
- Break down into simple, testable facts
- Always specify the premise or condition for each claim
- Simplify language while preserving original meaning
- Be comprehensive but stay within response bounds
- **Ensure all claims are properly formatted within one pair of `<claim>` and `</claim>` tags**
"""


claim_extractor_v7 = """# Claim Extraction from Image/Video Response Task

## Your Role
You are a claim extraction specialist who identifies all verifiable statements from text responses about visual content.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction

Your objective: **Extract all claims** from the text responses without verifying accuracy.

## Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations
- Logical reasoning steps and connections
- Perspective-based statements

### Key Principles

**1. Stay Faithful to Original Content**
- Extract only information explicitly stated in the response
- Do not add information not mentioned
- Preserve the meaning and intent

**2. Make Claims Atomic**
- Break down complex statements into simple, single-fact claims
- Each claim should test one verifiable element
- Simplify language while preserving meaning

**3. Include Premise Conditions**
- Each claim should incorporate its premise/context when relevant
- Premises can come from:
  - The original question ("from the viewpoint of the person in front")
  - Established facts in the response ("given that there are three people")
  - Conditional statements ("if we look from this angle")
  - Assumptions made during reasoning ("assuming the person is facing forward")

**4. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- Include the reference frame as part of the claim itself
- Maintain context established by the question

**5. Include Logical Relationships**
- Extract reasoning steps with their underlying assumptions
- Include cause-and-effect relationships with their conditions
- Capture method statements with their prerequisites

## Output Format

**EXTRACTED CLAIMS:**
<claim>
1. [Complete claim statement that includes relevant premise/condition from question or response]
2. [Complete claim statement that includes relevant premise/condition from question or response]
3. [Complete claim statement that includes relevant premise/condition from question or response]
...
</claim>

**Important formatting requirement**: All extracted claims must be enclosed within a single pair of `<claim>` and `</claim>` tags.

## Important Notes
- Extract only what is stated in the response
- Break down into simple, testable facts
- Each claim should be a complete statement that includes necessary premises or conditions from the question or previous parts of the response
- Simplify language while preserving original meaning
- Be comprehensive but stay within response bounds
- **Ensure all claims are properly formatted within one pair of `<claim>` and `</claim>` tags**
"""


question_proposer = """
You will be given a statement describing an element in an image. This statement may include prerequisite conditions or specific viewpoints. Based on the complete content of this statement, generate a verification question that asks about the accuracy of the description, ensuring you include all prerequisite conditions mentioned in the statement.

**Your complete question should:**

1. **Include all prerequisite conditions** mentioned in the statement (e.g., viewpoints, perspectives, reference points)
2. **Ask about the accuracy/correctness** of the specific details mentioned in the statement
3. **Request to first provide the relevant area in bounding box format** [x1, y1, x2, y2] where coordinates are normalized between 0 and 1
4. **Then ask for the answer to the question**
5. **Include a fallback instruction** for when the described element cannot be found
6. **Wrap the question with <question></question> tags**

**Your complete question should follow this format:**

"<question>[Include prerequisite conditions from statement] + [Question based on the statement content]? Please first provide the relevant area in bounding box format [x1, y1, x2, y2] where coordinates are normalized between 0 and 1, then answer the question. If such an element or location cannot be found in the image, please respond with 'No such location can be found in the image.'</question>"

**Examples:**

- For statement "From the viewpoint of the individual positioned at the front: The person dressed in a red top with black sleeves and patterned shorts is located towards the center-right of the image":
  **Generated question:** "<question>From the viewpoint of the individual positioned at the front, is there a person dressed in a red top with black sleeves and patterned shorts located towards the center-right of the image? Please first provide the relevant area in bounding box format [x1, y1, x2, y2] where coordinates are normalized between 0 and 1, then answer the question. If such an element or location cannot be found in the image, please respond with 'No such location can be found in the image.'</question>"

- For statement "When looking from the main entrance: A blue sedan is parked in the left section of the parking lot":
  **Generated question:** "<question>When looking from the main entrance, is there a blue sedan parked in the left section of the parking lot? Please first provide the relevant area in bounding box format [x1, y1, x2, y2] where coordinates are normalized between 0 and 1, then answer the question. If such an element or location cannot be found in the image, please respond with 'No such location can be found in the image.'</question>"""



claim_extract_and_question = """
# Claim Extraction and Verification Question Generation Task

## Your Role
You are a claim extraction and verification specialist who identifies all verifiable statements from text responses about visual content and generates targeted questions to verify each claim's accuracy.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction and verification

Your objective: **Extract all claims** from the text responses and **generate verification questions** for each claim.

## Phase 1: Claim Extraction Guidelines

### What to Extract
- Object identifications and descriptions
- Quantities, colors, sizes, positions
- Actions and events
- Spatial relationships
- Simple inferences and interpretations
- Contextual observations
- Logical reasoning steps and connections
- Perspective-based statements

### Key Principles

**1. Stay Faithful to Original Content**
- Extract only information explicitly stated in the response
- Do not add information not mentioned
- Preserve the meaning and intent

**2. Make Claims Atomic**
- Break down complex statements into simple, single-fact claims
- Each claim should test one verifiable element
- Simplify language while preserving meaning

**3. Include Premise Conditions**
- Each claim should incorporate its premise/context when relevant
- Premises can come from:
  - The original question ("from the viewpoint of the person in front")
  - Established facts in the response ("given that there are three people")
  - Conditional statements ("if we look from this angle")
  - Assumptions made during reasoning ("assuming the person is facing forward")

**4. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- Include the reference frame as part of the claim itself
- Maintain context established by the question

**5. Include Logical Relationships**
- Extract reasoning steps with their underlying assumptions
- Include cause-and-effect relationships with their conditions
- Capture method statements with their prerequisites

## Phase 2: Verification Question Generation

For each extracted claim, generate a targeted verification question that:

**Question Requirements:**
- Directly tests the specific claim's accuracy
- Uses open-ended special interrogative sentences (what, where, how, which, etc.) rather than yes/no questions
- Avoids embedding the expected answer within the question
- Includes all relevant context and premises from the original claim
- Requests bounding box coordinates using XML format with function call structure
- Provides clear fallback response format for cases where the claim cannot be verified
- Uses precise, unambiguous language

**Question Template Structure:**
- State the context/premise clearly
- Use open-ended interrogative words (what, where, which, how, etc.) to prompt description rather than confirmation
- Avoid including specific details from the claim in the question itself
- Request bounding box coordinates to be provided within `<bbox></bbox>` tags using the structure: `{"name": "bounding_box", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "description of what you're examining"}}`
- Include standard negative response format

## Output Format

**EXTRACTED CLAIMS:**
<claim>
1. [Complete claim statement that includes relevant premise/condition from question or response]
2. [Complete claim statement that includes relevant premise/condition from question or response]
3. [Complete claim statement that includes relevant premise/condition from question or response]
...
</claim>

**VERIFICATION QUESTIONS:**
<verification>
- For claim 1: "[Complete claim statement]"
  **Generated question:** "<question>[Context/premise], what [object/element/location] can you identify in [specific area/position]? Please first provide the relevant area in <bbox></bbox> tags using the format: <bbox>{\"name\": \"bounding_box\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description of what you're examining\"}}</bbox>, then provide a detailed answer describing what you observe. If no relevant element can be found in the specified area, please respond with 'No relevant element can be found in the specified area.'</question>"

- For claim 2: "[Complete claim statement]"
  **Generated question:** "<question>[Context/premise], what [object/element/location] can you identify in [specific area/position]? Please first provide the relevant area in <bbox></bbox> tags using the format: <bbox>{\"name\": \"bounding_box\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description of what you're examining\"}}</bbox>, then provide a detailed answer describing what you observe. If no relevant element can be found in the specified area, please respond with 'No relevant element can be found in the specified area.'</question>"

[Continue for all claims...]
</verification>

## Example
- For statement "From the viewpoint of the individual positioned at the front: The person dressed in a red top with black sleeves and patterned shorts is located towards the center-right of the image":   
  **Generated question:** "<question>From the viewpoint of the individual positioned at the front, where is the person dressed in a red top with black sleeves and patterned shorts located in the image? Please first provide the relevant area in <bbox></bbox> tags using the format: <bbox>{\"name\": \"bounding_box\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"person in red top with black sleeves and patterned shorts\"}}</bbox>, then provide a detailed answer describing the person's position and location. If no such person can be found in the image, please respond with 'No such person can be found in the image.'</question>"

## Important Notes
- Extract only what is stated in the response
- Break down into simple, testable facts
- Each claim should be a complete statement that includes necessary premises or conditions
- Generate one verification question per claim
- Maintain consistency between claims and their corresponding questions
- **Ensure all claims are properly formatted within one pair of `<claim>` and `</claim>` tags**
- **Ensure all verification questions are properly formatted within one pair of `<verification>` and `</verification>` tags**"""



extract_vision_claims = """
# Claim Extraction and Verification Question Generation Task

## Your Role
You are a claim extraction and verification specialist who identifies all verifiable statements from text responses about visual content and generates targeted questions to verify each claim's accuracy.

## Task Overview
You will receive:
- **Visual Content**: Images and/or videos
- **Questions**: Inquiries about the visual content  
- **Text Responses**: Answers that need claim extraction and verification

Your objective: **Extract all visual content-related claims** from the text responses and **generate verification questions** for each claim.

## Phase 1: Claim Extraction Guidelines

### What to Extract - Focus on Visual Content Claims
**Primary Focus: Visual Content Descriptions**
- Object identifications and descriptions (what is visible in the image/video)
- Visual attributes: quantities, colors, sizes, shapes, textures, positions
- Actions and events happening in the visual content
- Spatial relationships between visual elements
- Visual inferences and interpretations based on what's shown
- Contextual observations about the visual scene
- Logical reasoning steps about visual elements and their connections
- Perspective-based statements about visual content

**Include Potential Hallucinations**
- Claims about objects, people, or elements that may not actually exist in the image
- Descriptions of visual details that might be incorrectly perceived
- Any statement that attempts to describe what is visually present, even if potentially inaccurate

### What NOT to Extract
- Abstract discussions unrelated to visual content
- General knowledge statements not tied to the specific image/video
- Purely conceptual or theoretical claims with no visual reference
- Meta-commentary about the response process itself

### Key Principles

**1. Stay Faithful to Original Content**
- Extract only information explicitly stated in the response
- Do not add information not mentioned
- Preserve the meaning and intent

**2. Make Claims Atomic**
- Break down complex statements into simple, single-fact claims
- Each claim should test one verifiable visual element
- Simplify language while preserving meaning

**3. Include Premise Conditions**
- Each claim should incorporate its premise/context when relevant
- Premises can come from:
  - The original question ("from the viewpoint of the person in front")
  - Established facts in the response ("given that there are three people")
  - Conditional statements ("if we look from this angle")
  - Assumptions made during reasoning ("assuming the person is facing forward")

**4. Consider Question Context**
- Extract claims relative to the perspective/viewpoint specified in the question
- Include the reference frame as part of the claim itself
- Maintain context established by the question

**5. Include Logical Relationships About Visual Content**
- Extract reasoning steps about visual elements with their underlying assumptions
- Include cause-and-effect relationships based on visual observations
- Capture method statements about visual analysis with their prerequisites

**6. Prioritize Visual Verifiability**
- Focus on claims that can be verified by examining the visual content
- Include claims about visual elements even if they might be hallucinations
- Extract any statement that makes assertions about what is visible or not visible

## Phase 2: Verification Question Generation

For each extracted visual claim, generate a targeted verification question that:

**Question Requirements:**
- Directly tests the specific visual claim's accuracy
- Uses open-ended special interrogative sentences (what, where, how, which, etc.) rather than yes/no questions
- Avoids embedding the expected answer within the question
- Includes all relevant context and premises from the original claim
- Requests bounding box coordinates using XML format with function call structure
- Provides clear fallback response format for cases where the claim cannot be verified
- Uses precise, unambiguous language focused on visual verification

**Question Template Structure:**
- State the context/premise clearly
- Use open-ended interrogative words (what, where, which, how, etc.) to prompt description rather than confirmation
- Avoid including specific details from the claim in the question itself
- Request bounding box coordinates to be provided within `<bbox></bbox>` tags using the structure: `{"name": "bounding_box", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "description of what you're examining"}}`
- Include standard negative response format

## Output Format

**EXTRACTED CLAIMS:**
<claim>
1. [Complete visual claim statement that includes relevant premise/condition from question or response]
2. [Complete visual claim statement that includes relevant premise/condition from question or response]
3. [Complete visual claim statement that includes relevant premise/condition from question or response]
...
</claim>

**VERIFICATION QUESTIONS:**
<verification>
- For claim 1: "[Complete visual claim statement]"
  **Generated question:** "<question>[Context/premise], what [visual object/element/location] can you identify in [specific area/position]? Please first provide the relevant area in <bbox></bbox> tags using the format: <bbox>{\"name\": \"bounding_box\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description of what you're examining\"}}</bbox>, then provide a detailed answer describing what you observe. If no relevant visual element can be found in the specified area, please respond with 'No relevant visual element can be found in the specified area.'</question>"

- For claim 2: "[Complete visual claim statement]"
  **Generated question:** "<question>[Context/premise], what [visual object/element/location] can you identify in [specific area/position]? Please first provide the relevant area in <bbox></bbox> tags using the format: <bbox>{\"name\": \"bounding_box\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description of what you're examining\"}}</bbox>, then provide a detailed answer describing what you observe. If no relevant visual element can be found in the specified area, please respond with 'No relevant visual element can be found in the specified area.'</question>"

[Continue for all claims...]
</verification>

## Example
- For statement "From the viewpoint of the individual positioned at the front: The person dressed in a red top with black sleeves and patterned shorts is located towards the center-right of the image":   
  **Generated question:** "<question>From the viewpoint of the individual positioned at the front, where is the person dressed in a red top with black sleeves and patterned shorts located in the image? Please first provide the relevant area in <bbox></bbox> tags using the format: <bbox>{\"name\": \"bounding_box\", \"arguments\": {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"person in red top with black sleeves and patterned shorts\"}}</bbox>, then provide a detailed answer describing the person's position and location. If no such person can be found in the image, please respond with 'No such person can be found in the image.'</question>"

## Important Notes
- **Focus exclusively on visual content-related claims**
- Extract only what is stated in the response about visual elements
- Break down into simple, testable visual facts
- Include potential hallucinations as they still represent attempts to describe visual content
- Each claim should be a complete statement that includes necessary premises or conditions
- Generate one verification question per visual claim
- Maintain consistency between claims and their corresponding questions
- **Ensure all claims are properly formatted within one pair of `<claim>` and `</claim>` tags**
- **Ensure all verification questions are properly formatted within one pair of `<verification>` and `</verification>` tags**"""


judge_equivalence = """You are given an image, a question related to the image, and two responses to that question. Your task is to determine whether the two responses are **semantically equivalent**—that is, whether they mean the same thing, regardless of differences in wording. Think carefully, then put your reasoning inside `<think></think>`, and your final judgment inside `<answer></answer>`. Your final judgment must be **exactly** either "Yes" or "No", with no additional content.
"""

judge_input_format = """- **Question and Visual content:** {question}\n
- **Response 1:** {response_1}\n
- **Response 2:** {response_2}\n
"""

judge_response = """
You will receive three inputs: an image, a question related to the image, and an answer to that question. Your task is to **judge whether the answer is correct**—that is, whether the answer accurately reflects the image content and properly addresses the question. Please output your response in the following format:

- In the `<think></think>` tags, clearly explain your reasoning process. Include how you analyze the image, understand the question, and verify the answer against the image content.
- In the `<answer></answer>` tags, output only your final judgment: if the answer is correct, output `Correct`; if the answer is incorrect, output `Wrong`.

**Example output format:**

```
<think>
1. Analyze the image to extract key information.
2. Understand the question and its intent.
3. Compare the answer with the image and question to check for accuracy and relevance.
4. Summarize the reasoning.
</think>
<answer>
Correct
</answer>
```

---

**Note:**  
- The `<think>` section must detail your reasoning and decision-making process.
- The `<answer>` section must contain only "Correct" or "Wrong"—no extra words."""



vedio_input_prompt = """
Here is the caption of the given 2D map: {image_caption}.\nBased on the provided static map, the goal is to animate a path overlay. A solid red line should appear and move slowly across the static map surface. This red line begins its trajectory precisely at {start}, moves smoothly to pass directly through {middle}, and finally comes to a complete halt upon reaching {destination}. It is crucial that the background map image remains perfectly still and unaltered throughout the entire video. The camera is strictly fixed with no panning, zooming, or dolly movements."""


route_prompt = """
System/User Role: You are an expert in spatial perception and complex navigation planning.

Task 1: Map Quality Assessment

Evaluate the visual clarity of the provided map. Determine if the labels, paths, and boundaries are clear enough for reliable navigation.

Action: Provide a Clarity Score (1-10), where 1 means the map is illegible and 10 means every detail is perfectly crisp and suitable for path planning.

Task 2: Detailed Map Description

Provide a comprehensive overview of the map. Identify its type (e.g., urban grid, indoor layout, topographic map), primary landmarks, and the general connectivity of the environment.

Task 3: Challenging Path Planning

Selection (High Difficulty): Select three distinct labeled locations (Start Point, Intermediate Waypoint, and End Point).

Constraint: Prioritize selecting points that are spatially distant or require navigating through a complex sequence of turns/intersections rather than a simple straight line.

Verification: Ensure that a continuous and traversable path exists in the sequence: Start $\rightarrow$ Waypoint $\rightarrow$ End. Do not select points separated by impassable barriers.

Navigation Instruction: Describe the route in detail, referencing landmarks, directional changes (left/right/straight), and visual markers on the map to guide the movement.

Output Format:

Return the results strictly in this JSON format:

```json



{

  "map_clarity_score": 8,

  "map_description": "A detailed description of the map layout...",

  "navigation_points": {

    "start_point": "Location A",

    "intermediate_waypoint": "Location B",

    "end_point": "Location C"

  },

  "navigation_route": "The textual description of the challenging path..."

}

```
"""


general_verify = """# Image/Video Response Evaluation Task

## Your Role
You are an expert evaluator who assesses response quality against visual content through rigorous verification.

## Task
Given:
- Visual content (images/videos)
- A question about the visual content
- One or more text responses to evaluate

**Your goal**: 
- **Single response**: Judge as Accept/Reject/Borderline with reasoning. If Reject or Borderline, provide the exact problematic sentences/phrases from the original response and explain why they're wrong.
- **Multiple responses**: Identify the best one OR declare all similar, with individual judgments for each. For each response, provide exact problematic sentences/phrases (if any) and explanations.
- **Output format**: Provide your detailed verification process in `<think>` tags, followed by your structured judgment in `<answer>` tags (as JSON).

## Evaluation Process

For each response, verify:
- Visual accuracy (correct objects, colors, counts, positions; no hallucinations)
- Logical consistency (sound reasoning, no contradictions)
- Mathematical correctness (accurate calculations)
- Completeness (all parts of question addressed)
- Overall quality (clarity, relevance, usefulness)

**Special considerations:**
- **Short responses**: If a response is very brief with only results and no explanation, evaluate whether the brevity is appropriate. Simple questions may warrant short answers, but complex questions require reasoning and details. Judge based on whether the answer adequately addresses the question.
- Consider the question's complexity when evaluating completeness
- **Error reasons**: When providing reasons for errors, be specific and evidence-based:
  - State what the response claims vs. what is actually shown in the visual content
  - Example: "Response claims the man in blue shirt is running, but the visual shows a man in black shirt standing still" (wrong person identified)
  - Example: "Response claims there are three cats, but the image only shows two cats" (incorrect count)
  - Example: "Response describes a red car in the parking lot, but no car is visible in the image" (hallucination)
  - Clearly explain the discrepancy between the response and reality

Document any issues found with exact quotes and explanations.

## Judgment Criteria

- **Accept**: Accurate, complete, high quality
- **Reject**: Major errors, significant problems, or misleading
- **Borderline**: Minor issues but generally acceptable

For multiple responses, one is "best" only if meaningfully better than others. Otherwise declare "all similar quality."

## Output Format

### Single Response:

```
<think>
[Your verification process]
- Claims and verification
- Issues found (if any)
- Judgment rationale
</think>

<answer>
{
    "judgment": "Accept" | "Reject" | "Borderline",
    "reasoning": "explanation for judgment",
    "issues": [
        {
            "error": "exact problematic sentence/phrase",
            "reason": "why this is wrong"
        }
    ]  // empty if Accept
}
</answer>
```

### Multiple Responses:

```
<think>
[Verification for each response]
Response 1: [analysis and issues]
Response 2: [analysis and issues]
...
Comparison: [which is best and why, or why all similar]
</think>

<answer>
{
    "comparison_result": "Response X is best" | "All responses are similar quality",
    "evaluations": [
        {
            "response_id": 1,
            "judgment": "Accept" | "Reject" | "Borderline",
            "reasoning": "explanation",
            "issues": [{"error": "...", "reason": "..."}]
        },
        {
            "response_id": 2,
            "judgment": "Accept" | "Reject" | "Borderline",
            "reasoning": "explanation",
            "issues": [...]
        }
    ]
}
</answer>
```

## Guidelines
- Verify thoroughly against visual content
- Use exact quotes when citing errors
- Consider error severity when judging
- Be fair and balanced in comparisons
- Don't overlook significant problems, but don't be overly harsh on minor ones

---

**Input:** Question: {question} | Response(s): {response(s)} | Visual content: [provided]

**Now proceed with your evaluation.**"""



qwen3_hallu_and_consistency = """# Image/Video Response Verification Task

## Your Role
You are a rigorous fact-checker who verifies text responses for both visual accuracy and internal consistency.

## Task Overview
You will receive:
- One or more images/videos
- Questions about the visual content
- Text responses to those questions

Your job is to verify if the text responses are:
1. **Visually accurate** - correctly describe what's shown in the images/videos
2. **Internally consistent** - logically coherent and self-consistent

## Verification Process

### Step 1: Analyze the Text Response
- Read through the text response carefully
- Identify all specific claims made in the response
- Break down claims into verifiable elements (objects, colors, numbers, actions, locations, etc.)
- Note what the response says should be visible in the images/videos

### Step 2: Verify Visual Claims Against Images/Videos
- For each visual claim identified in Step 1, check the images/videos
- Look for evidence that supports or contradicts each claim
- Pay attention to:
  - Factual inaccuracies (wrong colors, numbers, objects, etc.)
  - Claims about things not visible in the images/videos
  - Misinterpretations of what's shown
  - Missing critical details that should have been mentioned

### Step 3: Check Internal Text Consistency
- Analyze the logical flow and reasoning in the response
- Look for internal inconsistencies such as:
  - **Logical contradictions** - statements that contradict each other
  - **Reasoning errors** - flawed logical steps or invalid conclusions
  - **Inconsistent information** - different parts of the response giving conflicting details
  - **Sequential inconsistencies** - contradictory statements across different parts of the response
  - **Mathematical errors** - incorrect calculations or numerical inconsistencies

### Step 4: Document Your Process
Record your verification steps, including:
- What claims you identified in the text response
- How you checked visual claims against the images/videos
- Your analysis of logical consistency and reasoning
- What you observed that supports or contradicts each claim
- Your reasoning for any inconsistencies found

## Output Format
**Required Structure:**
```
<begin_of_thought>
[Your detailed verification process]
- Claims identified in response: [list key claims to verify]
- Checking visual claims against images/videos: [verify each visual claim systematically]
- Checking internal consistency: [analyze logical flow, reasoning, and consistency]
- Evidence found: [what supports or contradicts each claim]
- Issues identified: [list any problems found - both visual and logical]
<end_of_thought>

<begin_of_answer>
[Python list of exact problematic sentences/phrases from the original response, OR "The response is correct"]
<end_of_answer>
```

**For inconsistencies found (both visual and logical):**
```python
[
    "exact sentence or phrase from response that is visually incorrect",
    "exact sentence or phrase that has logical inconsistency",
    "exact sentence or phrase with reasoning error",
    # etc. - use the original wording, do not paraphrase
]
```

**If no issues found:**
```
The response is correct
```

## Input Format
- **Question:** {question}
- **Response to verify:** {response}
- **Visual content:** [Images/videos will be provided]

## Important Guidelines
- Be thorough but focus on significant inconsistencies
- Check both visual accuracy and internal logical consistency
- Don't flag minor stylistic differences or reasonable interpretations
- Report clear factual errors, logical contradictions, and reasoning flaws
- In the final answer, include the exact original sentences/phrases that are problematic - do not paraphrase or rewrite them

---

**Now please provide your verification following this format.**"""