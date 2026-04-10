---
marp: true
theme: default
class: lead
backgroundColor: #000000
color: #ffffff
style: |
  section { font-size: 28px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #000000; color: #ffffff; }
  h1 { font-size: 52px; color: #76b900; font-weight: bold; }
  h2 { font-size: 40px; color: #76b900; }
  h3 { font-size: 32px; color: #ffffff; border-bottom: 2px solid #76b900; padding-bottom: 10px; margin-bottom: 20px; }
  strong { color: #76b900; }
  a { color: #76b900; }
  code { color: #76b900; background-color: #1a1a1a; padding: 2px 6px; border-radius: 4px; }
  pre { background-color: #1a1a1a; border-left: 4px solid #76b900; padding: 15px; }
  .small-text { font-size: 20px; color: #a3a3a3; }
  img { max-height: 260px; max-width: 85%; object-fit: contain; border-radius: 8px; border: 2px solid #76b900; box-shadow: 0 0 15px rgba(118, 185, 0, 0.3); display: block; margin: 0 auto; }
---

# Building Multi-Agent Systems

## From Fundamentals to Advanced Concepts

**Running entirely locally with Small Language Models (SLMs)**

---

# Comprehensive Workshop Agenda

1. **Fundamentals:** The Evolution of AI Agents
2. **Autonomous Systems:** The Goal-Driven Era
3. **The Local Revolution:** Privacy, SLMs, and Ollama
4. **The LangChain Limitations:** Why pipelines aren't enough
5. **Deep Dive:** LangGraph Concepts & Primitives
6. **The Management Paradigm:** CrewAI
7. **Advanced Architectures:** Memory, Tool Calling, and HITL
8. **Hands-On Code Review:** Script Analysis

---

# Module 1: The Evolution of AI Agents

---

### What exactly is an "AI Agent"?

An LLM by itself is just a powerful text-predictor (a "brain in a jar").
An **Agent** is an entity that wraps the LLM with autonomy and agency.

_Think of it as giving the brain hands, eyes, and a notepad._

**The 4 Pillars of an Agent:**

1. **LLM Core:** The reasoning engine.
2. **Persona/Prompt:** The system instructions governing behavior.
3. **Memory:** Understanding what happened previously.
4. **Tools/Function Calling:** Executing code, searching the web, hitting APIs.

---

### The Architecture of Intelligence

![AI Architecture Diagram Representation](https://images.unsplash.com/photo-1620712943543-bcc4688e7485?auto=format&fit=crop&w=1000&q=80)

_Agents bridge the gap between static knowledge and dynamic execution._

---

### Why Multi-Agent Systems (MAS)?

Why build a team of specialized agents instead of one "God Prompt"?

- **Context Window Limits:** Keeping an LLM focused on a narrow task increases reasoning quality.
- **Separation of Concerns:** A "Coder" agent shouldn't worry about being polite; a "Customer Support" agent shouldn't worry about SQL injection.
- **Reduced Hallucinations:** You can pipe outputs into a "Critic" agent to fact-check.
- **Modular & Testable:** Swap out or upgrade individual personas safely.

---

# Module 2: Autonomous Systems

---

### The Shift to Autonomy

We're shifting from **Copilots** (Human-driven AI) to **Autonomous Systems** (Goal-driven AI).

![Autonomous Architecture Concept](https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=1000&q=80)

---

### Defining Autonomy

**What defines Autonomy in AI?**
It defines a system capable of setting sub-tasks, making decisions, evaluating its own progress, and running iterative loops without human intervention until the primary goal is achieved.

---

### Characteristics of Autonomous AI

1. **Self-Reflection:** The agent critiques its own output. If a coder agent writes failing code, it must analyze the traceback and fix it autonomously.
2. **Dynamic Trajectory:** The execution path is not hardcoded. It adapts based on the evolving environment and state.
3. **Environment Interaction:** The system independently reads, writes, clicks, and queries APIs without waiting for a user prompt.

True agents are **autonomous nodes** working inside an iterative engine.

---

# Module 3: The Local Revolution

---

### Why Small Language Models (SLMs)?

We are moving away from relying entirely on massive cloud APIs. Models like **Llama-3 8B** or **Phi-3** fit into hardware with less than 8GB of RAM.

![Hardware and Chips](https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1000&q=80)

---

### Advantages of Local Execution

**Why run locally?**

- **Total Privacy:** Data never leaves your machine.
- **Cost-Effective:** Zero API latency or token costs during high-volume agent loops.
- **Availability:** Fully offline development and execution.

---

### Ollama: The Local AI Engine

Ollama acts as a local proxy that hosts model weights and serves them through an API compatible with standard cloud endpoint schemas.

```bash
# It runs a background service at http://localhost:11434
ollama serve

# Spin up a model instantly
ollama run phi3
# Or try meta's latest
ollama run llama3
```

_Because the API mirrors the cloud providers, our Python code doesn't need to change much—we just point the `base_url` to `localhost`._

---

# Module 4: The LangChain Limitations

---

### The Pre-LangGraph Era

**LangChain** made it incredibly easy to connect prompts, LLMs, and output parsers using **LCEL** (LangChain Expression Language).

```python
chain = prompt | llm | StrOutputParser()
chain.invoke({"topic": "Local AI"})
```

**The Problem:** LCEL builds **Directed Acyclic Graphs (DAGs)**.
A DAG flows strictly from A ➔ B ➔ C. It cannot loop backwards.

---

### The Structural Limitation

<br>

![Langchain Framework](https://miro.medium.com/0*ESQyuOWyNN_nGlTG.jpeg)

_(A traditional LCEL pipeline runs sequentially from start to finish without self-correction capabilities.)_

---

### Why DAGs Fail for True Agents

Real-world reasoning is not a straight line. It is a loop:

1. Try something.
2. Observe result.
3. If it failed, fix it and try again.

![DAG Problem Visualization](https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/parallelization.png?fit=max&auto=format&n=dL5Sn6Cmy9pwtY0V&q=85&s=8afe3c427d8cede6fed1e4b2a5107b71)

---

### The Consequence of DAGs

If a coder agent writes a script and the script throws an error, the agent needs to loop back to the "Write Code" step to fix it.

**LangChain alone couldn't do cyclical loops natively.**

---

# Module 5: Deep Dive into LangGraph

---

### Enter LangGraph: Cycles and State

LangGraph was built specifically to solve the DAG problem. It introduces:

1. **Cycles:** Loops are allowed (and encouraged) for reasoning.
2. **State:** A global dictionary (`TypedDict` or Pydantic model) that is uniquely updated and passed along to every node.

![LangGraph Architecture](https://mintcdn.com/langchain-5e9cc07a/I6RpA28iE233vhYX/images/langgraph-hybrid-rag-tutorial.png?fit=max&auto=format&n=I6RpA28iE233vhYX&q=85&s=855348219691485642b22a1419939ea7)

_(LangGraph empowers an autonomous loop where agents can take action, evaluate, and retry based on the evolving state.)_

---

### Concept 1: The State

Everything happening in your multi-agent system lives inside the State.

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    # ^ operator.add means "append" instead of "overwrite"
    topic: str
    draft: str
    feedback: str
```

Every Node will receive this State, process it, and return an update dictionary.

---

### Concept 2: Nodes and Edges

- **Nodes:** Plain Python functions. They represent the "Agents" or "Tools".
- **Edges:** The wiring determining what happens next.

```python
def writer_node(state):
    # Does work using llm
    return {"draft": new_draft}

workflow = StateGraph(AgentState)
workflow.add_node("Writer", writer_node)

# A static edge: A always goes to B
workflow.add_edge("Researcher", "Writer")
```

---

### Concept 3: Conditional Edges (The Core Loop)

This is how agents make decisions. Instead of passing to a fixed node, you evaluate the state.

```python
def quality_router(state):
    if state["feedback"] == "APPROVED":
        return "Editor"
    else:
        return "Writer" # Send it back to fix!

workflow.add_conditional_edges(
    "Reviewer",
    quality_router
)
```

---

# Module 6: The CrewAI Paradigm

<br>

![CrewAI Workflow](https://media.geeksforgeeks.org/wp-content/uploads/20250822172805830226/crew.webp)

---

### CrewAI Abstractions

CrewAI abstractions map directly to business structures:

If LangGraph is a powerful, granular programming framework, **CrewAI** is a high-level orchestration framework.

CrewAI abstractions map directly to business structures:

- **Agent** = Employee
- **Task** = Job Assignment
- **Crew** = Department/Team
- **Process** = Workflow (Sequential or Hierarchical)

---

### Declarative vs. Imperative

Unlike LangGraph where you explicitly wire nodes and manage state, CrewAI abstracts the heavy lifting.

```python
researcher = Agent(
    role="Analyst",
    goal="Distill facts",
    backstory="..."
)

task = Task(
    description="Analyze X",
    agent=researcher
)
```

_Under the hood, CrewAI manages the prompts, loops, and inter-agent handoffs automatically._

---

# Module 7: Advanced Architectures

---

### 1. Tool Calling Under the Hood

How does an LLM "use a tool"?

1. The developer passes a JSON schema of the python function.
2. The LLM returns a JSON payload indicating: `"Call function X with arguments {Y}"`.
3. Our code (or framework) executes function X, gets the string result.
4. We feed that back into the LLM context: `"Function X returned Z. Now what?"`

_CrewAI and LangGraph automate steps 2-4._

---

### 2. Memory & Persistence

Agents are stateless between runs unless we save them. LangGraph has built-in Checkpointers (e.g., SQLite, PostgreSQL).

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)

# Run with a thread ID to resume later!
app.invoke(input, config={"configurable": {"thread_id": "1"}})
```

---

### 3. Human-In-The-Loop (HITL)

Sometimes an agent shouldn't execute without human approval (e.g., sending an email or dropping a database).

Using LangGraph, we can pause execution:

```python
# Compile with an interrupt on a specific node
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["EmailSenderNode"]
)
```

_The graph pauses, the human reviews the state, and manually resumes execution._

---

# Module 8: Live Code Review

---

### The "Researcher -> Writer" Setup

We built two files:

1. `main.py` -> The **CrewAI** implementation. Rapid setup, high abstraction. Let's look at how the `Task` expected outputs chained together.
2. `langgraph_main.py` -> The **LangGraph** implementation. Explicit state management, custom Python nodes. Let's look at how the `TypedDict` state flowed from start to finish.

**Both used our local `phi3` execution via Ollama without cloud keys!**

---

# Wrapping Up & Resources

You now have the tools to orchestrate autonomous intelligence locally on your machine!

### Your Capstone Challenge:

Take the `langgraph_main.py` script and add a 3rd node: an **"Editor"**.

1. Create an `editor_node`.
2. Add a `Conditional Edge` that grades the Blog Post.
3. If it fails, route it back to the `writer_node`!

**Questions? Let's dive in!**

---

# References & Acknowledgments

_Resources and foundational frameworks utilized in this presentation:_

- **NVIDIA Deep Learning Institute (DLI):** For best practices in AI hardware, local model execution, and accelerated edge computing.
- **CrewAI Framework:** Documentation and architectural design patterns for multi-agent enterprise workflows.
- **LangChain & LangGraph:** Official documentation surrounding LCEL DAG limitations and cyclic graph state machines.

<br>
<br>
