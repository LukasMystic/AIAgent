from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# What makes LangGraph special:
# 1. State Management: It explicitly passes a `State` dictionary between nodes. Every node can read and write to this shared memory.
# 2. Granular Control: Unlike CrewAI which acts like a "manager" that abstracts the workflow, LangGraph lets you build an exact graph diagram (like connecting Legos). You define exactly which node runs and when.

class State(TypedDict):
    # This state dictionary is LangGraph's superpower. It tracks the exact data moving through the pipeline.
    topic: str
    insane_facts: str
    viral_post: str


llm = ChatOllama(
    model="phi3", 
    base_url="http://localhost:11434",
    temperature=0.7,
)

def researcher_node(state: State):
    print("[Node Execution] Conspiracy Theorist is scouring the timeline...")
    topic = state.get("topic", "Default Topic")
    
    prompt = (
        f"You are a Tinfoil Hat Conspiracy Theorist.\n"
        f"Goal: Connect mundane things to massive intergalactic plots.\n"
        f"Task: Generate 3 completely unhinged but highly detailed 'facts' connecting '{topic}' to a secret reptilian society.\n"
        f"Expected Output: Only the 3 bullet points."
    )
    
    response = llm.invoke(prompt)
    # Update the LangGraph State
    return {"insane_facts": response.content}


def writer_node(state: State):
    print("[Node Execution] Gen-Z Influencer is typing furiously...")
    facts = state.get("insane_facts", "")
    
    prompt = (
        f"You are a hyperactive Gen-Z TikTok Influencer.\n"
        f"Goal: Go totally viral at all costs using brain-rot slang.\n"
        f"Task: Turn these unhinged facts into a 2-paragraph viral post to expose the truth to your chat. No emojis, just pure slang text!\n\n"
        f"Facts:\n{facts}\n\n"
        f"Expected Output: A 2-paragraph post."
    )
    
    response = llm.invoke(prompt)

    # Update the LangGraph State
    return {"viral_post": response.content}



workflow = StateGraph(State)

# Add our agent nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# Define the flow 
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# Compile the graph into a runnable application
app = workflow.compile()


def main():
    # Interactive User Input
    print("=" * 60)
    print("             LANGGRAPH: EXPLICIT STATE PIPELINE             ")
    print("=" * 60)

    user_topic = input("\nEnter a mundane topic to uncover the truth (e.g. Toasters, Math, Shoelaces): ").strip()
    if not user_topic:
        user_topic = "Shoelaces"

    print(f"\n[Execution Started] Mapping '{user_topic}' to the Shadow Realm...")
    
    initial_state = {
        "topic": user_topic
    }
    
    final_output = None
    
    # LangGraph streams the explicit state updates
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n--- Finished Node: {node_name.upper()} ---")
            
            # Print the incremental state updates to screen
            if "insane_facts" in state_update:
                print(">> INSANE FACTS GENERATED <<")
                print(state_update["insane_facts"])
            if "viral_post" in state_update:
                final_output = state_update["viral_post"]

    print("\n" + "=" * 60)
    print("      LANGGRAPH FINAL OUTPUT SUMMARY (THE VIRAL POST)       ")
    print("=" * 60)
    
    # Clean string print, resembling generic terminal
    print(final_output)

if __name__ == "__main__":
    main()
