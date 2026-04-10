import os
import re
os.environ["OTEL_SDK_DISABLED"] = "true"


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple_AI_Agents_Autonomous_System"

from crewai import Agent, Crew, Process, Task, LLM
from rich.console import Console
from rich.markdown import Markdown

def build_local_llm(model_name: str = "phi3"):
    return LLM(
        model=f"ollama/{model_name}",
        base_url="http://localhost:11434",
        temperature=0.0
    )

def main():
    # What makes this FULLY Autonomous compared to the previous fixed pipelines?
    # 1. Dynamic Routing: Instead of a hardcoded structure (e.g. Research -> Write -> Review), 
    #    an 'Orchestrator' LLM analyzes the current context on *every* loop and decides 
    #    which agent needs to act next based on the dynamic state.
    # 2. Self-Determined Termination: The system loops infinitely (up to a safety max) until 
    #    the Orchestrator evaluates the work and explicitly determines the user's goal is met.

    llm = build_local_llm(model_name="phi3")
    
    print("\n" + "=" * 60)
    print("    FULLY AUTONOMOUS SYSTEM: DYNAMIC ORCHESTRATOR    ")
    print("=" * 60)
    
    user_topic = input("\nEnter a goal for the autonomous system (e.g. Write a heavily researched post about AI): ").strip()
    topic = user_topic if user_topic else "Write a deeply analyzed post about AI Agents"

    print(f"\n--- Starting Fully Autonomous Agent Loop for: '{topic}' ---")

    # 1. Define the Action Agents (The Workforce)
    researcher = Agent(
        role="Research Analyst",
        goal="Provide extensive and logical information based on the requested task.",
        backstory="You are a precise data researcher.",
        llm=llm,
        verbose=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Draft high-quality text, articles, or summaries.",
        backstory="You are a skilled technical writer.",
        llm=llm,
        verbose=False,
    )

    critic = Agent(
        role="Editorial Reviewer",
        goal="Critique drafts and identify missing logic or poor flow.",
        backstory="You are a strict editor who provides actionable feedback.",
        llm=llm,
        verbose=False,
    )

    # 2. Define the Orchestrator (The Manager)
    orchestrator = Agent(
        role="System Orchestrator",
        goal="Deploy workers to achieve the user's ultimate goal. Decide who acts next and when the goal is complete.",
        backstory="You are a brilliant manager. You read the context history and dynamically decide the next logical move.",
        llm=llm,
        verbose=True,
    )

    # Memory state (Scratchpad)
    context_history = "No work has been done yet."
    is_done = False
    iteration = 1
    max_iterations = 6
    final_result = ""

    # Step 3: The Autonomous Cognitive Loop
    while not is_done and iteration <= max_iterations:
        print(f"\n[System] ================= Orchestrator Planning Cycle {iteration} =================")
        
        # The prompt forces the Orchestrator to decide the architecture on the fly
        orchestrator_prompt = f"""
        User's Ultimate Goal: {topic}
        
        Past Work Context:
        {context_history}
        
        You are the Orchestrator manager. Based on the Past Work, decide the next step.
        Choose exactly ONE worker from this list: [RESEARCHER, WRITER, CRITIC, DONE].

        Example 1:
        WORKER: RESEARCHER
        TASK: Research the core concepts of the user's topic and give 2 bullet points.

        Example 2:
        WORKER: DONE
        TASK: The user's goal has been thoroughly answered. Here is the final summary.

        YOUR RESPONSE MUST BE EXACTLY 2 LINES. NO CONVERSATION. FOLLOW THIS FORMAT:
        WORKER: [Choose one: RESEARCHER, WRITER, CRITIC, DONE]
        TASK: [Give the instruction or final summary]
        """

        orch_task = Task(
            description=orchestrator_prompt,
            expected_output="Exactly two lines. WORKER: <choice> and TASK: <instructions>",
            agent=orchestrator
        )

        orch_crew = Crew(agents=[orchestrator], tasks=[orch_task], verbose=False)
        orch_response = str(orch_crew.kickoff()).strip()

        # Parse the Orchestrator's autonomous decision
        worker_match = re.search(r"WORKER:\s*(RESEARCHER|WRITER|CRITIC|DONE)", orch_response, re.IGNORECASE)
        task_match = re.search(r"TASK:\s*(.*)", orch_response, re.IGNORECASE | re.DOTALL)
        
        # Fallbacks in case the SLM hallucinates format
        if worker_match:
            next_worker = worker_match.group(1).upper()
        else:
            if "DONE" in orch_response.upper(): next_worker = "DONE"
            elif "RESEARCH" in orch_response.upper(): next_worker = "RESEARCHER"
            elif "WRITE" in orch_response.upper(): next_worker = "WRITER"
            elif "CRITIC" in orch_response.upper() or "REVIEW" in orch_response.upper(): next_worker = "CRITIC"
            else: 
                # Never default to DONE if it's just hallucinating gibberish
                next_worker = "RESEARCHER" if iteration == 1 else "WRITER"

        task_desc = task_match.group(1).strip() if task_match else orch_response

        print(f"\n>> Orchestrator Selected Worker : {next_worker}")
        print(f">> Assigned Task                : {task_desc[:120]}...\n")

        if next_worker == "DONE":
            is_done = True
            final_result = task_desc
            print("[System] Orchestrator determined the goal is met! Terminating loop.")
            break

        # Route the task to the dynamically chosen worker
        worker_agent = None
        if next_worker == "RESEARCHER": worker_agent = researcher
        elif next_worker == "WRITER": worker_agent = writer
        elif next_worker == "CRITIC": worker_agent = critic
        else: worker_agent = writer # Safe fallback
        
        print(f"[System] Executing {next_worker} action...")
        action_task = Task(
            description=task_desc,
            expected_output="Complete the task to the best of your ability.",
            agent=worker_agent
        )
        action_crew = Crew(agents=[worker_agent], tasks=[action_task], verbose=False)
        worker_response = str(action_crew.kickoff())
        
        # Append worker's output into the global working memory!
        context_history += f"\n\n--- Output from {next_worker} (Cycle {iteration}) ---\n{worker_response}"
            
        iteration += 1

    if not is_done:
        print("\n[System] Reached max iterations before completion.")
        final_result = context_history

    print("\n" + "=" * 60)
    print("          FULLY AUTONOMOUS FINAL OUTPUT             ")
    print("=" * 60)
    Console().print(Markdown(final_result))

if __name__ == "__main__":
    main()
