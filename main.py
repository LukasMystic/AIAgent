import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Crew, Process, Task, LLM
from rich.console import Console
from rich.markdown import Markdown

def build_local_llm(model_name: str = "phi3"):
    return LLM(
        model=f"ollama/{model_name}",
        base_url="http://localhost:11434"
    )

def main():
    llm = build_local_llm(model_name="phi3")

    print("\n--- System Initialization: Agent Pipeline Started ---\n")
    topic = input("Enter target topic for processing: ")
    print(f"\nExecuting pipeline for topic: '{topic}'...\n")

    researcher = Agent(
        role="Research Analyst",
        goal=f"Define the core concept of '{topic}' and specify 2 primary use-cases.",
        backstory=(
            "You are a concise data analyst. You rely on internal knowledge to quickly define "
            "concepts and identify their most practical applications."
        ),
        llm=llm,
        verbose=True,
    )

    engineer = Agent(
        role="Systems Architect",
        goal="Design a brief, high-level technical architecture or stack for the proposed use-cases.",
        backstory=(
            "You are a pragmatic systems architect. You translate business use-cases into "
            "practical, high-level technical requirements and system designs."
        ),
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Technical Documenter",
        goal="Synthesize the research and architecture into a short, single-paragraph project summary.",
        backstory=(
            "You are an efficient technical writer. You combine research and engineering notes "
            "into clean, professional, and extremely brief developer documentation without exaggerations."
        ),
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Define the topic '{topic}' in one sentence, then list exactly 2 potential use-cases."
        ),
        expected_output="A one-sentence definition and a 2-item list of use-cases.",
        agent=researcher,
    )

    architecture_task = Task(
        description=(
            "Review the 2 use-cases proposed by the Research Analyst. Suggest a realistic "
            "tech stack (3-4 technologies) to build them, and state one major technical hurdle."
        ),
        expected_output="A short list outlining a tech stack and one sentence describing a technical hurdle.",
        agent=engineer,
    )

    writing_task = Task(
        description=(
            "Combine the research definition, use-cases, and tech stack into one clean, "
            "markdown-formatted paragraph that would serve as a project README introduction. Do not use buzzwords."
        ),
        expected_output="A single paragraph of Markdown text.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, engineer, writer],
        tasks=[research_task, architecture_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    final_output = crew.kickoff()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE: FINAL OUTPUT")
    print("=" * 60)
    
    Console().print(Markdown(str(final_output)))

if __name__ == "__main__":
    main()
