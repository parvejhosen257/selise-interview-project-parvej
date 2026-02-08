"""
Crew: one agent, one task (retrieve + answer) for fast responses.
"""
from crewai import Crew

from app.tasks import create_tasks


def run_crew(question: str):
    tasks = create_tasks(question)
    crew = Crew(agents=[tasks[0].agent], tasks=tasks, verbose=False)
    return crew.kickoff()
