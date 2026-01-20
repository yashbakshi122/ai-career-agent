from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="""
    You are an AI Career Mentor.
    Analyze resume and career goal.
    Give strengths, skill gaps, learning plan and interview tips.
    """
)




