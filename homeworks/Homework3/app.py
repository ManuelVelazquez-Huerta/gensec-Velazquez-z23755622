"""Run a simple LangChain agent with a Python REPL tool."""

import os

from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


def create_homework_agent():
	"""Create an agent configured with a Python REPL tool.

	The API key is read from ``OPENAI_API_KEY``. Set ``OPENAI_MODEL`` to
	choose a model and ``OPENAI_BASE_URL`` to use another OpenAI-compatible API.
	"""
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")

	model = ChatOpenAI(
		model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
		api_key=api_key,
		base_url=os.environ.get("OPENAI_BASE_URL"),
	)
	python_tool = PythonREPLTool()

	return create_agent(
		model=model,
		tools=[python_tool],
		system_prompt=(
			"You are a helpful assistant. Use the Python REPL tool when "
			"calculations or executable Python are useful."
		),
	)


def main():
	"""Run the interactive command-line conversation loop."""
	agent = create_homework_agent()
	print('Ask a question, or type "exit" to quit.')

	while True:
		try:
			question = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break

		if question.lower() == "exit":
			break
		if not question:
			continue

		result = agent.invoke({"messages": [{"role": "user", "content": question}]})
		print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
	main()
