"""Run a simple LangChain agent with a Python REPL tool."""

import os
import hashlib

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


@tool
def sha256_file(file_path: str) -> str:
	"""Calculate and return the SHA-256 hexadecimal digest of a file.

	Read the file in binary chunks so that large files do not need to fit
	in memory. Return a useful error message if the file cannot be read.
	"""
	digest = hashlib.sha256()
	try:
		with open(file_path, "rb") as file:
			for chunk in iter(lambda: file.read(1024 * 1024), b""):
				digest.update(chunk)
	except OSError as error:
		return f"Unable to hash '{file_path}': {error.strerror or error}."

	return digest.hexdigest()


def create_homework_agent():
	"""Create an agent configured with Python REPL and file hashing tools.

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
		tools=[python_tool, sha256_file],
		system_prompt=(
			"You are a helpful assistant. Use the Python REPL tool for calculations "
			"and the SHA-256 file tool to calculate a file's SHA-256 digest."
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
