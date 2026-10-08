"""Run a simple LangChain agent with a Python REPL tool."""

import hashlib
import os
from getpass import getpass

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


@tool
def analyze_password_strength(password: str) -> str:
	"""Assess password strength without storing or logging the password.

	Check length and the presence of uppercase, lowercase, numeric, and
	special characters. For real passwords, use the local ``/password``
	command instead of sending a password in a chat message.
	"""
	checks = {
		"at least 12 characters": len(password) >= 12,
		"uppercase letters": any(character.isupper() for character in password),
		"lowercase letters": any(character.islower() for character in password),
		"numbers": any(character.isdigit() for character in password),
		"special characters": any(not character.isalnum() for character in password),
	}
	passed_checks = sum(checks.values())
	if passed_checks == len(checks):
		strength = "Strong"
	elif passed_checks >= 3:
		strength = "Moderate"
	else:
		strength = "Weak"

	suggestions = []
	if not checks["at least 12 characters"]:
		suggestions.append("Use at least 12 characters; a longer passphrase is even better.")
	for check, present in checks.items():
		if check != "at least 12 characters" and not present:
			suggestions.append(f"Add {check}.")
	if not suggestions:
		suggestions.append("No changes needed for these basic checks; use a unique password.")

	return (
		f"Assessment: {strength} ({passed_checks}/{len(checks)} checks passed).\n"
		f"Length: {len(password)} characters.\n"
		"Suggestions: " + " ".join(suggestions)
	)


def create_homework_agent():
	"""Create an agent configured with Python, file hashing, and password tools.

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
		tools=[python_tool, sha256_file, analyze_password_strength],
		system_prompt=(
			"You are a helpful assistant. Use the Python REPL tool for calculations "
			"and the SHA-256 file tool to calculate a file's SHA-256 digest. "
			"Never ask for or analyze a real password in chat; direct the user to "
			"the local /password command. Use the password tool only for dummy examples."
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
		if question.lower() == "/password":
			password = getpass("Password to assess (input hidden): ")
			print(analyze_password_strength.invoke({"password": password}))
			continue
		if not question:
			continue

		result = agent.invoke({"messages": [{"role": "user", "content": question}]})
		print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
	main()
