from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import Request
import logging
from dataclasses import dataclass
from time import time
import json
import aiohttp
import re

from open_webui.utils.misc import get_last_user_message


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.set_name("secure_code_generation")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


from pydantic import BaseModel
from typing import Any, Optional


class User(BaseModel):
    id: str
    email: str
    name: str
    role: str
    settings: Any = None
    info: Any = None
    profile_image_url: Optional[str] = None
    last_active_at: Optional[int] = None
    updated_at: Optional[int] = None
    created_at: Optional[int] = None
    api_key: Optional[int] = None

    class Config:
        extra = "allow"  # Accept unknown fields without error


class Pipe:
    class Valves(BaseModel):
        MODEL: str = (
            Field(
                default="llama3:70b",
                description="The model to use for the pipeline.",
            ),
        )
        GENERATION_MODE: str = Field(
            default="ollama",
            description="The generation",
        )

    def create_prompt(self, test_case):
        prompt = f"You are a Scala code generator. You will be given a task description and you will generate the code for it. The code should start with ```scala and end with ```. The task is: \n {test_case}"
        return prompt

    ################## GENERAL SETTING ##################
    def prompt_for_vulnerabilities(self, test_case, previous_code):
        prepend = "You will be given a task description and generated code. Your task is to find potential vulnerabilities in the code that could lead to security issues or unexpected behavior. Solely describe the vulnerabilities, do not give me any code."
        prompt = f"{prepend}\n\nHere is the task: {test_case}\n\nHere is the previous code: {previous_code}"
        return prompt

    def create_final_prompt_vulnerabilities_notypes(
        self, task, previous_code, vulnerabilities
    ):
        prepend = "You are a Scala code generator. You will be given a task description, generated code, and vulnerabilities that should be addressed. Your task is to improve the code. The code should start with ```scala and end with ```."
        prompt = f"{prepend}\n\nHere is the task: {task}\n\nHere is the previous code: {previous_code}\n\nHere are the vulnerabilities: {vulnerabilities}"
        return prompt

    def parse_output(self, response):
        # Take the first match beginning with ```scala and ending with ```
        match = re.search(r"```scala\s*(.*?)```", response, re.DOTALL)
        if match:
            value = match.group(1).strip()
            return value
        else:
            # We want to look for "object GeneratedFunctions {"
            # If it is there, we start counting the number of { and } and we want to return the code between the first { and the last }
            target = "object GeneratedFunctions {"
            start = response.find(target)
            i = start
            if start == -1:
                return None
            opening_bracket_count = 0
            while i < len(response):
                if response[i] == "{":
                    opening_bracket_count += 1
                elif response[i] == "}":
                    opening_bracket_count -= 1
                    if opening_bracket_count == 0:
                        return response[start : i + 1]
                i += 1
        # If we don't find any match, we return None
        return None

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.__user__ = None
        self._json_buffer = ""
        if self.valves.GENERATION_MODE == "ollama":
            from open_webui.routers.ollama import generate_chat_completion

            self.generate_chat_completion = generate_chat_completion

        if self.valves.GENERATION_MODE == "openai":
            from open_webui.routers.openai import generate_chat_completion

            # from open_webui.main import chat_completion

            self.generate_chat_completion = generate_chat_completion

        if self.valves.GENERATION_MODE not in ["ollama", "openai"]:
            raise ValueError(
                f"Unsupported generation mode: {self.valves.GENERATION_MODE}. "
                "Supported modes are 'ollama' and 'openai'."
            )

    def pipes(self):
        """Define available pipes"""
        name = f"SafePilot, model: {self.valves.MODEL}"
        return [{"name": name, "id": name}]

    def get_chunk_content(self, chunk: bytes):
        """
        Accumulate chunk data in a buffer and extract complete JSON objects
        from the buffer.
        """
        self._json_buffer += chunk.decode("utf-8")

        while True:
            newline_index = self._json_buffer.find("\n")
            if newline_index == -1:
                break

            line = self._json_buffer[:newline_index].strip()
            self._json_buffer = self._json_buffer[newline_index + 1 :]

            if not line:
                continue

            if line.startswith("data: "):
                line = line[len("data: ") :]

            if line == "[DONE]":
                break

            try:
                chunk_data = json.loads(line)

                # Defensive check: make sure choices exist
                choices = chunk_data.get("choices", [])
                if choices and isinstance(choices, list):
                    delta = choices[0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

                elif "message" in chunk_data and "content" in chunk_data["message"]:
                    yield chunk_data["message"]["content"]

                if chunk_data.get("done", False):
                    break

            except json.JSONDecodeError as e:
                logger.error(f'ChunkDecodeError: unable to parse "{line[:100]}": {e}')
                # Re-append for future re-parse
                self._json_buffer = line + "\n" + self._json_buffer
                break

    async def get_response(
        self, model: str, messages: List[Dict[str, str]], stream: bool
    ):
        """Generate response from the appropriate API."""

        response = await self.generate_chat_completion(
            self.__request__,
            {"model": model, "messages": messages, "stream": stream},
            user=self.__user__,
        )
        return response

    async def stream_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        __event_emitter__: Callable,
    ) -> AsyncGenerator[str, None]:

        response = None
        try:
            response = await self.get_response(model, messages, True)

            if not hasattr(response, "body_iterator"):
                if isinstance(response, dict) and "choices" in response:
                    yield response["choices"][0]["message"]["content"]
                return

            while True:
                chunk = await response.body_iterator.read(1024)
                if not chunk:
                    break

                for content in self.get_chunk_content(chunk):
                    yield content

        except Exception as e:
            api_type = "Ollama"
            error_msg = f"Error with {api_type} API: {str(e)}"
            logger.error(error_msg)
            await self.set_status_end(error_msg, __event_emitter__)

        finally:
            if response and hasattr(response, "close"):
                await response.close()

    async def run_generation(
        self,
        model: str,
        messages: List[Dict[str, str]],
        __event_emitter__: Callable,
        step_name: str,
        regenerating: bool = False,
    ) -> str:
        """Run the subprompt generation."""
        output = ""
        token_count = 0

        async for chunk in self.stream_response(model, messages, __event_emitter__):
            output += chunk
            token_count += 1
            status_msg = (
                f"Re-generating the code ({token_count} tokens)"
                if regenerating
                else f"Generating {step_name} ({token_count} tokens)"
            )
            await self.set_status(status_msg, __event_emitter__)

        return output.strip()

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Callable,
    ):

        try:
            self.__user__ = User(**__user__)
            self.__request__ = __request__
            messages = body["messages"]
            query = get_last_user_message(messages)
            start_time = time()

            ### STAGE 1: RUN
            logger.info("Generating initial code for your question... ")
            await self.set_status(
                "Generating initial code for your question...",
                __event_emitter__,
            )
            await self.set_file(
                "https://raw.githubusercontent.com/AlexanderSternfeld/images/refs/heads/main/safepilot_stage0_llm.gif",
                __event_emitter__,
            )

            initial_prompt = self.create_prompt(query)

            initial_code_message = [
                {
                    "role": "user",
                    "content": initial_prompt,
                }
            ]
            initial_code = await self.run_generation(
                model=self.valves.MODEL,
                messages=initial_code_message,
                __event_emitter__=__event_emitter__,
                step_name="initial code",
            )

            logger.info(f"Initial code: {initial_code}")

            parsed_initial_code = self.parse_output(initial_code)

            status_update = "The ***initial code*** is:\n"
            status_update += f"```scala\n{parsed_initial_code}\n```"
            status_initial_code = status_update

            await self.send_data(status_update, __event_emitter__)
            logger.info(f"Initial code parsed: {parsed_initial_code}")

            ## STAGE 2: RUN VULNERABILITY DETECTION
            logger.info("Generating the code vulnerabilities...")
            await self.set_status(
                "Finding potential vulnerabilities in the code...",
                __event_emitter__,
            )
            await self.set_file(
                "https://github.com/AlexanderSternfeld/images/blob/main/safepilot_stage1_llm.gif?raw=true",
                __event_emitter__,
            )
            vulnerabilities_prompt = self.prompt_for_vulnerabilities(
                query, parsed_initial_code
            )
            vulnerabilities_message = [
                {
                    "role": "user",
                    "content": vulnerabilities_prompt,
                }
            ]
            vulnerabilities = await self.run_generation(
                model=self.valves.MODEL,
                messages=vulnerabilities_message,
                __event_emitter__=__event_emitter__,
                step_name="the vulnerabilities in the code.",
            )
            logging.info(f"Vulnerabilities: {vulnerabilities}")
            vulnerabilities = vulnerabilities.strip()

            vulnerabilities_update = "The ***vulnerabilities*** are:\n"
            if vulnerabilities:
                vulnerabilities_update += f"{vulnerabilities}\n"
            else:
                vulnerabilities_update += "No vulnerabilities found.\n"

            vulnerabilities_update = (
                vulnerabilities_update + "\n\n" + status_initial_code
            )

            await self.send_data(
                vulnerabilities_update, __event_emitter__, replace=True
            )
            ## STAGE 3: REGENERATE CODE
            logger.info("Regenerating the code to address vulnerabilities...")
            await self.set_status(
                "Addressing the vulnerabilities...",
                __event_emitter__,
            )
            await self.set_file(
                "https://raw.githubusercontent.com/AlexanderSternfeld/images/refs/heads/main/safepilot_stage2_llm.gif",
                __event_emitter__,
            )
            final_prompt = self.create_final_prompt_vulnerabilities_notypes(
                query, parsed_initial_code, vulnerabilities
            )
            final_code_message = [
                {
                    "role": "user",
                    "content": final_prompt,
                }
            ]
            final_answer = await self.run_generation(
                model=self.valves.MODEL,
                messages=final_code_message,
                __event_emitter__=__event_emitter__,
                step_name="the final code.",
            )
            final_answer = final_answer.strip()
            final_answer = self.parse_output(final_answer)

            status_update = "The ***final code*** is:\n"
            status_update += f"```scala\n{final_answer}\n```"
            status_update += "\n\n" + vulnerabilities_update
            logger.info(f"Final code: {final_answer}")
            await self.send_data(status_update, __event_emitter__)

            # Log the final prompt and answer

            logger.info(f"Final code: {final_answer}")
            await self.set_status(
                "The final code has been generated.",
                __event_emitter__,
            )

            await self.set_file(
                "https://raw.githubusercontent.com/AlexanderSternfeld/images/refs/heads/main/safepilot_final_llm.png",
                __event_emitter__,
            )

            end_time = time()
            total_time = end_time - start_time
            logger.info(f"Pipeline completed in {total_time:.2f} seconds.")

            # make sure the code stops now, and does not start looping
            await self.set_status_end(
                "Pipeline completed successfully.", __event_emitter__
            )

            return status_update

        except Exception as e:
            error_msg = f"Error in pipe execution: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"

    async def send_data(
        self, data: str, __event_emitter__: Callable, replace: bool = False
    ):
        """Send data to the UI."""
        if replace:
            await __event_emitter__(
                {
                    "type": "chat:message",
                    "data": {
                        "content": data,
                        "role": "assistant",
                    },
                }
            )
        else:
            await __event_emitter__(
                {
                    "type": "chat:message:delta",
                    "data": {
                        "content": data,
                        "role": "assistant",
                    },
                }
            )

    async def set_status(self, description: str, __event_emitter__: Callable):
        """Set in-progress status."""
        await __event_emitter__(
            {"type": "status", "data": {"description": description, "done": False}}
        )

    async def set_status_end(self, description: str, __event_emitter__: Callable):
        """Set final status."""
        await __event_emitter__(
            {"type": "status", "data": {"description": description, "done": True}}
        )

    async def set_file(self, url: str, __event_emitter__: Callable):
        await __event_emitter__(
            {
                "type": "files",
                "data": {
                    "files": [
                        {
                            "type": "image",
                            "url": url,
                        }
                    ]
                },
            }
        )
