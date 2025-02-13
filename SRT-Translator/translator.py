import asyncio
import time
import logging
from pathlib import Path
import google.generativeai as genai
from google.api_core import exceptions
import random
import re
import json


class APIKeyRotator:
    """
    Manages a rotation of API keys, handling key unavailability.
    """

    def __init__(self, api_keys):
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        self.api_keys = api_keys
        self.current_key_index = 0
        self.unavailable_keys = set()
        logging.info(f"API Key Rotator initialized with {len(api_keys)} keys.")

    def get_current_key(self):
        """Returns the current API key."""
        return self.api_keys[self.current_key_index]

    def rotate_key(self):
        """Rotates to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logging.info(f"Rotating API key to index {self.current_key_index}")

    def mark_key_unavailable(self, key):
        """Marks an API key as unavailable due to quota exhaustion or other issues."""
        if key not in self.unavailable_keys:
            self.unavailable_keys.add(key)
            logging.warning(
                f"API key marked as unavailable: {key[:10]}... (Total unavailable: {len(self.unavailable_keys)}/{len(self.api_keys)})"
            )
            if len(self.unavailable_keys) == len(self.api_keys):
                logging.error("All API keys have been marked as unavailable.")

    def has_available_keys(self):
        """Checks if there are any API keys available for use."""
        return len(self.unavailable_keys) < len(self.api_keys)


class SRTTranslator:
    """
    Translates SRT subtitle files using the Gemini API with API key rotation,
    retry logic, and progress saving.
    """

    def __init__(
        self, api_keys, base_delay=30, max_backoff=300, context_blocks_count=2
    ):  # Added context_blocks_count
        self.key_rotator = APIKeyRotator(api_keys)
        self.model = self._initialize_model()
        self.last_request_time = 0
        self.base_delay = base_delay
        self.max_retries = 5
        self.max_backoff = max_backoff
        self.context_blocks_count = context_blocks_count

        self.srt_block_count_regex = re.compile(
            r"^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->", re.MULTILINE
        )
        self.srt_first_block_regex = re.compile(
            r"^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}", re.MULTILINE
        )
        self.srt_block_split_regex = re.compile(r"\n\s*\n")
        self.srt_block_number_regex = re.compile(r"^\d+\s*$")
        self.srt_time_regex = re.compile(
            r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}"
        )

    def _initialize_model(self):
        """Initializes the Gemini generative model with API key and settings."""
        genai.configure(api_key=self.key_rotator.get_current_key())
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        logging.info("Gemini model initialized.")
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    async def _make_api_request(
        self,
        chat_session,
        message,
        block_number,
        retry_count=0,
    ):
        """
        Makes an API request with retry and error handling, including key rotation.
        """
        original_key_index = self.key_rotator.current_key_index

        try:
            self._apply_backoff_delay(retry_count)
            response = await chat_session.send_message_async(message)
            return response

        except (
            exceptions.PermissionDenied,
            exceptions.ResourceExhausted,
            exceptions.ServerError,
        ) as e:
            error_type = (
                "Permission denied (403)"
                if isinstance(e, exceptions.PermissionDenied) and e.code == 403
                else (
                    "Resource exhausted"
                    if isinstance(e, exceptions.ResourceExhausted)
                    else (
                        f"Server error ({e.code})"
                        if isinstance(e, exceptions.ServerError)
                        and (e.code == 500 or e.code == 504)
                        else "Unknown error"
                    )
                )
            )

            if retry_count < self.max_retries and self.key_rotator.has_available_keys():
                if isinstance(e, exceptions.ResourceExhausted):
                    self._handle_quota_exhaustion()
                    if not self.key_rotator.has_available_keys():
                        raise Exception(
                            "All API keys exhausted due to quota limits."
                        ) from e

                if self.key_rotator.has_available_keys():
                    self.key_rotator.rotate_key()
                    genai.configure(api_key=self.key_rotator.get_current_key())
                    logging.warning(
                        f"{error_type} for block {block_number}, retrying with a different key (attempt {retry_count + 1}/{self.max_retries})"
                    )
                    return await self._make_api_request(
                        chat_session, message, block_number, retry_count + 1
                    )
                else:
                    raise Exception(
                        "No API keys available for retry after quota exhaustion."
                    ) from e

            logging.error(
                f"{error_type} for block {block_number} after max retries. Error: {e}"
            )
            raise Exception(
                f"Failed to process block {block_number} after multiple retries due to API errors."
            ) from e

        except Exception as e:
            logging.error(f"Unexpected error for block {block_number}: {e}")
            raise

    def _calculate_backoff(self, retry_count):
        """Calculates exponential backoff with jitter."""
        delay = min(self.max_backoff, self.base_delay * (2**retry_count))
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter

    def _apply_backoff_delay(self, retry_count):
        """Applies a backoff delay before retrying an API request."""
        if retry_count > 0:
            delay = self._calculate_backoff(retry_count - 1)
            logging.info(
                f"Backing off for {delay:.2f}s (attempt {retry_count}/{self.max_retries})"
            )
            time.sleep(delay)

    def _handle_quota_exhaustion(self):
        """Handles quota exhaustion by marking the current key as unavailable."""
        current_key = self.key_rotator.get_current_key()
        self.key_rotator.mark_key_unavailable(current_key)
        logging.info("Current API key marked as unavailable due to quota exhaustion.")

    def _create_chat(self):
        """Creates a new chat session with the Gemini model."""
        return self.model.start_chat()

    def count_srt_blocks(self, content):
        """Counts the number of SRT blocks in a given content string."""
        blocks = self.srt_block_count_regex.findall(content)
        return len(blocks)

    def extract_srt_blocks(self, content):
        """Extracts valid SRT blocks from a content string."""
        first_block_match = self.srt_first_block_regex.search(content)
        if first_block_match:
            content = content[first_block_match.start() :]
        blocks = [
            block.strip()
            for block in self.srt_block_split_regex.split(content)
            if block.strip()
        ]
        valid_blocks = []
        for block in blocks:
            lines = block.splitlines()
            if (
                len(lines) >= 2
                and self.srt_block_number_regex.match(lines[0])
                and self.srt_time_regex.match(lines[1])
            ):
                valid_blocks.append(block)
        return valid_blocks

    def _extract_dialogue_snippet(self, input_content, num_lines=5):
        """Extracts a snippet of dialogue from the SRT content for context setting."""
        blocks = self.extract_srt_blocks(input_content)
        dialogue_lines = []
        line_count = 0
        for block in blocks:
            lines = block.splitlines()
            if len(lines) > 2:
                for i in range(2, len(lines)):
                    dialogue_lines.append(lines[i])
                    line_count += 1
                    if line_count >= num_lines:
                        return "\n".join(dialogue_lines)
        return "\n".join(dialogue_lines)

    async def translate_file(
        self,
        input_path,
        output_path,
        input_lang="English",
        output_lang="Persian",
        save_progress=True,
        progress_callback=None,
        status_callback=None,
        cancel_check=None,
    ):
        """
        Translates an SRT file, handling progress saving, cancellation, and callbacks.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        progress_path = output_path.with_suffix(".progress")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                input_content = f.read()
        except UnicodeDecodeError:
            logging.warning(
                f"UTF-8 decoding failed for input file, trying latin-1 encoding."
            )
            with open(input_path, "r", encoding="latin-1") as f:
                input_content = f.read()

        total_blocks = self.count_srt_blocks(input_content)
        logging.info(f"Total SRT blocks to translate: {total_blocks}")
        if status_callback:
            status_callback(f"Total SRT blocks: {total_blocks}")

        translated_content = {}
        start_block_number = 1
        if save_progress and progress_path.exists():
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    progress_content = json.load(f)
                    if isinstance(progress_content, dict):
                        translated_content = progress_content
                    else:
                        logging.warning(
                            "Progress file corrupted/old format. Discarding."
                        )
            except (FileNotFoundError, json.JSONDecodeError):
                logging.warning("Progress file not found/corrupted. Starting fresh.")

            if translated_content:
                last_block_number = max(map(int, translated_content.keys()), default=0)
                start_block_number = last_block_number + 1
                current_blocks = len(translated_content)
                logging.info(
                    f"Loaded {current_blocks}/{total_blocks} blocks from progress. Resuming from block {start_block_number}."
                )
                if status_callback:
                    status_callback(
                        f"Loaded {current_blocks}/{total_blocks}, resuming from {start_block_number}."
                    )
                if progress_callback:
                    progress_callback(current_blocks, total_blocks)
            else:
                logging.info("Starting translation from scratch.")
                if status_callback:
                    status_callback("Starting translation from scratch.")

        chat = self._create_chat()
        block_retry_counts = {}
        expected_block_number = start_block_number

        system_message_base = f"""You are an expert subtitle translator, specializing in {input_lang} to {output_lang} translations. Your goal is to produce subtitles of the highest possible quality, indistinguishable from those created by a professional human translator.  Pay close attention to cultural nuances, idiomatic expressions, and the natural flow of conversation in {output_lang}. Ensure perfect SRT format, including numbering from {{}}, accurate timings, and proper structure.  Avoid adding any extra text or commentary. Translate with a style that is consistent with the context of the video content.""".format(
            "{}"
        )

        try:
            if not translated_content:
                if status_callback:
                    status_callback("Analyzing video context...")

                dialogue_snippet = self._extract_dialogue_snippet(input_content)
                context_prompt_message = f"""Analyze the essence of the following dialogue snippet from a video. Identify the underlying topic, theme, and overall tone. Provide a concise Persian summary (maximum 1 sentence) that captures the general context and nuances of the conversation. This summary will be used to ensure the subtitle translation vocabulary and style are perfectly aligned with the original content's spirit and feel.\n\nDialogue Snippet:\n{dialogue_snippet}"""
                context_response = await self._make_api_request(
                    chat,
                    context_prompt_message,
                    0,
                )
                translation_context = context_response.text.strip()
                logging.info(f"Detected context: {translation_context}")
                if status_callback:
                    status_callback(f"Detected context: {translation_context[:50]}...")

                system_message_full = (
                    system_message_base
                    + f" The context of the video is understood to be: {translation_context}."
                )
            else:
                system_message_full = (
                    system_message_base
                    + " Continue to translate with a style consistent with the established context for this project."
                )

            all_blocks_english = self.extract_srt_blocks(input_content)

            block_index = start_block_number - 1
            while block_index < len(all_blocks_english):
                block_english = all_blocks_english[block_index]
                current_block_number_english = block_english.splitlines()[0]

                if cancel_check and cancel_check():
                    logging.info("Translation cancelled during block processing")
                    if status_callback:
                        status_callback("Translation cancellation requested")
                    return

                system_message = system_message_full.format(expected_block_number)

                context_blocks = []
                for i in range(
                    max(0, block_index - self.context_blocks_count), block_index + 1
                ):
                    context_blocks.append(all_blocks_english[i])
                english_blocks_batch = "\n\n".join(context_blocks)
                current_block_in_batch_number_english = context_blocks[-1].splitlines()[
                    0
                ]

                try:
                    prompt_message = f"""{system_message}\n\nContext English SRT blocks (including block number {current_block_in_batch_number_english}):\n{english_blocks_batch}\n\nTranslate ONLY the LAST English SRT block from the above context into a natural and idiomatic {output_lang} SRT block, ensuring it reads as if originally written in {output_lang} by a native speaker. The output should be in perfect SRT format, starting with block number {expected_block_number}, and should capture the nuances and tone of the original English dialogue based on the context of the provided blocks, without sounding like a machine translation. Focus on delivering human-quality subtitles.\n\nOutput ONLY the {output_lang} SRT block in perfect format for block number {expected_block_number}."""
                    response = await self._make_api_request(
                        chat, prompt_message, expected_block_number
                    )

                    extracted_blocks = self.extract_srt_blocks(response.text)
                    if extracted_blocks:
                        block_persian = extracted_blocks[0]
                        block_lines = block_persian.splitlines()
                        block_persian_renumbered = "\n".join(
                            [str(expected_block_number)] + block_lines[1:]
                        )
                        translated_content[str(expected_block_number)] = (
                            block_persian_renumbered
                        )

                        current_blocks = len(translated_content)
                        logging.info(
                            f"Translated block {current_block_number_english} as {expected_block_number} ({current_blocks}/{total_blocks})"
                        )
                        if status_callback:
                            status_callback(
                                f"Translated {current_blocks}/{total_blocks}"
                            )
                        if progress_callback:
                            progress_callback(current_blocks, total_blocks)

                        if save_progress:
                            with open(progress_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    translated_content, f, ensure_ascii=False, indent=4
                                )

                        block_index += 1
                        expected_block_number += 1
                        block_retry_counts = {}

                    else:
                        block_retry_counts[current_block_number_english] = (
                            block_retry_counts.get(current_block_number_english, 0) + 1
                        )
                        if (
                            block_retry_counts[current_block_number_english]
                            <= self.max_retries
                        ):
                            retry_attempt = block_retry_counts[
                                current_block_number_english
                            ]
                            logging.warning(
                                f"No SRT block in response for block {current_block_number_english}, retry {retry_attempt}/{self.max_retries}."
                            )
                            await asyncio.sleep(
                                self._calculate_backoff(retry_attempt - 1)
                            )
                            continue
                        else:
                            error_message = f"Failed to get SRT block for block {current_block_number_english} after {self.max_retries} retries."
                            logging.error(error_message)
                            if status_callback:
                                status_callback(
                                    f"Failed block {current_block_number_english} after retries (No SRT response)."
                                )
                            raise Exception(error_message)

                except Exception as e:
                    logging.error(
                        f"Error translating block {current_block_number_english}: {e}"
                    )
                    block_retry_counts[current_block_number_english] = (
                        block_retry_counts.get(current_block_number_english, 0) + 1
                    )
                    if (
                        block_retry_counts[current_block_number_english]
                        <= self.max_retries
                    ):
                        retry_attempt = block_retry_counts[current_block_number_english]
                        logging.warning(
                            f"Exception translating block {current_block_number_english}, retry {retry_attempt}/{self.max_retries}."
                        )
                        await asyncio.sleep(self._calculate_backoff(retry_attempt - 1))
                        continue
                    else:
                        error_message = f"Failed to translate block {current_block_number_english} after {self.max_retries} retries due to errors. Error: {e}"
                        logging.error(error_message)
                        if status_callback:
                            status_callback(
                                f"Failed block {current_block_number_english} after retries due to errors."
                            )
                        raise Exception(error_message)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(translated_content.values()))

            if save_progress and progress_path.exists():
                progress_path.unlink()

            logging.info(f"Translation completed, saved to {output_path}")
            if status_callback:
                status_callback(f"Translation completed, saved to {output_path}")

        except Exception as e:
            logging.error(f"Error during translation process: {e}")
            raise


async def main():
    api_keys = [
        "API-1",
        "API_2",
    ]
    translator = SRTTranslator(
        api_keys, base_delay=20, max_backoff=240, context_blocks_count=2
    )
    try:
        await translator.translate_file(
            "input.srt",
            "output_fa_humanlike.srt",
            input_lang="English",
            output_lang="Persian",
            save_progress=True,
            status_callback=print,
            progress_callback=lambda c, t: print(f"Progress: {c}/{t}"),
        )
    except Exception as e:
        print(f"Translation halted due to error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
