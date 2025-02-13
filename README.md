# SRT Translator

A robust Python application for translating subtitle files (SRT format) using Google's Gemini AI API. This tool is designed to handle large subtitle files with automatic progress saving, API key rotation, and intelligent error handling.

## Features

- Translates SRT files while maintaining precise timing and formatting
- Supports multiple API keys with automatic rotation
- Intelligent context detection for maintaining translation consistency
- Automatic progress saving and resumption capability
- Comprehensive error handling and retry mechanism
- Exponential backoff strategy for API requests
- Real-time progress tracking and status updates

## Requirements

- Python 3.7+
- `google.generativeai` library
- `asyncio`

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install google-generativeai
```

## Configuration

Set up your API keys in the script (head over to [AI Studio](https://aistudio.google.com/app/apikey) to get an free api) :

```python
api_keys = [
    "YOUR_API_KEY_1",
    "YOUR_API_KEY_2",
    # Add more API keys as needed
]
```

## Usage

Basic usage:

```python
import asyncio
from srt_translator import SRTTranslator

async def translate():
    translator = SRTTranslator(api_keys)
    await translator.translate_file(
        "input.srt",
        "output.srt",
        input_lang="English",
        output_lang="Persian",
        save_progress=True
    )

asyncio.run(translate())
```

Advanced usage with callbacks:

```python
def status_callback(status):
    print(f"Status: {status}")

def progress_callback(current, total):
    print(f"Progress: {current}/{total} blocks")

await translator.translate_file(
    "input.srt",
    "output.srt",
    input_lang="English",
    output_lang="Persian",
    save_progress=True,
    status_callback=status_callback,
    progress_callback=progress_callback
)
```

## Key Features Explained

### API Key Rotation
The system automatically rotates between multiple API keys when encountering rate limits or errors. This ensures continuous operation even if some API keys become temporarily unavailable.

### Progress Saving
Translation progress is automatically saved to a `.progress` file, allowing the translation to resume from where it left off if interrupted.

### Context Detection
The translator analyzes a snippet of the dialogue at the start to understand the context and maintain consistent translation style throughout the file.

### Error Handling
- Implements exponential backoff with jitter
- Automatically retries failed requests
- Handles API quota exhaustion by rotating keys
- Manages various API errors and network issues

## Parameters

### SRTTranslator Class

- `api_keys`: List of Google API keys
- `base_delay`: Initial delay for retries (default: 30 seconds)
- `max_backoff`: Maximum delay between retries (default: 300 seconds)
- `max_retries`: Maximum number of retry attempts (default: 5)

### translate_file Method

- `input_path`: Path to input SRT file
- `output_path`: Path for translated SRT file
- `input_lang`: Source language (default: "English")
- `output_lang`: Target language (default: "Persian")
- `save_progress`: Enable/disable progress saving (default: True)
- `progress_callback`: Function for progress updates
- `status_callback`: Function for status messages
- `cancel_check`: Function to check for cancellation

## Error Prevention

The translator includes several safeguards:
- Validates SRT format before processing
- Ensures correct block numbering
- Maintains original timing information
- Handles different file encodings (UTF-8 and latin-1)
- Validates API responses

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.
