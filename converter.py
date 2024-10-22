import whisper
import openai
import os
import json
import argparse
import sys

# Function to load configuration from env.json
def load_config(file_path="libs/env.json"):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        print(f"Configuration loaded from {file_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse configuration file '{file_path}'. Ensure it's valid JSON.")
        sys.exit(1)

# Initialize OpenAI API Key from config file
def initialize_openai(api_key):
    try:
        openai.api_key = api_key
        print("OpenAI API key loaded.")
    except Exception as e:
        print(f"Error setting OpenAI API key: {e}")
        sys.exit(1)

# Function to transcribe audio using Whisper
def transcribe_audio(file_path, model_type="base"):
    try:
        print("Loading Whisper model...")
        model = whisper.load_model(model_type)  # Model options: tiny, base, small, medium, large
        print(f"Model '{model_type}' loaded successfully.")

        print("Transcribing audio...")
        result = model.transcribe(file_path)
        print("Transcription completed.")
        return result['text']

    except FileNotFoundError:
        print(f"Error: Audio file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        sys.exit(1)

# Function to translate text using GPT (ChatGPT)
def translate_text(text, target_language="Japanese"):
    try:
        print(f"Translating text to {target_language}...")
        response = openai.Completion.create(
            model="gpt-4",
            prompt=f"Translate the following text to {target_language}:\n\n{text}",
            temperature=0.5,
            max_tokens=1500
        )
        print(f"Translation to {target_language} completed.")
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        sys.exit(1)

# Save output to file
def save_to_file(text, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Output successfully saved to {filename}.")
    except Exception as e:
        print(f"Error saving file: {e}")

# Main function to handle transcribing and translating
def process_audio(audio_file, model_type="base", output_file="transcription.txt", language="Japanese"):
    # Step 1: Transcribe the audio to text using Whisper
    transcribed_text = transcribe_audio(audio_file, model_type=model_type)
    print(f"\nTranscribed Text:\n{transcribed_text}")

    # Step 2: Translate the transcribed text to the target language using ChatGPT
    translated_text = translate_text(transcribed_text, target_language=language)
    print(f"\nTranslated Text ({language}):\n{translated_text}")

    # Step 3: Save both transcriptions and translations to a file
    transcribed_file = f"transcribed_{output_file}"
    translated_file = f"translated_{language.lower()}_{output_file}"
    save_to_file(transcribed_text, transcribed_file)
    save_to_file(translated_text, translated_file)

# CLI Interface using argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio and translate text using Whisper and ChatGPT")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., .wav, .mp3, .m4a)")
    parser.add_argument("-c", "--config", type=str, default="libs/env.json",
                        help="Path to the configuration file (default: libs/env.json)")
    return parser.parse_args()

# Entry point
if __name__ == "__main__":
    args = parse_arguments()

    # Load configuration from env.json
    config = load_config(args.config)

    # Initialize OpenAI API Key from config
    initialize_openai(config['openai_api_key'])

    # Validate audio file existence
    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' does not exist.")
        sys.exit(1)

    # Process the audio file using the config settings
    process_audio(
        audio_file=args.audio_file, 
        model_type=config.get('whisper_model', 'base'), 
        output_file=config.get('output_file', 'output.txt'),
        language=config.get('target_language', 'Japanese')
    )
