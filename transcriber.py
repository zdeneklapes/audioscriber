#!/usr/bin/env python3
"""
Audio transcriber using OpenAI's Whisper speech recognition model.
Usage: python3 transcriber.py -u, --url <URL>
"""
import getopt
import json
import os
import re
import sys
from pprint import pprint

import torch
import whisper

from googletrans import Translator
import yt_dlp as youtube_dl
import argparse
from pathlib import Path

from distutils.util import strtobool

ROOT_DIR = Path(__file__).resolve().parent
TMP_DIR = ROOT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def banner(text):
    """Display a message when the script is working in the background"""
    print(f"# {text} #")

def parse_args():
    """Parse command line arguments."""
    banner("Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Audio transcriber using OpenAI's Whisper speech recognition model.")
    parser.add_argument("-u", "--url", type=str, nargs='?', help="YouTube video URL")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        default=False,
                        help="Display more information")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        default=os.getenv("DEBUG", False),
                        help="Display debug information")
    parser.add_argument("--audio-file", type=Path, nargs='?', default=TMP_DIR / "audio.mp3",
                        help="Path to audio file on disk")
    parser.add_argument("--transcription-file", type=Path, nargs='?', default=TMP_DIR / "transcription.txt",
                        help="Path to transcript file on disk")
    parser.add_argument("--translation-file", type=Path, nargs='?', default=TMP_DIR / "translation.txt",
                        help="Path to translation file on disk")
    parser.add_argument("--modelname", type=str, nargs='?', default="large",
                        help=f"Select speech recognition model name: {whisper.available_models()}")
    parser.add_argument("--translate", type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        default=False,
                        help="Translate audio transcription to English")
    args = parser.parse_args()

    if args.debug:
        pprint(vars(args))
    return args


def match_pattern(pattern, arg):
    """If YouTube shorts URL is given, convert it to standard URL."""
    match = re.search(pattern, arg)
    if bool(match):
        url = re.sub(pattern, "watch?v=", arg)
    else:
        url = arg
    return url


def get_audio(args):
    """
    Download mp3 audio of a YouTube video. Credit to Stokry.
    https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
    """
    # try:
    #     opts, args = getopt.getopt(argv, "u:", ["url="])
    # except:
    #     print("Usage: python3 transcriber.py -u <url>")
    # for opt, arg in opts:
    #     if opt in ['-u', '--url']:
    #         url = match_pattern("shorts/", arg)
    video_info = youtube_dl.YoutubeDL().extract_info(url=args.url, download=False)
    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': args.audio_file,
    }
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])



def check_device():
    """Check CUDA availability."""
    if torch.cuda.is_available() == 1:
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_result(args):
    """Get speech recognition model."""
    banner("Loading speech recognition model")
    model = whisper.load_model(args.modelname, device=check_device())
    banner("Transcribing audio")
    result = model.transcribe(
        audio=args.audio_file.absolute().__str__(),
        verbose=args.verbose,
    )
    format_result(args, result["text"])


def format_result(args, text):
    """Put a newline character after each sentence and prompt user for translation."""
    banner("Formatting transcription")
    format_text = re.sub('\.', '.\n', text)
    with open(args.translation_file, 'a', encoding="utf-8") as file:
        banner("Writing transcription to text file")
        file.write(format_text)
    if args.translate:
        translate_result(args)


def translate_result(args):
    """
    Translate transcribed text. Credit to Harsh Jain at educative.io
    https://www.educative.io/answers/how-do-you-translate-text-using-python
    """
    translator = Translator()  # Create an instance of Translator() class
    with open(args.transcription_file, 'r', encoding="utf-8") as transcription:
        banner("Translating text")
        contents = transcription.read()
        translation = translator.translate(contents)
    with open(args.translation_file, 'a', encoding="utf-8") as file:
        banner("Writing translation to text file")
        file.write(translation.text)


def main():
    """Main function."""
    args = parse_args()
    if args.url:
        get_audio(args)
    get_result(args)  # Get audio transcription and translation if needed

if __name__ == "__main__":
    main()
