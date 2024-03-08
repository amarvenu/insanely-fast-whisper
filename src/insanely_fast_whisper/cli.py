import json
from pathlib import Path
from typing import TypedDict

import click
import torch
from pyannote.audio import Pipeline as PyannotePipeline
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from transformers import pipeline, Pipeline as HfPipeline

from .utils.diarize import (
    post_process_segments_and_transcripts,
    preprocess_inputs,
    diarize_audio,
)


@click.command()
@click.option(
    "--file-name",
    "-f",
    required=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    help="Path or URL to the audio file to be transcribed, or directory containing audio files. Allows multiple.",
)
@click.option(
    "--device-id",
    default="0",
    type=click.STRING,
    help="Device ID for your GPU. Just pass the device number when using CUDA, or 'mps' for Macs with Apple Silicon. (default: '0')",
    show_default=True,
)
@click.option(
    "--transcript-path",
    "-o",
    default="output.jsonl",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to save the transcription output JSONL. If multiple audio files are processed, one row per file will be written. (default: output.jsonl)",
    show_default=True,
)
@click.option(
    "--model-name",
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
    show_default=True,
)
@click.option(
    "--task",
    default="transcribe",
    type=click.Choice(["transcribe", "translate"]),
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
    show_default=True,
)
@click.option(
    "--language",
    required=False,
    type=str,
    help="Language of the input audio. (default: 'None' (Whisper auto-detects the language))",
)
@click.option(
    "--batch-size",
    default=24,
    type=int,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
    show_default=True,
)
@click.option(
    "--flash",
    is_flag=True,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
    show_default=True,
)
@click.option(
    "--timestamp",
    default="chunk",
    type=click.Choice(["chunk", "word"]),
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
    show_default=True,
)
@click.option(
    "--hf_token",
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
    show_default=True,
)
@click.option(
    "--diarize",
    is_flag=True,
    default=False,
    help="Whether to perform speaker diarization on the audio file. (default: False)",
    show_default=True,
)
@click.option(
    "--diarization_model",
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
    show_default=True,
)
@click.option(
    "--min-speakers",
    default=None,
    type=int,
    help="Sets the minimum number of speakers that the system should consider during diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be less than or equal to --max-speakers if both are specified. (default: None)",
    show_default=True,
)
@click.option(
    "--max-speakers",
    default=None,
    type=int,
    help="Defines the maximum number of speakers that the system should consider in diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be greater than or equal to --min-speakers if both are specified. (default: None)",
    show_default=True,
)
def main(
    file_name: list[Path],
    transcript_path: Path,
    device_id: str,
    model_name: str,
    task: str,
    language: str | None,
    batch_size: int,
    flash: bool,
    timestamp: str,
    hf_token: str,
    diarize: bool,
    diarization_model: str,
    min_speakers: int | None,
    max_speakers: int | None,
):
    if diarize:
        _check_diarization_args(max_speakers, min_speakers)

    audio_files = _get_audio_files(file_name)
    print(f"Found {len(audio_files)} audio files to process.")

    if transcript_path.exists():
        already_processed = _get_already_processed_files(transcript_path)
        prev_len = len(audio_files)
        audio_files = [
            file
            for file in audio_files
            if file.absolute().as_posix() not in already_processed
        ]
        print(
            f"Found {prev_len - len(audio_files)} already processed files. Skipping them."
        )

    transcription_pipeline, diarization_pipeline = _get_pipelines(
        model_name, diarization_model, diarize, device_id, flash, hf_token
    )

    generate_kwargs = {"task": task, "language": language}
    if model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    with Progress(
        TextColumn("[blue][progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(pulse_style="yellow"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[blue]{task.fields[curr_task]}"),
    ) as pbar, open(transcript_path, "a", encoding="utf8") as output_f:
        task = pbar.add_task(
            "Processing files...", total=len(audio_files), curr_task=""
        )
        for input_file_path in audio_files:
            try:
                pbar.update(task, curr_task=f"Transcribing {input_file_path}...")
                tr_outputs = transcription_pipeline(
                    str(input_file_path),
                    chunk_length_s=30,
                    batch_size=batch_size,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=("word" if timestamp == "word" else True),
                )
                diarize_outputs = []
                if diarize:
                    pbar.update(task, curr_task=f"Diarizing {input_file_path}...")
                    inputs, diarizer_inputs = preprocess_inputs(
                        inputs=str(input_file_path)
                    )
                    segments = diarize_audio(
                        diarizer_inputs,
                        diarization_pipeline,
                        None,
                        min_speakers,
                        max_speakers,
                    )
                    diarize_outputs = post_process_segments_and_transcripts(
                        segments, tr_outputs["chunks"], group_by_speaker=False
                    )
                result = {
                    "speakers": diarize_outputs,
                    "chunks": tr_outputs["chunks"],
                    "text": tr_outputs["text"],
                    "file_path": Path(input_file_path).absolute().as_posix(),
                }
                output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_f.flush()
                pbar.update(task, advance=1, curr_task=f"Processed {input_file_path}.")
            except Exception as e:
                pbar.console.print_exception()
                pbar.console.print(f"Error processing {input_file_path}: {e}")
    print(f"Transcription complete. Output written to {transcript_path}")


def _check_diarization_args(max_speakers, min_speakers):
    if min_speakers is not None and max_speakers is not None:
        assert (
            min_speakers <= max_speakers
        ), "min-speakers must be less than or equal to max-speakers."
    if min_speakers is not None or max_speakers is not None:
        assert (
            min_speakers is not None and max_speakers is not None
        ), "Both min-speakers and max-speakers must be specified together."
        assert (
            max_speakers >= min_speakers
        ), "max-speakers must be greater than or equal to min-speakers."


def _get_already_processed_files(transcript_path: Path) -> set[str]:
    print("Found existing transcript file. Checking for already processed files...")
    already_processed = set()
    with open(transcript_path, "r", encoding="utf8") as f:
        for line in f:
            already_processed.add(
                Path(json.loads(line)["file_path"]).absolute().as_posix()
            )
    return already_processed


def _get_pipelines(
    model_name, diarization_model, do_diarization, device_id, flash, hf_token
) -> tuple[HfPipeline, PyannotePipeline | None]:
    transcription_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16,
        device="mps" if device_id == "mps" else f"cuda:{device_id}",
        model_kwargs={"attn_implementation": "flash_attention_2"}
        if flash
        else {"attn_implementation": "sdpa"},
    )
    diarization_pipeline = None
    if do_diarization:
        diarization_pipeline = PyannotePipeline.from_pretrained(
            checkpoint_path=diarization_model,
            use_auth_token=hf_token,
        )
        diarization_pipeline.to(
            torch.device("mps" if device_id == "mps" else f"cuda:{device_id}")
        )
    if device_id == "mps":
        torch.mps.empty_cache()
    return transcription_pipeline, diarization_pipeline


def _get_audio_files(file_args: list[Path]) -> list[Path]:
    files = []
    for file in file_args:
        if file.is_dir():
            files.extend(
                [
                    file
                    for file in Path(file).rglob("*")
                    if file.suffix
                    in [
                        ".wav",
                        ".mp3",
                        ".ogg",
                        ".flac",
                        ".m4a",
                        # Should also work with video files:
                        ".mp4",
                        ".mkv",
                        ".webm",
                        ".avi",
                        ".mov",
                    ]
                ]
            )
        else:
            files.append(file)
    return sorted(files)


class JsonTranscriptionResult(TypedDict):
    speakers: list
    chunks: list
    text: str
    file_path: str
