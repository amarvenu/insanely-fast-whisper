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
    "--output-path",
    "-o",
    default=".",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, path_type=Path),
    help="Path at which to save the output JSON(s). If multiple audio files are processed, one file will be written per input file. (default: current directory)",
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
    type=click.Choice(["transcribe", "diarize", "both"]),
    help="Task to perform: transcribe, diarize, or both (default: transcribe)",
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
    "--chunk-length",
    default=30,
    type=int,
    help="Length of chunk (in seconds) when using chunked timestamps. (default: 30)",
    show_default=True,
)
@click.option(
    "--diarization-config",
    default="pyannote_diarization_config.yaml",
    type=str,
    help="Path to (offline) pyannote diairzation config",
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
    output_path: Path,
    device_id: str,
    model_name: str,
    task: str,
    language: str | None,
    batch_size: int,
    flash: bool,
    timestamp: str,
    chunk_length: int,
    diarization_config: str,
    min_speakers: int | None,
    max_speakers: int | None,
):
    if task in ['diarize', 'both']:
        _check_diarization_args(max_speakers, min_speakers)

    audio_files = _get_audio_files(file_name)
    print(f"Found {len(audio_files)} audio files to process.")

    if output_path.exists():
        already_processed = _get_already_processed_files(output_path)
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
        task, model_name, device_id, flash, diarization_config
    )

    if transcription_pipeline is not None:
        generate_kwargs = {"task": 'transcribe', "language": language}
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
    ) as pbar:
        task = pbar.add_task(
            "Processing files...", total=len(audio_files), curr_task=""
        )
        for input_file_path in audio_files:
            outpath = output_path / '.'.join(input_file_path.name.split('.')[:-1] + ['json'])
            with open(outpath, "a", encoding="utf8") as output_f:
                try:
                    chunks = []
                    text = ''
                    if transcription_pipeline is not None:
                        pbar.update(task, curr_task=f"Transcribing {input_file_path}...")
                        tr_outputs = transcription_pipeline(
                            str(input_file_path),
                            chunk_length_s=chunk_length,
                            batch_size=batch_size,
                            generate_kwargs=generate_kwargs,
                            return_timestamps=("word" if timestamp == "word" else True),
                        )

                        chunks = tr_outputs['chunks']
                        text = tr_outputs['text']

                    diarize_outputs = []
                    if diarization_pipeline is not None:
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

                        if transcription_pipeline is not None:  # if both, do the merge
                            diarize_outputs = post_process_segments_and_transcripts(
                                segments, tr_outputs["chunks"], group_by_speaker=False
                            )
                        else:
                            diarize_outputs = segments

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
            print(f"Transcription complete. Output written to {outpath}")


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


def _get_already_processed_files(output_path: Path) -> set[str]:
    print("Found existing transcript directory. Checking for already processed files...")
    files = [x for x in output_path.glob('**/*') if x.is_file()]
    already_processed = set([x.absolute().as_posix() for x in files])
    return already_processed

def _get_pipelines(
    task, model_name, device_id, flash, diarization_config
) -> tuple[HfPipeline | None, PyannotePipeline | None]:
    diarization_pipeline = None
    transcription_pipeline = None

    if task in ['transcribe', 'both']:
        transcription_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device="mps" if device_id == "mps" else f"cuda:{device_id}",
            model_kwargs={"attn_implementation": "flash_attention_2"}
            if flash
            else {"attn_implementation": "sdpa"},
        )

    if task in ['diarize', 'both']:
        diarization_pipeline = PyannotePipeline.from_pretrained(diarization_config)

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