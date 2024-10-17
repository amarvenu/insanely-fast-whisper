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

@click.command()
@click.option(
    "--file-name",
    "-f",
    required=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    help="Path or URL to the audio file to be diarized, or directory containing audio files. Allows multiple.",
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
    help="Path at which to save the diarization output RTTM(s). If multiple audio files are processed, one file will be written per input file. (default: current directory)",
    show_default=True,
)
@click.option(
    "--diarization-config",
    default="models/pyannote_diarization_config.yaml",
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
    diarization_config: str,
    min_speakers: int | None,
    max_speakers: int | None,
):
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

    diarization_pipeline = _get_pipeline(
        device_id, diarization_config
    )

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
            outpath = output_path / '.'.join(input_file_path.name.split('.')[:-1] + ['rttm'])
            with open(outpath, 'w') as rttm:
                try:
                    pbar.update(task, curr_task=f"Diarizing {input_file_path}...")

                    diarization = diarization_pipeline(input_file_path)
                    diarization.write_rttm(rttm)

                    pbar.update(task, advance=1, curr_task=f"Processed {input_file_path}.")
                except Exception as e:
                    pbar.console.print_exception()
                    pbar.console.print(f"Error processing {input_file_path}: {e}")

            print(f"Diarization complete. Output written to {outpath}")


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
    print("Found existing output directory. Checking for already processed files...")
    files = [x for x in output_path.glob('**/*') if x.is_file()]
    already_processed = set([x.absolute().as_posix() for x in files])
    return already_processed


def _get_pipeline(
    device_id, diarization_config
) -> PyannotePipeline:
    diarization_pipeline = PyannotePipeline.from_pretrained(diarization_config)

    diarization_pipeline.to(
        torch.device("mps" if device_id == "mps" else f"cuda:{device_id}")
    )
    if device_id == "mps":
        torch.mps.empty_cache()

    return diarization_pipeline


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
