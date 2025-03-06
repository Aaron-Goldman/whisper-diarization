import logging
import os
import re
import gradio as gr

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mtypes = {"cpu": "int8", "cuda": "float16"}

def diarize(audio, no_stem, suppress_numerals, model_name, batch_size, language, device):
    language = process_language_arg(language, model_name)

    if not no_stem:
        # Isolate vocals from the rest of the audio
        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o temp_outputs --device "{device}"'
        )
        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = audio
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(audio))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio

    # Transcribe the audio file
    whisper_model = faster_whisper.WhisperModel(
        model_name, device=device, compute_type=mtypes[device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=batch_size,
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    # convert audio to mono for NeMo combatibility
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        torch.from_numpy(audio_waveform).unsqueeze(0).float(),
        16000,
        channels_first=True,
    )

    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    txt_output = f"{os.path.splitext(audio)[0]}.txt"

    with open(txt_output, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    cleanup(temp_path)
    return txt_output

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Audio Diarization")
        audio = gr.Audio(label="Upload Audio", type="filepath")
        with gr.Accordion("Advanced", open=False):
            no_stem = gr.Checkbox(label="Disable Source Separation", value=True)
            suppress_numerals = gr.Checkbox(label="Suppress Numerical Digits", value=True)
            whisper_model = gr.Dropdown(label="Whisper Model", choices=faster_whisper.available_models(), value="medium.en")
            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=8)
            language = gr.Dropdown(label="Language", choices=whisper_langs, value="en")
            device = gr.Dropdown(label="Device", choices=["cpu", "cuda"], value="cuda" if torch.cuda.is_available() else "cpu")
        diarize_button = gr.Button("Diarize")
        txt_output = gr.File(label="Transcript Output")

        diarize_button.click(diarize, inputs=[audio, no_stem, suppress_numerals, whisper_model, batch_size, language, device], outputs=[txt_output])

    demo.launch()

if __name__ == "__main__":
    main()
