import torch
import torchaudio
from unsloth import FastLanguageModel
from snac import SNAC
import click
import re
from pathlib import Path


def redistribute_codes(code_list, snac_model):
    """将 7-code 序列重新分配到 3 层 SNAC codes"""
    layer_1 = []
    layer_2 = []
    layer_3 = []

    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))

    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0)
    ]

    audio_hat = snac_model.decode(codes)
    return audio_hat


def split_text_smart(text, max_length=150):
    """智能分割文本，按句子边界分割"""
    # 句子结束标记
    sentence_endings = r'([。！？.!?;；]+[\s]*)'

    sentences = re.split(sentence_endings, text)

    # 重新组合句子和标点
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])

    # 如果最后一个元素不是标点，添加它
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        combined_sentences.append(sentences[-1])

    # 将句子组合成不超过 max_length 的块
    chunks = []
    current_chunk = ""

    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 如果单个句子就超过 max_length，强制分割
        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # 按逗号分割长句子
            sub_parts = re.split(r'([,，]+[\s]*)', sentence)
            temp_chunk = ""
            for part in sub_parts:
                if len(temp_chunk) + len(part) <= max_length:
                    temp_chunk += part
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk)
                    temp_chunk = part
            if temp_chunk:
                current_chunk = temp_chunk
        elif len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_audio_chunk(model, tokenizer, snac_model, text, speaker,
                         max_new_tokens, temperature, top_p, repetition_penalty):
    """生成单个文本块的音频"""

    # 准备提示
    if speaker:
        prompt = f"{speaker}: {text}"
    else:
        prompt = text

    # 编码文本
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # 添加特殊 tokens
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    input_ids = modified_input_ids.to("cuda")
    attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64).to("cuda")

    # 生成
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            eos_token_id=128258,
            use_cache=True
        )

    # 提取音频 codes
    token_to_find = 128257
    token_to_remove = 128258

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t.item() - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    if len(code_lists) == 0 or len(code_lists[0]) == 0:
        return None

    # 解码为音频
    code_list = code_lists[0]
    audio_waveform = redistribute_codes(code_list, snac_model)
    audio_waveform = audio_waveform.detach().squeeze().cpu()

    return audio_waveform


def concatenate_audio(audio_chunks, silence_duration=0.3, sample_rate=24000):
    """连接多个音频片段，中间添加静音"""
    if not audio_chunks:
        return torch.zeros(0)

    silence_samples = int(silence_duration * sample_rate)
    silence = torch.zeros(silence_samples)

    result = []
    for i, chunk in enumerate(audio_chunks):
        result.append(chunk)
        if i < len(audio_chunks) - 1:
            result.append(silence)

    return torch.cat(result)


def process_line(model, tokenizer, snac_model, line, speaker, chunk_size,
                 max_new_tokens, temperature, top_p, repetition_penalty,
                 line_silence, chunk_silence):
    """
    处理单行文本，如果超长则分块处理
    返回该行的完整音频
    """
    line = line.strip()
    if not line:
        return None

    # 检查行是否需要分块
    if len(line) <= chunk_size:
        chunks = [line]
    else:
        chunks = split_text_smart(line, chunk_size)

    # 生成每个块的音频
    chunk_audios = []
    for i, chunk in enumerate(chunks):
        audio = generate_audio_chunk(
            model, tokenizer, snac_model, chunk, speaker,
            max_new_tokens, temperature, top_p, repetition_penalty
        )

        if audio is not None:
            chunk_audios.append(audio)
            duration = audio.shape[0] / 24000
            if len(chunks) > 1:
                print(f"      Chunk {i + 1}/{len(chunks)}: {duration:.2f}s - {chunk[:40]}...")
            else:
                print(f"      Generated: {duration:.2f}s")
        else:
            print(f"      ⚠ Warning: Failed to generate chunk {i + 1}")

    if not chunk_audios:
        return None

    # 如果有多个块，用短静音连接
    if len(chunk_audios) > 1:
        line_audio = concatenate_audio(chunk_audios, silence_duration=chunk_silence)
    else:
        line_audio = chunk_audios[0]

    return line_audio


@click.command()
@click.option("--model_path", required=True, help="Path to fine-tuned LoRA model")
@click.option("--text_file", required=True, help="Path to text file (one line per audio segment)")
@click.option("--output_dir", default="./output", help="Output directory for audio files")
@click.option("--speaker", default=None, help="Speaker name (for multi-speaker models)")
@click.option("--chunk_size", default=150, type=int, help="Max characters per chunk (for long lines)")
@click.option("--line_silence", default=0.8, type=float, help="Silence duration between lines (seconds)")
@click.option("--chunk_silence", default=0.2, type=float,
              help="Silence duration between chunks within a line (seconds)")
@click.option("--save_lines", is_flag=True, help="Save each line as separate audio file")
@click.option("--merge_all", is_flag=True, default=True, help="Merge all lines into one final audio file")
@click.option("--max_new_tokens", default=1200, help="Maximum tokens to generate per chunk")
@click.option("--temperature", default=0.6, type=float, help="Sampling temperature")
@click.option("--top_p", default=0.95, type=float, help="Top-p sampling")
@click.option("--repetition_penalty", default=1.1, type=float, help="Repetition penalty")
@click.option("--max_seq_length", default=2048, help="Maximum sequence length")
def inference(model_path, text_file, output_dir, speaker, chunk_size, line_silence,
              chunk_silence, save_lines, merge_all, max_new_tokens, temperature, top_p,
              repetition_penalty, max_seq_length):
    """
    使用微调后的 Orpheus TTS 模型进行分行文本语音合成
    文本文件每行独立生成，超长行自动分块
    """

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取文本文件
    print(f"Reading text from: {text_file}")
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print("=" * 70)
    print("Orpheus TTS Line-by-Line Inference")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Text file: {text_file}")
    print(f"Total lines: {len(lines)}")
    if speaker:
        print(f"Speaker: {speaker}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Line silence: {line_silence}s")
    print(f"Chunk silence: {chunk_silence}s")
    print(f"Save individual lines: {save_lines}")
    print(f"Merge all lines: {merge_all}")
    print("=" * 70 + "\n")

    # 显示文本预览
    print("Text preview:")
    for i, line in enumerate(lines[:5], 1):
        preview = line[:60] + '...' if len(line) > 60 else line
        needs_chunk = " [WILL BE CHUNKED]" if len(line) > chunk_size else ""
        print(f"  Line {i}: {preview}{needs_chunk}")
    if len(lines) > 5:
        print(f"  ... and {len(lines) - 5} more lines")
    print()

    # 加载模型
    print("Loading base model (orpheus-3b-0.1-ft)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/orpheus-3b-0.1-ft",
        max_seq_length=max_seq_length,
        dtype="bfloat16",
        load_in_4bit=False,
        load_in_8bit=False,
    )
    print("✓ Base model loaded\n")

    print(f"Loading LoRA adapters from {model_path}...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    FastLanguageModel.for_inference(model)
    print("✓ LoRA adapters loaded\n")

    print("Loading SNAC decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cpu")
    snac_model.eval()
    print("✓ SNAC decoder loaded\n")

    # 逐行生成音频
    line_audios = []

    print("=" * 70)
    print("Generating audio line by line...")
    print("=" * 70 + "\n")

    for i, line in enumerate(lines, 1):
        print(f"[Line {i}/{len(lines)}] {line[:50]}{'...' if len(line) > 50 else ''}")

        line_audio = process_line(
            model, tokenizer, snac_model, line, speaker, chunk_size,
            max_new_tokens, temperature, top_p, repetition_penalty,
            line_silence, chunk_silence
        )

        if line_audio is None:
            print(f"  ⚠ Warning: Failed to generate audio for line {i}, skipping...\n")
            continue

        line_audios.append(line_audio)

        # 保存单独的行音频
        if save_lines:
            line_path = output_path / f"line_{i:04d}.wav"
            torchaudio.save(
                str(line_path),
                line_audio.unsqueeze(0),
                sample_rate=24000,
                encoding="PCM_S",
                bits_per_sample=16
            )
            duration = line_audio.shape[0] / 24000
            print(f"  💾 Saved to {line_path} ({duration:.2f}s)")

        print()

    if len(line_audios) == 0:
        print("\n❌ Error: No audio was generated successfully")
        return

    print(f"✓ Successfully generated {len(line_audios)}/{len(lines)} lines\n")

    # 合并所有行
    if merge_all:
        print("Merging all lines into final audio...")
        final_audio = concatenate_audio(line_audios, silence_duration=line_silence)

        # 保存最终音频
        final_path = output_path / "merged_all.wav"
        torchaudio.save(
            str(final_path),
            final_audio.unsqueeze(0),
            sample_rate=24000,
            encoding="PCM_S",
            bits_per_sample=16
        )

        duration = final_audio.shape[0] / 24000
        print(f"✓ Final audio saved to {final_path}")
        print(f"  Total duration: {duration:.2f}s ({duration / 60:.2f} minutes)")

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Lines processed: {len(line_audios)}/{len(lines)}")
    print(f"Output directory: {output_dir}")
    if save_lines:
        print(f"Individual line files: line_0001.wav to line_{len(line_audios):04d}.wav")
    if merge_all:
        print(f"Merged file: merged_all.wav")
    print("=" * 70)


if __name__ == "__main__":
    inference()