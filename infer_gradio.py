import os

os.environ["GRADIO_TEMP_DIR"] = "./temp"

import torch
import torchaudio
import gradio as gr
from unsloth import FastLanguageModel
from snac import SNAC
import re
import tempfile

# 全局变量存储模型
model = None
tokenizer = None
snac_model = None
model_loaded = False


def redistribute_codes(code_list, snac_model):
    """将 7-code 序列重新分配到 3 层 SNAC codes"""
    layer_1, layer_2, layer_3 = [], [], []

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
    sentence_endings = r'([。！？.!?;；]+[\s]*)'
    sentences = re.split(sentence_endings, text)

    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])

    if len(sentences) % 2 == 1 and sentences[-1].strip():
        combined_sentences.append(sentences[-1])

    chunks = []
    current_chunk = ""

    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
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


def generate_audio_chunk(text, speaker, max_new_tokens, temperature, top_p, repetition_penalty):
    """生成单个文本块的音频"""
    global model, tokenizer, snac_model

    prompt = f"{speaker}: {text}" if speaker else text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    input_ids = modified_input_ids.to("cuda")
    attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64).to("cuda")

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


def load_model(model_path, max_seq_length=2048):
    """加载模型"""
    global model, tokenizer, snac_model, model_loaded

    if model_loaded:
        return "模型已加载"

    try:
        # 加载基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/orpheus-3b-0.1-ft",
            max_seq_length=max_seq_length,
            dtype="bfloat16",
            load_in_4bit=False,
            load_in_8bit=False,
        )

        # 加载 LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        FastLanguageModel.for_inference(model)

        # 加载 SNAC
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        snac_model = snac_model.to("cpu")
        snac_model.eval()

        model_loaded = True
        return "✅ 模型加载成功！"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"


def generate_tts(text, speaker, chunk_size, line_silence, chunk_silence,
                 max_new_tokens, temperature, top_p, repetition_penalty):
    """主生成函数"""
    global model_loaded

    if not model_loaded:
        return None, "❌ 请先加载模型！"

    if not text.strip():
        return None, "❌ 请输入文本！"

    try:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        line_audios = []
        log_messages = []

        for i, line in enumerate(lines, 1):
            log_messages.append(f"[Line {i}/{len(lines)}] {line[:50]}...")

            # 分块处理
            if len(line) <= chunk_size:
                chunks = [line]
            else:
                chunks = split_text_smart(line, chunk_size)

            chunk_audios = []
            for j, chunk in enumerate(chunks):
                audio = generate_audio_chunk(
                    chunk, speaker if speaker else None,
                    max_new_tokens, temperature, top_p, repetition_penalty
                )
                if audio is not None:
                    chunk_audios.append(audio)
                    duration = audio.shape[0] / 24000
                    log_messages.append(f"  Chunk {j + 1}: {duration:.2f}s")

            if chunk_audios:
                if len(chunk_audios) > 1:
                    line_audio = concatenate_audio(chunk_audios, silence_duration=chunk_silence)
                else:
                    line_audio = chunk_audios[0]
                line_audios.append(line_audio)

        if not line_audios:
            return None, "❌ 生成失败，没有产生音频"

        # 合并所有行
        final_audio = concatenate_audio(line_audios, silence_duration=line_silence)

        # 保存到临时文件（当前目录）
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir)
        torchaudio.save(
            temp_file.name,
            final_audio.unsqueeze(0),
            sample_rate=24000,
            encoding="PCM_S",
            bits_per_sample=16
        )

        duration = final_audio.shape[0] / 24000
        log_messages.append(f"\n✅ 生成完成！总时长: {duration:.2f}s")

        return temp_file.name, "\n".join(log_messages)

    except Exception as e:
        return None, f"❌ 生成出错: {str(e)}"


# 创建 Gradio 界面
with gr.Blocks(title="Orpheus TTS") as demo:
    gr.Markdown("# 🎙️ Orpheus TTS 语音合成")
    gr.Markdown("基于微调的 Orpheus 模型进行文本转语音")

    with gr.Row():
        with gr.Column(scale=1):
            model_path_input = gr.Textbox(
                label="模型路径",
                placeholder="输入 LoRA 模型路径...",
                value=""
            )
            load_btn = gr.Button("加载模型", variant="primary")
            load_status = gr.Textbox(label="加载状态", interactive=False)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="输入要合成的文本，支持多行...",
                lines=8
            )
            speaker_input = gr.Textbox(
                label="说话人 (可选)",
                placeholder="例如: tara, leo 等",
                value=""
            )

        with gr.Column(scale=1):
            with gr.Accordion("高级参数", open=False):
                chunk_size = gr.Slider(50, 300, value=150, step=10, label="分块大小")
                line_silence = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="行间静音(秒)")
                chunk_silence = gr.Slider(0.1, 1.0, value=0.2, step=0.1, label="块间静音(秒)")
                max_new_tokens = gr.Slider(500, 2000, value=1200, step=100, label="最大生成tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.05, label="Top-p")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="重复惩罚")

    generate_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")

    with gr.Row():
        audio_output = gr.Audio(label="生成的音频", type="filepath",interactive=True, buttons=['download'])
        log_output = gr.Textbox(label="生成日志", lines=10, interactive=False)

    # 绑定事件
    load_btn.click(
        fn=load_model,
        inputs=[model_path_input],
        outputs=[load_status]
    )

    generate_btn.click(
        fn=generate_tts,
        inputs=[
            text_input, speaker_input, chunk_size, line_silence, chunk_silence,
            max_new_tokens, temperature, top_p, repetition_penalty
        ],
        outputs=[audio_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", root_path="https://orp.portal.1395812.xyz/")