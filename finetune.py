import locale
import os.path
import traceback

import datasets
from unsloth import FastLanguageModel
import click
import numpy as np
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from snac import SNAC
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq


def get_optimal_training_config(num_samples):
    """根据样本数自动选择最佳训练配置"""
    if num_samples < 1000:
        return {
            "lr_scheduler": "cosine_with_restarts",
            "epochs": 8,
            "batch_size": 4,
            "grad_accum": 4,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.15,
            "eval_steps": 50,
            "save_steps": 50,
            "note": "极小数据集 - 高 epoch + 慢学习率避免过拟合"
        }
    elif num_samples < 5000:
        return {
            "lr_scheduler": "cosine_with_restarts",
            "epochs": 5,
            "batch_size": 6,
            "grad_accum": 3,
            "learning_rate": 1.5e-4,
            "warmup_ratio": 0.1,
            "eval_steps": 100,
            "save_steps": 100,
            "note": "小数据集 - 多次重启优化"
        }
    elif num_samples < 15000:
        return {
            "lr_scheduler": "cosine_with_restarts",
            "epochs": 4,
            "batch_size": 8,
            "grad_accum": 2,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.05,
            "eval_steps": 200,
            "save_steps": 200,
            "note": "中等数据集 - 平衡性能和质量"
        }
    elif num_samples < 30000:
        return {
            "lr_scheduler": "cosine",
            "epochs": 3,
            "batch_size": 10,
            "grad_accum": 2,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.05,
            "eval_steps": 300,
            "save_steps": 300,
            "note": "大数据集 - 标准 cosine 调度"
        }
    else:
        return {
            "lr_scheduler": "cosine",
            "epochs": 2,
            "batch_size": 12,
            "grad_accum": 2,
            "learning_rate": 2.5e-4,
            "warmup_ratio": 0.03,
            "eval_steps": 500,
            "save_steps": 500,
            "note": "超大数据集 - 快速收敛"
        }


@click.command()
@click.option("--data_file", required=True, help="Path to training data JSON file")
@click.option("--output_dir", default="lora_model", help="Output directory for model")
@click.option("--batch_size", default=None, type=int, help="Training batch size (auto if not set)")
@click.option("--gradient_accumulation", default=None, type=int, help="Gradient accumulation steps (auto if not set)")
@click.option("--epochs", default=None, type=int, help="Number of training epochs (auto if not set)")
@click.option("--learning_rate", default=None, type=float, help="Learning rate (auto if not set)")
@click.option("--lr_scheduler", default=None, type=str,
              help="LR scheduler: linear/cosine/cosine_with_restarts (auto if not set)")
@click.option("--max_seq_length", default=2048, help="Maximum sequence length")
@click.option("--auto_config", is_flag=True, default=True, help="Auto-configure based on dataset size")
def finetune(data_file, output_dir, batch_size, gradient_accumulation, epochs,
             learning_rate, lr_scheduler, max_seq_length, auto_config):
    print("=" * 70)
    print("Orpheus TTS Fine-tuning - RTX 5090 Optimized (BF16)")
    print("=" * 70)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Auto-configuration: {'Enabled' if auto_config else 'Disabled'}")
    print("=" * 70 + "\n")

    # Load model with BF16 for best quality on RTX 5090
    print("Loading model in BF16 precision...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/orpheus-3b-0.1-ft",
        max_seq_length=max_seq_length,
        dtype="bfloat16",
        load_in_4bit=False,
        load_in_8bit=False,
    )
    print("✓ Model loaded successfully\n")

    # Configure LoRA for fine-tuning
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("✓ LoRA configured\n")

    # Load SNAC audio codec model
    print("Loading SNAC audio codec (24kHz)...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")
    snac_model.eval()
    print("✓ SNAC model loaded\n")

    if os.path.exists("./preprocess_cache"):
        dataset = datasets.load_from_disk("./preprocess_cache")
    else:
        # Load dataset
        print(f"Loading dataset from {data_file}...")
        dataset = load_dataset("json", data_files=data_file, split="train")
        locale.getpreferredencoding = lambda: "UTF-8"

        initial_sample_count = len(dataset)
        print(f"✓ Dataset loaded: {initial_sample_count} samples\n")

        # Check if audio field contains paths or actual audio data
        first_audio = dataset[0]["audio"]
        audio_is_path = isinstance(first_audio, str)

        if audio_is_path:
            print("Detected audio paths in dataset. Loading audio files...")
            import librosa

            def load_audio_file(example):
                """Load audio from file path"""
                try:
                    audio_path = example["audio"]
                    # Load audio using librosa
                    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                    example["audio"] = {
                        "array": waveform,
                        "sampling_rate": sample_rate
                    }
                except Exception as e:
                    print(f"⚠ Error loading audio file {example.get('audio', 'unknown')}: {e}")
                    example["audio"] = None
                return example

            dataset = dataset.map(load_audio_file, desc="Loading audio files")

            # Filter out failed audio loads
            dataset = dataset.filter(lambda x: x["audio"] is not None)
            print(f"✓ Audio files loaded: {len(dataset)} successful\n")

            if len(dataset) == 0:
                raise ValueError("No audio files could be loaded. Please check your file paths.")

        ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
        print(f"✓ Sample rate: {ds_sample_rate} Hz\n")

        def tokenise_audio(waveform):
            """Convert audio waveform to SNAC tokens"""
            waveform = torch.from_numpy(np.array(waveform)).unsqueeze(0)
            waveform = waveform.to(dtype=torch.float32)

            if ds_sample_rate != 24000:
                resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
                waveform = resample_transform(waveform)

            waveform = waveform.unsqueeze(0).to("cuda")

            with torch.inference_mode():
                codes = snac_model.encode(waveform)

            all_codes = []
            for i in range(codes[0].shape[1]):
                all_codes.append(codes[0][0][i].item() + 128266)
                all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
                all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
                all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
                all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
                all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
                all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

            return all_codes

        failed_count = 0

        def add_codes(example):
            """Process audio and add SNAC codes to example"""
            nonlocal failed_count
            codes_list = None

            try:
                answer_audio = example.get("audio")
                if answer_audio and "array" in answer_audio:
                    audio_array = answer_audio["array"]
                    codes_list = tokenise_audio(audio_array)
            except Exception as e:
                traceback.print_exc()
                print(f"⚠ Error processing audio: {e}")
                failed_count += 1

            example["codes_list"] = codes_list
            return example

        # Process all audio files to SNAC codes
        print("Processing audio files to SNAC codes...")
        dataset = dataset.map(add_codes, remove_columns=["audio"], desc="Encoding audio")
        dataset.save_to_disk("./preprocess_cache")

        if failed_count > 0:
            print(f"⚠ Failed to process {failed_count} samples")
        print(f"✓ Audio processing complete\n")

    # Free SNAC model from GPU memory
    print("Releasing SNAC model from GPU...")
    del snac_model
    torch.cuda.empty_cache()
    print("✓ GPU memory freed\n")

    # Define special tokens
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    end_of_ai = tokeniser_length + 6
    pad_token = tokeniser_length + 7

    # Filter out invalid samples
    print("Filtering dataset...")
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x["codes_list"] is not None)
    dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
    filtered_count = original_len - len(dataset)
    if filtered_count > 0:
        print(f"⚠ Removed {filtered_count} invalid samples")
    print(f"✓ Valid samples: {len(dataset)}\n")

    def remove_duplicate_frames(example):
        """Remove consecutive duplicate audio frames to reduce sequence length"""
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")

        result = vals[:7]
        removed_frames = 0

        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]

            if current_first != previous_first:
                result.extend(vals[i:i + 7])
            else:
                removed_frames += 1

        example["codes_list"] = result
        return example

    print("Removing duplicate frames...")
    dataset = dataset.map(remove_duplicate_frames, desc="Deduplicating frames")
    print("✓ Frame deduplication complete\n")

    print("=" * 70)
    print("TEXT PROMPT CONFIGURATION")
    print("=" * 70)
    print("Your dataset format will determine the prompt structure:")
    print("• Single-speaker: Uses only 'text' field")
    print("• Multi-speaker: Uses 'source' + 'text' fields")
    print("Example multi-speaker: 'Speaker1: Hello, how are you?'")
    print("=" * 70 + "\n")

    too_long_count = 0

    def create_input_ids(example):
        """Create input sequences with special tokens"""
        nonlocal too_long_count

        if "source" in example and example["source"]:
            text_prompt = f"{example['source']}: {example['text']}"
        else:
            text_prompt = example["text"]

        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)

        example["text_tokens"] = text_ids

        input_ids = (
                [start_of_human]
                + example["text_tokens"]
                + [end_of_human]
                + [start_of_ai]
                + [start_of_speech]
                + example["codes_list"]
                + [end_of_speech]
                + [end_of_ai]
        )

        if len(input_ids) > max_seq_length:
            too_long_count += 1
            return None

        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)

        return example

    print("Creating input sequences...")
    dataset = dataset.map(
        create_input_ids,
        remove_columns=["text", "codes_list"],
        desc="Tokenizing"
    )

    dataset = dataset.filter(lambda x: x is not None)

    if too_long_count > 0:
        print(f"⚠ Removed {too_long_count} sequences exceeding max length ({max_seq_length})")

    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    final_sample_count = len(dataset)
    print(f"✓ Final dataset size: {final_sample_count} samples\n")

    # Get optimal configuration based on sample count
    if auto_config:
        print("=" * 70)
        print("AUTO-CONFIGURATION")
        print("=" * 70)
        optimal_config = get_optimal_training_config(final_sample_count)

        # Override with user-specified values if provided
        batch_size = batch_size or optimal_config["batch_size"]
        gradient_accumulation = gradient_accumulation or optimal_config["grad_accum"]
        epochs = epochs or optimal_config["epochs"]
        learning_rate = learning_rate or optimal_config["learning_rate"]
        lr_scheduler = lr_scheduler or optimal_config["lr_scheduler"]
        warmup_ratio = optimal_config["warmup_ratio"]
        eval_steps = optimal_config["eval_steps"]
        save_steps = optimal_config["save_steps"]

        print(f"Dataset size: {final_sample_count} samples")
        print(f"Strategy: {optimal_config['note']}")
        print(f"\nSelected configuration:")
        print(f"  • LR Scheduler: {lr_scheduler}")
        print(f"  • Epochs: {epochs}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Gradient accumulation: {gradient_accumulation}")
        print(f"  • Effective batch size: {batch_size * gradient_accumulation}")
        print(f"  • Learning rate: {learning_rate}")
        print(f"  • Warmup ratio: {warmup_ratio}")
        print("=" * 70 + "\n")
    else:
        # Use defaults if auto_config is disabled
        batch_size = batch_size or 8
        gradient_accumulation = gradient_accumulation or 2
        epochs = epochs or 3
        learning_rate = learning_rate or 2e-4
        lr_scheduler = lr_scheduler or "cosine"
        warmup_ratio = 0.05
        eval_steps = 200
        save_steps = 200

        print(f"Using manual configuration:")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Gradient accumulation: {gradient_accumulation}")
        print(f"  • Epochs: {epochs}")
        print(f"  • Learning rate: {learning_rate}")
        print(f"  • LR Scheduler: {lr_scheduler}\n")

    # Split into train/validation sets
    print("Splitting dataset (95% train, 5% validation)...")
    dataset_split = dataset.train_test_split(test_size=0.05, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(eval_dataset)}\n")

    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Calculate total training steps
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"Training schedule:")
    print(f"  • Steps per epoch: {steps_per_epoch}")
    print(f"  • Total training steps: {total_steps}")
    print(f"  • Warmup steps: {warmup_steps}")
    print(f"  • Evaluation every: {eval_steps} steps")
    print(f"  • Save checkpoint every: {save_steps} steps\n")

    # Configure Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            # Batch size and accumulation
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,

            # Training duration
            num_train_epochs=epochs,
            max_steps=-1,

            # Learning rate schedule
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler,

            # Optimization
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,

            # Precision - BF16 for RTX 5090
            bf16=True,
            fp16=False,
            bf16_full_eval=True,

            # Memory optimization
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            # Logging
            logging_steps=50,
            logging_first_step=True,

            # Evaluation
            eval_strategy="steps",
            eval_steps=eval_steps,

            # Checkpointing
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,

            # Output
            output_dir="outputs",
            overwrite_output_dir=True,

            # Reporting
            report_to="tensorboard",

            # Reproducibility
            seed=3407,
            data_seed=3407,

            # Performance
            dataloader_num_workers=4,
            dataloader_pin_memory=True,

            # Disable features we don't need
            push_to_hub=False,
            remove_unused_columns=False,
        ),
    )

    # Start training
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Monitor training with: tensorboard --logdir outputs")
    print(f"Press Ctrl+C to interrupt training safely\n")

    try:
        trainer.train()
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70 + "\n")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

    # Save final model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")

    # Save training info
    info_file = f"{output_dir}/training_info.txt"
    with open(info_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Orpheus TTS Fine-tuning Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset Information:\n")
        f.write(f"  • Initial samples: {initial_sample_count}\n")
        f.write(f"  • Final samples: {final_sample_count}\n")
        f.write(f"  • Training samples: {len(train_dataset)}\n")
        f.write(f"  • Validation samples: {len(eval_dataset)}\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  • LR Scheduler: {lr_scheduler}\n")
        f.write(f"  • Epochs: {epochs}\n")
        f.write(f"  • Batch size: {batch_size}\n")
        f.write(f"  • Gradient accumulation: {gradient_accumulation}\n")
        f.write(f"  • Effective batch size: {batch_size * gradient_accumulation}\n")
        f.write(f"  • Learning rate: {learning_rate}\n")
        f.write(f"  • Warmup ratio: {warmup_ratio}\n")
        f.write(f"  • Precision: BF16\n")
        f.write(f"  • LoRA rank: 64\n")
        f.write(f"  • Max sequence length: {max_seq_length}\n")
        f.write(f"  • Total training steps: {total_steps}\n")
        if auto_config:
            f.write(f"\nAuto-configuration was enabled based on {final_sample_count} samples\n")

    print(f"✓ Training info saved to {info_file}\n")

    print("=" * 70)
    print("ALL DONE! 🎉")
    print("=" * 70)
    print(f"Your fine-tuned model is ready at: {output_dir}")
    print(f"Check training logs with: tensorboard --logdir outputs\n")


if __name__ == '__main__':
    finetune()