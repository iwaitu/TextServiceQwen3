import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def parse_args():
    parser = argparse.ArgumentParser(description="Export embedding model to dynamic-batch ONNX")
    parser.add_argument("--model-dir", required=True, help="Source Hugging Face model directory")
    parser.add_argument("--output-dir", required=True, help="Output ONNX directory")
    parser.add_argument("--max-length", type=int, default=512, help="Dummy export max sequence length")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def resolve_export_device():
    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    target_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if target_arch not in supported_arches:
        print(
            f"CUDA export fallback: device architecture {target_arch} is not supported by current PyTorch build."
        )
        return "cpu"
    return "cuda"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        padding_side="left",
        trust_remote_code=True,
    )

    device = resolve_export_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)

    wrapper = EmbeddingEncoderWrapper(model)
    sample_inputs = tokenizer(
        [
            "What is the capital of China?",
            "Beijing is the capital of China.",
        ],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = sample_inputs["input_ids"].to(device)
    attention_mask = sample_inputs["attention_mask"].to(device)

    output_path = os.path.join(args.output_dir, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
            external_data=True,
            dynamo=False,
        )

    tokenizer.save_pretrained(args.output_dir)
    print(f"Export device: {device}")
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
