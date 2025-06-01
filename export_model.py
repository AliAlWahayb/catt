import torch
from ed_pl import TashkeelModel as EDTashkeelModel
from eo_pl import TashkeelModel as EOTashkeelModel
from tashkeel_tokenizer import TashkeelTokenizer
import sys

# Usage: python export_model.py <model_type: eo|ed> <ckpt_path> <output_prefix>
# Example: python export_model.py ed models/best_ed_mlm_ns_epoch_178.pt exported_ed

def export_to_onnx(model, dummy_input, output_path):
    # If dummy_input is a tuple, set input_names accordingly
    if isinstance(dummy_input, tuple):
        input_names = ['input_ids', 'target_ids']
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'target_ids': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size', 1: 'seq_len'}
        }
    else:
        input_names = ['input_ids']
        dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size', 1: 'seq_len'}}
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"Exported to ONNX: {output_path}")

def export_to_torchscript(model, dummy_input, output_path):
    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)
    print(f"Exported to TorchScript: {output_path}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python export_model.py <model_type: eo|ed> <ckpt_path> <output_prefix>")
        return
    model_type, ckpt_path, output_prefix = sys.argv[1:4]
    device = 'cpu'
    tokenizer = TashkeelTokenizer()
    max_seq_len = 128  # Use a reasonable default for export
    if model_type == 'eo':
        model = EOTashkeelModel(tokenizer, max_seq_len=max_seq_len, n_layers=6, learnable_pos_emb=False)
    else:
        model = EDTashkeelModel(tokenizer, max_seq_len=max_seq_len, n_layers=6, learnable_pos_emb=False)
    # Load checkpoint correctly (Lightning checkpoint contains 'state_dict')
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval().to(device)
    # Create dummy input (batch_size=1, seq_len=max_seq_len)
    # For encoder-decoder, both x and y are required
    if model_type == 'ed':
        dummy_x = torch.randint(0, 100, (1, max_seq_len), dtype=torch.long)
        dummy_y = torch.randint(0, 100, (1, max_seq_len), dtype=torch.long)
        dummy_input = (dummy_x, dummy_y)
    else:
        dummy_input = torch.randint(0, 100, (1, max_seq_len), dtype=torch.long)
    export_to_onnx(model, dummy_input, f"{output_prefix}.onnx")
    export_to_torchscript(model, dummy_input, f"{output_prefix}_scripted.pt")

if __name__ == '__main__':
    main()
