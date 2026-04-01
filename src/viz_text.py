import argparse
import importlib
import torch
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

def main():
    parser = argparse.ArgumentParser(description="Data-agnostic Energon Text Inspector.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Energon dataset.")
    parser.add_argument("--encoder-module", type=str, default="task_encoders", help="Module containing the TaskEncoder.")
    parser.add_argument("--encoder-class", type=str, required=True, help="TaskEncoder class name.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID for the encoder.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--output", type=str, default="token_map.png", help="Ignored in this script, kept for arg compatibility.")
    args = parser.parse_args()

    # Dynamic loading of the TaskEncoder
    module = importlib.import_module(args.encoder_module)
    encoder_cls = getattr(module, args.encoder_class)
    encoder = encoder_cls(model_id=args.model_id, max_length=args.max_length)

    print(f"Using {args.encoder_class} from {args.encoder_module}")
    print(f"Loading dataset from {args.dataset}...\n")
    
    dataset = get_train_dataset(
        args.dataset,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        task_encoder=encoder,
        batch_size=args.batch_size,
        shuffle_buffer_size=10,
        max_samples_per_sequence=None,
        packing_buffer_size=getattr(encoder, "packing_buffer_size", None)
    )
    loader = get_loader(dataset)

    print(f"🔄 Processing 1 batch...\n")
    batch = next(iter(loader))
    
    input_ids = batch['input_ids']
    cu_seqlens = batch.get('cu_seqlens')

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.unsqueeze(0) if cu_seqlens.dim() == 1 else cu_seqlens

    batch_size = input_ids.shape[0]

    for i in range(batch_size):
        print(f"=== Batch Item {i} ===")
        
        if cu_seqlens is not None:
            seq_lens = cu_seqlens[i].tolist()
            for j in range(len(seq_lens) - 1):
                start = seq_lens[j]
                end = seq_lens[j+1]
                if start == end: 
                    continue
                    
                seq_ids = input_ids[i, start:end]
                text = encoder.tokenizer.decode(seq_ids)
                print(f"\n[Sequence {j} | Indices {start}:{end}]")
                print(text)
        else:
            text = encoder.tokenizer.decode(input_ids[i])
            print(f"\n[Unpacked Sequence]")
            print(text)
            
        print("-" * 50)

if __name__ == "__main__":
    main()