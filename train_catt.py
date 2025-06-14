import torch
import pickle
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tashkeel_dataset import TashkeelDataset, PrePaddingDataLoader
from tashkeel_tokenizer import TashkeelTokenizer
import os
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

if __name__ == "__main__":
    # Model's Configs
    model_type = 'ed' # 'eo' for Encoder-Only OR 'ed' for Encoder-Decoder
    dl_num_workers = 10
    batch_size = 8
    max_seq_len = 512
    threshold = 0.6

    # Pretrained Char-Based BERT
    pretrained_mlm_pt = None # Use None if you want to initialize weights randomly OR the path to the char-based BERT
    #pretrained_mlm_pt = 'char_bert_model_pretrained.pt'

    train_txt_folder_path = 'dataset/train'
    val_txt_folder_path = 'dataset/val'
    test_txt_folder_path = 'dataset/test'

    if model_type == 'ed':
        from ed_pl import TashkeelModel
    else:
        from eo_pl import TashkeelModel

    tokenizer = TashkeelTokenizer()

    print('Creating Train Dataset...')
    train_dataset = TashkeelDataset(train_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
    print('Creating Train Dataloader...')
    train_dataloader = PrePaddingDataLoader(tokenizer, train_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=True, persistent_workers=True)

    print('Creating Validation Dataset...')
    val_dataset = TashkeelDataset(val_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
    print('Creating Validation Dataloader...')
    val_dataloader = PrePaddingDataLoader(tokenizer, val_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=False, persistent_workers=True)

    print('Creating Test Dataset...')
    test_dataset = TashkeelDataset(test_txt_folder_path, tokenizer, max_seq_len, tashkeel_to_text_ratio_threshold=threshold)
    print('Creating Test Dataloader...')
    test_dataloader = PrePaddingDataLoader(tokenizer, test_dataset, batch_size=batch_size, num_workers=dl_num_workers, shuffle=False, persistent_workers=True)

    print('Creating Model...')
    model = TashkeelModel(tokenizer, max_seq_len=max_seq_len, n_layers=6, learnable_pos_emb=False)

    # Use the pretrained weights of the char-based BERT model to initialize the model
    if not pretrained_mlm_pt is None:
        missing = model.transformer.load_state_dict(torch.load(pretrained_mlm_pt), strict=False)
        print(f'Missing layers: {missing}')

    # This is to freeze the encoder weights
    #freeze(model.transformer.encoder)

    dirpath = f'catt_{model_type}_model_v1/'

    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, save_top_k=1, save_last=True,
                                          monitor='val_der',
                                          filename=f'catt_{model_type}_model' + '-{epoch:02d}-{val_loss:.5f}-{val_der:.5f}',
                                          save_weights_only=True,
                                          )

    # Add EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    class PrintStatsCallback(Callback):
        def on_validation_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            msg = f"Epoch {trainer.current_epoch} | "
            msg += ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
            rank_zero_info(msg)

    print('Creating Trainer...')

    logs_path = f'{dirpath}/logs'

    print('#'*100)
    print(model)
    print('#'*100)

    trainer = Trainer(
        #accelerator="cpu",
        accelerator="cuda",
        devices=-1,
        max_epochs=500,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, early_stop_callback, PrintStatsCallback()],
        logger=CSVLogger(save_dir=logs_path),
    #    strategy="ddp_find_unused_parameters_false"
        )

    # Find last checkpoint if exists
    last_ckpt = None
    ckpt_dir = os.path.join(dirpath, 'last.ckpt')
    if os.path.exists(ckpt_dir):
        last_ckpt = ckpt_dir

    #ckpt_path = 'YOUR_CKPT_PATH_GOES_HERE'
    #trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    if last_ckpt:
        print(f"Resuming from last checkpoint: {last_ckpt}")
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)

    # # Automatically run analytics after training
    # os.system('python analyze_training.py')

    # # Automatically export to ONNX and TorchScript after training
    # from sys import executable
    # import subprocess
    # model_type_str = 'ed' if model_type == 'ed' else 'eo'
    # ckpt_path = os.path.join(dirpath, 'last.ckpt')
    # output_prefix = os.path.join(dirpath, f'exported_{model_type_str}')
    # if os.path.exists(ckpt_path):
    #     print(f'Exporting model to ONNX and TorchScript: {ckpt_path}')
    #     subprocess.run([executable, 'export_model.py', model_type_str, ckpt_path, output_prefix])
    # else:
    #     print('No last.ckpt found for export.')
