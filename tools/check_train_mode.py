# PL 2.x 학습 중 내부 BART가 train 모드(dropout 활성)인지 확인. 저장소 루트에서 실행.
import os, sys, torch
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration
sys.path.insert(0, os.getcwd())


class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('skt/kobart-base-v1')
        print('from_pretrained 직후 inner.training =', self.model.training, '/ LightningModule.training =', self.training)
        self.seen = []

    def training_step(self, batch, i):
        self.seen.append((self.training, self.model.training, self.model.model.encoder.layers[0].training))
        ids = batch[0]
        return self.model(input_ids=ids, decoder_input_ids=ids, labels=ids).loss

    def validation_step(self, batch, i):
        self.model.eval()  # model.py generate()와 같은 호출

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


ids = torch.randint(7, 29000, (8, 16))
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ids), batch_size=4)
m = M()
pl.Trainer(accelerator='cpu', max_epochs=2, logger=False, enable_checkpointing=False, enable_progress_bar=False,
           num_sanity_val_steps=0).fit(m, dl, dl)
print('training_step 중 (LightningModule, inner BART, encoder layer0) training 상태:', m.seen)
print('PL', pl.__version__)
