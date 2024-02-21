from pytorch_lightning import Callback
import torch
import os
from core.storage import ensure_dir, create_path
from torchmetrics.functional import confusion_matrix


class EmbeddingCollectorCallback(Callback):
    def __init__(
        self,
        path: str,
        file_name: str,
    ):
        ensure_dir(path)
        self.base_path = path
        self.file_name = file_name

    def on_test_start(self, trainer, pl_module):
        self.embeddings = {}
        self.encoder_embeddings = {}

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Assuming you return the embeddings from test_step with key 'embedding'
        if "encoder_embedding" not in outputs:
            return
        for i, label in enumerate(outputs["label"]):
            label = str(label.to("cpu").item())
            if label not in self.embeddings:
                self.embeddings[label] = [outputs["embedding"][i].to("cpu")]
                continue
            if label not in self.encoder_embeddings:
                self.encoder_embeddings[label] = [
                    outputs["encoder_embedding"][i].to("cpu")
                ]
                continue
            self.encoder_embeddings[label].append(
                outputs["encoder_embedding"][i].to("cpu")
            )
            self.embeddings[label].append(outputs["embedding"][i].to("cpu"))

    def on_test_end(self, trainer, pl_module):
        if self.embeddings:
            return
        for label, tensors in self.embeddings.items():
            all_embeddings = torch.stack(tensors, dim=0)
            new_path = create_path(self.base_path, label)
            ensure_dir(new_path)
            torch.save(
                all_embeddings, os.path.join(new_path, f"{self.file_name}_linear.pt")
            )
        for label, tensors in self.encoder_embeddings.items():
            all_embeddings = torch.stack(tensors, dim=0)
            new_path = create_path(self.base_path, label)
            ensure_dir(new_path)
            torch.save(
                all_embeddings, os.path.join(new_path, f"{self.file_name}_encoder.pt")
            )
        # Now all_embeddings is a tensor of shape (Batchsize*Stepsize, 1024)
        # Do whatever you need to do with all_embeddings
        # You can also set it as an attribute to the pl_module if needed

        # trainer.all_test_embeddings = all_embeddings


class ConfusionMatrixCallback(Callback):
    def __init__(
        self,
        path: str,
        file_name: str,
    ):
        ensure_dir(path)
        self.base_path = path
        self.file_name = file_name

    def on_test_start(self, trainer, pl_module):
        self.test_step_ys = []
        self.test_step_y_hats = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_step_ys.append(outputs["label"])
        self.test_step_y_hats.append(outputs["preds"])

    def on_test_end(self, trainer, pl_module):
        y_hat = torch.cat(self.test_step_y_hats)
        y = torch.cat(self.test_step_ys)
        conf_matrix = confusion_matrix(y_hat.view(-1), y.int(), task="binary")
        print(conf_matrix)
        torch.save(
            conf_matrix, os.path.join(self.base_path, f"{self.file_name}_encoder.pt")
        )
