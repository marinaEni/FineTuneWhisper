# -*- coding: utf-8 -*-
"""
Created on Mon July 08 


@author: marinamu
"""
import numpy as np
import torch
import random

from sklearn.utils import class_weight
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler

from early_stopper import EarlyStopper
from torch_dataset import TorchDataset
from tic_toc_class import tic_toc
from whisper_for_classification import WhisperForClassification


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For DataLoader seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed
seed = 42
set_seed(seed)


# -------------------------------------------------------------------------------------------------
class WhisperFineTune:

    def __init__(self, config: dict, train_data: dict):

        self.x_train, self.y_train = train_data.get("X", None), train_data.get("y", None)  # Data for training
        self.unfreeze_layers = int(config.get("unfreeze_layers", 0))
        self.whisper_checkpoint = config["model_extractor"]  # "whisper-tiny"
        self.n_classes_out = config["n_out"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.freeze_epochs = config["freeze_epochs"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.decay_rate = config["decay_rate"]
        # unfreeze last # self.unfreeze_layers layers or first:
        self.unfreeze_first_last = config.get("unfreeze_first_last", "last")
        self.early_stop = config['early_stop'].get('evaluate', True)
        self.patience = config["early_stop"]["patience"]

        self.model = None
        self.flag_print_model = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_input_shape = (1, 80, 3000)

    # ---------------------------------------------------------------------------------------------    
    def define_optimizer(self, model):

        # Define optimizer:
        if self.optimizer.lower() == "adam":
            optimizer = Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer.lower() == "rmsprop":
            optimizer = RMSprop(model.parameters(), lr=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=self.learning_rate)
        else:
            raise AssertionError("Mission aborted: no such optimizer: " + self.optimizer)

        # Define learning rate:
        lr_scheduler = ExponentialLR(optimizer, gamma=self.decay_rate)

        return optimizer, lr_scheduler

    # ---------------------------------------------------------------------------------------------    
    def compile_model_freezed(self):
        # Load the pre-trained Whisper model
        self.model = WhisperForClassification(model_name=f'openai/{self.whisper_checkpoint}',
                                              n_out=self.n_classes_out)

        # Set to GPU/CPU:
        self.model.to(self.device)

        # Define loss function and optimizer for the classifier only:
        self.criterion = CrossEntropyLoss() if self.n_classes_out > 1 else BCELoss()
        self.optimizer, self.scheduler = self.define_optimizer(self.model.classifier)

        # Print the model architecture:
        if self.flag_print_model:
            # summary(self.model, self.whisper_input_shape) # whisper input shape
            self.flag_print_model = 0

        print("Done compiling freezed model")

    # ---------------------------------------------------------------------------------------------
    def compile_unfreeze_encoder(self, num_layers=3):
        print("Unfreeze the first 3 layers of the Whisper model")
        # "Whisper in Focus: Enhancing Stuttered Speech Classification with Encoder Layer Optimization", 2023
        # # Unfreeze all the layers of the Whisper model
        # for param in self.whisper_model.parameters():
        #     param.requires_grad = True

        # Unfreeze the first three layers of Whisper:
        # Unfreeze specific layers
        if self.unfreeze_layers > 0:
            # Calculate total number of layers in the encoder
            num_layers = len(self.model.encoder.layers)

            if self.unfreeze_layers > num_layers:
                self.unfreeze_layers = num_layers

            if self.unfreeze_first_last == "first":
                # Unfreeze the first `self.unfreeze_layers` layers
                for i in range(self.unfreeze_layers):
                    for param in self.model.encoder.layers[i].parameters():
                        param.requires_grad = True
            else:
                # Unfreeze the last `self.unfreeze_layers` layers
                for i in range(-self.unfreeze_layers, 0):
                    for param in self.model.encoder.layers[i].parameters():
                        param.requires_grad = True

        # Optimizer for the whole model:must do it because of model.parameters()
        self.optimizer, self.scheduler = self.define_optimizer(self.model)
        print("Done unfreezing whisper encoder model")

    # ---------------------------------------------------------------------------------------------
    def train_models(self):
        ### Define weighting:
        classes = np.unique(self.y_train.tolist())
        # Calculate weights:
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=np.squeeze(self.y_train).tolist())
        weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        # Each sample in the dataset is assigned a weight based on its class: 
        sample_weights = np.array([weights_dict[y] for y in np.squeeze(self.y_train)])
        """
        Sampling with Replacement: 
            The sampler draws samples from the dataset based on the assigned weights, with replacement. 
            This means that samples can be selected more than once in an epoch, 
            especially those with higher weights.
        Creating a Balanced Batch: 
            The goal is to create batches that are more balanced in terms of class distribution, 
            which can help the model learn better representations for all classes.
        """
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))  # replacement=True

        train_dataset = TorchDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=int(self.batch_size),
                                  sampler=sampler, shuffle=(sampler is None))
        # Visualize datadistribution in each batch:
        self.visualize_data_classes(sampler)

        # Compile the Whisper+additional model while freezing all layers of Whisper:
        self.compile_model_freezed()

        # Train the model for few epochs with frozen layers:
        self._train(train_loader, int(self.freeze_epochs))

        # Unfreeze the Whisper model
        self.compile_unfreeze_encoder()

        # Continue to train the model while unfreezing Whisper:
        self._train(train_loader, int(self.self.epochs))

        return self.model, {"history": {"loss": self.loss_values}}

    # ---------------------------------------------------------------------------------------------
    def _train(self, train_loader, epochs, verbose=2):
        """
        model.train(): the model in training mode, Dropout layers and BatchNorm layers behave as 
        intended for training, applying dropout and using batch statistics, respectively.
        model.eval(): the model in evaluation mode. Dropout layers are disabled, and BatchNorm 
        layers use running statistics (computed during training) instead of batch statistics.
        """

        # Define early stopper:    
        early_stopper = EarlyStopper(patience=self.patience) if self.early_stop else None

        self.loss_values = []
        self.lrs = []

        for epoch in range(epochs):
            tic_toc.tic()
            # Set the model to training mode:
            self.model.train()
            total_correct = 0
            total_samples = 0
            epoch_loss = 0.0
            for i_batch, (x_batch, y_batch) in enumerate(train_loader):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                """
                Gradient Accumulation: During the backward pass (loss.backward()), 
                gradients are accumulated in the .grad attributes of the parameters (tensors) of 
                your model. If you don't zero out the gradients before each backward pass, 
                the gradients from the previous batch will be accumulated with the gradients of the
                current batch. This can lead to incorrect gradient updates and unstable training.
                
                Avoiding Gradient Stale Issues: By zeroing the gradients before each backward pass,
                you ensure that the gradients are fresh and only reflect the loss calculated on the
                current batch of data. This prevents stale gradients from influencing the optimization
                process.
                """
                self.optimizer.zero_grad()  # Clear the gradients

                # Run the input through the model
                outputs = self.model(x_batch)  # Forward pass. (batch_size, 1)
                # Calculate loss:
                if self.n_classes_out == 1:
                    loss = self.criterion(outputs, y_batch.float())  # y_batch: probabilities of class=1
                    # Convert probabilities to binary class labels using a threshold=0.5
                    preds = (outputs > 0.5).long()
                else:
                    loss = self.criterion(outputs, y_batch)  # y_batch: one-hot dummies
                    # For multi-class classification, get the class with the highest probability
                    preds = outputs.argmax(dim=1)

                # print("Outputs requires_grad:", outputs.requires_grad)
                # print("Outputs grad_fn:", outputs.grad_fn)
                # print("Loss requires_grad:", loss.requires_grad)
                # print("Loss grad_fn:", loss.grad_fn)

                # Back propagation:
                loss.backward()
                # Update the model weights (parameters) based on the computed gradients:
                self.optimizer.step()

                epoch_loss += loss.item()

                acc_batch = (preds == y_batch).sum().item()
                total_correct += acc_batch
                total_samples += y_batch.size(0)
                if not (i_batch + 1) % verbose:
                    print(
                        f"**Epoch #{epoch + 1}/{epochs}: Batch #{i_batch + 1}/{len(train_loader)}, Loss: {loss / len(x_batch):.4f}, acc = {100 * acc_batch / y_batch.size(0):.2f}%")
            time_epoch = tic_toc.toc()
            # Calculate the accuracy for this epoch
            accuracy = 100 * total_correct / total_samples
            # Save the values of learning rates:
            self.lrs.append(self.optimizer.param_groups[0]["lr"])
            # Update the learning rate scheduler after each epoch:
            self.scheduler.step()
            print(
                f"Done Epoch #{epoch + 1}/{epochs} ({time_epoch / 60:.2f}min): Loss: {epoch_loss / len(train_loader):.4f}, Accuracy = {accuracy:.2f}%")
            # Save the loss values for loss plot:
            self.loss_values.append(epoch_loss / len(train_loader))
            # Early stopping check:

            if early_stopper and early_stopper.early_stop(epoch_loss):
                print("Early stopping")
                break

    def visualize_data_classes(self, sampler):
        # https://gist.github.com/Chris-hughes10/260c70650c5a6f322d273a8a8728b91a
        ds = TensorDataset(torch.as_tensor([(idx, l) for idx, l in enumerate(np.squeeze(self.y_train))]))
        dl = DataLoader(ds, batch_size=int(self.batch_size),
                        worker_init_fn=seed_worker, generator=g)

        fig_no_balan = self.visualise_dataloader(dl)
        dl = DataLoader(ds, batch_size=int(self.batch_size),
                        sampler=sampler, worker_init_fn=seed_worker, generator=g)
        fig_with_balan = self.visualise_dataloader(dl)
        return fig_no_balan, fig_with_balan

    def visualise_dataloader(self, dl, id_to_label=None, with_outputs=True):
        idxs_seen = []
        class_0_batch_counts = []
        class_1_batch_counts = []

        for i, batch in enumerate(dl):

            idxs = batch[0][:, 0].tolist()
            classes = batch[0][:, 1]
            class_ids, class_counts = classes.unique(return_counts=True)
            class_ids = set(class_ids.tolist())
            class_counts = class_counts.tolist()

            idxs_seen.extend(idxs)

            if len(class_ids) == 2:
                class_0_batch_counts.append(class_counts[0])
                class_1_batch_counts.append(class_counts[1])
            elif len(class_ids) == 1 and 0 in class_ids:
                class_0_batch_counts.append(class_counts[0])
                class_1_batch_counts.append(0)
            elif len(class_ids) == 1 and 1 in class_ids:
                class_0_batch_counts.append(0)
                class_1_batch_counts.append(class_counts[0])
            else:
                raise ValueError("More than two classes detected")

        if with_outputs:
            fig, ax = plt.subplots(1, figsize=(15, 15))
            ind = np.arange(len(class_0_batch_counts))
            width = 0.35
            ax.bar(ind, class_0_batch_counts, width, label=(id_to_label[0] if id_to_label is not None else "0"))
            ax.bar(ind + width, class_1_batch_counts, width, label=(id_to_label[1] if id_to_label is not None else "1"))
            ax.set_xticks(ind, ind + 1)
            ax.set_xlabel("Batch index", fontsize=12)
            ax.set_ylabel("No. of images in batch", fontsize=12)
            ax.set_aspect("equal")
            plt.legend()
            print(
                f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 10).mean()}'
            )
            print(
                f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 10).mean()}'
            )
            print("=============")

        return fig
