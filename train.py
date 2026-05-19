from datasets import load_dataset 

dataset = dataset = load_dataset("imagefolder", data_dir="archive/crop_data")

from datasets import load_metric

metric = load_metric("accuracy")

from transformers import AutoImageProcessor

image_processor  = AutoImageProcessor.from_pretrained("ktp-crop-clip")

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    RandomRotation
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            
            Resize((224, 224)),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# split up training into training + validation
#splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = dataset['train']
val_ds = dataset['validation']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print(label2id, id2label)    
model = AutoModelForImageClassification.from_pretrained(
    'ktp-crop-clip', 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

model_checkpoint = 'ktp-kk-crop'
batch_size = 16
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    "document-crop",
    remove_unused_columns=False,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=250,
    warmup_ratio=0.1,
    logging_steps=20,
    eval_steps=20,
    save_total_limit=1,  # Only keep the best model
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Use accuracy to determine the best model
    greater_is_better=True,  # Higher accuracy is better
    push_to_hub=True,
)

import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
