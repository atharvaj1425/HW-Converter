import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
from PIL import Image

# Step 1: Load IAM Dataset
dataset = load_dataset("Teklia/IAM-line")

# Step 2: Load Pretrained Processor and Model
# Step 2: Load Pretrained Processor and Model
# Step 2: Load Pretrained Processor and Model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Set required token IDs to prevent errors
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id  # Fixes decoder start token issue
model.config.pad_token_id = processor.tokenizer.pad_token_id  # Fixes pad token issue


# Step 3: Define Custom Dataset Class
class IAMDataset(Dataset):
    def __init__(self, dataset_split):
        self.dataset = dataset_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Ensure the image is in RGB format
        image = item["image"].convert("RGB") if isinstance(item["image"], Image.Image) else Image.open(item["image"]).convert("RGB")

        # Convert image to tensor
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)  # Remove batch dim

        # Tokenize text and convert to tensor
        labels = processor.tokenizer(item["text"], return_tensors="pt", padding="max_length", truncation=True).input_ids
        labels = torch.tensor(labels).squeeze(0)  # Ensure correct shape

        return {"pixel_values": pixel_values, "labels": labels}

# Step 4: Prepare Datasets
train_dataset = IAMDataset(dataset["train"])
val_dataset = IAMDataset(dataset["validation"])

# Step 5: Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned_iam",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

# Step 6: Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Model
model.save_pretrained("./trocr_finetuned_iam")
processor.save_pretrained("./trocr_finetuned_iam")

