import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from Data_Processing import *
import time

#define the DUMA_Layer
class DUMA_Layer(nn.Module):

    def __init__(self, d_model_size, num_heads):

        super(DUMA_Layer, self).__init__()
        self.attn_qa = MultiheadAttention(d_model_size, num_heads)
        self.attn_p = MultiheadAttention(d_model_size, num_heads)

    def forward(self, qa_seq_representation, p_seq_representation, qa_mask=None, p_mask=None):

        qa_seq_representation = qa_seq_representation.permute([1, 0, 2])
        p_seq_representation = p_seq_representation.permute([1, 0, 2])
        enc_output_qa, _ = self.attn_qa(
            value=qa_seq_representation,
            key=qa_seq_representation,
            query=p_seq_representation
        )
        enc_output_p, _ = self.attn_p(
            value=p_seq_representation,
            key=p_seq_representation,
            query=qa_seq_representation
        )

        return enc_output_qa.permute([1, 0, 2]), enc_output_p.permute([1, 0, 2])

#define our model for the Strange Stories Task
class DUMA(nn.Module):

    def __init__(self, freeze_bert=False):
        super(DUMA, self).__init__()

        D_in, H, D_out = 768, 50, 3
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.duma = DUMA_Layer(D_in, num_heads=self.config.num_attention_heads)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids_p, attention_mask_p, input_ids_qa, attention_mask_qa):

        # input_ids_p, input_ids_qa, attention_mask_p, attention_mask_qa = seperate(input_ids, attention_mask)
        outputs_qa = self.bert(input_ids=input_ids_qa,
                               attention_mask=attention_mask_qa)
        outputs_p = self.bert(input_ids=input_ids_p,
                              attention_mask=attention_mask_p)
        last_outputs_qa = outputs_qa.last_hidden_state
        last_outputs_p = outputs_p.last_hidden_state
        enc_outputs_qa_0, enc_outputs_p_0 = self.duma(last_outputs_qa,
                                                  last_outputs_p,
                                                  attention_mask_qa,
                                                  attention_mask_p)
        # enc_outputs_p_1, enc_outputs_qa_1 = self.duma(enc_outputs_qa_0,
        #                                           enc_outputs_p_0,
        #                                           attention_mask_qa,
        #                                           attention_mask_p)
        # try different number of DUMA Layers, default number = 1
        # enc_outputs_p_2, enc_outputs_qa_2 = self.duma(enc_outputs_qa_1,
        #                                               enc_outputs_p_1,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        # enc_outputs_p_3, enc_outputs_qa_3 = self.duma(enc_outputs_qa_2,
        #                                               enc_outputs_p_2,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        # enc_outputs_p_4, enc_outputs_qa_4 = self.duma(enc_outputs_qa_3,
        #                                               enc_outputs_p_3,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        # enc_outputs_p_5, enc_outputs_qa_5 = self.duma(enc_outputs_qa_4,
        #                                               enc_outputs_p_4,
        #                                               attention_mask_qa,
        #                                               attention_mask_p)
        fuse_output = torch.cat([enc_outputs_qa_0, enc_outputs_p_0], dim=1)
        # fuse_output = torch.cat([enc_outputs_qa_1, enc_outputs_p_1], dim=1)
        pooled_output = torch.mean(fuse_output, dim=1)
        logits = self.classifier(pooled_output)

        return logits

#read the data
dataset = get_dataset()
#split the train/val/test dataset
df_train_0, df_test = split_train(dataset, 0.1)
df_train, df_val = split_train(df_train_0, 0.1)
X_p_train = df_train.Passage.values
X_p_val = df_val.Passage.values
X_qa_train = df_train.Question.values + df_train.Answer.values
X_qa_val = df_val.Question.values + df_val.Answer.values
y_train = df_train.Score.values
y_val = df_val.Score.values

#choose the GPU if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#hyper-parameter settings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN_P = 128
MAX_LEN_QA = 32
batch_size = 24

#functions for tokenizing the data
def preprocessing_for_bert_p(data):

    input_ids = []
    attention_masks = []

    for sent in data:

        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN_P,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def preprocessing_for_bert_qa(data):

    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN_QA,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

print('Tokenizing data...')
train_inputs_p, train_masks_p = preprocessing_for_bert_p(X_p_train)
train_inputs_qa, train_masks_qa = preprocessing_for_bert_qa(X_qa_train)
val_inputs_p, val_masks_p = preprocessing_for_bert_p(X_p_val)
val_inputs_qa, val_masks_qa = preprocessing_for_bert_qa(X_qa_val)

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

#use DataLoader to package the data
train_data = TensorDataset(train_inputs_p, train_masks_p, train_inputs_qa, train_masks_qa, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs_p, val_masks_p, val_inputs_qa, val_masks_qa, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

#initialize the model
def initialize_model_d(epochs=4):

    duma_classifier = DUMA(freeze_bert=False)

    duma_classifier.to(device)

    optimizer = AdamW(duma_classifier.parameters(),
                      lr=5e-5,
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return duma_classifier, optimizer, scheduler

#define the loss function
loss_fn = nn.CrossEntropyLoss()

#define the train function
def train_d(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids_p, b_attn_mask_p, b_inputs_ids_qa, b_attn_mask_qa, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids_p, b_attn_mask_p, b_inputs_ids_qa, b_attn_mask_qa)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate_d(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")

#define the evaluate function
def evaluate_d(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids_p, b_attn_mask_p, b_input_ids_qa, b_attn_mask_qa, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids_p, b_attn_mask_p, b_input_ids_qa, b_attn_mask_qa)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

#instance the model and train
duma_classifier, optimizer, scheduler = initialize_model_d(epochs=3)
train_d(duma_classifier, train_dataloader, val_dataloader, epochs=3, evaluation=True)

#calculate the accuracy on the test dataset
text = df_test.Passage.values
qa = df_test.Question.values + df_test.Answer.values
test_inputs, test_masks = preprocessing_for_bert_p(text)
qa_inputs, qa_masks = preprocessing_for_bert_qa(qa)
y_test = df_test.Score.values
y_label = torch.tensor(y_test)

test_dataset = TensorDataset(test_inputs, test_masks, qa_inputs, qa_masks, y_label)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

test_loss, test_accuracy = evaluate_d(duma_classifier, test_dataloader)
print(test_accuracy)

#case study
case_study = '/home/yyl/code/Labs/Data/case_study/ss.xlsx'
data = pd.read_excel(case_study)
X_p = data.Passage.values
X_qr = data.Question.values + data.Answer.values
inputs_p, masks_p = preprocessing_for_bert_p(X_p)
inputs_qr, masks_qr = preprocessing_for_bert_qa(X_qr)
inputs_p = inputs_p.to(device)
masks_p = masks_p.to(device)
inputs_qr = inputs_qr.to(device)
masks_qr = masks_qr.to(device)
results = duma_classifier(inputs_p, masks_p, inputs_qr, masks_qr)
preds = torch.argmax(results, dim=1).flatten()
print(preds)
#
