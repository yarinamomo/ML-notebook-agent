# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
from transformers import AutoModel,AutoConfig,AutoTokenizer,AutoModelForMultipleChoice

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
MODEL_NAME = 'microsoft/deberta-v2-xlarge'
MODEL_NAME_CHOICE = 'vinai/phobert-base'

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
from  transformers.modeling_outputs import MultipleChoiceModelOutput

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from torch import nn

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# class CustomModelMultichoice(nn.Module):
#     def __init__(self,config,num_choice):
#         super(CustomModelMultichoice,self).__init__()
#         model = AutoModelForMultipleChoice.from_config(config)
#         model.classifier = nn.Linear(768,2)
#         self.model = model
#         ## add activation
#         self.sigmoid = nn.Sigmoid()
#         self.num_choice = num_choice
#     def forward(self,input_ids = None,token_type_ids = None ,attention_mask = None,labels = None):
#         outputs = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
#         logits = self.sigmoid(outputs.logits)
#         loss = None
#         if labels is not None:
#             loss_func = nn.NLLLoss()
#             loss = loss_func(logits.view(-1,self.num_choice),labels.view(-1))
#         return MultipleChoiceModelOutput(loss = loss,logits=logits,hidden_states= None,attentions =None)

# === AFTER (edited) ===
class CustomModelMultichoice(nn.Module):
    def __init__(self, config, num_choice):
        super(CustomModelMultichoice, self).__init__()
        # Use base AutoModel instead of AutoModelForMultipleChoice to avoid internal processing
        base_model = AutoModel.from_config(config)
        # Manually create the multiple choice model structure
        self.model = AutoModelForMultipleChoice(base_model.config)
        # Override the classifier with correct dimensions
        hidden_size = config.hidden_size
        self.model.classifier = nn.Linear(hidden_size, 1)  # 1 logit per choice
        self.num_choices = num_choice
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Flatten inputs for processing through the model
        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)
        
        # Reshape: (batch, num_choices, seq_len) -> (batch * num_choices, seq_len)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        
        # Get logits for each choice
        inputs = {'input_ids': input_ids}
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Reshape back: (batch * num_choices, 1) -> (batch, num_choices)
        logits = logits.view(batch_size, num_choices)
        
        loss = None
        if labels is not None:
            # Reshape labels to remove extra dimensions
            labels = labels.view(-1)
            # Use CrossEntropyLoss for multi-class classification
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
        
        return MultipleChoiceModelOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 6}
config = AutoConfig.from_pretrained(MODEL_NAME_CHOICE)

CustomModel = CustomModelMultichoice(config,3)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
prompt = "Bác Hồ là người nước nào ?."
candidate1 = "Việt Nam"
candidate2 = "Mỹ"
candidate3 = 'Việt Nam'

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_CHOICE)
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2],[prompt, candidate3]], return_tensors="pt", padding=True)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import torch
labels = torch.tensor(0).unsqueeze(0)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
inputs['token_type_ids'] = inputs['token_type_ids'].unsqueeze(0)
inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
CustomModel.eval()

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
target = torch.tensor([[1, 0, 1]])
target

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
out = CustomModel(**inputs,labels = target)