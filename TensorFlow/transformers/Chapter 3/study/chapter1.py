from transformers import  pipeline
fill_mask = pipeline("fill-mask", model="camembert-base")
results=fill_mask("Le camembert est <mask> :)")

results=fill_mask("Le camembert est <mask> :)",top_k=3)

results=fill_mask("Le camembert est <mask> :)",targets=['good','bad'])

from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer_a = CamembertTokenizer.from_pretrained("camembert-base")
model_a = CamembertForMaskedLM.from_pretrained("camembert-base")


from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer_b = AutoTokenizer.from_pretrained("camembert-base")
model_b = AutoModelForMaskedLM.from_pretrained("camembert-base")

