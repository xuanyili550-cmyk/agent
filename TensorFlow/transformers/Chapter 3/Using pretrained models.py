#使用预训练模型
from transformers import pipeline

camembert_fill_mask = pipeline( "fill-mask" , model= "camembert-base" )
results = camembert_fill_mask( "Le camembert est <mask> :)" )
# [{'score': 0.4909169375896454, 'token': 7200, 'token_str': 'délicieux',
#   'sequence': 'Le camembert est délicieux :)'},
#  {'score': 0.10557052493095398, 'token': 2183, 'token_str':
#      'excellent', 'sequence': 'Le camembert est excellent :)'},
#  {'score': 0.03453364968299866, 'token': 26202, 'token_str':
#      'succulent', 'sequence': 'Le camembert est succulent :)'},
#  {'score': 0.0330316536128521, 'token': 528, 'token_str':
#      'meilleur', 'sequence': 'Le camembert est meilleur :)'},
#  {'score': 0.03007686696946621, 'token': 1654, 'token_str':
#      'parfait', 'sequence': 'Le camembert est parfait :)'}]

print(results)

#使用模型架构实例化检查点
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")