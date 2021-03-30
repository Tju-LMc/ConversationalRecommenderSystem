import os
import torch
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentRuleConfig():
    def __init__(self):
        self.all_facet_list = ["categories", "state", "city", "price", "stars"] # facet list
        self.facet_dim_list = [189, 11, 189, 4, 9] # value kinds of each facet
        self.utt_gen_dict_path = root_path + "/data/agents/utt_gen_dict.json"
        # self.use_gpu = True # use gpu or not
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")