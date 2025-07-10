from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass


class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt, model_name='gemini-1.5-flash'):
        prompt = Template(prompt).render(text=ex['text'])
        #response = utils.chatgpt(
            #prompt, max_tokens=4, n=1, timeout=2, 
            #temperature=self.opt['temperature'])[0]
        response = utils.gemini(
            prompt, max_tokens=1024, n=1, 
            temperature=self.opt['temperature'],
            model_name='gemini-1.5-flash')[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred


class SGLReasoningPredictor(GPT4Predictor):
    """
    Generate responses given the question.
    """

    def inference(self, ex, prompt):
        assert 'sglang' in self.opt['engine'].lower(), "This class is desinged for SGLang-serving models"
        prompt = Template(prompt).render(text=ex['text'])
        # prompt = prompt + " " + ex['text']
        response = utils.sglang_model(
            # input_list_dict=input_dict_list,
            prompt=prompt,
            host=self.opt['host'], 
            n=self.opt['num_rollout'], 
            temperature=self.opt['engine_temperature'], 
            top_p=self.opt['top_p'], 
            max_tokens=self.opt['max_tokens'], 
            model_name=self.opt['engine'],
        )
        return response