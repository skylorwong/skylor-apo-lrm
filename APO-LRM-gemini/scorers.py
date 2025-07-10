import utils
from collections import defaultdict
import numpy as np
from liquid import Template
from tqdm import tqdm
import concurrent.futures
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    pred = predictor.inference(ex, prompt)
    return prompt, ex, pred

class Cached01Scorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(predict_on_example, ex) for ex in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
                    prompt, ex, pred = future.result()            
                    if pred == ex['label']:
                        out_scores[f'{ex}-{prompt}'] = 1
                    else:
                        out_scores[f'{ex}-{prompt}'] = 0
            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)


def logprob_on_example(inputs):
    ex, predictor, base_prompt, prompt, temperature = inputs
    lps = utils.instructGPT_logprobs(prompt, temperature=temperature)
    # last log prob is the log prob of answer (assuming single token responses)
    return base_prompt, ex, lps[0]['logprobs']['token_logprobs'][-1]


class CachedLogLikelihoodScorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = []
            for prompt, ex in prompts_exs:
                inputs.append((
                    ex,
                    predictor,
                    prompt,
                    Template(
                        prompt + ' ' + predictor.categories[ex['label']]
                        ).render(text=ex['text']),
                            predictor.opt['temperature']
                ))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(logprob_on_example, input) for input in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)
                                                ), total=len(futures), desc='ll scorer'):
                    prompt, ex, pred = future.result()            
                    out_scores[f'{ex}-{prompt}'] = pred
            return out_scores


        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))

        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)


class CachedMathVerifyScorer:
    
    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(predict_on_example, ex) for ex in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='math_verify scorer'):
                    prompt, ex, pred = future.result()
                    label_parsed = parse(ex['label'], extraction_mode="first_match")
                    pred_parsed = parse(
                        pred[0],
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    equations=True,
                                    boxed="all",
                                    units=True,
                                ),
                                # Ensures that boxed is tried first
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )

                    if verify(label_parsed, pred_parsed):
                        out_scores[f'{ex}-{prompt}'] = 1
                    else:
                        out_scores[f'{ex}-{prompt}'] = 0
            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)