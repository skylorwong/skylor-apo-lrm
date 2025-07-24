"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string
import openai
from openai import OpenAI, BadRequestError

import sys
sys.path.append("./sglang/python")
sys.path.append("./../sglang/python")


def fix_seed(seed: int):
    """Sets the seed for reproducibility across various libraries."""
    import random
    import numpy as np
    import torch
    from transformers import set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }
    retries = 0
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]

def gemini_flash_logprobs(prompt, temperature=0.7):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.GOOGLE_API_KEY}"
    }

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1
        },
        "safetySettings": [],  # Optional: Add safety settings if needed
        "tools": []  # Optional
    }

    retries = 0
    while True:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)

    response_json = r.json()
    return response_json.get('candidates', [])

def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']


def gpt(prompt, temperature=0.7, n=1, top_p=1, max_tokens=1024,
          presence_penalty=0, frequency_penalty=0, logit_bias={}, model="gpt-4o-2024-08-06"):
    client = OpenAI(api_key=config.OPENAI_KEY,
                    organization=config.OPENAI_ORG)

    messages = [{"role": "user", "content": prompt}]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            response = client.chat.completions.create(model=model,
                                                      messages=messages,
                                                      temperature=temperature,
                                                      n=n,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      presence_penalty=presence_penalty,
                                                      frequency_penalty=frequency_penalty,
                                                      logit_bias=logit_bias
                                                      )
            num_attempts = 5
            return [response.choices[i].message.content for i in range(n)]

        except BadRequestError as be:
            print(f"BadRequestError: {be}")
            continue
        except openai.RateLimitError as e:
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"OpenAI server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)  # Wait for 5 seconds before the next attempt
            continue



def gemini(prompt, temperature=0.7, n=1, top_p=1, max_tokens=1024, model_name='gemini-1.5-flash'):
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    from google.generativeai.types.generation_types import StopCandidateException
    from google.generativeai import protos

    genai.configure(api_key=config.GEMINI_KEY)

    safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": 'BLOCK_NONE'}]

    #model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    messages = prompt
    #print(f"LOGGING model : {model}")
    #print(f"LOGGING prompt : {prompt}")

    num_attempts = 0
    while num_attempts < 10:
        #print(f"LOGGING num_attempts : {num_attempts}")
        num_attempts += 1
        try:
            #print("LOGGING trying to get response")
            response = model.generate_content(messages,
                                              generation_config=genai.GenerationConfig(temperature=temperature,
                                                                                       top_p=top_p,
                                                                                       max_output_tokens=max_tokens),
                                              safety_settings=safety_settings,
                                              )
            #print("LOGGING got response")
            #print(f"LOGGING response text: {response.text}")
            FinishReason = protos.Candidate.FinishReason
            if response.candidates:
                if (response.candidates[0].finish_reason == FinishReason.STOP
                        or response.candidates[0].finish_reason == FinishReason.MAX_TOKENS):
                    out = response.text
                    num_attempts = 10
                    return [out]
                else:
                    if not response.candidates:
                        print("Generate issue: No candidates returned in response.")
                    else:
                        print(f"Generate issue {response.candidates[0].finish_reason}")
                    time.sleep(1)

        except StopCandidateException as e:
            if e.args[0].finish_reason == 3:  # Block reason is safety
                print('Blocked for Safety Reasons')
                time.sleep(1)
        except ResourceExhausted as e:  # Too many requests, wait for a minute
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
        except Exception as e:
            print(f"Other issue: {e}")
            time.sleep(1)
    return [None]


def sglang_model(prompt, host, n=1, temperature=0.6, top_p=1., max_tokens=8192, model_name='sglang_deepseek_r1_1.5b'):
    import sglang as sgl
    from sglang import function
    from sglang.lang.chat_template import get_chat_template

    if 'deepseek_r1' in model_name.lower():
        template = "deepseek-v3"
    else:
        raise Exception(f'Unsupported model: {model_name}')

    backend = sgl.RuntimeEndpoint(host)
    chat_template = get_chat_template(template)
    backend.chat_template = chat_template

    @function
    def qa(s, question):
        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer"))
        # s += sgl.assistant(sgl.gen("answer", max_tokens, temperature=temperature, top_p=top_p))
    
    # state = qa.run_batch(
    #     input_dict_list,
    #     backend=backend,
    #     num_threads=96, 
    #     progress_bar=True,
    # )
    # return [state["answer"].strip()]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            state = qa.run(question=prompt,
                            backend=backend,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p)
            num_attempts = 5
            return [state["answer"].strip()]

        except Exception as e:
            print(f"SGLang server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)
            continue