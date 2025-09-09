# -*- coding: utf-8 -*-
from typing import List
import openai
import time

class OpenAI_GPT4_mini_API:
    def __init__(self):
        self.model = 'gpt-4o-mini'
        self.llm = openai.OpenAI(api_key='???')
        self.auto_max_tokens = False

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format":{ "type": "json_object" },
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> str:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.RateLimitError as e: 
            print("OpenAI API request exceeded rate limit")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APITimeoutError as e:
            print("OpenAI API request timed out")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APIConnectionError as e:
            print("OpenAI API connect error")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))


        return rsp.choices[0].message.content
    
class OpenAI_GPT4_API:
    def __init__(self):
        self.model = 'gpt-4o'
        self.llm = openai.OpenAI(api_key='???')
        self.auto_max_tokens = False

    def _cons_kwargs(self, messages: List[dict], return_json) -> dict:
        if return_json:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format":{ "type": "json_object" },
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
                "n": 1,
                "stop": None,
                "temperature": 0.3,
                "timeout": 60
            }
            
        return kwargs

    def completion(self, messages: List[dict], return_json=False) -> str:
        try:
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.RateLimitError as e: 
            print("OpenAI API request exceeded rate limit")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APITimeoutError as e:
            print("OpenAI API request timed out")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except openai.APIConnectionError as e:
            print("OpenAI API connect error")
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))
        except Exception as e:
            print(e)
            time.sleep(60)
            rsp = self.llm.chat.completions.create(**self._cons_kwargs(messages, return_json))


        return rsp.choices[0].message.content
