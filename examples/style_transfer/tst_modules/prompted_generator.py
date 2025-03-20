from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import tiktoken
import deepspeed
from typing import Optional, List
import torch
import openai
import time
import os
from rlprompt.utils.utils import colorful_print
from vllm import LLM, SamplingParams
#os.environ['CUDA_VISIBLE_DEVICES'] ="2,3,4,5"
VLLM_MODELS = ['facebook/opt-iml-30b', 'gemma', 'gpt2', 'gemma-it', 'llama_2_chat_13', 'mistral', 'llama2_13', 'llama3', 'llama3-it', 'llama3-it-70B', 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', 'openlm-research/open_llama_3b', 'openlm-research/open_llama_3b_v2'] #
    
def retrieve_model_link(config, model):
    #
    flag = "workgroup_dev"
  
    if model == "gemma":
        return f"/mnt/workspace/{flag}/zhiqiang/huggingface_models/hub/models--google--gemma-7b/snapshots/7646584ed746494da9e1058b1be53d1be8b2ee73"
    elif model == "llama_2_chat_13":
        return f"/mnt/workspace/{flag}/zhiqiang/huggingface_models/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423"
    elif model == "mistral":
        return f"/mnt/workspace/{flag}/zhiqiang/huggingface_models/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873"
    elif model == "llama2_13":
        return f"/mnt/workspace/{flag}/phi/pret_models/models--meta-llama--Llama-2-13b-chat-hf/snapshots/4021a3b5608262f386b2bee683b6348e9228325d"
    elif model == "gemma-it":
        return f"/mnt/workspace/{flag}/phi/pret_models/gemma-7b-it"
    elif model == 'llama3':
        return f"/mnt/workspace/{flag}/zhiqiang/huggingface_models/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b"
    elif model == 'llama3-it':
        return f'/mnt/workspace/{flag}/huggingface/meta-llama/Meta-Llama-3-8B-Instruct'
    elif model == 'llama3-it-70B':
        return f'/mnt/workspace/{flag}/huggingface/meta-llama/Meta-Llama-3-70B-Instruct'
    elif model == 'gpt2':
        return f'/mnt/workspace/{flag}/jianyu/gpt2'
    
    
class PromptedGenerator:
    def __init__(
        self,
        config,
        model: str,
        template: str,
        end_punct: str,
        pad_token: str,
        device_id: int,
        lower_outputs: bool,
        control_output_length: bool,
        use_deepspeed: Optional[bool] = False, #True,
        mp_size=4,
        batch_size=1
        #llm=None
    ):
        #device_id=4
        openai.api_key="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" #"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"
        if model != 'chatgpt' and model not in VLLM_MODELS:
            self.tokenizer = AutoTokenizer.from_pretrained(model,
                                                        pad_token=pad_token,
                                                        padding_side='left')
            #with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                #self.config = AutoConfig.from_pretrained(model)
            self.generator = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).to(0)
            
            #self.generator = pipeline("text-generation",
            #                        model=model,
            #                        torch_dtype=torch.float16,
            #                        tokenizer=self.tokenizer,
            #                        #device_map="auto",
            #                        device=device_id,
            #                        trust_remote_code=True)
            
            if use_deepspeed == True:
                self.generator = deepspeed.init_inference(self.generator,
                    mp_size=mp_size,
                    dtype=torch.half,
                    replace_method="auto",
                    replace_with_kernel_inject=True)
        
        elif model in VLLM_MODELS:
            #os.environ['CUDA_VISIBLE_DEVICES'] ="0,1,2,3"
            self.tokenizer = AutoTokenizer.from_pretrained(retrieve_model_link(config, model),
                                                pad_token=pad_token)
            self.generator = LLM(retrieve_model_link(config, model), dtype='float16', tensor_parallel_size=config.num_devices) # TODO
         
        else:
            self.tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo")
        #self.generator = AutoModelForCausalLM.from_pretrained(model).to(device_id)
        #self.generator = deepspeed.init_inference(self.generator,
        #    mp_size=2,
        #    replace_method="auto",
        #    replace_with_kernel_inject=True)
        self.model = model
        self.template = template
        self.batch_size=batch_size
        self.end_punct = end_punct
        self.lower_outputs = lower_outputs
        self.device_id = device_id
        self.use_deepspeed = use_deepspeed
        self.control_output_length = control_output_length

    def _get_max_new_tokens(
        self,
        seq_len: int,
        control_output_length: Optional[bool] = None
    ) -> Optional[int]:
        if control_output_length is None:
            control_output_length = self.control_output_length
        if control_output_length:
            # This hack tends to speed up generation compared to default
            return max(1.5 * seq_len, seq_len + 10)
        else:
            return None

    def sample_generate(
        self,
        prompt: str,
        source_text: str,
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[str]:
        
        # Used for controlling output length
        formatted_template = self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_text)#self.template.format(prompt=prompt,
                             #                     sentence_1=source_text)

        if self.model != 'chatgpt' and self.model not in VLLM_MODELS:
            src_len = len(self.tokenizer(source_text)['input_ids'])
            max_new_tokens = self._get_max_new_tokens(
                src_len, control_output_length=control_output_length)
            pad_token_id = self.tokenizer.pad_token_id
            prompt_len = len(self.tokenizer.tokenize(formatted_template))
            #inputs = self.tokenizer(formatted_template, return_tensors="pt").to(f"cuda:{self.device_id}")
            input_tokens = self.tokenizer.encode(formatted_template,return_tensors="pt")#, padding=True)
            #print(input_tokens.shape)
            #generate_kwargs = dict(max_new_tokens=int(max_new_tokens),top_k=top_k, top_p=top_p, do_sample=True, num_return_sequences=num_samples)
            sample_outputs = self.generator.generate(input_tokens.to(self.device_id), max_new_tokens=int(max_new_tokens),top_k=top_k, do_sample=True, num_return_sequences=num_samples)
            #print(len(sample_outputs), len(sample_outputs[0]))
            #print(sample_outputs.shape, sample_outputs[0].shape)
            sample_outputs = sample_outputs.tolist()
            for i, sample_output in enumerate(sample_outputs):
                #print(int(input_tokens.shape[1]))
                #print(len(sample_output))
                #print(prompt_len)
                sample_outputs[i] = sample_output[int(input_tokens.shape[1]):]
            
            sample_outputs = self.tokenizer.batch_decode(torch.tensor(sample_outputs).to(self.device_id), skip_special_tokens=True)
            '''
            sample_outputs = self.generator(formatted_template,
                                            pad_token_id=pad_token_id,
                                            do_sample=True,
                                            max_new_tokens=int(max_new_tokens),
                                            top_k=top_k,
                                            top_p=top_p,
                                            num_return_sequences=num_samples,
                                            # Only return generated text, without the prompt
                                            return_full_text=False,
                                            **kwargs)
            '''
            generated_texts = []
            for output in sample_outputs:
                generated_texts.append(self.postprocess_output(output))
            return generated_texts
        
        else:
            src_len = len(self.tokenizer(source_text)['input_ids'])
            max_new_token = self._get_max_new_tokens(
                src_len, control_output_length=control_output_length)
            
            sampling_params = SamplingParams(temperature=1, top_k=top_k, max_tokens=max_new_token, n=num_samples)
            outputs = self.generator.generate([formatted_template], sampling_params)
            
            all_generated_texts = []
            for output in outputs:
                generated_texts = []
                for i in range(len(output.outputs)):
                    generated_texts.append(self.postprocess_output(output.outputs[i].text))
            
                all_generated_texts.append(generated_texts)
            return all_generated_texts

            
            """
            src_len = len(self.tokenizer.encode(source_text))
            prompt_len = len(self.tokenizer.encode(formatted_template))
            max_new_tokens = self._get_max_new_tokens(
                src_len, control_output_length=control_output_length)
        
            inference_not_done = True
            while inference_not_done:
                try:
                    print(1)
                    colorful_print(f"Send Prompt {formatted_template} to ChatGPT...", fg="green")
                    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages = [{"role": "user", "content": formatted_template}],
                            max_tokens=30,#max_new_tokens,
                            temperature=0,
                            stream=True,
                            stop=None,
                            request_timeout=5,
                            )
                    collected_messages = []
                    for chunk in response:
                        chunk_message = chunk['choices'][0]['delta']
                        collected_messages.append(chunk_message)

                    inference_not_done = False
                except:
                    print("Current Prompt:", formatted_template)
                    print(f"Waiting 2 seconds")
                    time.sleep(2)
            completion = ''.join([m.get('content', '') for m in collected_messages]) #response["choices"][0]["message"]['content']
            colorful_print(completion, fg='yellow')#return self.postprocess_output(completion)
            return self.postprocess_output(completion)
            """
    
    def parallel_sample_generate_batch(
        self,
        input_prompts: List[str],
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        all_whole_generated_texts = []
                    
        prompts = []
        whole_max_new_tokens = []
        
        prompt_separate_indices = []

        separate_len = len(source_texts)
        for prompt_index, input_prompt in enumerate(input_prompts):
            max_lengths = []
            for i, source_text in enumerate(source_texts):
                
                prompts.append(input_prompt.replace("{prompt}", input_prompt).replace("{sentence_1}", source_text))#format(prompt=prompt,sentence_1=source_text))

                src_len = len(self.tokenizer(source_text)['input_ids'])
                max_new_tokens = self._get_max_new_tokens(
                                    src_len, control_output_length=control_output_length)
                max_lengths.append(max_new_tokens)

            max_new_token=max(max_lengths)
            whole_max_new_tokens.append(max_new_token)
            prompt_separate_indices.append(prompt_index * len(source_texts))
        
        max_new_token = max(whole_max_new_tokens)
        # no sampled slicing, top_k
        if top_k == 1:
            temperature = 0

        else:
            temperature = 1

        sampling_params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_new_token, n=num_samples)
       
        outputs = self.generator.generate(prompts, sampling_params)
        
        for _ in prompt_separate_indices:
            all_generated_texts = []
            for output in outputs[_:_+len(source_texts)]:
                generated_texts = []
                for i in range(len(output.outputs)):
                    generated_texts.append(self.postprocess_output(output.outputs[i].text))
                #print(len(generated_texts))
                #print(generated_texts)
                all_generated_texts.append(generated_texts)
            all_whole_generated_texts.append(all_generated_texts)


        '''
        outputs = [self.postprocess_output(output.outputs[0].text) for output in outputs]
        for i in range(0, len(outputs), num_samples):
            temp = outputs[i:i+num_samples]
            all_generated_texts.append(temp)

        for i, prompt_repeations in enumerate(prompts):
            outputs = self.generator.generate(prompt_repeations, sampling_params)
            outputs = [output.outputs[0].text for output in outputs]
            all_generated_texts.append(outputs)
        '''
        
        return all_whole_generated_texts
    
    
    def sample_generate_batch(
        self,
        prompt: str,
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        all_generated_texts = []
        if self.use_deepspeed == True and self.model != "chatgpt" and self.model not in VLLM_MODELS:
            formatted_templates = []
            max_lengths = []
            # print(self.generator.model.__dir__)
            for i, source_text in enumerate(source_texts):
                formatted_templates.append(self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_text))#.format(prompt=prompt,sentence_1=source_text))
                src_len = len(self.tokenizer(source_text)['input_ids'])
                max_new_tokens = self._get_max_new_tokens(
                                    src_len, control_output_length=control_output_length)
                max_lengths.append(max_new_tokens)
        
            max_length=max(max_lengths)
            count = 0
            batchsize=self.batch_size
            
            #for i in tqdm(range(0, len(formatted_templates), batchsize)):
            if True:
                batch = formatted_templates#[i: i+batchsize]
                #print(batch)
                with torch.no_grad():
                    encoding = self.tokenizer(batch, padding=True, return_tensors = 'pt').to(self.device_id)
                
                    sample_outputs = self.generator.generate(**encoding, #formatted_templates,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    do_sample=True,
                                    #padding=True,
                                    max_new_tokens=max_length,
                                    top_k=top_k,
                                    num_return_sequences=num_samples,
                                )
                    sample_outputs = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
                for i in range(0,len(sample_outputs),num_samples):
                    input=self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_texts[count])#.format(prompt=prompt,sentence_1=source_texts[count])
                    generated_texts_per_input = []
                    for output in sample_outputs[i:i+num_samples]:
                        #if count == 0:
                        #    print(output, output.replace(input, ''))
                        text = self.postprocess_output(output.replace(input, ''))
                        generated_texts_per_input.append(text)
                    count += 1
                    all_generated_texts.append(generated_texts_per_input)
                    print(generated_texts_per_input)
                    
        elif self.model in VLLM_MODELS:
            #print(self.model)
            prompts = []
            max_lengths = []
            # print(self.generator.model.__dir__)
            for i, source_text in enumerate(source_texts):
                prompts.append(self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_text))#format(prompt=prompt,sentence_1=source_text))
            
                src_len = len(self.tokenizer(source_text)['input_ids'])
                max_new_tokens = self._get_max_new_tokens(
                                    src_len, control_output_length=control_output_length)
                max_lengths.append(max_new_tokens)
        
            max_new_token=max(max_lengths)
            #print(max_new_tokens)
            # adding prompts
            #print(prompts[0], len(prompts[0]))
            #print(source_texts[0],prompts[0])
            #print(len(source_texts), len(prompts))
        
            # no sampled slicing, top_k
            if top_k == 1:
                temperature = 0
               
            else:
                temperature = 1
              
            sampling_params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_new_token, n=num_samples)
            #print(prompts)
            #for i in range(0, len(prompts), 100):
            if True:
                #print(len(prompts))
                outputs = self.generator.generate(prompts, sampling_params)
                #print(outputs[0].outputs[0].text)
            
                for output in outputs:
                    generated_texts = []
                    for i in range(len(output.outputs)):
                        generated_texts.append(self.postprocess_output(output.outputs[i].text))
                    #print(len(generated_texts))
                    #print(generated_texts)
                    all_generated_texts.append(generated_texts)
                    

            '''
            outputs = [self.postprocess_output(output.outputs[0].text) for output in outputs]
            for i in range(0, len(outputs), num_samples):
                temp = outputs[i:i+num_samples]
                all_generated_texts.append(temp)
            
            for i, prompt_repeations in enumerate(prompts):
                outputs = self.generator.generate(prompt_repeations, sampling_params)
                outputs = [output.outputs[0].text for output in outputs]
                all_generated_texts.append(outputs)
            '''
        else:
            for i, source_text in tqdm(enumerate(source_texts),
                                   total=len(source_texts)):
                generated_texts = self.sample_generate(
                    prompt, source_text, num_samples, top_k, top_p,
                    lower_outputs=lower_outputs,
                    control_output_length=control_output_length)
                all_generated_texts.append(generated_texts)
                #print(all_generated_texts[-2:])
        return all_generated_texts

    def postprocess_output(
        self,
        text: str,
        end_punct: Optional[str] = None,
        lower_outputs: Optional[bool] = None
    ) -> str:
        if end_punct is None:
            end_punct = self.end_punct
        if lower_outputs is None:
            lower_outputs = self.lower_outputs

        try:
            end = text.index(end_punct)
        except ValueError:
            end = len(text)
        text = text[:end].strip()

        try:
            end = text.index('.')
        except ValueError:
            end = len(text)
        try:
            end = min(end, text.index('!'))
        except ValueError:
            end = end
        try:
            end = min(end, text.index('?'))
        except ValueError:
            end = end

        text = text[:end+1].strip()
        if lower_outputs:
            text = text.lower()

        return text
