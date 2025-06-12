# Changelog

所有模型分割出现的问题更改都将记录在这个文件中。

## [Unreleased]

### Added
- 无

### Changed
- ***❌显存不足-ChatGLM2DistributedInference.load_models***：
    在 `ChatGLM2DistributedInference.load_models` 方法中，由于显示**显存不足**，
    因此将设备直接换成了 `cpu`，同时将 `strict` 设置为 `False`。
    ```python
    self.first_half = ChatGLM2FirstHalf(AutoModel.from_config(self.config, trust_remote_code=True), self.split_info['split_layer'])
            state_dict_first = torch.load(
                os.path.join(self.first_half_dir, "pytorch_model.bin"),
                map_location=torch.device('cpu')
            )
            # 遍历状态字典，对每个张量调用 pin_memory 方法
            state_dict_first = {k: v.pin_memory() for k, v in state_dict_first.items()}
            self.first_half.load_state_dict(state_dict_first, strict=False)# 设置 strict 为 False
            # self.first_half.to(first_device)
            self.first_half.to('cpu')
            self.first_half.eval()
    ```

- ***❌Error：ChatGLMTokenizer***：
  - TypeError: ChatGLMTokenizer._pad() got an unexpected keyword argument 'padding_side';
            官方[issue](https://github.com/THUDM/ChatGLM3/issues/1324)提到与transformer版本有关，故切换了版本
  ```bash
    pip uninstall transformers -y; pip install transformers==4.44.2
    ```

  - setattr(self, key, value) AttributeError: property 'eos_token' of 'ChatGLMTokenizer' object has no setter
              出现了好几个与tokenizer有关错误，查找到一篇[博客](https://blog.csdn.net/qq_37085158/article/details/137118237)
              对tokenizer_config.json进行修改，修改后问题解决。
  ```json
          "added_tokens_decoder": {},
            "auto_map": {
              "AutoTokenizer": [
                "tokenization_chatglm.ChatGLMTokenizer",
                null
              ]
            },
            "clean_up_tokenization_spaces": false,
            "do_lower_case": false,
            "eos_token": "</s>",  删除
            "extra_special_tokens": {},
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": "<unk>",删除
            "padding_side": "left",
            "remove_space": false,
            "tokenizer_class": "ChatGLMTokenizer",
            "unk_token": "<unk>" 删除
        }
    ```

- ***❌GLMBlock layer参数错误***:
      由于官方的GLMBlock层的参数错误，导致在加载模型时出现了错误，因此在`ChatGLM2FirstHalf`中修改了`GLMBlock`层的参数，
      同时在`ChatGLM2SecondHalf`也修改了`GLMBlock`层的参数。
    ```python
    # 前半部分
      class ChatGLM2FirstHalf(nn.Module):
      def __init__(self, original_model, split_layer):
          super().__init__()
          self.config = original_model.config
          # 通过 transformer 访问 embedding
          self.embedding = original_model.transformer.embedding
          # 通过 transformer 访问 rotary_pos_emb
          self.rotary_pos_emb = original_model.transformer.rotary_pos_emb
          self.encoder = nn.ModuleList(list(original_model.transformer.encoder.layers)[:split_layer])
          # 通过 transformer 访问 final_layernorm
          self.final_layernorm = original_model.transformer.encoder.final_layernorm

      def forward(
              self,
              input_ids: torch.LongTensor,
              #position_ids: Optional[torch.LongTensor] = None,
              attention_mask: Optional[torch.Tensor] = None,
              #past_key_values: Optional[List[torch.FloatTensor]] = None,
              inputs_embeds: Optional[torch.FloatTensor] = None,
              use_cache: bool = False,
              # output_attentions: bool = False,
              output_hidden_states: bool = False,
              return_dict: bool = True,
      ):
      ......已有代码
   ```

    ```python
    # 后半部分
    def forward(
            self,
            hidden_states: torch.FloatTensor,
            rotary_pos_emb,
            #position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            #past_key_values: Optional[List[torch.FloatTensor]] = None,
            # use_cache: bool = False,
            # output_attentions: bool = False,
            # output_hidden_states: bool = False,
            # return_dict: bool = True,
    ):
    ......已有代码
    ```
  

- ***❌数据类型有误 RuntimeError: addmm_impl_cpu_ not implemented for Half***：
    由于在`ChatGLM2SecondHalf`中出现了`RuntimeError: addmm_impl_cpu_ not implemented for Half`错误，
    加载配置config时将数据类型改为了`torch.float32`。
    ```python
    # 加载配置
            # config_path = os.path.join(self.dir, "config.json").replace('\\', '/')  # 替换反斜杠为正斜杠
            self.config = AutoConfig.from_pretrained(self.dir,
                                                     trust_remote_code=self.trust_remote_code,
                                                     # torch_dtype = torch.float16
                                                     torch_dtype = torch.float32
                                                     )
    ```
  

- ***❌内存不足***：
    经过上述修改后还是报错：RuntimeError: [enforce fail at ..\c10\core\impl\alloc_cpu.cpp:72] data. DefaultCPUAllocator: not enough memory: you tried to allocate 224395264 bytes.