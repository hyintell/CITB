from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from AdapterT5 import AdapterT5
import random


class Seq2SeqCL:

    def __init__(self, model_args, data_args, training_args):
        super().__init__()

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        # config = AutoConfig.from_pretrained(
        #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     use_fast=model_args.use_fast_tokenizer, # default True
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )

        if training_args.cl_method == "ADAPTERCL":

            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer, # default True
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

            model = AdapterT5.from_pretrained(model_args.model_name_or_path)
            model.add_adapters(bottleneck_size=training_args.bottleneck_size, adapter_num=training_args.number_of_adpt)

            for n, p in model.named_parameters():
                if 'adapter' not in n:
                    p.requires_grad = False
                # if p.requires_grad:
                #     print(f"--- param: {n}------")

        else:

            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer, # default True
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        total_params = sum(p.numel() for p in model.parameters())
        total_tunable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tunable_params_ration = round(total_tunable_params / total_params, 4)

        # x = []
        # for n, p in model.named_parameters():
        #     print(n)
        #     x.append(n)
        #     # if 'adapter' not in n:
        #     #     p.requires_grad = False
        #     # if p.requires_grad:
        #     #     print(f"--- param: {n}------")
        # print(f"total x: {len(x)}")
        """
            Adapter+T5-LM, 512*100 for 40 tasks total params: 143544192, tunable params: 66583040, percent: 0.4639, same as init multi
            Other CL method:  total params: 76932480, tunable params: 76932480, percent: 1.0
            143544192 - 76932480 = 66611715 (total 40 tasks)
            (66611715 / 40 ) / 76932480 = 0.022 addition

            bottleneck-100 ****** total params: 108559424, tunable params: 31626944, percent: 0.2913 (19 tasks, still 2.2%)

            bottleneck-200 total params: 139748096, tunable params: 62786944, percent: 0.4493
            62786944 / 19 / 76932480 = 0.043, 4.3%

            Other CL method + multi-init/T5-LM, total named params = 190
            Adapter + multi-init/T5-LM: total named params = 2014 = 190 + 19 task * 6 layers * (8+8) encoder and decoder
        """
        print()
        print(f" ******** total params: {total_params}, tunable params: {total_tunable_params}, percent: {tunable_params_ration}")  
        print()

        # import sys
        # sys.exit(0)

        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                    f"to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )

        self.model = model
        self.tokenizer = tokenizer
        self.task_list_seen = []
        self.task_list_future = [] # to record the next task id for Adapter
        self.replay_memory = []

        self.fisher = defaultdict(list)
        self.optpar = defaultdict(list)
        self.episodic_mem = defaultdict(list)
        self.reg = training_args.reg

        # total_params = sum(p.numel() for p in self.model.parameters())
        # total_tunable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # tunable_params_ration = round(total_tunable_params / total_params, 4)
        
        # print()
        # print(f" ******** total params: {total_params}, tunable params: {total_tunable_params}, percent: {tunable_params_ration}")  
        # print()
