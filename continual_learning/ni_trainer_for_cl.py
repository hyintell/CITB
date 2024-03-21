import string
import re
from random import sample
from collections import defaultdict
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from datasets import load_metric
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control


class NITrainer(Seq2SeqTrainer):

    # rewrite the evaluation loop, with customized call to compute_metrics
    def __init__(self, optpar=None, fisher=None, episodic_mem=None, task_list_seen=None, task_list_future=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.optpar = optpar
        self.fisher = fisher
        self.episodic_mem = episodic_mem
        self.task_list_seen = task_list_seen
        self.task_list_future = task_list_future

    
    def select_task_id_for_adapter(self, inputs):
        # in case of tasks that do not have task_id (e.g. init training tasks)
        # we randomly select a task id from current list

        if inputs['task_id'][0] not in self.task_list_future:
            
            selected_task_id = len(self.task_list_future) - 1

            # selected_task_id = random.randint(0, len(self.task_list_seen))
            print(f" +++++++ task id not in the list, randomly select: {selected_task_id}")

            return selected_task_id
        else:
            return self.task_list_future.index(inputs['task_id'][0])


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        
        # logger.info(f"----------------- compute loss -----------------")

        if self.args.cl_method == 'AGEM' and not self.args.first_task:
            # logger.info(f"----------------- AGEM, not first task -----------------")
            dev = next(model.parameters()).device
            batch_mem = sample(self.episodic_mem["all"], 1)[0]  # ==> we sample one batch from episodic memory
            model.zero_grad()
            outputs_ = model(input_ids=batch_mem["input_ids"].to(dev),
                                attention_mask=batch_mem["attention_mask"].to(dev),
                                labels=batch_mem["labels"].to(dev)
                                )
            outputs_.loss.backward(retain_graph=True)
            grad_ref = []
            for p in model.parameters():
                if p.requires_grad:
                    grad_ref.append(p.grad.view(-1))
            grad_ref = torch.cat(grad_ref)  ## from eq. 10 of AGEM Paper

            model.zero_grad()


        ## origin T5
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        
        # print(f"------ current task id: {self.task_list_future.index(inputs['task_id'][0])} ------")
        if self.args.cl_method == "ADAPTERCL":
            outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"],
                        task_id=self.select_task_id_for_adapter(inputs),
                        # task_id=self.task_list_future.index(inputs["task_id"][0]),
                        # return_dict=False
                    )
        else:
            # inputs.pop('task_id')
            outputs = model(**inputs)
        # print(f"------ LOSS: {outputs.loss} ------")


        if self.args.cl_method == 'L2' and not self.args.first_task:
            dev = next(model.parameters()).device
            l2_reg = 0

            for n,p in model.named_parameters():
                l = self.args.reg * (p - self.optpar[n].to(dev)).pow(2)
                l2_reg += l.sum()
            loss = outputs.loss + l2_reg

        elif self.args.cl_method == "EWC" and not self.args.first_task:
            
            dev = next(model.parameters()).device
            ewc_loss = 0

            for n, p in model.named_parameters():
                ## Eq (3) of https://arxiv.org/pdf/1612.00796.pdf
                l = self.args.reg * self.fisher[n].to(dev) * (p - self.optpar[n].to(dev)).pow(2)
                ewc_loss += l.sum()

            # logger.info(f"================ ewc_loss={ewc_loss}")
            loss = outputs.loss + ewc_loss
        
        elif self.args.cl_method == 'AGEM' and not self.args.first_task:
            ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
            outputs.loss.backward(retain_graph=True)
            grad_cur = []
            for p in model.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_ref).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_ref * grad_ref).sum()
                grad_proj = grad_cur - (angle / length_rep) * grad_ref
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param

            # for AGEM, return the origin T5 loss as the return loss, but we don't need this loss, so we skip
            # the loss.backward() in the training_step()
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        else:

            ################################ HF default ################################
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    # modify loss.backward()
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        # this is true
        if self.do_grad_scaling:
            # print(f"----- self.do_grad_scaling: {self.do_grad_scaling} -----------")
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            # print(f"----- self.do_grad_scaling: {self.do_grad_scaling} -----------")
            ### Modify  ##### may need to put it above
            if self.args.cl_method == 'AGEM' and not self.args.first_task:
                # logger.info(f"----------------- AGEM skip loss.backward() @@@@@@@@")
                pass
            else:
                loss.backward()

        return loss.detach()


    # def create_optimizer(self):
    #     """
    #     Setup the optimizer.
    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    #     """
    #     opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    #     if self.optimizer is None:
    #         decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    #         decay_parameters = [name for name in decay_parameters if "bias" not in name]

    #         ## ******** Origin: num1 - 1426 num2 - 2604 ********
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters],
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters],
    #                 "weight_decay": 0.0,
    #             },
    #         ]

    #         # print()
    #         # num1 = len(optimizer_grouped_parameters[0]['params'])
    #         # num2 = len(optimizer_grouped_parameters[1]['params'])
    #         # print(f" ******** Adapter: num1 - {num1} num2 - {num2} ********")
    #         # print()
    #         # import sys 
    #         # sys.exit(0)

    #         optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    #         if self.sharded_ddp == ShardedDDPOption.SIMPLE:
    #             self.optimizer = OSS(
    #                 params=optimizer_grouped_parameters,
    #                 optim=optimizer_cls,
    #                 **optimizer_kwargs,
    #             )
    #         else:
    #             self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #             if optimizer_cls.__name__ == "Adam8bit":
    #                 import bitsandbytes

    #                 manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

    #                 for module in opt_model.modules():
    #                     if isinstance(module, nn.Embedding):
    #                         manager.register_module_override(module, "weight", {"optim_bits": 32})
    #                         logger.debug(f"bitsandbytes: will optimize {module} in fp32")

    #     if is_sagemaker_mp_enabled():
    #         self.optimizer = smp.DistributedOptimizer(self.optimizer)

    #     return self.optimizer


    # Transformers v4.25.1
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.include_inputs_for_metrics
                # could use this arg
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                # 只有这里和原来的transformers的Trainer中不一样，使用的自定义的函数
                metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
                # metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        # print(f"++++++++++++++++++++++ inputs: {inputs['task_id']}")
        # print(f"++++++++++++++++++++++ generation_inputs: {generation_inputs}")
        # import sys
        # sys.exit(0)

        if self.args.cl_method == "ADAPTERCL":
            
            print(f"-------- generation_inputs: {set(inputs['task_id'])}, selected task id: {self.select_task_id_for_adapter(inputs)}")

            generated_tokens = self.model.generate(
                generation_inputs,
                # task_id=self.task_list_future.index(inputs["task_id"][0]),
                task_id=self.select_task_id_for_adapter(inputs),
                **gen_kwargs,
            )
        else:
            generated_tokens = self.model.generate(
                generation_inputs,
                **gen_kwargs,
            )

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    
                    # print(f"----- {inputs['task_id']}")
                    if self.args.cl_method == "ADAPTERCL":
                        print(f"-------- current input: {set(inputs['task_id'])}, selected task id: {self.select_task_id_for_adapter(inputs)}")

                        selected_task_id = self.select_task_id_for_adapter(inputs)
                        # remove it from the dict
                        inputs.pop("task_id")
                        outputs = model(task_id=selected_task_id, **inputs)
                    else:
                        # inputs.pop("task_id")
                        outputs = model(**inputs)

                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)
    

    