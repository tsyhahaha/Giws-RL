# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for vLLM 0.8.5 - Uses VllmConfig instead of separate config objects

from functools import partial
from typing import Callable, Dict, Optional, Type, Union

import torch
import torch.nn as nn
from vllm.config import DecodingConfig, ObservabilityConfig, VllmConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine as VllmLLMEngine, SchedulerContext, SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.sequence import Sequence
from vllm.tracing import init_tracer
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils import Counter, weak_bind
from vllm.version import __version__ as VLLM_VERSION

from .tokenizer import TokenizerGroup

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


def _load_generation_config_dict(model_config):
    """Load generation config from model config."""
    return model_config.try_get_generation_config()


class LLMEngine:
    """An LLM engine adapted for vLLM 0.8.5 using VllmConfig.
    
    This engine receives requests and generates texts. For verl, it supports
    custom model weights injection and SPMD execution.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],  # model itself or its parameter dict
        tokenizer: nn.Module,
        vllm_config: VllmConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
        # Store vllm_config
        self.vllm_config = vllm_config
        
        # Extract individual configs
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.load_config = vllm_config.load_config
        self.decoding_config = vllm_config.decoding_config or DecodingConfig()
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config or ObservabilityConfig()
        
        self.log_stats = log_stats
        self.use_cached_outputs = use_cached_outputs

        logger.info(
            "Initializing vLLM 0.8.5 LLM engine (v%s) with config: "
            "model=%r, dtype=%s, max_seq_len=%d, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d",
            VLLM_VERSION,
            self.model_config.model,
            str(self.model_config.dtype),
            self.model_config.max_model_len,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )

        # Initialize tokenizer
        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer(tokenizer)
            self.detokenizer = Detokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        def get_tokenizer_for_seq(sequence: Sequence) -> AnyTokenizer:
            assert tokenizer_group, "tokenizer_group cannot be None"
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(self.model_config)

        self.input_preprocessor = InputPreprocessor(self.model_config, self.tokenizer)
        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(self.model_config)

        # Initialize executor with VllmConfig
        self.model_executor = executor_class(
            model=model,
            vllm_config=vllm_config,
        )

        if self.model_config.runner_type != "pooling":
            self._initialize_kv_caches()

        # Usage stats
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import get_architecture_class_name

            usage_message.report_usage(
                get_architecture_class_name(self.model_config),
                usage_context,
                extra_kvs={
                    "dtype": str(self.model_config.dtype),
                    "tensor_parallel_size": self.parallel_config.tensor_parallel_size,
                    "block_size": self.cache_config.block_size,
                    "gpu_memory_utilization": self.cache_config.gpu_memory_utilization,
                    "quantization": self.model_config.quantization,
                    "kv_cache_dtype": str(self.cache_config.cache_dtype),
                    "enable_lora": bool(self.lora_config),
                    "enable_prompt_adapter": bool(self.prompt_adapter_config),
                    "enable_prefix_caching": self.cache_config.enable_prefix_caching,
                    "enforce_eager": self.model_config.enforce_eager,
                    "disable_custom_all_reduce": self.parallel_config.disable_custom_all_reduce,
                },
            )

        if self.tokenizer:
            self.tokenizer.ping()

        self.cached_scheduler_outputs = [
            SchedulerOutputState() for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.multi_step_stream_outputs)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        if self.model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)
            self.async_callbacks = [
                partial(process_model_outputs, ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]
        else:
            self.async_callbacks = []

        self.process_request_outputs_callback: Optional[Callable] = None

        # Create scheduler
        self.scheduler = [
            Scheduler(
                self.scheduler_config,
                self.cache_config,
                self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id] if self.model_config.use_async_output_proc else None,
            )
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

        # Metric logging
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger

                self.stat_loggers = {
                    "logging": LoggingStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        vllm_config=vllm_config,
                    ),
                    "prometheus": PrometheusStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        labels=dict(model_name=self.model_config.served_model_name),
                        vllm_config=vllm_config,
                    ),
                }
                self.stat_loggers["prometheus"].info("cache_config", self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer("vllm.llm_engine", self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor
        self.output_processor = SequenceGroupOutputProcessor.create_output_processor(
            self.scheduler_config,
            self.detokenizer,
            self.scheduler,
            self.seq_counter,
            get_tokenizer_for_seq,
            stop_checker=StopChecker(
                self.scheduler_config.max_model_len,
                get_tokenizer_for_seq,
            ),
        )

    def _initialize_kv_caches(self) -> None:
        """Initialize the KV cache."""
        num_gpu_blocks, num_cpu_blocks = self.model_executor.determine_num_available_blocks()
        
        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks = self.cache_config.num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self.model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def _init_tokenizer(self, tokenizer, **tokenizer_init_kwargs):
        init_kwargs = dict(
            enable_lora=bool(self.lora_config),
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_input_length=None,
        )
        init_kwargs.update(tokenizer_init_kwargs)
        return TokenizerGroup(tokenizer, **init_kwargs)

    def get_tokenizer_group(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer

    def init_cache_engine(self):
        """Re-initialize cache engine for verl weight offloading."""
        self.model_executor.init_cache_engine()

    def free_cache_engine(self):
        """Free cache engine for verl weight offloading."""
        self.model_executor.free_cache_engine()

    @classmethod
    def _get_executor_cls(cls, vllm_config: VllmConfig) -> Type[ExecutorBase]:
        from .spmd_gpu_executor import SPMDGPUExecutor
        return SPMDGPUExecutor

    @classmethod
    def from_engine_args(
        cls,
        model,
        tokenizer,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create VllmConfig from engine args
        vllm_config = engine_args.create_engine_config(usage_context)
        
        executor_class = cls._get_executor_cls(vllm_config)

        # Create the LLM engine
        engine = cls(
            model=model,
            tokenizer=tokenizer,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        """Sync model weights for verl."""
        self.model_executor.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        """Offload model weights for verl."""
        self.model_executor.offload_model_weights()

    def _process_model_outputs(self, ctx, request_id=None, output=None):
        """Process model outputs - placeholder for async processing."""
        pass
