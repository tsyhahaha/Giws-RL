# TinyZero 深度技术解析

## 目录

1. [veRL 框架核心概念](#verl-框架核心概念)
2. [Single Controller 设计模式](#single-controller-设计模式)
3. [Worker 通信机制](#worker-通信机制)
4. [PPO 训练流程详解](#ppo-训练流程详解)
5. [模型并行策略](#模型并行策略)
6. [关键代码路径分析](#关键代码路径分析)

---

## veRL 框架核心概念

### 设计哲学

veRL (Volcano Engine RL) 采用 **HybridFlow** 设计，核心理念是：

> "将 RL 训练的控制逻辑与计算逻辑分离，让开发者可以用几行代码定义复杂的 Post-Training 数据流"

#### 传统方法的问题

```
┌──────────────────────────────────────────┐
│           传统 Multi-Controller          │
│                                          │
│  Worker 1 ◀───────────────▶ Worker 2    │
│     │                           │        │
│     └───────────────────────────┘        │
│              直接通信                     │
│                                          │
│  问题：                                   │
│  - 复杂的跨 worker 同步                   │
│  - 难以表达复杂的数据依赖                  │
│  - 扩展性差                              │
└──────────────────────────────────────────┘
```

#### HybridFlow 解决方案

```
┌──────────────────────────────────────────┐
│           HybridFlow 架构                │
│                                          │
│  ┌─────────────────────────────────┐     │
│  │      Single Controller          │     │
│  │   (轻量级 Python 编程模型)       │     │
│  │                                 │     │
│  │   - 定义高层数据流              │     │
│  │   - 协调各个 WorkerGroup        │     │
│  │   - 执行轻量级计算 (advantage)  │     │
│  └────────────┬────────────────────┘     │
│               │ RPC                      │
│     ┌─────────┴─────────┐                │
│     ▼                   ▼                │
│  ┌─────────┐       ┌─────────┐          │
│  │WorkerGp1│       │WorkerGp2│          │
│  │(Actors) │       │(Critics)│          │
│  └─────────┘       └─────────┘          │
│                                          │
│  优势：                                   │
│  - 简洁的控制流                          │
│  - 高效的并行执行                        │
│  - 灵活的资源分配                        │
└──────────────────────────────────────────┘
```

---

## Single Controller 设计模式

### 核心组件

#### 1. WorkerGroup

`WorkerGroup` 是一组逻辑上相关的 Worker 的抽象：

```python
# verl/single_controller/base/worker_group.py

class WorkerGroup:
    def __init__(self, resource_pool: ResourcePool, **kwargs):
        self._workers = []           # 实际的 Worker 实例列表
        self._worker_names = []      # Worker 名称
    
    @property
    def world_size(self):
        return len(self._workers)
    
    def _bind_worker_method(self, user_defined_cls, func_generator):
        """
        关键方法：将 Worker 的方法绑定到 WorkerGroup
        """
        for method_name in dir(user_defined_cls):
            method = getattr(user_defined_cls, method_name)
            
            if hasattr(method, MAGIC_ATTR):
                # 获取装饰器配置
                attribute = getattr(method, MAGIC_ATTR)
                dispatch_mode = attribute['dispatch_mode']
                execute_mode = attribute['execute_mode']
                blocking = attribute['blocking']
                
                # 创建分发函数
                dispatch_fn = get_predefined_dispatch_fn(dispatch_mode)
                execute_fn = getattr(self, execute_fn_name)
                
                # 绑定新方法
                func = func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking)
                setattr(self, method_name, func)
```

#### 2. @register 装饰器

`@register` 装饰器将普通方法转换为分布式方法：

```python
# verl/single_controller/base/decorator.py

def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True):
    def decorator(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        
        # 附加元数据
        setattr(inner, MAGIC_ATTR, {
            'dispatch_mode': dispatch_mode,
            'execute_mode': execute_mode,
            'blocking': blocking,
        })
        return inner
    return decorator
```

### Dispatch 模式详解

| 模式 | 描述 | 使用场景 |
|------|------|----------|
| `ONE_TO_ALL` | 相同数据发送到所有 Worker | 模型初始化、广播配置 |
| `DP_COMPUTE_PROTO` | 按 batch 维度分片，自动合并结果 | 生成序列、计算 log prob |
| `ALL_TO_ALL` | 每个 Worker 接收/返回独立数据 | 自定义分片逻辑 |
| `MEGATRON_COMPUTE` | Megatron DP/TP/PP 分发 | Megatron 后端 |

#### DP_COMPUTE_PROTO 实现

```python
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    """
    将 DataProto 按 batch 维度分片
    """
    dp_size = worker_group.world_size
    
    def split_to_dp(data):
        if isinstance(data, DataProto):
            return data.chunk(chunks=dp_size)
        elif isinstance(data, DataProtoFuture):
            return data.chunk(chunks=dp_size)
        else:
            return [data] * dp_size
    
    # 分片所有参数
    split_args = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    return split_args

def collect_dp_compute_data_proto(worker_group, output):
    """
    合并各个 Worker 的输出
    """
    return _concat_data_proto_or_future(output)
```

---

## Worker 通信机制

### RayWorkerGroup

基于 Ray 框架实现的 WorkerGroup：

```python
# verl/single_controller/ray/base.py

class RayWorkerGroup(WorkerGroup):
    def __init__(self, resource_pool, ray_cls_with_init, ...):
        # 1. 获取 placement groups
        pgs = resource_pool.get_placement_groups()
        
        # 2. 创建 Ray actors
        for pg_idx, pg in enumerate(pgs):
            for bundle_idx in range(len(pg.bundle_specs)):
                actor = ray_cls_with_init(
                    placement_group=pg,
                    placement_group_bundle_idx=bundle_idx,
                    num_gpus=1
                )
                self._workers.append(actor)
    
    def execute_all_async(self, method_name: str, *args, **kwargs):
        """
        异步执行所有 Worker 的方法
        """
        futures = []
        for worker, arg in zip(self._workers, args):
            method = getattr(worker, method_name)
            future = method.remote(*arg, **kwargs)
            futures.append(future)
        return futures
    
    def execute_all_sync(self, method_name: str, *args, **kwargs):
        """
        同步执行（等待所有结果）
        """
        futures = self.execute_all_async(method_name, *args, **kwargs)
        return ray.get(futures)
```

### 函数生成器

为 WorkerGroup 生成实际的调用函数：

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    def func(*args, **kwargs):
        # 1. 分发数据
        dispatched_args = dispatch_fn(self, *args, **kwargs)
        
        # 2. 执行
        output = execute_fn(method_name, dispatched_args)
        
        # 3. 收集结果
        if blocking:
            output = ray.get(output) if isinstance(output[0], ray.ObjectRef) else output
            output = collect_fn(self, output)
        
        return output
    
    return func
```

---

## PPO 训练流程详解

### RayPPOTrainer 主循环

```python
# verl/trainer/ppo/ray_trainer.py

class RayPPOTrainer:
    def fit(self):
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 转换为 DataProto
                batch = DataProto.from_single_dict(batch_dict)
                
                # ========== 生成阶段 ==========
                gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                
                # 重复以匹配多个 response
                batch = batch.repeat(repeat_times=self.config.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)
                
                # 负载均衡
                self._balance_batch(batch, metrics=metrics)
                
                # ========== 计算参考 log prob ==========
                if self.use_reference_policy:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                
                # ========== 计算价值 ==========
                if self.use_critic:
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
                
                # ========== 计算奖励 ==========
                reward_tensor = self.reward_fn(batch)
                batch.batch['token_level_scores'] = reward_tensor
                
                # ========== KL 惩罚 + Advantage 计算 ==========
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl)
                batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator)
                
                # ========== 更新 Critic ==========
                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(batch)
                
                # ========== 更新 Actor ==========
                if self.config.trainer.critic_warmup <= self.global_steps:
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                
                self.global_steps += 1
```

### 数据准备流程

```python
# verl/utils/dataset/rl_dataset.py

class RLHFDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, prompt_key, max_prompt_length, ...):
        # 1. 加载 parquet 文件
        self.dataframe = pd.read_parquet(parquet_files)
        
        # 2. 提取 prompt
        self.prompts = self.dataframe[prompt_key].tolist()
        
        # 3. Tokenize
        self.tokenized = []
        for prompt in self.prompts:
            input_ids = tokenizer(prompt, truncation=True, max_length=max_prompt_length)
            self.tokenized.append(input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized[idx]['input_ids'],
            'attention_mask': self.tokenized[idx]['attention_mask'],
            'reward_model': self.dataframe.iloc[idx]['reward_model'],
            'data_source': self.dataframe.iloc[idx]['data_source'],
        }
```

---

## 模型并行策略

### FSDP (Fully Sharded Data Parallel)

```python
# verl/workers/fsdp_workers.py

def _build_model_optimizer(self, model_path, fsdp_config, ...):
    # 1. 加载预训练模型
    actor_module = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )
    
    # 2. 配置 FSDP
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    )
    
    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config)
    
    # 3. 包装为 FSDP 模型
    actor_module_fsdp = FSDP(
        actor_module,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        mixed_precision=mixed_precision,
        device_mesh=self.device_mesh
    )
    
    return actor_module_fsdp, optimizer, lr_scheduler, config
```

### vLLM 推理引擎

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py

class vLLMRollout:
    def __init__(self, actor_module, config, tokenizer, model_hf_config):
        # 初始化 vLLM LLM 引擎
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.prompt_length + config.response_length,
        )
        
        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.response_length,
        )
    
    def generate_sequences(self, prompts: DataProto):
        # 1. 预处理输入
        prompt_token_ids = _pre_process_inputs(
            self.pad_token_id, 
            prompts.batch['input_ids']
        )
        
        # 2. vLLM 生成
        output = self.inference_engine.generate(
            prompts=None,
            sampling_params=self.sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
        
        # 3. 后处理
        response_ids = [out.outputs[0].token_ids for out in output]
        return DataProto.from_dict(tensors={'responses': response_ids, ...})
```

### FSDP ↔ vLLM 权重转换

```python
# verl/workers/sharding_manager/fsdp_vllm.py

class FSDPVLLMShardingManager(BaseShardingManager):
    def __enter__(self):
        """
        进入推理模式：将 FSDP 权重同步到 vLLM
        """
        if self.full_params:
            # 收集完整权重
            with FSDP.state_dict_type(self.module, StateDictType.FULL_STATE_DICT):
                state_dict = self.module.state_dict()
            
            # 加载到 vLLM
            self.inference_engine.sync_model_weights(
                actor_weights=state_dict,
                load_format='hf'
            )
        else:
            # 使用 FSDP 分片权重直接加载 (更高效)
            self.inference_engine.sync_model_weights(
                actor_weights=self.module,
                load_format='dtensor'
            )
    
    def __exit__(self, ...):
        """
        退出推理模式：释放 vLLM 的临时缓存
        """
        self.inference_engine.free_cache_engine()
        torch.cuda.empty_cache()
```

---

## 关键代码路径分析

### 路径 1：序列生成

```
User Code                    Framework                        Worker
    │                            │                               │
    │  trainer.fit()             │                               │
    │──────────────────────────▶ │                               │
    │                            │                               │
    │           actor_rollout_wg.generate_sequences(prompts)     │
    │                            │───────────────────────────────▶
    │                            │                               │
    │                            │   1. @register(DP_COMPUTE_PROTO)
    │                            │   2. dispatch: prompts.chunk(dp_size)
    │                            │   3. execute_all_async()
    │                            │                               │
    │                            │                    ┌──────────┴──────────┐
    │                            │                    │ Worker.generate_sequences()
    │                            │                    │                            
    │                            │                    │ with rollout_sharding_manager:
    │                            │                    │   # FSDP → vLLM 权重同步
    │                            │                    │   output = vllm_rollout.generate()
    │                            │                    │   # vLLM → FSDP 权重恢复
    │                            │                    │                            
    │                            │                    │ if recompute_log_prob:
    │                            │                    │   old_log_probs = actor.compute_log_prob()
    │                            │                    └──────────┬──────────┘
    │                            │                               │
    │                            │   4. collect: DataProto.concat()
    │                            │◀──────────────────────────────│
    │                            │                               │
    │◀───────────────────────────│                               │
```

### 路径 2：Actor 更新

```
User Code                    Framework                        Worker
    │                            │                               │
    │       actor_rollout_wg.update_actor(batch)                │
    │                            │───────────────────────────────▶
    │                            │                               │
    │                            │   @register(DP_COMPUTE_PROTO) │
    │                            │                               │
    │                            │                    ┌──────────┴──────────┐
    │                            │                    │ Worker.update_actor()
    │                            │                    │                            
    │                            │                    │ # 1. 加载 offload 的参数/梯度
    │                            │                    │ load_fsdp_param_and_grad()
    │                            │                    │                            
    │                            │                    │ # 2. 多个 epoch 的 mini-batch 更新
    │                            │                    │ for epoch in ppo_epochs:
    │                            │                    │   for mini_batch in data.make_iterator():
    │                            │                    │     loss = compute_policy_loss()
    │                            │                    │     loss.backward()
    │                            │                    │     optimizer.step()
    │                            │                    │                            
    │                            │                    │ # 3. 卸载参数以节省显存
    │                            │                    │ offload_fsdp_param_and_grad()
    │                            │                    │                            
    │                            │                    │ return DataProto(meta_info={'metrics': ...})
    │                            │                    └──────────┬──────────┘
    │                            │                               │
    │◀───────────────────────────│◀──────────────────────────────│
```

---

## 调试与监控

### 日志配置

```python
# 设置日志级别
os.environ['VERL_PPO_LOGGING_LEVEL'] = 'DEBUG'

# verl/utils/debug.py
def log_gpu_memory_usage(message, logger=None):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    if logger:
        logger.info(f"{message}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
```

### Metrics 追踪

```python
# verl/trainer/ppo/ray_trainer.py

metrics = {
    # Critic 指标
    'critic/kl': current_kl,
    'critic/kl_coeff': beta,
    'critic/score/mean': torch.mean(sequence_score),
    'critic/advantages/mean': torch.mean(valid_adv),
    'critic/values/mean': torch.mean(valid_values),
    'critic/vf_explained_var': (1 - var(returns - values) / var(returns)),
    
    # Actor 指标
    'actor/lr': lr,
    'actor/loss': policy_loss,
    'actor/clipfrac': pg_clipfrac,
    
    # 性能指标
    'mfu/actor': model_flops_utilization,
    'timing_s/gen': generation_time,
    'timing_s/update_actor': update_time,
    'timing_per_token_ms/gen': gen_time_per_token,
    
    # 长度统计
    'response_length/mean': torch.mean(response_length),
    'response_length/clip_ratio': clip_ratio,
}
```

---

## 常见问题排查

### 1. OOM (Out of Memory)

```bash
# 启用 gradient checkpointing
critic.model.enable_gradient_checkpointing=True

# 减小 micro batch size
actor_rollout_ref.actor.ppo_micro_batch_size=4

# 启用参数卸载
actor.fsdp_config.param_offload=True
actor.fsdp_config.optimizer_offload=True
```

### 2. 训练不稳定

```python
# 检查 KL 散度是否过大
if kl > 0.1:
    # 增大 KL 惩罚系数
    algorithm.kl_ctrl.kl_coef *= 1.5
    
# 检查奖励分布
if reward_std < 0.01:
    # 奖励信号太弱，检查奖励函数
    pass
```

### 3. 生成质量差

```python
# 调整采样参数
actor_rollout_ref.rollout.temperature = 1.0  # 增加多样性
actor_rollout_ref.rollout.top_p = 0.95       # 使用 nucleus sampling

# 增加 response 长度限制
data.max_response_length = 2048
```

---

## 总结

TinyZero/veRL 的技术架构体现了以下设计原则：

1. **关注点分离**：Controller 负责流程控制，Worker 负责计算
2. **抽象一致性**：统一的 DataProto 协议简化了组件间通信
3. **性能优先**：混合引擎、动态 batch、序列均衡等优化
4. **易于扩展**：装饰器模式和模块化设计支持快速添加新功能

理解这些核心概念后，你可以：
- 快速定位和修复问题
- 添加新的任务和奖励函数
- 实现新的 RL 算法
- 优化训练效率
