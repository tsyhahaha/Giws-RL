# TinyZero 代码库架构详解

## 📖 项目概述

**TinyZero** 是一个基于 [veRL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLM) 框架的项目，旨在复现 [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) 的自主推理能力训练方法。

### 核心目标
- 通过强化学习 (RL) 使 3B 参数的基座语言模型自主发展出 **自我验证** 和 **搜索推理** 能力
- 提供低成本（< $30）的实验方案
- 支持 countdown（倒计时算术）和 multiply（乘法）任务

---

## 🏗️ 整体架构设计思想

### 1. HybridFlow 混合编程模型

veRL 的核心设计思想是 **HybridFlow**，结合了两种分布式编程范式的优势：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Single Controller                           │
│  (Driver Process - 负责调度和轻量级计算)                           │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    RayPPOTrainer                           │   │
│  │  - 创建数据加载器                                           │   │
│  │  - 协调各个 WorkerGroup                                     │   │
│  │  - 计算 Advantage (轻量级)                                  │   │
│  │  - 管理训练循环                                             │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ RPC 调用
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Multi Controller / Workers                     │
│  (GPU Workers - 负责重计算任务)                                    │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ActorRollout  │  │   Critic     │  │   RefPolicy  │            │
│  │   Worker     │  │   Worker     │  │   Worker     │            │
│  │              │  │              │  │              │            │
│  │ - 生成序列   │  │ - 计算价值   │  │ - 计算参考   │            │
│  │ - 更新Actor │  │ - 更新Critic │  │   log prob   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 设计原则

1. **计算与数据解耦**：通过 `DataProto` 协议统一数据交换格式
2. **灵活的设备映射**：支持多种模型在不同 GPU 集群上的部署策略
3. **模块化集成**：无缝集成 PyTorch FSDP、Megatron-LM、vLLM 等框架
4. **混合引擎 (Hybrid Engine)**：训练和推理共享模型权重，通过 3D-HybridEngine 实现高效切换

---

## 📁 目录结构详解

```
TinyZero/
├── verl/                          # 核心框架代码
│   ├── protocol.py                # 数据传输协议 (DataProto)
│   ├── single_controller/         # 单控制器模式实现
│   │   ├── base/                  # 基础抽象类
│   │   │   ├── decorator.py       # 分发/执行装饰器
│   │   │   ├── worker.py          # Worker 基类
│   │   │   └── worker_group.py    # WorkerGroup 管理
│   │   └── ray/                   # Ray 后端实现
│   │       └── base.py            # RayWorkerGroup
│   │
│   ├── trainer/                   # 训练器
│   │   ├── ppo/                   # PPO 算法
│   │   │   ├── core_algos.py      # 核心算法 (GAE, KL, Policy Loss)
│   │   │   └── ray_trainer.py     # Ray 分布式 PPO Trainer
│   │   └── main_ppo.py            # 主入口
│   │
│   ├── workers/                   # Worker 实现
│   │   ├── fsdp_workers.py        # FSDP 后端 Workers
│   │   ├── megatron_workers.py    # Megatron-LM 后端
│   │   ├── actor/                 # Actor 模型
│   │   ├── critic/                # Critic 模型
│   │   ├── rollout/               # 推理引擎
│   │   │   ├── vllm_rollout/      # vLLM 推理
│   │   │   └── hf_rollout.py      # HuggingFace 推理
│   │   └── sharding_manager/      # 权重分片管理
│   │       ├── fsdp_vllm.py       # FSDP ↔ vLLM 权重转换
│   │       └── fsdp_ulysses.py    # FSDP + Ulysses SP
│   │
│   ├── models/                    # 模型适配
│   └── utils/                     # 工具函数
│       ├── reward_score/          # 奖励计算
│       │   ├── countdown.py       # Countdown 任务奖励
│       │   └── gsm8k.py           # GSM8K 任务奖励
│       └── dataset/               # 数据处理
│
├── examples/                      # 示例代码
│   ├── data_preprocess/           # 数据预处理脚本
│   │   ├── countdown.py           # Countdown 数据准备
│   │   └── gsm8k.py               # GSM8K 数据准备
│   └── ppo_trainer/               # PPO 训练配置
│
├── scripts/                       # 运行脚本
│   └── train_tiny_zero.sh         # TinyZero 训练入口
│
└── tests/                         # 测试代码
```

---

## 🔄 核心数据流架构

### DataProto 协议

`DataProto` 是 veRL 中统一的数据交换协议，用于在不同组件间传递数据：

```python
@dataclass
class DataProto:
    batch: TensorDict = None          # 张量数据 (PyTorch TensorDict)
    non_tensor_batch: Dict = {}       # 非张量数据 (numpy arrays)
    meta_info: Dict = {}              # 元信息

    # 核心方法
    def chunk(self, chunks: int)      # 分片 (用于数据并行)
    def concat(data: List)            # 合并
    def union(self, other)            # 合并两个 DataProto
    def make_iterator(...)            # 创建迭代器
```

### PPO 训练数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            PPO Training Loop                              │
└──────────────────────────────────────────────────────────────────────────┘

1. 数据准备
   ┌─────────────┐
   │ DataLoader  │─────▶ prompts (input_ids, attention_mask)
   └─────────────┘

2. 生成阶段 (Rollout)
   ┌─────────────────────────────────────────────────────────────┐
   │  ActorRolloutWorker.generate_sequences(prompts)              │
   │                                                               │
   │  ┌────────────────────────────────────────────────────────┐  │
   │  │  ShardingManager                                        │  │
   │  │  FSDP Weights ──▶ vLLM Weights                         │  │
   │  └────────────────────────────────────────────────────────┘  │
   │                          │                                    │
   │                          ▼                                    │
   │  ┌────────────────────────────────────────────────────────┐  │
   │  │  vLLM Rollout                                           │  │
   │  │  - 自回归生成 responses                                  │  │
   │  │  - 返回 old_log_probs                                   │  │
   │  └────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
   output: {responses, old_log_probs, attention_mask}

3. 参考策略计算 (可选)
   ┌─────────────────────────────────────────────────────────────┐
   │  RefPolicyWorker.compute_ref_log_prob(data)                  │
   │  ──▶ ref_log_prob                                            │
   └─────────────────────────────────────────────────────────────┘

4. 价值估计
   ┌─────────────────────────────────────────────────────────────┐
   │  CriticWorker.compute_values(data)                           │
   │  ──▶ values                                                  │
   └─────────────────────────────────────────────────────────────┘

5. 奖励计算 (Driver Process)
   ┌─────────────────────────────────────────────────────────────┐
   │  RewardManager(data)                                         │
   │  - 基于规则的奖励 (countdown: 方程正确性)                      │
   │  - 或模型奖励 (RewardModelWorker)                            │
   │  ──▶ token_level_scores                                      │
   └─────────────────────────────────────────────────────────────┘

6. 优势估计 (Driver Process - 轻量计算)
   ┌─────────────────────────────────────────────────────────────┐
   │  apply_kl_penalty()    # KL 惩罚                            │
   │  compute_advantage()   # GAE 或 GRPO                        │
   │  ──▶ advantages, returns                                    │
   └─────────────────────────────────────────────────────────────┘

7. Critic 更新
   ┌─────────────────────────────────────────────────────────────┐
   │  CriticWorker.update_critic(data)                           │
   │  - Value Loss = (V_pred - returns)²                         │
   └─────────────────────────────────────────────────────────────┘

8. Actor 更新
   ┌─────────────────────────────────────────────────────────────┐
   │  ActorRolloutWorker.update_actor(data)                      │
   │  - PPO Clipped Policy Loss                                   │
   │  - Entropy Bonus (可选)                                      │
   └─────────────────────────────────────────────────────────────┘
```

---

## 🧠 核心算法实现

### 1. PPO 核心算法 (`verl/trainer/ppo/core_algos.py`)

#### GAE (Generalized Advantage Estimation)

```python
def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
    """
    计算 token 级别的 advantage 和 returns
    
    优势估计公式：
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = δ_t + γλ * A_{t+1}
    """
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = masked_whiten(advantages, eos_mask)  # 归一化
    return advantages, returns
```

#### PPO Policy Loss

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    PPO Clipped Objective
    
    L^{CLIP}(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
    其中 r(θ) = π_θ(a|s) / π_θ_old(a|s)
    """
    ratio = torch.exp(log_prob - old_log_prob)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
```

#### GRPO (Group Relative Policy Optimization)

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index):
    """
    GRPO: 基于组内相对奖励计算 advantage
    
    对于相同 prompt 的多个 response:
    advantage = (score - mean(group_scores)) / std(group_scores)
    """
    # 按 prompt index 分组计算 mean 和 std
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    
    # 归一化
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

### 2. KL 控制器

```python
class AdaptiveKLController:
    """
    自适应 KL 系数控制器
    根据当前 KL 散度动态调整惩罚系数
    """
    def update(self, current_kl, n_steps):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

---

## 🔧 Worker 系统设计

### 1. Worker 角色定义

```python
class Role(Enum):
    Actor = 0           # 策略网络
    Rollout = 1         # 推理引擎
    ActorRollout = 2    # Actor + Rollout 混合
    Critic = 3          # 价值网络
    RefPolicy = 4       # 参考策略
    RewardModel = 5     # 奖励模型
    ActorRolloutRef = 6 # Actor + Rollout + Ref 混合
```

### 2. Dispatch 模式

veRL 通过装饰器系统实现灵活的数据分发和收集：

```python
class Dispatch:
    RANK_ZERO = 0        # 只在 rank 0 执行
    ONE_TO_ALL = 1       # 广播到所有 worker
    ALL_TO_ALL = 2       # 全对全通信
    DP_COMPUTE = 8       # 数据并行计算
    DP_COMPUTE_PROTO = 9 # 数据并行 + DataProto 自动分片
```

使用示例：

```python
class ActorRolloutRefWorker(Worker):
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """广播初始化命令到所有 worker"""
        ...
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        自动将 prompts 按 batch 维度分片到各个 worker，
        执行后自动合并结果
        """
        ...
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """分布式策略更新"""
        ...
```

### 3. FSDP Worker 实现

`ActorRolloutRefWorker` 是核心的多功能 Worker，可以根据配置扮演不同角色：

```python
class ActorRolloutRefWorker(Worker):
    def __init__(self, config: DictConfig, role: str):
        # 角色判断
        self._is_actor = role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = role in ['ref', 'actor_rollout_ref']
        
        # 初始化 FSDP Device Mesh
        self.device_mesh = init_device_mesh('cuda', (world_size,), ['fsdp'])
        
        # 初始化 Ulysses 序列并行 (可选)
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda', (dp, sp), ['dp', 'sp'])
```

---

## ⚡ Hybrid Engine 混合引擎

### 核心概念

混合引擎的目标是让同一套模型权重在 **训练** 和 **推理** 之间高效切换：

```
训练阶段 (FSDP)                      推理阶段 (vLLM)
┌─────────────────────┐             ┌─────────────────────┐
│  FSDP Sharded       │             │  vLLM TP Sharded    │
│  Parameters         │ ◀───────▶  │  Parameters + KV    │
│                     │  Resharding │  Cache              │
│  - Full Shard       │             │  - Tensor Parallel  │
│  - Mixed Precision  │             │  - Paged Attention  │
└─────────────────────┘             └─────────────────────┘
```

### ShardingManager

`FSDPVLLMShardingManager` 负责 FSDP 和 vLLM 之间的权重同步：

```python
class FSDPVLLMShardingManager(BaseShardingManager):
    def __enter__(self):
        """进入推理模式"""
        # 1. 从 FSDP 收集完整权重
        # 2. 根据 vLLM TP 策略重新分片
        # 3. 加载到 vLLM 引擎
        self.inference_engine.sync_model_weights(...)
    
    def __exit__(self, ...):
        """退出推理模式"""
        # 释放临时缓存，恢复 FSDP 状态
```

---

## 📊 TinyZero 特定功能

### 1. Countdown 任务

目标：给定目标数字和一组数字，找出能得到目标的算术表达式。

#### 数据格式

```python
# examples/data_preprocess/countdown.py
prompt = f"""Using the numbers {numbers}, create an equation that equals {target}. 
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your work in <think> </think> tags. 
And return the final answer in <answer> </answer> tags."""
```

#### 奖励函数

```python
# verl/utils/reward_score/countdown.py
def compute_score(solution_str, ground_truth):
    """
    奖励规则：
    - 无法提取答案: 0
    - 格式正确但答案错误: 0.1 (format_score)
    - 答案正确: 1.0
    """
    equation = extract_solution(solution_str)  # 从 <answer> 标签提取
    
    if not validate_equation(equation, numbers):  # 验证使用的数字
        return format_score
    
    result = evaluate_equation(equation)  # 安全地计算结果
    if abs(result - target) < 1e-5:
        return 1.0
    else:
        return format_score
```

### 2. 训练配置关键参数

```bash
# scripts/train_tiny_zero.sh

# 数据配置
data.train_batch_size=256
data.max_prompt_length=256
data.max_response_length=1024

# Actor 配置
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.actor.ppo_micro_batch_size=8

# Rollout 配置 (vLLM)
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# Critic 配置
critic.optim.lr=1e-5
critic.model.path=$BASE_MODEL  # 使用相同的预训练模型

# KL 惩罚
algorithm.kl_ctrl.kl_coef=0.001

# 训练设置
trainer.total_epochs=15
trainer.save_freq=100
trainer.test_freq=100
```

---

## 🌐 分布式训练架构

### 资源池管理

```python
@dataclass
class ResourcePoolManager:
    """
    管理 GPU 资源分配
    """
    resource_pool_spec: dict[str, list[int]]  # pool_name -> [每个节点的 GPU 数]
    mapping: dict[Role, str]                   # Role -> pool_name
    
    def create_resource_pool(self):
        for name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1  # FSDP: 合并到一个进程组
            )
            self.resource_pool_dict[name] = resource_pool
```

### 共置 Worker (Colocated Workers)

TinyZero 使用共置策略，将 Actor、Rollout、Ref Policy 放在同一组 GPU 上：

```python
# verl/trainer/ppo/ray_trainer.py
def init_workers(self):
    # 所有模型共享同一个资源池
    mapping = {
        Role.ActorRollout: 'global_pool',
        Role.Critic: 'global_pool',
        Role.RefPolicy: 'global_pool',
    }
    
    # 创建共置 Worker 类
    worker_dict_cls = create_colocated_worker_cls(class_dict={
        'actor_rollout': actor_rollout_cls,
        'critic': critic_cls,
        'ref': ref_policy_cls,
    })
```

---

## 📈 性能优化技术

### 1. 动态 Batch Size

```python
actor.use_dynamic_bsz = True  # 根据序列长度动态调整 batch
```

### 2. 序列长度均衡

```python
# verl/utils/seqlen_balancing.py
def get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=True):
    """
    将数据按序列长度均衡分配到各个 DP rank，
    避免长序列导致的计算负载不均
    """
```

### 3. 参数/梯度/优化器卸载

```python
# 支持将 FSDP 参数卸载到 CPU 以节省 GPU 显存
actor.fsdp_config.param_offload = True
actor.fsdp_config.grad_offload = True
actor.fsdp_config.optimizer_offload = True
```

### 4. Gradient Checkpointing

```python
# 减少显存使用
critic.model.enable_gradient_checkpointing = True
```

---

## 🔗 扩展指南

### 添加新任务

1. **创建数据预处理脚本** (`examples/data_preprocess/your_task.py`)
2. **实现奖励函数** (`verl/utils/reward_score/your_task.py`)
3. **在 `main_ppo.py` 中注册**：

```python
def _select_rm_score_fn(data_source):
    if "your_task" in data_source:
        return your_task.compute_score
```

### 添加新算法

1. **实现核心算法** (`verl/trainer/your_algo/core_algos.py`)
2. **创建 Trainer** (`verl/trainer/your_algo/ray_trainer.py`)
3. **继承并修改 Worker** 行为

---

## 📚 参考资料

- [HybridFlow 论文](https://arxiv.org/abs/2409.19256v2)
- [veRL 官方文档](https://verl.readthedocs.io/)
- [TinyZero 实验日志](https://wandb.ai/jiayipan/TinyZero)
- [DeepSeek R1 论文](https://github.com/deepseek-ai/DeepSeek-R1)

---

## 💡 关键洞察

### 为什么 TinyZero 能工作？

1. **自主发展推理能力**：通过 RL 训练，模型学会在 `<think>` 标签中进行多步推理
2. **简单但有效的奖励**：只基于最终答案的正确性，不需要过程监督
3. **充分的探索**：长达 1024 token 的 response 长度允许模型尝试多种推理路径
4. **稳定的 KL 约束**：防止模型偏离预训练分布太远

### 架构优势

1. **高吞吐量**：通过 vLLM 加速推理，FSDP 优化训练
2. **内存效率**：混合引擎避免了冗余的模型副本
3. **易于扩展**：模块化设计使得添加新任务和算法变得简单
