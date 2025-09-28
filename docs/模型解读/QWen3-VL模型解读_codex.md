# Qwen3-VL模型解读

## 1. 模型定位与核心特性
Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date. This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities. It is available in Dense and Mixture-of-Experts (MoE) variants that scale from edge to cloud, with both Instruct and reasoning-enhanced Thinking editions for flexible, on-demand deployment.

**关键能力亮点**
- Visual Agent：能够操作 PC / 移动端 GUI，识别界面元素、理解功能、调用工具并完成任务。
- Visual Coding Boost：可从图像或视频生成 Draw.io、HTML、CSS、JavaScript 等代码内容。
- Advanced Spatial Perception：具备更强的二维/三维空间理解，能够判断物体位置、视角与遮挡关系，并支持空间推理与 Embodied AI。
- Long Context & Video Understanding：原生 256K 上下文长度，可扩展至 1M，支持长文档和长视频的精确回溯与秒级定位。
- Enhanced Multimodal Reasoning：在 STEM / 数学等领域具备更强的因果分析与逻辑推理能力。
- Upgraded Visual Recognition：更广泛、更高质量的预训练让模型能识别几乎所有类别的对象，包括名人、动漫、商品、地标、植物、动物等。
- Expanded OCR：原生支持 32 种语言的 OCR，增强低光、模糊、倾斜场景的识别效果，并提升长文档结构解析能力。
- Text Understanding on par with pure LLMs：视觉与文本信息无缝融合，实现与纯文本大模型相当的语言理解与生成能力。

**模型架构更新**
1. Interleaved-MRoPE：通过在时间、高度、宽度维度进行全频率分配的多维旋转位置编码，显著增强长时序视频推理能力。
2. DeepStack：融合多层 ViT 特征以捕获细粒度视觉细节，同时提升图像与文本之间的对齐能力。
3. Text–Timestamp Alignment：在视频理解中引入时间戳对齐，使模型能够精确定位事件发生时间，超越传统的时间 RoPE（T-RoPE）。

## 2. 输入输出与调用流程
### 2.1 基础推理调用
推理时通过 Hugging Face `AutoModelForImageTextToText` 与 `AutoProcessor` 组合处理多模态对话输入，文本与视觉模态放在同一 `messages` 序列中，模型最终输出自回归文本：

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

该流程支持将图片、视频、文本在同一个消息中混合输入，`apply_chat_template` 会自动插入 `<|vision_start|>...<|vision_end|>` 占位符并拼接到文本序列。

### 2.2 多图片、视频与批量推理
- 多图输入：在 `content` 中连续放置多个 `{"type": "image"}` 项，模型会依次读取并生成描述或比较结果。
- 视频输入：通过 `{"type": "video", "video": "..."}` 或提供帧列表的方式传入，模型会在序列中插入 `<|video_pad|>` 占位符完成对齐。
- 批量推理：将多个独立对话包裹成列表后送入 `apply_chat_template`，并将 `processor.tokenizer.padding_side` 设置为 `left` 以支持批量生成。

示例（视频输入）：
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]
```

### 2.3 视觉内容编号与模板输出
处理多视觉输入时，可通过 `add_vision_id=True` 自动为各图片/视频编号，提示词示例：
```
Picture 1: <|vision_start|><|image_pad|><|vision_end|>
Picture 2: <|vision_start|><|image_pad|><|vision_end|>
Video 1: <|vision_start|><|video_pad|><|vision_end|>
```
模型在回答中可以明确引用 “Picture 2” 或 “Video 1”，便于复杂交互。

### 2.4 长上下文与 FlashAttention-2
- Qwen3-VL 默认 context length 为 256K，可通过修改配置启用 YaRN，将 `max_position_embeddings` 调到 1,000,000 并设置 `rope_scaling`：

```
{
    "max_position_embeddings": 1000000,
    "rope_scaling": {
        "rope_type": "yarn",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": true,
        "factor": 3.0,
        "original_max_position_embeddings": 262144
    }
}
```

> 由于 Interleaved-MRoPE 的位置 ID 增长更慢，扩展上下文时建议将 `factor` 控制在 2~3 之间。

- 可选开启 FlashAttention-2：安装 `flash-attn` 并在 `from_pretrained` 时指定 `attn_implementation="flash_attention_2"`，建议将模型加载到 `torch.bfloat16` 或 `torch.float16` 以兼容。

## 3. 视觉预处理与 token 控制
### 3.1 官方处理器调参
`AutoProcessor` 提供独立的图像与视频处理器，可直接设定视觉 token 预算：
```python
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")
processor.image_processor.size = {"longest_edge": 1280*32*32, "shortest_edge": 256*32*32}
processor.video_processor.size = {"longest_edge": 16384*32*32, "shortest_edge": 256*32*32}
```
- `size['longest_edge']` / `size['shortest_edge']` 对图像表示单张图的最大 / 最小像素数（忽略通道）。
- 对视频，该参数控制所有帧的像素总预算，常用于限制显存消耗。
- `apply_chat_template` 可额外接受 `fps` 或 `num_frames` 参数来设定抽帧策略（默认 2 FPS）。

### 3.2 qwen-vl-utils 关键实现
`qwen-vl-utils` 针对 Qwen3-VL 做了专门适配：
- 默认 `image_patch_size=16`，同时定义 `SPATIAL_MERGE_SIZE=2`，因此所有视觉尺寸会被对齐到 32 的倍数。
- 视觉 token 限制：
  - `IMAGE_MIN_TOKEN_NUM = 4`，`IMAGE_MAX_TOKEN_NUM = 16384`
  - `VIDEO_MIN_TOKEN_NUM = 128`，`VIDEO_MAX_TOKEN_NUM = 768`
- 其他重要常量：`FPS = 2.0`、`FRAME_FACTOR = 2`、`FPS_MIN_FRAMES = 4`、`FPS_MAX_FRAMES = 768`、`MAX_RATIO = 200`（限制长宽比），以及环境变量 `MODEL_SEQ_LEN`（默认 128000）用于估算视频总 token 上限。
- `smart_resize` 会在保持宽高比的前提下，将分辨率调整到上述约束范围内。
- `fetch_image` 针对 `http(s)`、`file://`、Base64 与 `PIL.Image` 输入统一转为 RGB，并在透明图像上填充白底后再 resize。
- `fetch_video` 支持三种解码后端（`torchvision`、`decord`、`torchcodec`），并根据 `fps` 或 `nframes` 参数抽帧；若输入为帧列表，则使用线程池并按 2 的倍数补齐帧数。返回值可以附带 `video_metadata`（帧率、帧索引、总帧数、使用的 backend），供 Text–Timestamp Alignment 或下游应用使用。
- 推荐安装 `qwen-vl-utils[decord]` 或手动部署 `torchcodec` 以加速视频加载；若需指定后台，可设置环境变量 `FORCE_QWENVL_VIDEO_READER`。

视频后台兼容性：
| Backend | HTTP | HTTPS |
|---------|------|-------|
| torchvision >= 0.19.0 | ✅ | ✅ |
| torchvision < 0.19.0  | ❌ | ❌ |
| decord                | ✅ | ❌ |
| torchcodec            | ✅ | ✅ |

### 3.3 视频输入示例
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 256 * 32 * 32,
                "total_pixels": 20480 * 32 * 32,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    return_tensors="pt",
    do_resize=False,
    **video_kwargs,
)
```

## 4. 模型结构与实现机制
- **视觉编码器 + 语言大模型双塔**：视觉塔输出多尺度特征，经多模态投影后融入 Qwen3 自回归 Transformer，实现统一的 token 序列建模。
- **Interleaved-MRoPE**：在时间、宽度、高度维度交替分配频率，使旋转位置编码在长距离仍保持稳定；结合 YaRN 可将上下文扩展到 1M。
- **DeepStack**：在视觉塔内部叠加多个层级的 ViT 特征，兼顾全局语义与细节描述，并配合更大的 patch size 提升高分辨率处理能力。
- **Text–Timestamp Alignment**：训练时结合帧级时间戳监督，推理阶段可借助 `video_metadata` 精确定位事件发生的时间段，在长视频说明、时序问答、事件定位任务上显著提升表现。

## 5. 训练数据构建
官方训练框架要求使用 JSON/JSONL 数据，将视觉内容通过特殊 token 引入对话：

**单图示例**
```json
{
    "image": "images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat's the main object in this picture?"
        },
        {
            "from": "gpt",
            "value": "A red apple on a wooden table"
        }
    ]
}
```

**多图示例**
```json
{
    "images": ["cats/001.jpg", "cats/002.jpg"],
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<image>\nWhat are the differences between these two cats?"
        },
        {
            "from": "gpt",
            "value": "The first cat is an orange tabby with short fur and green eyes, while the second is a gray Siamese with blue eyes and pointed coloration. They also appear to be in different environments - the first is indoors on a couch, the second is outdoors in a garden."
        }
    ]
}
```

**视频示例**
```json
{
    "video": "videos/005.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nWhat caused the blue object to move?\nOptions:\n(A) Gravity\n(B) Collision\n(C) Magnetic force"
        },
        {
            "from": "gpt",
            "value": "Answer: (B) Collision"
        }
    ]
}
```

**Grounding 示例**
```json
{
    "image": "demo/COCO_train2014_000000580957.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
        },
        {
            "from": "gpt",
            "value": "{\n"bbox_2d": [135, 114, 1016, 672]\n}"
        }
    ]
}
```

**数据配置要点**
- 在 `data/__init__.py` 中通过字典注册数据集：`{"annotation_path": ..., "data_path": ...}`。
- 数据集名称可追加 `%` 设置采样率，例如 `dataset_name%50` 表示抽样 50%。
- `<image>` / `<video>` 标签只能出现在问题中，并且与实际文件一一对应，回答侧必须为纯文本。
- 提供 `tools/process_bbox.ipynb` 用于转换检索框数据，`tools/pack_data.py` 可打包不同样本以减少 padding。

## 6. 训练流程与损失函数
### 6.1 依赖与环境
建议的依赖组合：
- torch 2.6.0
- torchvision 0.21.0
- transformers 4.50.0.dev0
- deepspeed 0.16.4
- flash_attn 2.7.4.post1
- triton 3.0.0
- accelerate 1.4.0
- torchcodec 0.2

### 6.2 启动脚本核心参数
官方示例脚本使用 `torchrun` 启动 `qwenvl/train/train_qwen.py`，关键参数如下：
- `--model_name_or_path`：预训练权重路径，例如 `Qwen/Qwen3-VL-235B-A22B-Instruct`。
- 组件调节：`--tune_mm_llm`、`--tune_mm_vision`、`--tune_mm_mlp` 控制是否微调语言模型、视觉塔、模态融合层。若同时训练图像和视频，建议保持 `tune_mm_vision=False` 并只调节投影层以降低显存。
- 精度与批大小：`--bf16`、`--per_device_train_batch_size`、`--gradient_accumulation_steps`。
- 学习率：`--learning_rate`（推荐 1e-6 到 2e-7）、`--mm_projector_lr`、`--vision_tower_lr`。
- 分辨率控制：`--max_pixels` / `--min_pixels`、`--video_max_frames` / `--video_min_frames`、`--video_max_frame_pixels` / `--video_min_frame_pixels`。
- 训练调度：`--num_train_epochs`、`--warmup_ratio`、`--lr_scheduler_type`、`--weight_decay`、`--logging_steps`、`--save_steps`。
- 高级选项：`--data_flatten True`（将一个 batch 内样本拼成连续序列）、`--data_packing True`（需要先调用 `tools/pack_data.py`）、`--deepspeed zero3.json`（启用 ZeRO Stage-3）。

### 6.3 训练脚本内部机制
`train_qwen.py` 中的 `set_model` 函数根据上述三个开关逐层设置 `requires_grad`：
- `tune_mm_vision=True` 时解冻视觉编码器 `model.visual`。
- `tune_mm_mlp=True` 时解冻视觉特征合并器 `model.visual.merger`。
- `tune_mm_llm=True` 时解冻语言骨干 `model.model` 与 `model.lm_head`。
默认情况下模型将视觉塔冻结，仅训练语言骨干和输出头，以减少显存与数据需求。

其他训练细节：
- 当启用 `gradient_checkpointing` 时，脚本会对输入嵌入注册钩子以确保梯度可回传。
- 训练使用 Hugging Face `Trainer`，数据模块来自 `make_supervised_data_module` 或 `make_supervised_data_module_packed`。
- 完成训练后会保存模型状态与图像处理器配置，以保持推理时的视觉预处理一致性。

### 6.4 Loss 设计
训练目标沿用标准自回归语言模型损失，即对所有输出 token 计算交叉熵。包括 Grounding、结构化输出（如 bbox JSON）在内的多模态任务均在文本空间展开，无需新增检测头，从而保持统一的训练范式。

## 7. 实践建议与注意事项
- **视觉 token 预算**：在多图、多视频或长文档场景下，务必结合 `processor.image_processor.size` / `processor.video_processor.size`、`min_pixels` / `max_pixels` 等参数预估视觉 token，用于避免超过 256K（或扩展后的上限）。
- **长视频理解**：启用 `add_vision_id` 与 `process_vision_info(..., return_video_metadata=True)`，保留帧索引和采样 FPS，结合提示词显式要求模型引用时间戳。
- **FlashAttention-2 与 ZeRO-3**：对大模型或多 GPU 训练尤为重要，可显著降低显存占用并提高吞吐。
- **微调策略**：若仅需适配领域知识，可冻结视觉塔，仅调 `tune_mm_llm=True`；需要更高精度的视觉目标时，再逐步解冻 `tune_mm_mlp` 或 `tune_mm_vision`。
- **数据清洗**：确保 `<image>` / `<video>` 标签与真实文件一一对应，缺失文件可通过 `tools/check_image.py` 预先排查。
- **环境变量**：通过设置 `MODEL_SEQ_LEN` 可以上调视频 token 总预算，使 `fetch_video` 在长序列场景中自动放宽 `total_pixels` 限制。

以上内容覆盖了 Qwen3-VL 的模型架构、输入输出、视觉预处理、训练流程与实践要点，可作为理解与二次开发该模型的集中参考。

## 8. 常见问题与答疑
**Gradient Checkpointing 有什么作用？**
- 在训练脚本中开启 `gradient_checkpointing` 后，模型只缓存关键节点的激活值，其余节点在反向传播阶段按需重新计算，显著降低显存占用。
- 代价是反向传播需要额外的前向重算，因此训练时间会略微增加。这在长上下文或大 batch 训练时尤为有用。

**JSONL 是什么格式？**
- JSON Lines（JSONL）是一种逐行存储 JSON 对象的文本格式，每一行是一条完整的 JSON 记录，用换行符分隔，不需要外层方括号。适合“一行一条样本”的大规模数据流式读取与追加写入。

**Grounding 任务的标签和 token 如何对应？**
- Grounding 标签就是回答端的整段文本（例如包含 bbox 的 JSON）。Tokenizer 会将该文本切分成一串 token，模型按自回归方式逐 token 预测，loss 仍然是交叉熵。
- 例：`{"bbox_2d": [135, 114, 1016, 672]}` 会被拆成由符号、词片和数字构成的序列。因为数字以字符串形式出现，模型生成的坐标可以是整数或小数。

**如果需要预测小数或 3D 坐标，样本怎么写？**
- 直接在答案文本中写入带小数的坐标即可，例如：
  ```json
  {
    "image": "demo/page_01.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n请给出发票上“总金额”文本框的归一化坐标，格式为 JSON。"
      },
      {
        "from": "gpt",
        "value": "{\n  \"bbox_2d_norm\": [0.183, 0.642, 0.487, 0.705],\n  \"text\": \"总金额\"\n}"
      }
    ]
  }
  ```
- 3D grounding 也类似，直接给出所需的 3D 参数，如中心点与尺寸或角点列表：
  ```json
  {
    "image": "demo/scene123_rgb.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n在 3D 场景中标注“红色椅子”的位置，给出中心点(x,y,z)和尺寸(w,h,d)，单位米。"
      },
      {
        "from": "gpt",
        "value": "{\n  \"object\": \"red chair\",\n  \"center_3d_m\": [1.42, 0.13, 2.08],\n  \"size_3d_m\": [0.55, 0.92, 0.60]\n}"
      }
    ]
  }
  ```
- 无论维度或精度如何变化，核心做法都是把目标参数线性化成文本，让模型通过 token 生成完成监督。
