<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# โหลด pipelines, models, และ schedulers

[[open-in-colab]]

วิธีง่ายๆในการใช้กระบวนการของ diffusion ในการอนุมาน (inference) นั้นเป็นสิ่งที่จำเป็นมากสำหรับ 🧨 Diffusers เนื่องจากกระบวนการของ diffusion มักประกอบด้วยองค์ประกอบหลายอย่าง เช่น โมเดลแบบกำหนดพารามิเตอร์ โทเค็นไนเซอร์ (tokenizer) และตัวกำหนดเวลา (scheduler) ที่มีการโต้ตอบในรูปแบบที่ซับซ้อน นั่นคือเหตุผลที่เราออกแบบ [`DiffusionPipeline`] เพื่อลดความซับซ้อนในกระบวนการของ diffusion ทั้งหมดไว้ในรูปแบบ API ที่ใช้งานง่าย ในขณะที่ยังคงความยืดหยุ่นเพียงพอที่จะปรับใช้สำหรับกรณีการใช้งานอื่นๆ เช่น การโหลดแต่ละส่วนประกอบแยกกัน เพื่อประกอบเข้ากับกระบวนการ diffusion ของคุณเอง

ทุกสิ่งที่คุณต้องการสำหรับการอนุมาน (inference) หรือการเทรนโมเดล (training) นั้นสามารถเข้าถึงได้ด้วยการใช้เมธอด `from_pretrained()`

คู่มือนี้จะแสดงวิธีการโหลด:

- การโหลดไปป์ไลน์ (pipelines) จาก Huggingface Hub และเครื่องของเรา
- การโหลดส่วนประกอบต่าง ๆ ลงในไปป์ไลน์ (pipeline)
- การโหลดตัวแปร checkpoint เช่น ประเภทจุดทศนิยมที่แตกต่างกันหรือน้ำหนักเฉลี่ยที่ไม่ใช่เลขชี้กำลัง (non-exponential mean averaged weights : EMA)
- การโหลดโมเดล (models) และตัวกำหนดเวลา (schedulers)

## การใช้ Diffusion Pipeline

<Tip>

คุณสามารถข้ามไปที่ส่วน [DiffusionPipeline explained](#diffusionpipeline-explained) หากคุณสนใจที่จะเรียนรู้รายละเอียดเพิ่มเติมเกี่ยวกับวิธีการทำงานของคลาส [`DiffusionPipeline`]

</Tip>

คลาส [`DiffusionPipeline`] เป็นวิธีที่ง่ายที่สุดและทั่วถึงที่สุดในการโหลดโมเดล diffusion จาก [Hub](https://huggingface.co/models?library=diffusers) ตัวของเมธอด [`DiffusionPipeline.from_pretrained`] จะตรวจจับคลาสไปป์ไลน์ (pipeline) ที่ถูกต้องจาก checkpoint โดยอัตโนมัติ ซึ่งจะดาวน์โหลดและแคชไฟล์ configuration และ weight ที่จำเป็นทั้งหมด และส่งคืนอินสแตนซ์ของไปป์ไลน์ที่พร้อมสำหรับการนำไปใช้อนุมาน (inference)

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```
คุณยังสามารถโหลด checkpoint ด้วยคลาสไปป์ไลน์เฉพาะ (specific pipeline) ได้ ตัวอย่างด้านบนเป็นการโหลดโมเดล Stable Diffusion; เพื่อให้ได้ผลลัพธ์เดียวกัน ให้ใช้คลาส [`StableDiffusionPipeline`]

```python
from diffusers import StableDiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```

ตัว checkpoint (เช่น [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) หรือ [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) นั้นอาจนำไปใช้งานได้มากกว่าหนึ่งงาน เช่น การแปลงข้อความเป็นรูปภาพ (text-to-image)หรือรูปภาพเป็นรูปภาพ (image-to-image) ซึ่งหากต้องการแยกแยะว่างานใดที่คุณต้องการใช้ checkpoint ที่กำหนด คุณต้องโหลดโดยตรงด้วยคลาสไปป์ไลน์เฉพาะงาน (task-specific pipeline class) ที่เกี่ยวข้อง:

```python
from diffusers import StableDiffusionImg2ImgPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
```

### การโหลด pipeline ที่มีอยู่ในเครื่องของเรา

To load a diffusion pipeline locally, use [`git-lfs`](https://git-lfs.github.com/) to manually download the checkpoint (in this case, [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) to your local disk. This creates a local folder, `./stable-diffusion-v1-5`, on your disk:

```bash
git-lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

Then pass the local path to [`~DiffusionPipeline.from_pretrained`]:

```python
from diffusers import DiffusionPipeline

repo_id = "./stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```

The [`~DiffusionPipeline.from_pretrained`] method won't download any files from the Hub when it detects a local path, but this also means it won't download and cache the latest changes to a checkpoint.

### Swap components in a pipeline

You can customize the default components of any pipeline with another compatible component. Customization is important because:

- Changing the scheduler is important for exploring the trade-off between generation speed and quality.
- Different components of a model are typically trained independently and you can swap out a component with a better-performing one.
- During finetuning, usually only some components - like the UNet or text encoder - are trained.

To find out which schedulers are compatible for customization, you can use the `compatibles` method:

```py
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
stable_diffusion.scheduler.compatibles
```

Let's use the [`SchedulerMixin.from_pretrained`] method to replace the default [`PNDMScheduler`] with a more performant scheduler, [`EulerDiscreteScheduler`]. The `subfolder="scheduler"` argument is required to load the scheduler configuration from the correct [subfolder](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/scheduler) of the pipeline repository.

Then you can pass the new [`EulerDiscreteScheduler`] instance to the `scheduler` argument in [`DiffusionPipeline`]:

```python
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler

repo_id = "runwayml/stable-diffusion-v1-5"

scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")

stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, use_safetensors=True)
```

### Safety checker

Diffusion models like Stable Diffusion can generate harmful content, which is why 🧨 Diffusers has a [safety checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) to check generated outputs against known hardcoded NSFW content. If you'd like to disable the safety checker for whatever reason, pass `None` to the `safety_checker` argument:

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, safety_checker=None, use_safetensors=True)
```

### Reuse components across pipelines

You can also reuse the same components in multiple pipelines to avoid loading the weights into RAM twice. Use the [`~DiffusionPipeline.components`] method to save the components:

```python
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

components = stable_diffusion_txt2img.components
```

Then you can pass the `components` to another pipeline without reloading the weights into RAM:

```py
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)
```

You can also pass the components individually to the pipeline if you want more flexibility over which components to reuse or disable. For example, to reuse the same components in the text-to-image pipeline, except for the safety checker and feature extractor, in the image-to-image pipeline:

```py
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
    vae=stable_diffusion_txt2img.vae,
    text_encoder=stable_diffusion_txt2img.text_encoder,
    tokenizer=stable_diffusion_txt2img.tokenizer,
    unet=stable_diffusion_txt2img.unet,
    scheduler=stable_diffusion_txt2img.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
```

## Checkpoint variants

A checkpoint variant is usually a checkpoint where it's weights are:

- Stored in a different floating point type for lower precision and lower storage, such as [`torch.float16`](https://pytorch.org/docs/stable/tensors.html#data-types), because it only requires half the bandwidth and storage to download. You can't use this variant if you're continuing training or using a CPU.
- Non-exponential mean averaged (EMA) weights which shouldn't be used for inference. You should use these to continue finetuning a model.

<Tip>

💡 When the checkpoints have identical model structures, but they were trained on different datasets and with a different training setup, they should be stored in separate repositories instead of variations (for example, [`stable-diffusion-v1-4`] and [`stable-diffusion-v1-5`]).

</Tip>

Otherwise, a variant is **identical** to the original checkpoint. They have exactly the same serialization format (like [Safetensors](./using_safetensors)), model structure, and weights have identical tensor shapes.

| **checkpoint type** | **weight name**                     | **argument for loading weights** |
|---------------------|-------------------------------------|----------------------------------|
| original            | diffusion_pytorch_model.bin         |                                  |
| floating point      | diffusion_pytorch_model.fp16.bin    | `variant`, `torch_dtype`         |
| non-EMA             | diffusion_pytorch_model.non_ema.bin | `variant`                        |

There are two important arguments to know for loading variants:

- `torch_dtype` defines the floating point precision of the loaded checkpoints. For example, if you want to save bandwidth by loading a `fp16` variant, you should specify `torch_dtype=torch.float16` to *convert the weights* to `fp16`. Otherwise, the `fp16` weights are converted to the default `fp32` precision. You can also load the original checkpoint without defining the `variant` argument, and convert it to `fp16` with `torch_dtype=torch.float16`. In this case, the default `fp32` weights are downloaded first, and then they're converted to `fp16` after loading.

- `variant` defines which files should be loaded from the repository. For example, if you want to load a `non_ema` variant from the [`diffusers/stable-diffusion-variants`](https://huggingface.co/diffusers/stable-diffusion-variants/tree/main/unet) repository, you should specify `variant="non_ema"` to download the `non_ema` files.

```python
from diffusers import DiffusionPipeline
import torch

# load fp16 variant
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
# load non_ema variant
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="non_ema", use_safetensors=True
)
```

To save a checkpoint stored in a different floating point type or as a non-EMA variant, use the [`DiffusionPipeline.save_pretrained`] method and specify the `variant` argument. You should try and save a variant to the same folder as the original checkpoint, so you can load both from the same folder:

```python
from diffusers import DiffusionPipeline

# save as fp16 variant
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16")
# save as non-ema variant
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="non_ema")
```

If you don't save the variant to an existing folder, you must specify the `variant` argument otherwise it'll throw an `Exception` because it can't find the original checkpoint:

```python
# 👎 this won't work
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
# 👍 this works
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
```

<!--
TODO(Patrick) - Make sure to uncomment this part as soon as things are deprecated.

#### Using `revision` to load pipeline variants is deprecated

Previously the `revision` argument of [`DiffusionPipeline.from_pretrained`] was heavily used to 
load model variants, e.g.:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", use_safetensors=True)
```

However, this behavior is now deprecated since the "revision" argument should (just as it's done in GitHub) better be used to load model checkpoints from a specific commit or branch in development.

The above example is therefore deprecated and won't be supported anymore for `diffusers >= 1.0.0`.

<Tip warning={true}>

If you load diffusers pipelines or models with `revision="fp16"` or `revision="non_ema"`, 
please make sure to update to code and use `variant="fp16"` or `variation="non_ema"` respectively
instead.

</Tip>
-->

## Models

Models are loaded from the [`ModelMixin.from_pretrained`] method, which downloads and caches the latest version of the model weights and configurations. If the latest files are available in the local cache, [`~ModelMixin.from_pretrained`] reuses files in the cache instead of redownloading them.

Models can be loaded from a subfolder with the `subfolder` argument. For example, the model weights for `runwayml/stable-diffusion-v1-5` are stored in the [`unet`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/unet) subfolder:

```python
from diffusers import UNet2DConditionModel

repo_id = "runwayml/stable-diffusion-v1-5"
model = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", use_safetensors=True)
```

Or directly from a repository's [directory](https://huggingface.co/google/ddpm-cifar10-32/tree/main):

```python
from diffusers import UNet2DModel

repo_id = "google/ddpm-cifar10-32"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

You can also load and save model variants by specifying the `variant` argument in [`ModelMixin.from_pretrained`] and [`ModelMixin.save_pretrained`]:

```python
from diffusers import UNet2DConditionModel

model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", variant="non-ema", use_safetensors=True
)
model.save_pretrained("./local-unet", variant="non-ema")
```

## Schedulers

Schedulers are loaded from the [`SchedulerMixin.from_pretrained`] method, and unlike models, schedulers are **not parameterized** or **trained**; they are defined by a configuration file.

Loading schedulers does not consume any significant amount of memory and the same configuration file can be used for a variety of different schedulers.
For example, the following schedulers are compatible with [`StableDiffusionPipeline`] which means you can load the same scheduler configuration file in any of these classes:

```python
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "runwayml/stable-diffusion-v1-5"

ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
dpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# replace `dpm` with any of `ddpm`, `ddim`, `pndm`, `lms`, `euler_anc`, `euler`
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=dpm, use_safetensors=True)
```

## DiffusionPipeline explained

As a class method, [`DiffusionPipeline.from_pretrained`] is responsible for two things:

- Download the latest version of the folder structure required for inference and cache it. If the latest folder structure is available in the local cache, [`DiffusionPipeline.from_pretrained`] reuses the cache and won't redownload the files.
- Load the cached weights into the correct pipeline [class](./api/pipelines/overview#diffusers-summary) - retrieved from the `model_index.json` file - and return an instance of it.

The pipelines underlying folder structure corresponds directly with their class instances. For example, the [`StableDiffusionPipeline`] corresponds to the folder structure in [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5).

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
print(pipeline)
```

You'll see pipeline is an instance of [`StableDiffusionPipeline`], which consists of seven components:

- `"feature_extractor"`: a [`~transformers.CLIPFeatureExtractor`] from 🤗 Transformers.
- `"safety_checker"`: a [component](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32) for screening against harmful content.
- `"scheduler"`: an instance of [`PNDMScheduler`].
- `"text_encoder"`: a [`~transformers.CLIPTextModel`] from 🤗 Transformers.
- `"tokenizer"`: a [`~transformers.CLIPTokenizer`] from 🤗 Transformers.
- `"unet"`: an instance of [`UNet2DConditionModel`].
- `"vae"` an instance of [`AutoencoderKL`].

```json
StableDiffusionPipeline {
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

Compare the components of the pipeline instance to the [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) folder structure, and you'll see there is a separate folder for each of the components in the repository:

```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
│   └── pytorch_model.bin
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
│   └── pytorch_model.bin
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
└── vae
    ├── config.json
    ├── diffusion_pytorch_model.bin
```

You can access each of the components of the pipeline as an attribute to view its configuration:

```py
pipeline.tokenizer
CLIPTokenizer(
    name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer",
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos_token": AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "eos_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "unk_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "pad_token": "<|endoftext|>",
    },
)
```

Every pipeline expects a `model_index.json` file that tells the [`DiffusionPipeline`]:

- which pipeline class to load from `_class_name`
- which version of 🧨 Diffusers was used to create the model in `_diffusers_version`
- what components from which library are stored in the subfolders (`name` corresponds to the component and subfolder name, `library` corresponds to the name of the library to load the class from, and `class` corresponds to the class name)

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```
