<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ภาพรวม

🧨 ใน Diffusers นั้นมีการเสนอ ไปป์ไลน์ (pipelines), โมเดล (models), และ ตัวกำหนดเวลา (schedulers) สำหรับใช้งานสร้างสรรค์ต่างๆ เพื่อให้การโหลดส่วนประกอบเหล่านี้ง่ายที่สุดเท่าที่จะเป็นไปได้ เราสามารถทำได้โดยการเรียกใช้เมธอด - `from_pretrained()` - ซึ่งจะโหลดส่วนประกอบเหล่านี้จาก Hugging Face [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) หรือเครื่องของคุณ และเมื่อไหร่ก็ตามที่คุณโหลดไปป์ไลน์หรือโมเดล ไฟล์ล่าสุดจะถูกดาวน์โหลดและแคชโดยอัตโนมัติ เพื่อให้คุณสามารถนำมาใช้ซ้ำได้อย่างรวดเร็วในครั้งถัดไปโดยไม่จำเป็นต้องดาวน์โหลดไฟล์ซ้ำ


ส่วนนี้จะแสดงทุกสิ่งที่คุณจำเป็นต้องรู้เกี่ยวกับการโหลดไปป์ไลน์ (pipeline) วิธีโหลดส่วนประกอบต่างๆ ในไปป์ไลน์ วิธีโหลด checkpoint และวิธีการโหลดไปป์ไลน์ของผู้อื่น (community pipeline) นอกจากนี้คุณยังจะได้เรียนรู้วิธีโหลดตัวกำหนดเวลา (schedulers)และเปรียบเทียบความเร็วและคุณภาพของการใช้ตัวกำหนดเวลาที่แตกต่างกัน สุดท้ายนี้ คุณจะเห็นวิธีการแปลงและโหลด checkpoint ของ KerasCV เพื่อให้คุณสามารถใช้ checkpoint เหล่านี้ใน PyTorch ไปพร้อมกับ Diffusers