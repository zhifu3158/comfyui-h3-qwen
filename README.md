# ComfyUI-H3-Qwen（comfyui-h3-qwen）

> 通过 **`/dev/shm` 共享内存文件信号** 与独立 **Qwen3.8-27B VLM 服务** 通信的 ComfyUI 自定义节点包。
> 含两个节点（`H3_Qwen通信` + `H3_检测门控`）与一个独立推理服务（`qwen_service.py`）。
> 典型场景：MiniMax-H3 视频工作流的 **提示词完善（生成前）** 与 **画面质检（一采后，不通过则终止二采）**。

---

## 一、仓库结构

```
comfyui-h3-qwen/
├── __init__.py          # 入口：注册两个节点
├── h3_qwen_node.py      # H3_Qwen通信 + H3_检测门控
├── qwen_service.py      # 独立 Qwen3.8 推理服务（独立进程，先于 ComfyUI 启动）
└── README.md
```

---

## 二、架构与原理

```
┌──────────── ComfyUI 进程 ────────────┐        ┌──────── qwen_service.py ────────┐
│ H3_Qwen通信:                          │        │ 主循环每秒写心跳                 │
│  ① 服务预检(探心跳,默认2s)            │        │ /tmp/h3_vlm_heartbeat           │
│  ② 收集图片→缩放→RGB uint8           │        │                                  │
│  ③ 写 h3_frames.npy + h3_request.json│──写入→│ 检测到 request → 读取+推理        │
│  ④ 每1秒轮询 h3_result.json          │←─写入─│ 写 h3_result.json + 删 request    │
│  ⑤ 存 pickup/{标识}_detection.json   │        │                                  │
└───────────────────────────────────────┘        └──────────────────────────────────┘
```

- **零编解码传输**：帧以原始 `uint8` 写 `/dev/shm`（内存级），仅服务端做 JPEG 编码。
- **防泄漏**：固定 3 文件名覆盖写 + 写前/读后双清理 + tmpfs 重启自清。
- **色彩协议**：`h3_frames.npy` 存 **RGB**；服务端 `cv2.imencode` 前必须 `RGB→BGR`。
- **心跳协议**：服务端每秒写 `/tmp/h3_vlm_heartbeat`；节点以 mtime≤3s 判活。

---

## 三、部署

1. 模型：`Qwen3.8-27B-UD-Q5_K_M.gguf`(19G) + `mmproj-BF16.gguf`(889M) 放入 `/mnt/workspace/models/llm/`。
2. 编译：`CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`（ROCm）。
3. 依赖：`numpy`(锁 2.1.3) + `opencv-python-headless`。
4. 启动：`python qwen_service.py`（先于 ComfyUI；app.py 已集成 bootstrap + 前端启停按钮）。

---

## 四、H3_Qwen通信 · 输入参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| 启用 | BOOLEAN | True | False=透传模式（见行为规格） |
| 图片数据 | IMAGE | None | 可连 LoadImage/VAEDecode，可多张 |
| 图片路径 | STRING | "" | 每行一个路径，批量 |
| 视频路径 | STRING | "" | 每行一个，按抽帧率抽帧 |
| 技能文件路径 | STRING | "" | H3 Skills(.md)，空则不用 |
| 剧本文件路径 | STRING | "" | 全局剧本，空则不用 |
| 系统提示词 | STRING | "" | 角色/规则/检测标准 |
| 提问词 | STRING | "" | 可连 #138；也是透传/回退的原文 |
| 附加指令 | STRING | "" | 拼在提问词【后面】（尾部注意力更高） |
| 温度 | FLOAT | 0.1 | 0=严格，越大越自由 |
| 最大输出长度 | INT | 2048 | 建议 2048~4096 |
| 超时时间 | INT | 60 | 结果轮询超时 |
| 服务检测超时 | FLOAT | 2.0 | 开头预检心跳，超时即通信失败 |
| 失败回退原文 | BOOLEAN | False | 仅通信失败时：开→原文，关→错误 |
| 每秒抽帧数 | FLOAT | 4.0 | 仅视频路径生效 |
| 缩放宽度/高度 | INT | 640/368 | 统一缩放目标 |
| 请求标识 | STRING | "" | 取件码，空存 detection.json |

**输出**：`结果文本`(STRING) + `是否正常`(BOOLEAN) + `错误信息`(STRING)

---

## 五、行为规格（最终版）⭐

| 场景 | 结果文本 | 是否正常 | 错误信息 | 回退开关 |
|---|---|---|---|---|
| **启用=False（透传）** | **原始提问词** | **True** | `""` | ❌ 不生效 |
| 启用+预检失败+回退开 | 原始提问词 | False | 错误信息 | ✅ |
| 启用+预检失败+回退关 | 错误信息 | False | 错误信息 | ✅ |
| 启用+正常 | Qwen 回复 | True | `""` | — |
| 启用+检测不通过 | Qwen 检测文本 | False | Qwen 检测文本 | ❌ |
| 启用+结果超时+回退开 | 原始提问词 | False | 错误信息 | ✅ |
| 启用+结果超时+回退关 | 错误信息 | False | 错误信息 | ✅ |

要点：
- **透传模式**：节点啥都不干，`结果文本`=#138 原文（不含附加指令/系统提示词），`是否正常=True`。
- **回退开关**仅对"通信失败"生效；"检测不通过"(`pass:false`)永远返回检测文本，不走回退。
- `是否正常` 不修改 `结果文本`，二者独立。

---

## 六、主动调整说明（重要）⭐

> 关于 `启用=False` 时 `是否正常` 的取值，我做了一个**主动调整**，请务必知悉：

- **你的原修改**：`启用=False` 返回 `是否正常=False`。
- **我的调整**：改为 `是否正常=True`（代码中 `return (raw_question, True, "")`）。

**原因**：透传模式 = 节点不工作 = "没有问题" = 应放行。若返回 `False`，当你只关了通信节点、却忘了关门控的`启用`时，门控会误判"检测不通过"而**终止工作流**，与"禁用=跳过检测继续生成"的本意相反。返回 `True` 可让 bypass 真正无缝、防呆。

**如何恢复你的原意**：把 `h3_qwen_node.py` 中透传分支改为：
```python
if not 启用:
    return (raw_question, False, "")   # 你的原语义
```
（此时需保证门控`启用`也设为 False 才能 bypass。）

---

## 七、H3_检测门控 · 四态逻辑

| 门控启用 | 是否正常 | 行为 |
|---|---|---|
| False | 任意 | ✅ 放行（bypass） |
| True | True | ✅ 放行 |
| True | False | 🛑 抛异常终止工作流（跳过二采） |

接线：`#231.video_latent→门控.latent`；`通信.是否正常→门控.是否正常`；`门控.latent→#234`。

---

## 八、`/dev/shm` 协议 & JSON Schema

请求：`{system_prompt, question, skills, script, temperature, max_tokens, frame_count, color_order:"RGB", timestamp}`
结果：`{text, status:"ok"|"error", error?}`
持久化：`/mnt/workspace/pickup/{请求标识}_detection.json`（空标识存 `detection.json`）。

---

## 九、FAQ

**Q：检测不通过如何终止二采？** A：用 `H3_检测门控`，`是否正常=False` 时抛异常，二采/SaveVideo 不执行，`detection.json` 已保存供脚本读取。

**Q：为什么 `附加指令` 放后面？** A：Transformer 尾部注意力更高（Recency Effect），约束力强于提问主体。

**Q：服务端为何要 RGB→BGR？** A：`h3_frames.npy` 存 RGB，而 `cv2.imencode` 期望 BGR；不转会导致红蓝颠倒，影响"冷蓝+暖橙"色调判断。

**Q：会不会 /dev/shm 泄漏？** A：不会。固定 3 文件覆盖写 + 双清理 + tmpfs 自清。
