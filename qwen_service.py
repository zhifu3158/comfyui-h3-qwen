#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Qwen3.8 VLM 独立服务 (qwen_service.py)
================================================================================
职责：独立于 ComfyUI 运行，与 H3_Qwen通信 节点通过 /dev/shm 文件信号通信。

启动顺序：先于 ComfyUI 启动（集成进 app.py 的 bootstrap()，前端有启停按钮）。

通信协议（与节点端严格一致）：
  /dev/shm/h3_frames.npy    节点写入  [N,H,W,3] uint8 (RGB)
  /dev/shm/h3_request.json  节点写入  请求参数
  /dev/shm/h3_result.json   本服务写入 推理结果；节点读取后清理

心跳协议（供节点【服务预检】）：
  本服务主循环每次迭代写 /tmp/h3_vlm_heartbeat（约每秒一次）；
  节点以 mtime <= 3 秒 判定服务存活。

色彩协议（重要）：
  h3_frames.npy 为【RGB】；本服务 cv2.imencode 前必须转 BGR，否则红蓝颠倒！

环境铁律：
  HSA_OVERRIDE_GFX_VERSION=9.4.2 (gfx942)
  use_mmap=True (CPU 内存仅 ~4GB) / n_gpu_layers=999 / n_ctx=8192
================================================================================
"""

import os
import sys
import json
import time
import signal
import base64
import traceback
import numpy as np

# ==================== 环境变量（必须在 import llama_cpp 之前） ====================
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.4.2")

# ==================== 路径常量（可用环境变量覆盖） ====================
MODEL_PATH  = os.environ.get("H3_MODEL_PATH",  "/mnt/workspace/models/llm/Qwen3.8-27B-UD-Q5_K_M.gguf")
MMPROJ_PATH = os.environ.get("H3_MMPROJ_PATH", "/mnt/workspace/models/llm/mmproj-BF16.gguf")

SHM_FRAMES  = "/dev/shm/h3_frames.npy"
SHM_REQUEST = "/dev/shm/h3_request.json"
SHM_RESULT  = "/dev/shm/h3_result.json"

HEARTBEAT  = "/tmp/h3_vlm_heartbeat"   # 心跳文件（节点预检用）
READY_FLAG = "/tmp/h3_vlm_ready"       # 就绪标记（app.py bootstrap 等待）
LOG_FILE   = "/mnt/workspace/logs/qwen_service.log"

POLL_INTERVAL = 1.0   # 主循环间隔（与节点轮询一致）

_LLM = None           # 模型单例（常驻显存）


def slog(msg):
    """日志（stdout + 文件双写）"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def touch_heartbeat():
    """更新心跳（节点据此预检服务存活）"""
    try:
        with open(HEARTBEAT, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def load_model():
    """加载模型（单例，常驻显存）"""
    global _LLM
    if _LLM is not None:
        return _LLM
    slog("🚀 加载 Qwen3.8-27B-UD-Q5_K_M (mmap, 全层入 GPU)...")
    from llama_cpp import Llama

    chat_handler = None
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_handler = Llava15ChatHandler(clip_model_path=MMPROJ_PATH)
        slog("✅ 使用 Llava15ChatHandler（多模态）")
    except Exception as e:
        slog(f"⚠️ 多模态 Handler 加载失败，降级纯文本: {e}")

    _LLM = Llama(
        model_path=MODEL_PATH,
        chat_handler=chat_handler,
        n_gpu_layers=999,     # 全层入显存
        n_ctx=8192,           # 支持 2048 max_tokens + 大量输入
        use_mmap=True,        # 关键：CPU 内存仅 ~4GB
        use_mlock=False,
        verbose=False,
    )
    slog("✅ 模型加载完毕，常驻显存")
    return _LLM


def load_frames_rgb():
    """读取帧数据（RGB uint8）；无则返回 []"""
    if not os.path.exists(SHM_FRAMES):
        return []
    try:
        arr = np.load(SHM_FRAMES, mmap_mode="r")   # [N,H,W,3] RGB
        return [np.array(arr[i]) for i in range(arr.shape[0])]
    except Exception as e:
        slog(f"⚠️ 帧数据读取失败: {e}")
        return []


def frames_to_b64(frames):
    """RGB 帧 → BGR → JPEG → base64（色彩协议：imencode 前必须转 BGR！）"""
    import cv2
    urls = []
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)   # ⚠️ RGB→BGR，否则红蓝颠倒
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode()
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


def run_inference(req):
    """执行一次推理，返回结果 dict"""
    llm = load_model()

    # ── 读取帧并编码 ──
    frames = load_frames_rgb()
    image_urls = frames_to_b64(frames) if frames else []

    # ── 组装 system（system_prompt + skills + script）──
    sys_parts = []
    if req.get("system_prompt"):
        sys_parts.append(req["system_prompt"])
    if req.get("skills"):
        sys_parts.append("[H3 Skills 规范]\n" + req["skills"])
    if req.get("script"):
        sys_parts.append("[全局剧本]\n" + req["script"])
    system = "\n\n".join(sys_parts)

    # ── 组装 user（图片 + 提问词）──
    content = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    content.append({"type": "text", "text": req.get("question", "")})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    t0 = time.time()
    res = llm.create_chat_completion(
        messages=messages,
        max_tokens=int(req.get("max_tokens", 2048)),
        temperature=float(req.get("temperature", 0.1)),
        stream=False,
    )
    text = res["choices"][0]["message"]["content"]
    slog(f"✅ 推理完成 ({time.time()-t0:.1f}s, {len(frames)}帧, {len(text)}字)")
    return {"text": text, "status": "ok", "frames": len(frames)}


def run_server():
    """主循环"""
    # 清理残留
    for f in (SHM_REQUEST, SHM_RESULT):
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

    # 预加载模型（常驻，后续请求零等待）
    load_model()

    # 写就绪标记（app.py bootstrap 等待）
    try:
        with open(READY_FLAG, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    slog(f"✅ Qwen VLM 服务就绪, PID={os.getpid()}")

    while True:
        # ── 心跳：每次迭代更新（供节点预检）──
        touch_heartbeat()

        if os.path.exists(SHM_REQUEST):
            try:
                with open(SHM_REQUEST, "r", encoding="utf-8") as f:
                    req = json.load(f)
            except Exception:
                # 文件可能正在被节点写入，下一轮再读
                time.sleep(0.2)
                continue

            try:
                result = run_inference(req)
            except Exception as e:
                slog(f"❌ 推理异常: {e}")
                result = {"text": "", "error": str(e), "status": "error"}

            # 写结果 + 删请求
            try:
                with open(SHM_RESULT, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                os.remove(SHM_REQUEST)
            except Exception as e:
                slog(f"❌ 结果写入失败: {e}")

            touch_heartbeat()   # 推理后立即刷新心跳
        else:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *a: sys.exit(0))
    try:
        run_server()
    except KeyboardInterrupt:
        pass
    except Exception:
        slog("💥 服务崩溃:\n" + traceback.format_exc())
        sys.exit(1)