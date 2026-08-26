# -*- coding: utf-8 -*-
"""
================================================================================
H3_Qwen通信节点 (comfyui-h3-qwen)
================================================================================
功能：与独立运行的 Qwen3.8-27B VLM 服务，通过 /dev/shm 文件信号进行通信。

架构示意：
    ┌──── ComfyUI 进程（本节点）────┐         ┌──── Qwen 独立服务 ────┐
    │ 1. 写帧数据  h3_frames.npy    │──写入──→│ 每1秒轮询 request     │
    │ 2. 写请求    h3_request.json  │         │ 读取后执行推理        │
    │ 3. 每1秒轮询 h3_result.json   │←─写入──│ 写 h3_result.json     │
    │ 4. 读结果 → 清理 shm 文件     │         │ 删 h3_request.json    │
    └───────────────────────────────┘         └───────────────────────┘

设计铁律：
  1. 节点【不判断】是第几次介入（提示词完善 / 画面检测），只负责"传数据+收结果"，
     业务含义由用户通过 系统提示词 / 提问词 / 附加指令 自行指定。
  2. 统一【阻塞式】：写请求后每 1 秒轮询结果，直到超时。
  3. 【异常标志】是输出（不是输入），由用户自行决定后续工作流走向：
       False = Qwen 正常工作（完善成功 / 检测通过）
       True  = 通信异常（未启动/超时/空结果）或 检测不通过（含 "pass": false）
  4. 防泄漏：/dev/shm 固定文件名覆盖写，永远只有 3 个文件，不会累积。

色彩协议（重要）：
  h3_frames.npy 统一存储 【RGB】 uint8。
  ⚠️ Qwen 服务端用 cv2.imencode 前必须先转 BGR：
     bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
  否则红蓝通道颠倒，"冷蓝+暖橙"色调检测会出错！

环境依赖：numpy(系统2.1.3) / cv2(opencv-headless) / /dev/shm / /mnt/workspace/pickup
================================================================================
"""

import os
import time
import json
import numpy as np

# ==================== /dev/shm 通信文件协议（固定文件名，覆盖写，防泄漏） ====================
SHM_FRAMES  = "/dev/shm/h3_frames.npy"     # 帧数据: [N, H, W, 3] uint8 (RGB)
SHM_REQUEST = "/dev/shm/h3_request.json"   # 请求参数: 提问词/系统提示词/skills/剧本/温度等
SHM_RESULT  = "/dev/shm/h3_result.json"    # 推理结果: Qwen服务写入, 本节点读取后清理

# ==================== 结果持久化目录（取件码机制，与 app.py 一致） ====================
PICKUP_DIR  = "/mnt/workspace/pickup"      # 结果存为 {请求标识}_detection.json，可经 /video/ 路由下载

# ==================== 轮询间隔（用户指定 1 秒，避免高频轮询造成系统卡顿） ====================
POLL_INTERVAL = 1.0

# ==================== 异常信息前缀（用于异常标志识别） ====================
ERR_PREFIX = "[H3_Qwen通信]"


class H3_QwenComm:
    """
    H3_Qwen通信节点：与独立 Qwen3.8 服务通信的"哑管道"。
    输入：图片/视频数据 + 提示词 + 技能/剧本文件路径 + 推理参数
    输出：结果文本(STRING) + 异常标志(BOOLEAN)
    """

    # 节点级描述（显示在 ComfyUI "信息" 面板顶部）
    DESCRIPTION = (
        "与独立运行的 Qwen3.8-27B VLM 服务通信（/dev/shm 文件信号，每1秒轮询）。\n"
        "不判断介入次数：第一次(提示词完善)或第二次(画面检测)由你的系统提示词/提问词决定。\n"
        "输出【异常标志】: False=Qwen正常; True=通信异常或服务未启动或检测不通过(含\"pass\": false)。\n"
        "检测结果自动保存到 /mnt/workspace/pickup/{请求标识}_detection.json。"
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                # ── 开关 ─────────────────────────────────────────────
                "启用": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "节点总开关。设为 False 时不与 Qwen 通信，直接返回(空文本, False)，不影响后续工作流，用于临时禁用。",
                }),

                # ── 数据输入（三种接入方式可同时使用，内部自动合并） ─────
                "图片数据": ("IMAGE", {
                    "tooltip": "接收 IMAGE 张量（可多张）。可连接 LoadImage / VAEDecode 等节点输出，如：一采解码后的视频帧、参考图。",
                }),
                "图片路径": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "图片文件绝对路径，每行一个，支持批量。与图片数据可同时使用，内部自动合并。",
                }),
                "视频路径": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "视频文件绝对路径，每行一个，支持批量。按【每秒抽帧数】自动抽帧后合并。",
                }),

                # ── 附加文件（路径为空或不存在则自动跳过） ──────────────
                "技能文件路径": ("STRING", {
                    "default": "",
                    "tooltip": "H3 官方 Skills(.md) 文件绝对路径。为空或不存在则不使用。默认/自定义由前台 app 决定。",
                }),
                "剧本文件路径": ("STRING", {
                    "default": "",
                    "tooltip": "完整剧本文件绝对路径。为空或不存在则不使用。为 Qwen 提供全局意境参照。",
                }),

                # ── 提示词三件套 ──────────────────────────────────────
                "系统提示词": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "给 Qwen 的系统级指令（角色设定/规则/检测标准）。例：'你是视频质检员，检测布局/站位/运镜/动作…'",
                }),
                "提问词": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "提问词主体。可直接连接 #138 提示词节点的输出。",
                }),
                "附加指令": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "拼接在提问词【后面】（Transformer 尾部注意力更高，约束力强于提问词主体）。用于告诉 Qwen 如何操作提问词。",
                }),

                # ── 推理参数 ──────────────────────────────────────────
                "温度": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "采样温度。0.0=严格服从指令；越大 Qwen 自由发挥空间越大。检测任务建议 0.1。",
                }),
                "最大输出长度": ("INT", {
                    "default": 2048, "min": 50, "max": 4096, "step": 10,
                    "tooltip": "Qwen 回复的最大 token 数。完善后的提示词可能几千字，建议 2048~4096。",
                }),
                "超时时间": ("INT", {
                    "default": 60, "min": 5, "max": 300, "step": 5,
                    "tooltip": "等待 Qwen 响应的最大秒数。超时返回异常标志 True，不会让工作流无限卡死。",
                }),

                # ── 视频抽帧 & 图片缩放 ───────────────────────────────
                "每秒抽帧数": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 24.0, "step": 0.5,
                    "tooltip": "视频路径的抽帧率（每秒抽几帧）。仅对【视频路径】生效；图片数据/图片路径不受影响。",
                }),
                "缩放宽度": ("INT", {
                    "default": 640, "min": 128, "max": 1920, "step": 32,
                    "tooltip": "所有图片统一缩放到的目标宽度。越小传输/推理越快，推荐 640。",
                }),
                "缩放高度": ("INT", {
                    "default": 368, "min": 96, "max": 1080, "step": 16,
                    "tooltip": "所有图片统一缩放到的目标高度。越小传输/推理越快，推荐 368。",
                }),

                # ── 标识 ─────────────────────────────────────────────
                "请求标识": ("STRING", {
                    "default": "",
                    "tooltip": "取件码（由 app.py 注入）。结果存为 {标识}_detection.json；留空则存为 detection.json。",
                }),
            }
        }

    # 输出：结果文本 + 异常标志
    RETURN_TYPES  = ("STRING", "BOOLEAN")
    RETURN_NAMES  = ("结果文本", "异常标志")
    FUNCTION      = "communicate"
    CATEGORY      = "H3/Qwen通信"

    # 输出描述（部分前端版本会展示）
    OUTPUT_TOOLTIPS = (
        "Qwen 的完整回复文本（正常时）或异常信息（异常时）。",
        "False=Qwen正常工作; True=通信异常/服务未启动/检测不通过。由你决定后续流程。",
    )

    # ================================================================
    #  主函数：收集数据 → 写 shm → 阻塞等待 → 存结果 → 返回
    # ================================================================
    def communicate(self, 启用=True, 图片数据=None, 图片路径="", 视频路径="",
                    技能文件路径="", 剧本文件路径="", 系统提示词="", 提问词="",
                    附加指令="", 温度=0.1, 最大输出长度=2048, 超时时间=60,
                    每秒抽帧数=4.0, 缩放宽度=640, 缩放高度=368, 请求标识="",
                    **kwargs):

        # ── 0. 禁用检查：不通信、不报错、不影响工作流 ──
        if not 启用:
            return ("", False)

        try:
            # ── 1. 组装最终提问词（附加指令拼在【后面】，注意力更高）──
            q = str(提问词 or "").strip()
            ext = str(附加指令 or "").strip()
            if ext:
                q = q + "\n" + ext

            # ── 2. 收集所有图片帧（三种来源合并，统一 RGB uint8）──
            frames = self._collect_frames(
                图片数据, 图片路径, 视频路径, float(每秒抽帧数))

            # ── 3. 统一缩放到目标尺寸（减小传输/推理开销）──
            if frames:
                frames = self._resize_frames(
                    frames, int(缩放宽度), int(缩放高度))

            # ── 4. 读取附加文件（skills / 剧本，路径无效则返回空串）──
            skills = self._read_file(技能文件路径)
            script = self._read_file(剧本文件路径)

            # ── 5. 清理旧的 shm 文件（防止上一轮残留干扰）──
            self._cleanup_shm()

            # ── 6. 写入 /dev/shm（帧数据 + 请求参数）──
            if frames:
                arr = np.stack(frames, axis=0)          # [N, H, W, 3] uint8 (RGB)
                np.save(SHM_FRAMES, arr)

            req = {
                "system_prompt": str(系统提示词 or ""),
                "question":      q,
                "skills":        skills,
                "script":        script,
                "temperature":   float(温度),
                "max_tokens":    int(最大输出长度),
                "frame_count":   len(frames),
                "color_order":   "RGB",                 # 告知服务端色彩顺序
                "timestamp":     time.time(),
            }
            with open(SHM_REQUEST, "w", encoding="utf-8") as f:
                json.dump(req, f, ensure_ascii=False)

            # ── 7. 阻塞等待结果（每 1 秒轮询一次）──
            result_text, result_data = self._wait_result(int(超时时间))

            # ── 8. 保存结果到取件码目录（供脚本/前端抓取）──
            self._save_result(result_data, str(请求标识 or "").strip())

            # ── 9. 清理 /dev/shm（用完即删，防泄漏）──
            self._cleanup_shm()

            # ── 10. 判断异常标志 ──
            anomaly = self._check_anomaly(result_text)

            return (result_text, anomaly)

        except Exception as e:
            # 节点内部任何未预期异常 → 记录+清理+返回异常标志 True
            err = f"{ERR_PREFIX} 节点内部异常: {e}"
            self._save_result({"text": err, "error": str(e)},
                              str(请求标识 or "").strip())
            self._cleanup_shm()
            return (err, True)

    # ================================================================
    #  图片收集：IMAGE张量 / 图片路径 / 视频路径 三源合并
    # ================================================================
    def _collect_frames(self, tensor_in, path_in, video_in, fps):
        frames = []

        # ── 来源1：IMAGE 张量（ComfyUI 为 RGB float32 0~1）──
        if tensor_in is not None:
            try:
                arr = tensor_in.cpu().numpy()               # [B, H, W, C]
                if arr.ndim == 3:                           # 单张 [H,W,C] 补维度
                    arr = arr[None, ...]
                if arr.shape[-1] == 4:                      # 带 alpha 通道则丢弃
                    arr = arr[..., :3]
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                for i in range(arr.shape[0]):
                    frames.append(arr[i])                   # RGB uint8
            except Exception:
                pass

        # ── 来源2：图片路径（cv2 读取为 BGR，转 RGB 统一）──
        for p in self._lines(path_in):
            if os.path.isfile(p):
                try:
                    import cv2
                    img = cv2.imread(p)
                    if img is not None:
                        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                except Exception:
                    pass

        # ── 来源3：视频路径（按帧率抽帧，BGR 转 RGB 统一）──
        for p in self._lines(video_in):
            if os.path.isfile(p):
                try:
                    frames.extend(self._video_frames(p, fps))
                except Exception:
                    pass

        return frames

    def _video_frames(self, path, fps):
        """从视频按指定帧率均匀抽帧，返回 RGB uint8 列表"""
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        if src_fps <= 0:
            src_fps = 24.0
        # 每隔 step 帧取 1 帧
        step = max(1, int(round(src_fps / max(fps, 0.1))))
        out, idx = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        return out

    # ================================================================
    #  缩放：统一目标尺寸（INTER_AREA 适合缩小，速度快）
    # ================================================================
    def _resize_frames(self, frames, tw, th):
        import cv2
        out = []
        for f in frames:
            h, w = f.shape[:2]
            if h != th or w != tw:
                f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
            out.append(f)
        return out

    # ================================================================
    #  工具函数
    # ================================================================
    @staticmethod
    def _lines(text):
        """多行文本 → 非空行列表（用于路径批量解析）"""
        if not text:
            return []
        return [l.strip() for l in str(text).splitlines() if l.strip()]

    @staticmethod
    def _read_file(path):
        """读取文本文件；路径为空/不存在/读失败 → 返回空串"""
        path = str(path or "").strip()
        if not path or not os.path.isfile(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    @staticmethod
    def _cleanup_shm():
        """删除 /dev/shm 中三个通信文件（不存在则跳过）"""
        for f in (SHM_FRAMES, SHM_REQUEST, SHM_RESULT):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    def _wait_result(self, timeout_sec):
        """阻塞轮询结果文件，每 1 秒一次；超时返回异常信息"""
        start = time.time()
        while time.time() - start < timeout_sec:
            if os.path.exists(SHM_RESULT):
                try:
                    with open(SHM_RESULT, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("text", ""), data
                except Exception:
                    pass    # 结果文件可能正在被写入，下一轮再读
            time.sleep(POLL_INTERVAL)
        # 超时：Qwen 服务未启动或推理过慢
        msg = f"{ERR_PREFIX} 等待超时，Qwen 服务未在 {timeout_sec} 秒内响应"
        return msg, {"text": "", "error": "timeout"}

    @staticmethod
    def _save_result(data, tag):
        """结果持久化到取件码目录；tag 为空存 detection.json"""
        try:
            os.makedirs(PICKUP_DIR, exist_ok=True)
            name = f"{tag}_detection.json" if tag else "detection.json"
            with open(os.path.join(PICKUP_DIR, name), "w", encoding="utf-8") as f:
                json.dump(data or {"text": ""}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _check_anomaly(text):
        """
        异常标志判断（核心逻辑）：
          True  = 通信失败 / 结果为空或过短 / 检测不通过(含 "pass": false)
          False = 一切正常
        注意：节点不判断业务含义，只识别 "pass": false 与通信异常。
        """
        t = str(text or "").strip()
        # 空或过短 → 异常
        if len(t) < 5:
            return True
        lo = t.lower()
        # 检测不通过（第二次介入的 JSON 结果）→ 异常
        if '"pass": false' in lo or '"pass":false' in lo:
            return True
        # 节点内部异常/超时前缀 → 异常
        if ERR_PREFIX.lower() in lo:
            return True
        return False


# ==================== 节点注册 ====================
NODE_CLASS_MAPPINGS        = {"H3_QwenComm": H3_QwenComm}
NODE_DISPLAY_NAME_MAPPINGS = {"H3_QwenComm": "H3_Qwen通信"}