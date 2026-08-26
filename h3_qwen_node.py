# -*- coding: utf-8 -*-
"""
================================================================================
H3_Qwen通信节点 (comfyui-h3-qwen) —— 最终版
================================================================================
功能：与独立运行的 Qwen3.8-27B VLM 服务，通过 /dev/shm 文件信号通信。

【启用=False 透传模式】
  节点啥都不干：结果文本 = 原始提问词(#138 原文，不含附加指令/系统提示词)，
  是否正常=True（放行），错误信息=""。二选一开关在此模式不生效。

【启用=True 工作模式】
  ① 服务预检：开头先探心跳，超过【服务检测超时】无有效心跳→通信失败。
  ② 通信失败时看【失败回退原文】：开→结果文本=原始提问词；关→结果文本=错误信息。
  ③ 检测不通过(含 "pass": false)：结果文本=Qwen检测文本，不走回退。
  ④ 正常：结果文本=Qwen回复，是否正常=True。

输出：结果文本(STRING) + 是否正常(BOOLEAN) + 错误信息(STRING)

设计铁律：
  1. 节点不判断介入次数，只负责"传数据+收结果"。
  2. 统一阻塞式：写请求后每 1 秒轮询结果，直到超时。
  3. 防泄漏：/dev/shm 固定文件名覆盖写，永远只有 3 个文件。

色彩协议：h3_frames.npy 统一存【RGB】uint8；服务端 imencode 前需转 BGR。
心跳协议：Qwen 服务主循环每秒写 /tmp/h3_vlm_heartbeat。
================================================================================
"""

import os
import time
import json
import numpy as np

# ==================== /dev/shm 通信文件协议 ====================
SHM_FRAMES  = "/dev/shm/h3_frames.npy"
SHM_REQUEST = "/dev/shm/h3_request.json"
SHM_RESULT  = "/dev/shm/h3_result.json"

# ==================== 服务心跳文件（预检用） ====================
HEARTBEAT       = "/tmp/h3_vlm_heartbeat"
HEARTBEAT_FRESH = 3.0   # 心跳新鲜度容忍(秒)

# ==================== 结果持久化目录 ====================
PICKUP_DIR = "/mnt/workspace/pickup"

# ==================== 轮询间隔 ====================
POLL_INTERVAL = 1.0

# ==================== 异常信息前缀 ====================
ERR_PREFIX = "[H3_Qwen通信]"


class H3_QwenComm:
    """
    H3_Qwen通信节点（哑管道）：
    输出：结果文本(STRING) + 是否正常(BOOLEAN) + 错误信息(STRING)
    """

    DESCRIPTION = (
        "与独立 Qwen3.8-27B VLM 服务通信（/dev/shm 文件信号，每1秒轮询）。\n"
        "启用=False：透传模式，结果文本=原始提问词，是否正常=True，不通信。\n"
        "启用=True：先探心跳预检；通信失败时按【失败回退原文】返回原文或错误；检测不通过返回检测文本。\n"
        "是否正常: True=正常/透传; False=通信异常或检测不通过(含\"pass\": false)。"
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                # ── 开关 ──
                "启用": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "总开关。False=透传模式：结果文本=原始提问词，是否正常=True，不通信、二选一开关不生效。",
                }),

                # ── 数据输入（三源可同时使用，自动合并）──
                "图片数据": ("IMAGE", {
                    "tooltip": "接收 IMAGE 张量（可多张），可连 LoadImage / VAEDecode 输出。",
                }),
                "图片路径": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "图片绝对路径，每行一个，支持批量，与图片数据自动合并。",
                }),
                "视频路径": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "视频绝对路径，每行一个，按【每秒抽帧数】抽帧后合并。",
                }),

                # ── 附加文件 ──
                "技能文件路径": ("STRING", {
                    "default": "",
                    "tooltip": "H3 Skills(.md) 绝对路径，为空/不存在则不使用。",
                }),
                "剧本文件路径": ("STRING", {
                    "default": "",
                    "tooltip": "完整剧本绝对路径，为空/不存在则不使用。",
                }),

                # ── 提示词三件套 ──
                "系统提示词": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "系统级指令（角色/规则/检测标准）。",
                }),
                "提问词": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "提问主体，可连 #138 输出。也是【透传/回退】的原文来源（不含附加指令）。",
                }),
                "附加指令": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "拼接在提问词【后面】（尾部注意力更高），告诉 Qwen 如何操作提问词。",
                }),

                # ── 推理参数 ──
                "温度": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "采样温度，0=严格服从，越大越自由。",
                }),
                "最大输出长度": ("INT", {
                    "default": 2048, "min": 50, "max": 4096, "step": 10,
                    "tooltip": "回复最大 token 数，建议 2048~4096。",
                }),
                "超时时间": ("INT", {
                    "default": 60, "min": 5, "max": 300, "step": 5,
                    "tooltip": "等待推理结果的最大秒数（结果轮询超时）。",
                }),

                # ── 服务预检 ──
                "服务检测超时": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5,
                    "tooltip": "【开头预检】编码/传输之前探心跳，超时立即返回通信失败，节省时间。",
                }),

                # ── 文本二选一 ──
                "失败回退原文": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅当 启用=True 且【通信失败】时生效：开→结果文本=原始提问词；关→结果文本=错误信息。检测不通过不受影响。启用=False 时不生效。",
                }),

                # ── 抽帧 & 缩放 ──
                "每秒抽帧数": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 24.0, "step": 0.5,
                    "tooltip": "视频抽帧率，仅对【视频路径】生效。",
                }),
                "缩放宽度": ("INT", {
                    "default": 640, "min": 128, "max": 1920, "step": 32,
                    "tooltip": "统一缩放目标宽，推荐 640。",
                }),
                "缩放高度": ("INT", {
                    "default": 368, "min": 96, "max": 1080, "step": 16,
                    "tooltip": "统一缩放目标高，推荐 368。",
                }),

                # ── 标识 ──
                "请求标识": ("STRING", {
                    "default": "",
                    "tooltip": "取件码，结果存为 {标识}_detection.json；留空存 detection.json。",
                }),
            }
        }

    # 输出：结果文本 + 是否正常 + 错误信息
    RETURN_TYPES  = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES  = ("结果文本", "是否正常", "错误信息")
    FUNCTION      = "communicate"
    CATEGORY      = "H3/Qwen通信"

    OUTPUT_TOOLTIPS = (
        "启用=False→原始提问词；正常→Qwen回复；通信失败→原文(回退开)或错误(回退关)；检测不通过→检测文本。",
        "True=正常/透传; False=通信异常或检测不通过。",
        "仅通信失败/检测不通过时有内容，正常/透传为空。",
    )

    # ================================================================
    #  主函数
    # ================================================================
    def communicate(self, 启用=True, 图片数据=None, 图片路径="", 视频路径="",
                    技能文件路径="", 剧本文件路径="", 系统提示词="", 提问词="",
                    附加指令="", 温度=0.1, 最大输出长度=2048, 超时时间=60,
                    服务检测超时=2.0, 失败回退原文=False,
                    每秒抽帧数=4.0, 缩放宽度=640, 缩放高度=368, 请求标识="",
                    **kwargs):

        tag = str(请求标识 or "").strip()
        raw_question = str(提问词 or "").strip()   # 原文（透传/回退用，不含附加指令）

        # ── 0. 透传模式：节点啥都不干，直接返回原始提问词 ──
        # 是否正常=True 表示"没有问题、放行"；二选一开关在此不生效。
        # （若你希望透传时 是否正常=False，把下面的 True 改为 False 即可）
        if not 启用:
            return (raw_question, True, "")

        # ── 1. 服务预检（在编码/传输之前！）──
        if not self._service_alive(float(服务检测超时)):
            err = f"{ERR_PREFIX} 未检测到 Qwen 服务（{服务检测超时}s 内无有效心跳），通信失败"
            return self._comm_fail(err, raw_question, 失败回退原文, tag, save=True)

        try:
            # ── 2. 组装最终提问词（附加指令拼后面）──
            q = raw_question
            ext = str(附加指令 or "").strip()
            if ext:
                q = q + "\n" + ext

            # ── 3. 收集图片帧 ──
            frames = self._collect_frames(图片数据, 图片路径, 视频路径, float(每秒抽帧数))

            # ── 4. 缩放 ──
            if frames:
                frames = self._resize_frames(frames, int(缩放宽度), int(缩放高度))

            # ── 5. 读取附加文件 ──
            skills = self._read_file(技能文件路径)
            script = self._read_file(剧本文件路径)

            # ── 6. 清理旧 shm ──
            self._cleanup_shm()

            # ── 7. 写 /dev/shm ──
            if frames:
                np.save(SHM_FRAMES, np.stack(frames, axis=0))

            req = {
                "system_prompt": str(系统提示词 or ""),
                "question":      q,
                "skills":        skills,
                "script":        script,
                "temperature":   float(温度),
                "max_tokens":    int(最大输出长度),
                "frame_count":   len(frames),
                "color_order":   "RGB",
                "timestamp":     time.time(),
            }
            with open(SHM_REQUEST, "w", encoding="utf-8") as f:
                json.dump(req, f, ensure_ascii=False)

            # ── 8. 阻塞等待结果 ──
            result_text, result_data = self._wait_result(int(超时时间))

            # ── 9. 存结果 + 清理 ──
            self._save_result(result_data, tag)
            self._cleanup_shm()

            # ── 10. 分类判断 ──
            if not self._check_anomaly(result_text):
                return (result_text, True, "")                      # 正常
            if self._is_detect_fail(result_text):
                return (result_text, False, result_text)            # 检测不通过
            # 通信失败（超时/空结果等）→ 走二选一
            return self._comm_fail(result_text, raw_question, 失败回退原文, tag, save=False)

        except Exception as e:
            err = f"{ERR_PREFIX} 节点内部异常: {e}"
            return self._comm_fail(err, raw_question, 失败回退原文, tag, save=True)

    # ================================================================
    #  服务预检（心跳）
    # ================================================================
    def _service_alive(self, wait_sec):
        """在 wait_sec 内等待一个新鲜心跳；有→True，超时→False"""
        deadline = time.time() + max(0.0, wait_sec)
        while True:
            try:
                age = time.time() - os.path.getmtime(HEARTBEAT)
                if age <= HEARTBEAT_FRESH:
                    return True
            except FileNotFoundError:
                pass
            except Exception:
                pass
            if time.time() >= deadline:
                return False
            time.sleep(0.2)

    # ================================================================
    #  通信失败统一返回（含二选一）
    # ================================================================
    def _comm_fail(self, err, raw_question, use_fallback, tag, save=True):
        if save:
            self._save_result({"text": err, "error": "comm_fail"}, tag)
        if use_fallback and raw_question:
            return (raw_question, False, err)   # 回退原文
        return (err, False, err)                # 返回错误信息

    # ================================================================
    #  异常分类
    # ================================================================
    @staticmethod
    def _is_detect_fail(text):
        lo = str(text or "").lower()
        return ('"pass": false' in lo) or ('"pass":false' in lo)

    @staticmethod
    def _check_anomaly(text):
        t = str(text or "").strip()
        if len(t) < 5:
            return True
        lo = t.lower()
        if '"pass": false' in lo or '"pass":false' in lo:
            return True
        if ERR_PREFIX.lower() in lo:
            return True
        return False

    # ================================================================
    #  图片收集 / 抽帧 / 缩放 / 工具
    # ================================================================
    def _collect_frames(self, tensor_in, path_in, video_in, fps):
        frames = []
        if tensor_in is not None:
            try:
                arr = tensor_in.cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[None, ...]
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                for i in range(arr.shape[0]):
                    frames.append(arr[i])
            except Exception:
                pass
        for p in self._lines(path_in):
            if os.path.isfile(p):
                try:
                    import cv2
                    img = cv2.imread(p)
                    if img is not None:
                        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                except Exception:
                    pass
        for p in self._lines(video_in):
            if os.path.isfile(p):
                try:
                    frames.extend(self._video_frames(p, fps))
                except Exception:
                    pass
        return frames

    def _video_frames(self, path, fps):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        if src_fps <= 0:
            src_fps = 24.0
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

    def _resize_frames(self, frames, tw, th):
        import cv2
        out = []
        for f in frames:
            h, w = f.shape[:2]
            if h != th or w != tw:
                f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
            out.append(f)
        return out

    @staticmethod
    def _lines(text):
        if not text:
            return []
        return [l.strip() for l in str(text).splitlines() if l.strip()]

    @staticmethod
    def _read_file(path):
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
        for f in (SHM_FRAMES, SHM_REQUEST, SHM_RESULT):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    def _wait_result(self, timeout_sec):
        start = time.time()
        while time.time() - start < timeout_sec:
            if os.path.exists(SHM_RESULT):
                try:
                    with open(SHM_RESULT, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("text", ""), data
                except Exception:
                    pass
            time.sleep(POLL_INTERVAL)
        msg = f"{ERR_PREFIX} 等待超时，Qwen 服务未在 {timeout_sec} 秒内响应"
        return msg, {"text": "", "error": "timeout"}

    @staticmethod
    def _save_result(data, tag):
        try:
            os.makedirs(PICKUP_DIR, exist_ok=True)
            name = f"{tag}_detection.json" if tag else "detection.json"
            with open(os.path.join(PICKUP_DIR, name), "w", encoding="utf-8") as f:
                json.dump(data or {"text": ""}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# ================================================================
#  H3_检测门控（配套节点）
# ================================================================
class H3_Gate:
    """
    H3_检测门控（四态逻辑）：
      ① 门控启用=False      → 放行（bypass）
      ② 门控启用+是否正常=True  → 放行
      ③ 门控启用+是否正常=False → 抛异常终止工作流（跳过二采）
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "一采 video_latent，原样传给二采链路"}),
                "是否正常": ("BOOLEAN", {"tooltip": "连接 H3_Qwen通信 的【是否正常】输出"}),
            },
            "optional": {
                "启用": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "与 H3_Qwen通信 的【启用】保持一致。False=bypass 直接放行。",
                }),
                "终止信息": ("STRING", {
                    "multiline": True,
                    "default": "H3 检测未通过或服务异常，终止本次工作流（跳过二采）。",
                    "tooltip": "终止时显示的错误信息。",
                }),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "gate"
    CATEGORY      = "H3/Qwen通信"
    DESCRIPTION   = "门控：不启用→放行；启用且是否正常=True→放行；启用且是否正常=False→抛异常终止工作流。"

    def gate(self, latent, 是否正常, 启用=True, 终止信息="H3 检测未通过或服务异常，终止本次工作流（跳过二采）。"):
        if not 启用:
            return (latent,)
        if 是否正常:
            return (latent,)
        raise RuntimeError(f"[H3_门控] {终止信息}")


# ==================== 注册 ====================
NODE_CLASS_MAPPINGS        = {"H3_QwenComm": H3_QwenComm, "H3_Gate": H3_Gate}
NODE_DISPLAY_NAME_MAPPINGS = {"H3_QwenComm": "H3_Qwen通信", "H3_Gate": "H3_检测门控"}
