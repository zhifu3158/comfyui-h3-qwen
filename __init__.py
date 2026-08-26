# -*- coding: utf-8 -*-
"""
comfyui-h3-qwen 插件入口
────────────────────────────────────────
作用：向 ComfyUI 注册：H3_Qwen通信 + H3_检测门控
ComfyUI 启动时会自动加载本文件，读取以下两个映射表：
  - NODE_CLASS_MAPPINGS        : 节点内部类名映射
  - NODE_DISPLAY_NAME_MAPPINGS : 节点在前端的中文显示名
"""

from .h3_qwen_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
