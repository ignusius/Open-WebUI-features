"""
title: Qwen3 Think Tags Fixer
author: local
version: 2.0
"""

from typing import Optional


class Filter:
    def __init__(self):
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Включаем размышления на уровне запроса (так как на сервере мы это отключили)
        body["chat_template_kwargs"] = {"enable_thinking": True}
        return body
