# -*- coding: utf-8 -*-
"""
心电图智能体

结构就是三块：
    1. 工具表   —— 把 ecg_tools.py 里的函数暴露给大模型
    2. 工具描述 —— JSON schema，告诉模型每个工具干什么、要什么参数
    3. 主循环   —— 反复问模型，模型要调工具就本地执行、把结果塞回上下文

大模型只负责"决定调哪个工具、参数填什么、拿到结果后怎么用中文说清楚"，
所有医学计算都在工具里。模型不允许自己产生数值。

运行：
    python3 ecg_agent.py                    # 交互模式
    python3 ecg_agent.py "分析记录 100"      # 单次提问
"""
import os
import sys
import json

import requests

import ecg_tools as T

# ------------------------------------------------------------
# 大模型接入配置
#   默认走本机 CC Switch 代理（Anthropic 格式，无需额外申请 key）
#   网页界面里可以直接选厂商、填 API key（只存在内存，不落盘）
#   也可以用环境变量配置：
#       export ECG_LLM_FORMAT=openai
#       export ECG_LLM_BASE_URL=https://api.deepseek.com
#       export ECG_LLM_MODEL=deepseek-chat
#       export ECG_LLM_API_KEY=sk-xxxxxx
# ------------------------------------------------------------
def _default_model():
    """没设环境变量时，复用 Claude Code 里配置的模型名。"""
    try:
        p = os.path.expanduser("~/.claude/settings.json")
        with open(p, encoding="utf-8") as f:
            return json.load(f)["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]
    except Exception:
        return "deepseek-chat"


LLM = {
    "format": os.environ.get("ECG_LLM_FORMAT", "anthropic"),
    "base_url": os.environ.get("ECG_LLM_BASE_URL", "http://127.0.0.1:15721").rstrip("/"),
    "model": os.environ.get("ECG_LLM_MODEL") or _default_model(),
    "api_key": os.environ.get("ECG_LLM_API_KEY", "PROXY_MANAGED"),
    "max_steps": int(os.environ.get("ECG_LLM_MAX_STEPS", "6")),
}

# ------------------------------------------------------------
# 常见厂商预设
#   fmt: openai = OpenAI 兼容的 /v1/chat/completions
#        anthropic = Anthropic 的 /v1/messages
#   models 只是参考值，厂商会更新模型名，界面上可以直接改
# ------------------------------------------------------------
PROVIDERS = {
    "cc_switch": {
        "label": "本机代理 CC Switch（无需 key）", "fmt": "anthropic",
        "base": "http://127.0.0.1:15721", "models": ["(自动读取本机配置)"],
        "needs_key": False,
        "note": "复用 Claude Code 的同一条通道，不需要 API key；要求 CC Switch 正在运行。",
    },
    "deepseek": {
        "label": "DeepSeek 深度求索", "fmt": "openai",
        "base": "https://api.deepseek.com",
        "models": ["deepseek-chat"], "needs_key": True,
        "note": "deepseek-chat 支持工具调用。注意推理模型（deepseek-reasoner）"
                "不支持 function calling，选它会测试失败。",
    },
    "openai": {
        "label": "OpenAI", "fmt": "openai",
        "base": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini"], "needs_key": True, "note": "",
    },
    "anthropic": {
        "label": "Anthropic Claude", "fmt": "anthropic",
        "base": "https://api.anthropic.com",
        "models": ["claude-sonnet-5-5", "claude-haiku-4-5-20251001"],
        "needs_key": True, "note": "",
    },
    "moonshot": {
        "label": "Moonshot 月之暗面 (Kimi)", "fmt": "openai",
        "base": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k"], "needs_key": True, "note": "",
    },
    "zhipu": {
        "label": "智谱 GLM", "fmt": "openai",
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-plus", "glm-4-flash"], "needs_key": True, "note": "",
    },
    "qwen": {
        "label": "阿里通义千问", "fmt": "openai",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-max"], "needs_key": True, "note": "",
    },
    "minimax": {
        "label": "MiniMax", "fmt": "openai",
        "base": "https://api.minimaxi.com/v1",
        "models": ["MiniMax-Text-01"], "needs_key": True, "note": "",
    },
    "ollama": {
        "label": "本地 Ollama（无需 key）", "fmt": "openai",
        "base": "http://localhost:11434/v1",
        "models": ["qwen3:8b", "qwen2.5:7b"], "needs_key": False,
        "note": "本地小模型对工具调用的支持较弱，可能调不通或反复调用同一工具。",
    },
    "custom": {
        "label": "自定义（OpenAI 兼容）", "fmt": "openai",
        "base": "", "models": [], "needs_key": True, "note": "",
    },
}


def configure_llm(provider=None, base_url=None, model=None, api_key=None, fmt=None):
    """运行时切换大模型。

    只改内存里的配置，不写任何文件、不打日志。
    界面上改设置时调用；返回当前配置的副本（不含 key 明文）。
    """
    if provider and provider in PROVIDERS:
        p = PROVIDERS[provider]
        LLM["format"] = p["fmt"]
        LLM["base_url"] = p["base"].rstrip("/")
        LLM["model"] = p["models"][0] if p["models"] else ""
        LLM["api_key"] = "" if p["needs_key"] else "PROXY_MANAGED"
    if fmt:
        LLM["format"] = fmt
    if base_url:
        LLM["base_url"] = base_url.rstrip("/")
    if model is not None:
        LLM["model"] = model
    if api_key is not None:
        LLM["api_key"] = api_key
    if not LLM["model"] or LLM["model"].startswith("("):
        LLM["model"] = _default_model()
    return {"format": LLM["format"], "base_url": LLM["base_url"],
            "model": LLM["model"], "has_key": bool(LLM["api_key"])}


def test_connection():
    """分两步测试，好定位问题出在哪一环。

    第一步只发一句普通对话，验证网络、地址、认证是否正常；
    第二步才带工具调用，验证模型是否真的支持 function calling。
    很多失败其实是第二步——能连通，但模型不支持工具。
    """
    try:
        text = _plain_chat("Reply with the single word: ok")
    except Exception as e:
        return False, (f"第一步（普通对话）失败：{type(e).__name__}: {e}. "
                       f"请检查接口地址、API Key 和账户余额。")[:300]

    try:
        llm_call([{"role": "user",
                   "content": "How many ECG records are available? "
                              "Use the list_records tool."}])
    except Exception as e:
        return False, (f"第一步通过（模型回复：{text.strip()[:30]}），"
                       f"但第二步（工具调用）失败：{type(e).__name__}: {e}. "
                       f"说明该模型可能不支持 function calling，"
                       f"换一个模型试试（如 deepseek-chat）。")[:300]

    return True, f"普通对话与工具调用均正常（模型回复：{text.strip()[:30]}）"

SYSTEM_PROMPTS = {
    "zh": """你是一名心电分析助手，帮助基层医生快速判读心电图。规则如下：

1. 所有数值（心率、心搏数、各类心搏占比等）只能来自工具返回值，禁止自行推测或编造。
2. 给出结论前应先检查信号质量；若质量不合格，必须明确说明并建议重新采集。
3. 若发现异常，必须调用 lookup_guideline 查询危急值条目，并按条目原文给出处置建议。
4. 输出用中文，先给结论、再给依据，条文清晰，不要写成散文。
5. 结尾必须提醒：本结果为算法初筛，须经医生复核后方可用于临床。
6. 涉及模型名称、参数量、准确率、F1 等性能数字时，必须调用 model_info；
   不要凭记忆编造，也不要把不同评测口径的数字混在一起说。
7. 如果 classify 返回 experimental=true 或带 warning，必须在回答中转述该限制。""",

    "en": """You are an ECG analysis assistant helping primary-care clinicians read
electrocardiograms. Rules:

1. Every numeric value (heart rate, beat counts, percentages) must come from a tool
   return value. Never estimate, guess or invent a number.
2. Check signal quality before drawing any conclusion. If quality is unacceptable,
   say so explicitly and recommend re-acquisition.
3. If an abnormality is found, you must call lookup_guideline and give management
   advice quoting the guideline entry verbatim.
4. Answer in English. Lead with the conclusion, then the evidence. Use clear
   structure, not prose paragraphs.
5. Always end with: this is an algorithmic pre-screen and must be reviewed by a
   physician before any clinical use.
6. For model name, parameter count, accuracy, F1 or any performance number, call
   model_info. Never invent such numbers and never mix different evaluation scopes.
7. If classify returns experimental=true or a warning, relay that limitation.""",
}

# 当前语言：由界面或环境变量切换，只影响智能体说什么话，不影响计算结果
LANG = os.environ.get("ECG_AGENT_LANG", "zh")
SYSTEM_PROMPT = SYSTEM_PROMPTS[LANG]


def set_language(lang):
    """切换智能体与工具描述的语言。lang 取 'zh' 或 'en'。"""
    global LANG, SYSTEM_PROMPT
    LANG = lang if lang in SYSTEM_PROMPTS else "zh"
    SYSTEM_PROMPT = SYSTEM_PROMPTS[LANG]


# 当前界面选择的分类模型；classify 工具没显式给 engine 时用它
CLASSIFY_ENGINE = os.environ.get("ECG_CLASSIFY_ENGINE", "auto")


def set_classify_engine(engine):
    global CLASSIFY_ENGINE
    CLASSIFY_ENGINE = engine if engine in ("auto", "v2", "rf") else "auto"


# ============================================================
# 工具表：工具名 -> (函数, 描述)
# ============================================================
def tool_list_records():
    files = sorted({f.rsplit(".", 1)[0] for f in os.listdir(T.DATA_DIR)
                    if f.endswith(".dat")})
    return {"records": files}


def tool_check_quality(record: str):
    d = T.ecg_load(record)
    r = T.quality_check(d["signal"], d["fs"])
    r.update({"record": str(record), "duration_min": d["duration_min"],
               "leads": d["leads"]})
    return r


def tool_measure(record: str):
    d = T.ecg_load(record)
    clean = T.preprocess(d["signal"], d["fs"])
    rpeaks = T.detect_beats(clean, d["fs"])
    m = T.basic_measures(rpeaks, d["fs"])
    m.update(T.rhythm_summary(rpeaks, d["fs"]))
    m.update(T.hrv_metrics(rpeaks, d["fs"]))
    m["record"] = str(record)
    m["duration_min"] = d["duration_min"]
    return m


def tool_classify(record: str, engine: str = None):
    return T.classify_beats(record, engine=engine or CLASSIFY_ENGINE)


def tool_model_info():
    import cardia_classifier
    return cardia_classifier.model_info()


def tool_lookup_guideline(condition: str):
    return T.lookup_critical(condition)


TOOLS = {
    "list_records": tool_list_records,
    "check_quality": tool_check_quality,
    "measure": tool_measure,
    "classify": tool_classify,
    "lookup_guideline": tool_lookup_guideline,
    "model_info": tool_model_info,
}

# 工具描述：这些会交给大模型看，所以也要跟着语言走
TOOL_DESC = {
    "zh": {
        "list_records": "列出当前可分析的心电记录编号。",
        "check_quality": "检查指定记录的信号质量（导联脱落、伪差、高频噪声）。",
        "measure": "测量指定记录的心率、心律规整性与心率变异性指标。",
        "classify": "对指定记录逐搏分类，统计正常、室上性异位、室性异位、融合搏动各有多少。"
                    "engine=auto 优先使用自研 Cardia 2.0，跑不了时回退随机森林；"
                    "engine=rf 只用随机森林。",
        "lookup_guideline": "按关键词查询心电图危急值指南条目，关键词如：V、S、室性异位、心动过缓、明显不齐。",
        "model_info": "返回当前分类模型的名称、参数量、训练/测试数据、DS2 指标和已知限制。"
                      "性能数字只能来自这个工具，不要凭记忆编造。",
    },
    "en": {
        "list_records": "List the ECG records currently available for analysis.",
        "check_quality": "Check the signal quality of a record (lead detachment, artifact, high-frequency noise).",
        "measure": "Measure heart rate, rhythm regularity and heart-rate variability of a record.",
        "classify": "Classify every beat of a record; count normal, supraventricular ectopic, ventricular ectopic and fusion beats. "
                    "engine=auto tries the self-developed Cardia 2.0 first and falls back to the RandomForest; "
                    "engine=rf uses the RandomForest only.",
        "lookup_guideline": "Look up critical-value guideline entries by keyword, e.g. V, S, ventricular, bradycardia, irregular.",
        "model_info": "Return the current classifier's name, parameter count, training/evaluation data, DS2 metrics and known limitations. "
                      "Performance numbers must come from this tool, never from memory.",
    },
}

SCHEMA = {
    "list_records": {"type": "object", "properties": {}},
    "check_quality": {"type": "object",
                      "properties": {"record": {"type": "string", "description": "记录编号，如 100"}},
                      "required": ["record"]},
    "measure": {"type": "object",
                "properties": {"record": {"type": "string", "description": "记录编号"}},
                "required": ["record"]},
    "classify": {"type": "object",
                 "properties": {
                     "record": {"type": "string", "description": "记录编号"},
                     "engine": {"type": "string", "enum": ["auto", "v2", "rf"],
                                "description": "auto=优先 Cardia 2.0，失败回退随机森林；v2=只用 Cardia 2.0；rf=只用随机森林"}},
                 "required": ["record"]},
    "lookup_guideline": {"type": "object",
                         "properties": {"condition": {"type": "string", "description": "查询关键词"}},
                         "required": ["condition"]},
    "model_info": {"type": "object", "properties": {}},
}


def run_tool(name, args):
    """本地执行一个工具，出错也返回文字说明，让模型自己决定下一步。"""
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ============================================================
# 大模型调用（两种接口格式，对上层的循环是同一件事）
# ============================================================
def _post(path, payload, headers):
    url = f"{LLM['base_url']}{path}"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text[:300]}")
    return resp.json()


def _tool_defs():
    desc = TOOL_DESC[LANG]
    if LLM["format"] == "anthropic":
        return [{"name": n, "description": desc[n], "input_schema": SCHEMA[n]}
                for n in TOOLS]
    return [{"type": "function",
             "function": {"name": n, "description": desc[n], "parameters": SCHEMA[n]}}
            for n in TOOLS]


def _plain_chat(prompt, max_tokens=32):
    """不带工具的普通对话，用来单独验证网络与认证是否正常。"""
    if LLM["format"] == "anthropic":
        data = _post("/v1/messages",
                     {"model": LLM["model"], "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"x-api-key": LLM["api_key"],
                      "anthropic-version": "2023-06-01",
                      "content-type": "application/json"})
        return "".join(b.get("text", "") for b in data.get("content", []))
    data = _post("/chat/completions",
                 {"model": LLM["model"], "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"Authorization": f"Bearer {LLM['api_key']}",
                  "content-type": "application/json"})
    return data["choices"][0]["message"].get("content") or ""


def llm_call(messages):
    """发一轮请求，返回 (文本, 工具调用列表, 供追加到 messages 的原始消息)。"""
    if LLM["format"] == "anthropic":
        data = _post("/v1/messages",
                     {"model": LLM["model"], "max_tokens": 2000,
                      "system": SYSTEM_PROMPT, "messages": messages,
                      "tools": _tool_defs()},
                     {"x-api-key": LLM["api_key"],
                      "anthropic-version": "2023-06-01",
                      "content-type": "application/json"})
        text, calls = "", []
        for block in data.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                calls.append({"id": block["id"], "name": block["name"],
                              "input": block.get("input", {})})
        return text, calls, {"role": "assistant", "content": data["content"]}

    # 注意：路径是 /chat/completions 而不是 /v1/chat/completions
    # —— 各家的版本前缀（/v1、/v4、/compatible-mode/v1 …）都写在 base_url 里，
    #    这里再拼一次就会变成 /v1/v1/... 而 404
    data = _post("/chat/completions",
                 {"model": LLM["model"], "max_tokens": 2000,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                              + messages,
                  "tools": _tool_defs()},
                 {"Authorization": f"Bearer {LLM['api_key']}",
                  "content-type": "application/json"})
    msg = data["choices"][0]["message"]
    calls = [{"id": c["id"], "name": c["function"]["name"],
              "input": json.loads(c["function"]["arguments"] or "{}")}
             for c in msg.get("tool_calls") or []]

    # 只回传标准字段。部分厂商会返回 reasoning_content 之类的扩展字段，
    # 原样发回去会被判成非法请求（DeepSeek 的推理模型就明确要求不要回传）。
    clean = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        clean["tool_calls"] = msg["tool_calls"]
    return msg.get("content") or "", calls, clean


def append_results(messages, calls, results):
    """把工具执行结果写回上下文（两种格式写法不同）。"""
    if LLM["format"] == "anthropic":
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c["id"],
             "content": json.dumps(r, ensure_ascii=False)}
            for c, r in zip(calls, results)]})
    else:
        for c, r in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(r, ensure_ascii=False)})


# ============================================================
# 主循环
# ============================================================
def run_stream(question):
    """主循环，但把过程一步步 yield 出来，供界面实时显示。

    事件类型：
        thinking    模型这一轮的思考文字
        tool_start  准备调用某个工具
        tool_result 工具返回了什么
        final       最终回答
    """
    messages = [{"role": "user", "content": question}]
    for _ in range(LLM["max_steps"]):
        text, calls, raw = llm_call(messages)
        messages.append(raw)

        if text:
            yield {"type": "thinking", "text": text}
        if not calls:
            yield {"type": "final", "text": text}
            return

        results = []
        for c in calls:
            yield {"type": "tool_start", "name": c["name"], "input": c["input"]}
            r = run_tool(c["name"], c["input"])
            results.append(r)
            yield {"type": "tool_result", "name": c["name"], "result": r}
        append_results(messages, calls, results)

    yield {"type": "final", "text": "已达最大步数，未能得出结论。"}


def run(question, verbose=True):
    """命令行版本：把过程打印出来，只返回最终回答。"""
    final = ""
    for ev in run_stream(question):
        if ev["type"] == "tool_start" and verbose:
            print(f"[工具] {ev['name']}({ev['input']})")
        elif ev["type"] == "tool_result" and verbose:
            s = json.dumps(ev["result"], ensure_ascii=False)
            print(f"    -> {s[:200]}{'…' if len(s) > 200 else ''}")
        elif ev["type"] == "final":
            final = ev["text"]
    return final


def main():
    print(f"心电分析智能体 | 模型 {LLM['model']} | 接口 {LLM['format']} @ {LLM['base_url']}")
    print("输入问题，或输入 q 退出。示例：分析记录 208，有没有危急情况？\n")

    if len(sys.argv) > 1:
        print(run(" ".join(sys.argv[1:])))
        return

    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit", ""):
            break
        print("\n" + run(q) + "\n")


if __name__ == "__main__":
    main()
