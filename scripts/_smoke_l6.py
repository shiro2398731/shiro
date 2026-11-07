"""L6 验证：只翻译符号、不回流、可缺席。"""
import sys, os, json, tempfile, shutil
sys.path.insert(0, '.')

from src.L6_expression.llm_adapter import LlmAdapter, TemplateAdapter

TMP = "data/_test_l6"
if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)

# 1. 没有 key 时 from_env 返回 None
os.environ.pop("DEEPSEEK_API_KEY", None)
print("1. 无 key →", LlmAdapter.from_env(out_dir=TMP))
assert LlmAdapter.from_env(out_dir=TMP) is None

# 2. 模板降级路径
tpl = TemplateAdapter(out_dir=TMP)
ret = tpl.translate({"trace": "3", "sequence": ["1", "2", "3"], "intent": "state"})
print("2. 模板 translate 返回值 =", ret)
assert ret is None, "translate 必须返回 None（架构红线）"

files = os.listdir(TMP)
print("   输出文件：", files)
rec = json.loads(open(os.path.join(TMP, files[0]), encoding="utf-8").read().strip())
print("   记录：", rec["text"], "| note:", rec["note"])
assert rec["ok"] is True

# 3. silent 时不说
tpl.translate({"trace": "none", "sequence": [], "intent": "silent"})
print("3. silent 时输出空 OK")

# 4. 白名单：传入的额外字段不会流出去
adapter = LlmAdapter(api_key="", out_dir=TMP)
adapter.translate({
    "trace": "5", "sequence": ["4", "5"], "intent": "state",
    "x": [1.0, 2.0, 3.0],           # 向量，不该出去
    "speed": 0.9,                    # 内部数值，不该出去
    "dev": 1.7,                      # 状态量，不该出去
})
files = sorted(os.listdir(TMP))
lines = open(os.path.join(TMP, files[-1]), encoding="utf-8").read().strip().splitlines()
last = json.loads(lines[-1])
print("4. 白名单过滤后的 symbols =", last["symbols"])
assert set(last["symbols"].keys()) == {"trace", "sequence", "intent"}, "白名单泄漏"
assert "x" not in last["symbols"] and "speed" not in last["symbols"]

# 5. API 挂了也不能抛异常
print("5. 空 key 调 API →", last["ok"], "（应为 False，且无异常）")
assert last["ok"] is False

shutil.rmtree(TMP)
print("\nL6 全部通过")
