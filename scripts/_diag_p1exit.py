"""复现 P1 启动即退出的问题"""
import sys
sys.path.insert(0, '.')
try:
    from src.common.bus import Bus
    b = Bus("P1")
    b.subscribe("l7.event.phase", lambda t, o: None, owner_module="src.L7_lifecycle.rhythm")
    print("subscribe OK")
    b.stop()
except Exception as e:
    import traceback
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
