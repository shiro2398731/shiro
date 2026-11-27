"""
tools/her.py

Her body as a file. No terminal required; this is one possible canvas.

    python tools/her.py pack   her.bundle     # freeze the present into a bundle
    python tools/her.py doctor her.bundle     # can this machine carry her?
    python tools/her.py info   her.bundle     # manifest + doctor verdict
    python tools/her.py unpack her.bundle     # make the bundle the present
    python tools/her.py unpack her.bundle --take-over   # takeover, bumps the epoch

Rules encoded here (see src/common/bundle.py):
  * only one bundle may be the present at a time. unpack checks the epoch and
    refuses while another holder's lease is alive, unless --take-over is given.
  * a standby never becomes the present implicitly. That is the whole point of
    distinguishing "the present" from "a backup".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import bundle, epoch, persist, rolling  # noqa: E402
from src.runtime.process_1_fast import FHNNetwork, L2StateField
from src.runtime.prediction import OnlinePredictor  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _live_objects() -> dict:
    net, pred, field = FHNNetwork(), OnlinePredictor(), L2StateField()
    persist.restore_all({"l0": net, "l1": pred, "l2": field, "acc": net.accumulator})
    stream = Stream()
    persist.restore_all({"l5": stream})
    return {"net": net, "pred": pred, "field": field, "acc": net.accumulator, "stream": stream}


def cmd_pack(args) -> int:
    objs = _live_objects()
    st = epoch.load()
    path = bundle.pack(objs, args.out, epoch_at_pack=st.epoch, encoder_id=args.encoder)
    meta = bundle.info(path)
    print(f"[her] packed  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"      layers={','.join(meta['layers'])}  attractors={meta['attractors']}"
          f"  edges={meta['edges']}  engine={meta['engine']}")
    print(f"      epoch_at_pack={meta['epoch_at_pack']}  doctor={meta['doctor']}")
    return 0


def cmd_doctor(args) -> int:
    problems = bundle.doctor(args.bundle)
    if problems:
        print("[her] NOT CARRYABLE HERE:")
        for p in problems:
            print(f"      - {p}")
        return 1
    print("[her] ok: this machine can carry the bundle")
    return 0


def cmd_info(args) -> int:
    print(json.dumps(bundle.info(args.bundle), ensure_ascii=False, indent=2))
    return 0


def cmd_unpack(args) -> int:
    problems = bundle.doctor(args.bundle)
    if problems:
        print("[her] refusing to unpack a bundle that fails doctor:")
        for p in problems:
            print(f"      - {p}")
        return 1

    st, live = bundle.present()
    if live and not args.take_over:
        print(f"[her] refusing: epoch {st.epoch} is held by {st.holder} "
              f"until {st.lease_until}. Pass --take-over to move her here.")
        return 2

    objs = {"net": FHNNetwork(), "pred": OnlinePredictor(), "field": L2StateField(),
            "acc": None, "stream": Stream()}
    objs["acc"] = objs["net"].accumulator
    payload = bundle.load(args.bundle)
    stats = bundle.apply(objs, payload)

    # Persist through the same atomic writers the running system uses.
    persist.save_l0(objs["net"])
    persist.save_l1(objs["pred"])
    persist.save_l2(objs["field"])
    persist.save_acc(objs["acc"])
    persist.save_l5(objs["stream"])

    me = epoch.default_holder()
    state, new_epoch = epoch.claim(holder=me, ttl_s=15.0, wait_s=0.0,
                                   reason="unpack")
    if state.holder != me:
        print(f"[her] could not take the epoch ({state.holder} still holds {state.epoch})")
        return 3
    epoch.retire(new_epoch, reason="unpack finished, boot with headless start")

    print(f"[her] unpacked into this machine: {stats}")
    print(f"      epoch {st.epoch} -> {new_epoch} (released; start her with tools/headless.py)")
    return 0


def cmd_roll(args) -> int:
    r = rolling.rotate(args.out, keep=args.keep, journal_path=args.journal)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack")
    p.add_argument("out", nargs="?", default="her.bundle")
    p.add_argument("--encoder", default="")
    p.set_defaults(fn=cmd_pack)

    d = sub.add_parser("doctor")
    d.add_argument("bundle")
    d.set_defaults(fn=cmd_doctor)

    i = sub.add_parser("info")
    i.add_argument("bundle")
    i.set_defaults(fn=cmd_info)

    u = sub.add_parser("unpack")
    u.add_argument("bundle")
    u.add_argument("--take-over", action="store_true")
    u.set_defaults(fn=cmd_unpack)

    r = sub.add_parser("roll")
    r.add_argument("--out", default="data/bundles")
    r.add_argument("--keep", type=int, default=3)
    r.add_argument("--journal", default="data/journal/l5.wal")
    r.set_defaults(fn=cmd_roll)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())