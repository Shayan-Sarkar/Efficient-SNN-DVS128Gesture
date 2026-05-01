from __future__ import annotations
import struct
from pathlib import Path
import numpy as np

_HDR = struct.Struct('<HHIIIIII')


def load_polarity_events(path: str | Path):
    path = Path(path)
    chunks_x, chunks_y, chunks_p, chunks_t = [], [], [], []

    with path.open('rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return _empty()
            if not line.startswith(b'#'):
                f.seek(pos)
                break

        while True:
            hdr_bytes = f.read(_HDR.size)
            if len(hdr_bytes) < _HDR.size:
                break
            (event_type, _event_source, event_size, _ts_offset,
             _ts_overflow, capacity, number, _valid) = _HDR.unpack(hdr_bytes)

            payload = f.read(capacity * event_size)

            if event_type != 1:
                continue
            if number == 0:
                continue

            n_bytes = number * 8
            if len(payload) < n_bytes:
                break

            arr = np.frombuffer(payload[:n_bytes], dtype='<u4').reshape(number, 2)
            data = arr[:, 0]
            ts   = arr[:, 1].astype(np.int64)

            xs = ((data >> 17) & 0x1FFF).astype(np.int16)
            ys = ((data >>  2) & 0x1FFF).astype(np.int16)
            ps = ((data >>  1) & 0x0001).astype(np.int8)

            chunks_x.append(xs)
            chunks_y.append(ys)
            chunks_p.append(ps)
            chunks_t.append(ts)

    if not chunks_t:
        return _empty()

    return {
        'x': np.concatenate(chunks_x),
        'y': np.concatenate(chunks_y),
        'p': np.concatenate(chunks_p),
        't': np.concatenate(chunks_t),
    }


def _empty():
    return {k: np.array([], dtype=dt)
            for k, dt in [('x', np.int16), ('y', np.int16),
                          ('p', np.int8),  ('t', np.int64)]}
