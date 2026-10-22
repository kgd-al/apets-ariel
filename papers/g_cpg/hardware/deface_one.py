import numpy as np
import deface.deface as d

_orig = d.anonymize_frame

def one_face(dets, frame, *args, **kwargs):
    if len(dets) > 1:
        dets = dets[[np.argmax(dets[:, 4])]]   # keep best-scoring face only
    return _orig(dets, frame, *args, **kwargs)

d.anonymize_frame = one_face   # video_detect looks this up at call time
d.main()
