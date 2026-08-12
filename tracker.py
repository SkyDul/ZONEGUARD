"""
tracker.py — Lightweight IoU-based multi-object tracker (ByteTrack-lite)
Replaces the need for Ultralytics .track() which only works with .pt models.
"""

import numpy as np
from collections import OrderedDict


def iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


class Track:
    def __init__(self, track_id, bbox, class_id, class_name, confidence):
        self.track_id   = track_id
        self.bbox       = bbox            # [x1, y1, x2, y2] pixel
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.age        = 0              # frames since last match
        self.hits       = 1

    def update(self, bbox, class_id, class_name, confidence):
        self.bbox       = bbox
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.age        = 0
        self.hits      += 1

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def foot(self):
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class ByteTracker:
    """
    Simple IoU-based tracker inspired by ByteTrack / SORT.
    - Matches detections to existing tracks via greedy IoU.
    - Unmatched tracks are kept for `max_age` frames before deletion.
    """

    def __init__(self, iou_threshold=0.35, max_age=10, min_hits=1):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self.min_hits      = min_hits
        self._next_id      = 1
        self.tracks: list[Track] = []

    def reset(self):
        self.tracks    = []
        self._next_id  = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of dicts with keys x1,y1,x2,y2,class_id,class_name,confidence
        Returns same list with 'track_id' added to each detection.
        """
        # Age all existing tracks
        for t in self.tracks:
            t.age += 1

        if not detections:
            # Remove stale tracks
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return []

        det_boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections])

        # ── Match detections → tracks ──────────────────────────────
        matched    = {}   # det_idx → track
        unmatched  = list(range(len(detections)))

        if self.tracks:
            trk_boxes = np.array([t.bbox for t in self.tracks])

            # Build IoU matrix (n_det × n_trk)
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for di, db in enumerate(det_boxes):
                for ti, tb in enumerate(trk_boxes):
                    iou_matrix[di, ti] = iou(db, tb)

            # Greedy matching: highest IoU first
            while True:
                if iou_matrix.size == 0:
                    break
                best = np.argmax(iou_matrix)
                di, ti = divmod(best, iou_matrix.shape[1])
                if iou_matrix[di, ti] < self.iou_threshold:
                    break
                if di not in matched and ti not in matched.values():
                    # Extra: prefer same class
                    if detections[di]["class_id"] == self.tracks[ti].class_id:
                        matched[di] = ti
                    elif iou_matrix[di, ti] >= 0.5:   # high overlap → accept anyway
                        matched[di] = ti
                iou_matrix[di, :] = -1
                iou_matrix[:, ti] = -1

            unmatched = [di for di in range(len(detections)) if di not in matched]

        # ── Update matched tracks ──────────────────────────────────
        for di, ti in matched.items():
            d = detections[di]
            self.tracks[ti].update(
                [d["x1"], d["y1"], d["x2"], d["y2"]],
                d["class_id"], d["class_name"], d["confidence"]
            )
            detections[di]["track_id"] = self.tracks[ti].track_id

        # ── Create new tracks for unmatched ───────────────────────
        for di in unmatched:
            d = detections[di]
            t = Track(
                self._next_id,
                [d["x1"], d["y1"], d["x2"], d["y2"]],
                d["class_id"], d["class_name"], d["confidence"]
            )
            self._next_id += 1
            self.tracks.append(t)
            detections[di]["track_id"] = t.track_id

        # ── Remove stale tracks ────────────────────────────────────
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return detections
