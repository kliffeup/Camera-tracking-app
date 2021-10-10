#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def generate_index(init_value):
    index = init_value
    while True:
        yield index

        index += 1


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    shi_tomasi_params = dict(qualityLevel=0.005,
                             minDistance=7,
                             blockSize=14)

    criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    lukas_kanade_params = dict(winSize =(10, 10),
                               maxLevel=2,
                               criteria=(criteria, 10, 0.03),
                               flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                               minEigThreshold=0.0005)

    max_corner_count = 2000
    radius = 7
    if frame_sequence[0].shape[0] < 1000:
        shi_tomasi_params["blockSize"] = 5
        lukas_kanade_params = dict(winSize=(5, 5),
                                   maxLevel=2,
                                   criteria=(criteria, 10, 0.03))

    corners_coords = cv2.goodFeaturesToTrack(image=frame_sequence[0],
                                             maxCorners=max_corner_count,
                                             **shi_tomasi_params)

    corners_coords = np.float32(corners_coords.reshape(-1, 2))

    corners = {i: corner_coord for i, corner_coord in enumerate(corners_coords)}

    index_gen = generate_index(len(corners))
    prev_frame = frame_sequence[0]

    builder.set_corners_at_frame(0, FrameCorners(
        np.fromiter(corners.keys(), dtype=np.uint32),
        corners_coords,
        np.array([shi_tomasi_params["blockSize"]] * len(corners_coords))))

    for i, cur_frame in enumerate(frame_sequence[1:], 1):
        tracked_corners_coords, status, error = cv2.calcOpticalFlowPyrLK(
            np.uint8(prev_frame * 255),
            np.uint8(cur_frame * 255),
            corners_coords, None, **lukas_kanade_params)

        tracked_corners_coords = tracked_corners_coords.reshape(-1, 2)
        tracked_corners_coords = np.float32(tracked_corners_coords)

        status = status.ravel()
        corners_ids_to_remove = list()

        for index in corners:
            cur_index = np.where((corners_coords == corners[index]).all(axis=1))
            cur_index = cur_index[0][0]
            if status[cur_index]:
                corners[index] = tracked_corners_coords[cur_index]
            else:
                corners_ids_to_remove.append(index)

        for index in corners_ids_to_remove:
            del corners[index]

        if max_corner_count - len(corners) > 50:
            mask = np.ones_like(cur_frame, dtype=np.uint8) * 255
            for corner_coords in corners.values():
                cv2.circle(mask, np.int0(corner_coords), radius, 0, -1)

            temp_corners_coords = cv2.goodFeaturesToTrack(
                image=cur_frame,
                maxCorners=max_corner_count - len(corners),
                mask=mask,
                **shi_tomasi_params)

            if temp_corners_coords is not None:
                temp_corners_coords = np.float32(temp_corners_coords.reshape(-1, 2))

                for corner_coords in temp_corners_coords:
                    corners[next(index_gen)] = corner_coords

        corners_coords = np.array([point for point in corners.values()])

        builder.set_corners_at_frame(i, FrameCorners(
            np.fromiter(corners.keys(), dtype=np.uint32),
            corners_coords,
            np.array([shi_tomasi_params["blockSize"]] * len(corners_coords))))

        prev_frame = cur_frame


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
