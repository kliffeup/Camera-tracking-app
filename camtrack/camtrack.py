#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    build_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    ratio = []

    if known_view_1 is None or known_view_2 is None:
        for i in range(0, len(corner_storage), (10 if len(corner_storage) > 50 else 1)):
            frame1 = i
            for j in range(
                    i + (10 if len(corner_storage) > 50 else 1),
                    len(corner_storage),
                    (10 if len(corner_storage) > 50 else 1)
            ):
                frame2 = j

                corrs = build_correspondences(
                    corner_storage[frame1],
                    corner_storage[frame2]
                )

                E, mask_essential = cv2.findEssentialMat(
                    corrs.points_1,
                    corrs.points_2,
                    intrinsic_mat,
                    method=cv2.RANSAC,
                    prob=0.9999,
                    threshold=1,
                    maxIters=5000,
                )

                H, mask_homography = cv2.findHomography(
                    corrs.points_1,
                    corrs.points_2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1,
                )

                ratio.append((i, j, np.count_nonzero(mask_essential, ) / np.count_nonzero(mask_homography, )))

        frame1, frame2, _ = sorted(ratio, key=lambda x: x[2])[-1]

        corrs = build_correspondences(
            corner_storage[frame1],
            corner_storage[frame2],
        )

        E, _ = cv2.findEssentialMat(
            corrs.points_1,
            corrs.points_2,
            intrinsic_mat,
            method=cv2.RANSAC,
            prob=0.9999,
            threshold=1,
            maxIters=5000,
        )

        _, R, t, _ = cv2.recoverPose(
            E,
            corrs.points_1,
            corrs.points_2,
            intrinsic_mat,
        )

        known_view_1 = (frame1, Pose(r_mat=np.eye(3, ), t_vec=np.zeros(3, )))
        known_view_2 = (frame2, Pose(R.T, R.T @ -t))

    np.random.seed(42)
    params = TriangulationParameters(1.0, 0.5, 0.6)
    dist_coefs = np.array([])

    frame_1 = known_view_1[0]
    frame_2 = known_view_2[0]
    processed_frames = [frame_1, frame_2]
    frames_to_extract_pose = [i for i in range(len(corner_storage))
                              if i != frame_1 and i != frame_2]

    view_mats = [None for i in range(len(corner_storage))]
    view_mats[frame_1] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[frame_2] = pose_to_view_mat3x4(known_view_2[1])

    corrs = build_correspondences(corner_storage[frame_1],
                                  corner_storage[frame_2])

    points_3d, ids, _ = triangulate_correspondences(corrs,
                                                    view_mats[frame_1],
                                                    view_mats[frame_2],
                                                    intrinsic_mat,
                                                    params)

    point_cloud_builder = PointCloudBuilder(ids=ids, points=points_3d)

    selected_frame = None
    ftep_copy = frames_to_extract_pose.copy()
    while frames_to_extract_pose and ftep_copy:
        if selected_frame is not None:
            cos_score = []
            for frame in processed_frames[:-1]:
                cur_corrs = build_correspondences(corner_storage[frame],
                                                  corner_storage[processed_frames[-1]])

                points_3d, ids, med_cos = triangulate_correspondences(cur_corrs,
                                                                      view_mats[frame],
                                                                      view_mats[processed_frames[-1]],
                                                                      intrinsic_mat,
                                                                      params)

                point_cloud_builder.add_points(ids, points_3d)
                cos_score.append(med_cos)

            # update points for top 10%
            min_index = np.argsort(cos_score)[:int(np.ceil(len(cos_score) * 0.1))][::-1]

            for index in min_index:
                frame = processed_frames[:-1][index]

                cur_corrs = build_correspondences(corner_storage[frame],
                                                  corner_storage[processed_frames[-1]])

                points_3d, ids, med_cos = triangulate_correspondences(cur_corrs,
                                                                      view_mats[frame],
                                                                      view_mats[processed_frames[-1]],
                                                                      intrinsic_mat,
                                                                      params)

                point_cloud_builder.add_points(ids, points_3d)

        print(f"triangulated points: {point_cloud_builder.points.shape[0]}")
        x_dim = [point[0] for point in point_cloud_builder.points]
        y_dim = [point[1] for point in point_cloud_builder.points]
        z_dim = [point[2] for point in point_cloud_builder.points]
        print(f"cloud point can be placed in parallelepiped with")
        print(f"length: {max(x_dim) - min(x_dim)}")
        print(f"width: {max(y_dim) - min(y_dim)}")
        print(f"height: {max(z_dim) - min(z_dim)}")
        print("")

        succeeded = False
        ftep_copy = frames_to_extract_pose.copy()
        while not succeeded and ftep_copy:
            selected_frame = np.random.choice(ftep_copy)
            selected_points_ids = np.intersect1d(point_cloud_builder.ids,
                                                 corner_storage[selected_frame].ids)

            selected_points_2d = []
            selected_points_3d = []

            for corner_coords, corner_id in zip(corner_storage[selected_frame].points,
                                                corner_storage[selected_frame].ids):
                if corner_id in selected_points_ids:
                    selected_points_2d.append(corner_coords)

            for point_coords, point_id in zip(point_cloud_builder.points,
                                              point_cloud_builder.ids):
                if point_id in selected_points_ids:
                    selected_points_3d.append(point_coords)

            print(f"in process: {selected_frame} frame")
            succeeded, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(selected_points_3d),
                                                                np.array(selected_points_2d),
                                                                intrinsic_mat, dist_coefs,
                                                                iterationsCount=5000,
                                                                reprojectionError=params.max_reprojection_error,
                                                                confidence=0.9999)

            if succeeded:
                print(f"{selected_frame} frame processing succeeded")
                print(f"{inliers.shape[0]} inlier(s) used in {selected_frame} frame processing")
                view_mats[selected_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                frames_to_extract_pose.remove(selected_frame)
                processed_frames.append(selected_frame)
            else:
                print(f"{selected_frame} frame processing failed, selecting another frame...")
                ftep_copy.remove(selected_frame)

            print("")

    count = 1
    last_t = view_mats[0][:, 3]
    next_t = view_mats[0][:, 3]
    for i in range(1, len(view_mats) - 1):
        if view_mats[i] is None:
            for j in range(i + 1, len(view_mats)):
                if view_mats[j] is not None:
                    next_t = view_mats[j][:, 3]
                    count = j - i
                    break
            view_mats[i] = view_mats[i - 1]
            view_mats[i][:, 3] = 1 / (count + 1) * last_t + count / (count + 1) * next_t
        else:
            last_t = view_mats[i][:, 3]
            next_t = view_mats[i][:, 3]
            count = 1


    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        params.max_reprojection_error
    )

    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
