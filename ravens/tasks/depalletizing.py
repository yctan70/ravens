# coding=utf-8
# Copyright 2023 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Palletizing Task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p


class Depalletizing(Task):
    """Depalletizing Task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = 30
        self.pattern = 0

    def alter(self, box_size, object_ids, object_points, env, zone_pose, stack_size):
        box_template = 'box/box-template.urdf'
        stack_dim = np.int32([3, 2, 2])
        offset = ((0, 0), (0, 0.5*box_size[2]))
        margin = 0.01

        for z in range(stack_dim[2]):
            # Transpose every layer.
            box_size[0], box_size[1] = box_size[1], box_size[0]
            for y in range(stack_dim[1]):
                for x in range(stack_dim[0]):
                    position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
                    position[0] += x * margin - stack_size[0] / 2 + offset[z][0]
                    position[1] += y * margin - stack_size[1] / 2 + offset[z][1]
                    position[2] += z * margin + 0.03
                    pose = (position, (0, 0, 0, 1))
                    pose = utils.multiply(zone_pose, pose)
                    urdf = self.fill_template(box_template, {'DIM': box_size})
                    box_id = env.add_object(urdf, pose)
                    os.remove(urdf)
                    object_ids.append((box_id, (1, None)))
                    self.color_random_brown(box_id)
                    object_points[box_id] = self.get_object_points(box_id)
            stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]-1

    def reset(self, env):
        super().reset(env)

        # Add pallet.
        zone_size = (0.3, 0.25, 0.25)
        zone_urdf = 'pallet/pallet.urdf'
        theta = np.random.rand() * np.pi / 4
        rotation = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        zone_pose = ((0.5, 0.25, 0.02), rotation)
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Add stack of boxes on pallet.
        margin = 0.01
        object_ids = []
        object_points = {}
        stack_size = (0.19, 0.19, 0.19)
        box_template = 'box/box-template.urdf'
        stack_dim = np.int32([2, 3, 3])
        # stack_dim = np.random.randint(low=2, high=4, size=3)
        box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim

        if self.pattern == 1:
            self.alter(box_size, object_ids, object_points, env, zone_pose, stack_size)
        else:
            for z in range(stack_dim[2]):
                # Transpose every layer.
                stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
                box_size[0], box_size[1] = box_size[1], box_size[0]
                for y in range(stack_dim[1]):
                    for x in range(stack_dim[0]):
                        position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
                        position[0] += x * margin - stack_size[0] / 2
                        position[1] += y * margin - stack_size[1] / 2
                        position[2] += z * margin + 0.03
                        pose = (position, (0, 0, 0, 1))
                        pose = utils.multiply(zone_pose, pose)
                        urdf = self.fill_template(box_template, {'DIM': box_size})
                        box_id = env.add_object(urdf, pose)
                        os.remove(urdf)
                        object_ids.append((box_id, (1, None)))
                        self.color_random_brown(box_id)
                        object_points[box_id] = self.get_object_points(box_id)

        # Randomly select top box on pallet and save ground truth pose.
        targets = []
        self.steps = []
        boxes = [i[0] for i in object_ids]
        while boxes:
            _, height, object_mask = self.get_true_image(env)
            top = np.argwhere(height > (np.max(height) - 0.03))
            rpixel = top[int(np.floor(np.random.random() * len(top)))]  # y, x
            box_id = int(object_mask[rpixel[0], rpixel[1]])
            if box_id in boxes:
                # remove the box to see the next level
                position, rotation = p.getBasePositionAndOrientation(box_id)
                rposition = np.float32(position) + np.float32([0, -10, 0])
                p.resetBasePositionAndOrientation(box_id, rposition, rotation)
                self.steps.append(box_id)
                boxes.remove(box_id)
        # self.steps.reverse()  # Time-reversed depalletizing.
        n = 0
        for box_id in self.steps:
            # move the box back for the initial condition
            position, rotation = p.getBasePositionAndOrientation(box_id)
            rposition = np.float32(position) + np.float32([0, 10, 0])
            p.resetBasePositionAndOrientation(box_id, rposition, rotation)
            position = [0.5, -0.25, 0.05]
            if (np.floor(n / 6) % 2) == 0:
                rotation = utils.eulerXYZ_to_quatXYZW((0, 0, 0))
            else:
                rotation = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
            targets.append((position, rotation))
            n += 1

        box_size[2], position[2] = 0.01, 0.01
        pose = (position, rotation)
        urdf = self.fill_template(box_template, {'DIM': box_size})
        box_id = env.add_object(urdf, pose)
        os.remove(urdf)

        self.color_random_brown(box_id)
        if self.mode == 'test2':
            matches = np.eye(len(object_ids))
            self.steps = []
        else:
            matches = np.zeros((len(object_ids), len(object_ids)))
        self.goals.append((
            object_ids, matches, targets, True, True,
            'pose', (object_points, [(zone_pose, zone_size)]), 1))

        self.spawn_box()

    def reward(self):
        reward, info = super().reward()
        self.progress = self._rewards
        if np.abs(1 - self._rewards) < 0.01:
            self.progress = 1  # Update task progress.
            self.goals.pop(0)
        self.spawn_box()
        return reward, info

    def spawn_box(self):
        """Palletizing: select the box to pick."""
        if self.goals:
            objs, matches, targs = self.goals[0][0], self.goals[0][1], self.goals[0][2]
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                for j in targets_i:
                    target_pose = targs[j]
                    pose = p.getBasePositionAndOrientation(object_id)
                    if self.is_match(pose, target_pose, symmetry):
                        # remove the box from the target position
                        box_id = objs[j][0]
                        position, rotation = p.getBasePositionAndOrientation(box_id)
                        rposition = np.float32(position) + np.float32([0, -10, 1]) + np.random.rand() * np.float32(
                            [1, 1, 0])
                        p.resetBasePositionAndOrientation(box_id, rposition, (0, 0, 0, 1))
                        matches[i, :] = 0
            if len(self.steps) > 0:
                obj = self.steps[0]
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    # matches[i, :] = np.zeros((1, len(objs)))
                    if object_id == obj:
                        matches[i, i] = 1
                self.steps.pop(0)

        # Wait until spawned box settles.
        for _ in range(480):
            p.stepSimulation()
