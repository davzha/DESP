from pathlib import Path
import json
import os

import h5py
import numpy as np
import torch


CLASSES = {
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
}


class CLEVRSet(torch.utils.data.Dataset):
    """From https://github.com/Cyanogenoid/dspn with slight modifications.
    """
    def __init__(self, mode, base_path=Path.home() / "data/clevr", n_points=10, n_objects=None, mem_feat=False, box=False, full=False):
        assert mode in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.base_path = base_path
        self.mode = mode
        self.n_objects = n_objects
        self.n_points = n_points
        self.mem_feat = mem_feat
        self.box = box  # True if clevr-box version, False if clevr-state version
        self.full = full  # Use full validation set?

        with self.img_db() as db:
            ids = db["image_ids"]
            self.image_id_to_index = {id: i for i, id in enumerate(ids)}
        self.image_db = None

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.scenes = self.prepare_scenes(scenes)

    def object_to_fv(self, obj):
        coords = [p / 3 for p in obj["3d_coords"]]
        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        return coords + material + color + shape + size

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        for scene in scenes_json:
            img_idx = scene["image_index"]
            # different objects depending on bbox version or attribute version of CLEVR sets
            if self.box:
                objects = self.extract_bounding_boxes(scene)
                objects = torch.tensor(objects)
            else:
                objects = [self.object_to_fv(obj) for obj in scene["objects"]]
                objects = torch.tensor(objects).transpose(0, 1)
            num_objects = objects.size(1)

            # skip if too many objects
            if self.n_objects is not None and num_objects not in self.n_objects:
                continue

            # pad with 0s
            if num_objects < self.n_points:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(objects.size(0), self.n_points - num_objects),
                    ],
                    dim=1,
                )
            # fill in masks
            mask = torch.zeros(self.n_points)
            mask[:num_objects] = 1

            img_ids.append(img_idx)
            scenes.append((objects, mask))
        return img_ids, scenes

    def extract_bounding_boxes(self, scene):
        """
        Code used for 'Object-based Reasoning in VQA' to generate bboxes
        https://arxiv.org/abs/1801.09718
        https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py#L51-L107
        """
        objs = scene["objects"]
        rotation = scene["directions"]["right"]

        num_boxes = len(objs)

        boxes = np.zeros((1, num_boxes, 4))

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []

        for i, obj in enumerate(objs):
            [x, y, z] = obj["pixel_coords"]

            [x1, y1, z1] = obj["3d_coords"]

            cos_theta, sin_theta, _ = rotation

            x1 = x1 * cos_theta + y1 * sin_theta
            y1 = x1 * -sin_theta + y1 * cos_theta

            height_d = 6.9 * z1 * (15 - y1) / 2.0
            height_u = height_d
            width_l = height_d
            width_r = height_d

            if obj["shape"] == "cylinder":
                d = 9.4 + y1
                h = 6.4
                s = z1

                height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
                height_d = height_u * (h - s + d) / (h + s + d)

                width_l *= 11 / (10 + y1)
                width_r = width_l

            if obj["shape"] == "cube":
                height_u *= 1.3 * 10 / (10 + y1)
                height_d = height_u
                width_l = height_u
                width_r = height_u

            obj_name = (
                obj["size"]
                + " "
                + obj["color"]
                + " "
                + obj["material"]
                + " "
                + obj["shape"]
            )
            ymin.append((y - height_d) / 320.0)
            ymax.append((y + height_u) / 320.0)
            xmin.append((x - width_l) / 480.0)
            xmax.append((x + width_r) / 480.0)

        return xmin, ymin, xmax, ymax

    @property
    def images_folder(self):
        return os.path.join(self.base_path, "images", self.mode)

    @property
    def scenes_path(self):
        if self.mode == "test":
            raise ValueError("Scenes are not available for test")
        return os.path.join(
            self.base_path, "scenes", "CLEVR_{}_scenes.json".format(self.mode)
        )

    def img_db(self):
        path = os.path.join(self.base_path, "{}-images.h5".format(self.mode))
        return h5py.File(path, "r")

    def load_image(self, image_id):
        if self.image_db is None:
            self.image_db = self.img_db()
        index = self.image_id_to_index[image_id]
        image = self.image_db["images"][index]
        return image

    def __getitem__(self, item):
        image_id = self.img_ids[item]
        image = self.load_image(image_id)
        objects, size = self.scenes[item]
        objects = objects.T

        if self.mem_feat:
            objects = torch.cat([objects, size.unsqueeze(1)], dim=1)

        return image, objects

    def __len__(self):
        if self.mode == "train" or self.full:
            return len(self.scenes)
        else:
            return len(self.scenes) // 10
