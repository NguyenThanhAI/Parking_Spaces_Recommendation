import json
import numpy as np
from skimage.draw import polygon
from parking_spaces_data.order_in_json_to_unified_id import cam_to_unified_id


class ParkingSpace(object):
    def __init__(self, unified_id, positions, reversed_considered_orients, adjacencies, active_cams: list, parking_ground="parking_ground_SA", shape=(720, 1080)):
        self.unified_id = unified_id
        self.positions = positions
        self.reversed_considered_orients = reversed_considered_orients
        self.adjacencies = adjacencies
        self.parking_ground = parking_ground
        self.positions_mask = {}

        cam_list = list(self.positions.keys())
        for cam in cam_list:
            if cam not in active_cams:
                self.positions.pop(cam)

        cam_list = list(self.reversed_considered_orients.keys())
        for cam in cam_list:
            if cam not in active_cams:
                self.reversed_considered_orients.pop(cam)

        for cam in self.positions:
            position_mask = np.zeros(shape=shape, dtype=np.bool)
            cc, rr = np.array(self.positions[cam], dtype=np.uint16).reshape(-1, 2).T
            rr, cc = polygon(rr, cc)
            position_mask[rr, cc] = True
            self.positions_mask[cam] = position_mask


class ParkingSpacesInitializer(object):
    def __init__(self, active_cams: list, parking_ground="parking_ground_SA", shape=(720, 1280), config_json_path="../parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json"):
        self.active_cams = active_cams
        self.parking_ground = parking_ground
        self.unified_id_list = []
        for cam in active_cams:
            self.unified_id_list.extend(cam_to_unified_id[cam])
        self.unified_id_list = list(set(self.unified_id_list))
        self.shape = shape

        with open(config_json_path, "r") as f:
            self.config_json = json.load(f)

    def initialize_parking_spaces(self):
        parking_spaces_list = []
        for unified_id in self.unified_id_list:
            positions = self.config_json[str(unified_id)]["positions"] # Sau khi sửa json sẽ phải thêm ["parking_ground_SA"] vào sau config_json
            reversed_considered_orients = self.config_json[str(unified_id)]["reversed_considered_orients"]
            adjacencies = self.config_json[str(unified_id)]["adjacencies"]
            parking_spaces_list.append(ParkingSpace(unified_id=unified_id,
                                                    positions=positions,
                                                    reversed_considered_orients=reversed_considered_orients,
                                                    adjacencies=adjacencies,
                                                    active_cams=self.active_cams,
                                                    shape=self.shape))
        return sorted(parking_spaces_list, key=lambda x: x.unified_id)

#initializer = ParkingSpacesInitializer(active_cams=["cam_1", "cam_2"])
#parking_spaces_list  = initializer.initialize_parking_spaces()
