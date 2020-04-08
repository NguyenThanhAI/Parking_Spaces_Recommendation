#Image id 4 - cam 1

order_in_json_to_unified_id = dict(zip(range(1, 33), range(39, 71)))

order_in_json_to_unified_id.update(dict(zip(range(33, 57), range(71, 95))))

order_in_json_to_unified_id.update(dict(zip(range(57, 81), range(95, 119))))

order_in_json_to_unified_id.update(dict(zip(range(81, 91), range(119, 129))))

# Image id 5 - cam 2

order_in_json_to_unified_id.update(dict(zip(range(91, 107), range(0, 16))))

order_in_json_to_unified_id.update(dict(zip(range(107, 130), range(16, 39))))

order_in_json_to_unified_id.update(dict(zip(range(130, 160), range(40, 70))))

order_in_json_to_unified_id.update({160: 39, 161: 70})

# Image id 6 - cam 3

order_in_json_to_unified_id.update(dict(zip(range(162, 170), range(126, 118, -1))))

order_in_json_to_unified_id.update({170: 82, 171: 83, 172: 81, 173: 84, 174: 80, 175: 85,
                                    176: 86, 177: 79, 178: 87, 179: 78, 180: 88, 181: 77,
                                    182: 89, 183: 76, 184: 90, 185: 75, 186: 91, 187: 74,
                                    188: 92, 189: 93, 190: 73, 191: 72, 192: 94, 193: 71})

order_in_json_to_unified_id.update({194: 106, 195: 105, 196: 108, 197: 104, 198: 109, 199: 103,
                                    200: 110, 201: 102, 202: 111, 203: 101, 204: 112, 205: 100,
                                    206: 113, 207: 99, 208: 114, 209: 98, 210: 115, 211: 97,
                                    212: 116, 213: 96, 214: 117, 215: 95, 216: 118})

cam_to_space_id = {"cam_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
                   "cam_2": [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161],
                   "cam_3": [162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216]}

#### Format {space_id: {"cam_1": polygon_in_cam_1, "cam_2": polygon_in_cam_2, "cam_3": polygon_in_cam_3},...}

unified_id_and_adjacency_ids = {0: {"adjacencies": {"eastern_adjacency": "",
                                                    "western_adjacency": 1,
                                                    "southern_adjacency": "",
                                                    "northern_adjacency": "",
                                                    "south_east_adjacency": "",
                                                    "south_west_adjacency": "",
                                                    "north_west_adjacency": "",
                                                    "north_east_adjacency": ""}},
                                1: {"adjacencies": {"eastern_adjacency": 0,
                                                    "western_adjacency": 2,
                                                    "southern_adjacency": "",
                                                    "northern_adjacency": "",
                                                    "south_east_adjacency": "",
                                                    "south_west_adjacency": "",
                                                    "north_west_adjacency": "",
                                                    "north_east_adjacency": ""}},
                                2: {"adjacencies": {"eastern_adjacency": 1,
                                                    "western_adjacency": 3,
                                                    "southern_adjacency": "",
                                                    "northern_adjacency": "",
                                                    "south_east_adjacency": "",
                                                    "south_west_adjacency": "",
                                                    "north_west_adjacency": "",
                                                    "north_east_adjacency": ""}},
                                3: {"adjacencies": {"eastern_adjacency": 2,
                                                    "western_adjacency": 4,
                                                    "southern_adjacency": "",
                                                    "northern_adjacency": "",
                                                    "south_east_adjacency": "",
                                                    "south_west_adjacency": "",
                                                    "north_west_adjacency": "",
                                                    "north_east_adjacency": ""}},
                                4: {"adjacencies": {"eastern_adjacency": 3,
                                                    "western_adjacency": 5,
                                                    "southern_adjacency": "",
                                                    "northern_adjacency": "",
                                                    "south_east_adjacency": "",
                                                    "south_west_adjacency": "",
                                                    "north_west_adjacency": "",
                                                    "north_east_adjacency": ""}},
                                5: {"adjacencies": {"eastern_adjacency": 4,
                                                    "western_adjacency": 6,
                                                    "southern_adjacency": "",
                                                    "northern_adjacency": "",
                                                    "south_east_adjacency": "",
                                                    "south_west_adjacency": "",
                                                    "north_west_adjacency": "",
                                                    "north_east_adjacency": ""}}}



