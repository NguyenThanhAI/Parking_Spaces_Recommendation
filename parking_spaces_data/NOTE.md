+ ```load_parking_space_annotation_visually_and_assign_space_id.py``` to display image from json file ```parking_spaces_annotation.json``` created on CVAT
+ ```order_in_json_to_unified_id.py``` create dictionary handly convert from id of parking spaces in json file created on CVAT to unified id
+ ```unified_id_to_polygon_in_cams.py``` create json file ```parking_spaces_unified_id_segmen_in_cameras.json``` with format:
```
{
    "parking_ground_SA":
        {
            "unified_id": {"positions": {
                            "cam_1": [[x1, y1, ...., xn, yn]]
                            "cam_2": ...
                            ...
                            },
                            "reversed_considered_orients": {"cam_1": ["west", ...],
                                                            "cam_2": ["north_west", ...],
                                                            .....
                                                            },
                            "adjacencies": {
                                "eastern_adjacency": 40,
                                "western_adjacency": null,
                                "southern_adjacency": null,
                                "northern_adjacency": 70,
                                "south_east_adjacency": null,
                                "south_west_adjacency": null,
                                "north_west_adjacency": null,
                                "north_east_adjacency": 69
                            }
                        }
            ...
        }
    ...
}
```
+ ```unified_id_to_polygons_in_cams_json_file_test.py``` test ```parking_spaces_unified_id_segmen_in_cameras.json``` and display segmentation of parking spaces in a white image