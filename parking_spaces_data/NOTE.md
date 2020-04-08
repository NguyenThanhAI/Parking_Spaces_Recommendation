+ ```load_parking_space_annotation_visually_and_assign_space_id.py``` to display image from json file ```parking_spaces_annotation.json``` created on CVAT
+ ```order_in_json_to_unified_id.py``` create dictionary handly convert from id of parking spaces in json file created on CVAT to unified id
+ ```unified_id_to_polygon_in_cams.py``` create json file ```parking_spaces_unified_id_segmen_in_cameras.json``` with format:
```
{
    "unified_id": {
                    "cam_1": [[x1, y1, ...., xn, yn]]
                    "cam_2": ...
                    ...
                }
    ...
}
```
+ ```unified_id_to_polygons_in_cams_json_file_test.py``` test ```parking_spaces_unified_id_segmen_in_cameras.json``` and display segmentation of parking spaces in a white image