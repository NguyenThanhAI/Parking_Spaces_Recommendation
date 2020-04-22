# Các việc cần làm:
* Mỗi parking space không cần mask nữa, chỉ cần lưu position từng đỉnh tương ứng với từng cam là đủ
* ParkingSpacesInitializer sẽ có thêm một phương thức tạo ra một mask (là một numpy array duy nhất bằng kích thước ảnh đào vào) biểu diễn vị trí các parking space (mỗi cam một mask là mảng numpy array).
Nền của array này là -1. Mask ứng với parking sace id i, những điểm thuộc parking space id i này sẽ có giá trị là i (các giá trị còn lại là -1)
* VehicleDetector cũng vậy, có thêm một phương thức tạo ra một mask biểu diễn vị trí các vehicle. Nền của array này là -1. Mask ứng với vehicle id j, những điểm thuộc vehicle id j này sẽ có giá trị j (các giá trị còn lại là -1)
* VehicleTracker cũng vậy, có thêm một phương thức tạo ra một mask biểu diễn vị trí các tracked vehicle. Nền của array này là -1. Mask ứng với tracked vehicle id k, những điểm thuộc tracked vehicle id k này có giá trị là k (các điểm còn lại giá trị là -1)
* Nghĩ cách tính intersection giữa parking space và vehicle nhanh nhất có thể

# Thuật toán tính intersection mới và match giữa parking space và vehicle:

* Phát hiện vehicle detection dưới dạng list các instance vehicle_detection
* Nếu có sử dụng tracking:
    * Sử dụng track lấy ra các vehicle track là list các instance của vehicle_track
* Lọc ra các parking spaces có trong cam hiện tại dưới dạng list các instance parking_space
* Tạo ra dictionary map từ unified_id sang instance của parking_space
* Tạo ra dictionary map từ vehicle_id (detection_id nếu không track, track_id nếu sử dụng track),
sang instance của vehicle_detection (nếu không sử dụng track) hoặc vehicle_track (nếu sử dụng track)
* So sánh kích thước của 2 dictionary tạo ở 2 bước trên, dictionary nào có kích thước nhỏ hơn,
ta sẽ sử dụng vòng lặp for trên dictionary đó
* Tạo hai dictionary unified_id_to_vehicle_id và vehicle_id_to_unified_id chứa thông tin ios (intersection over space) {unified_id1: {vehicle_id1: ..., vehicle_id2: ..., ...}, unified_id2: ...}, {vehicle_id1: {unified_id1:..., unified_id2:...,...}, vehicle_id2:...}
* Giả sử chọn vòng lặp for theo các key của dictionary của unified sang instance của parking_space:
    * Lấy crop của parkingspaceinitialzer.postions_mask[cam][unified_id] từ bbox của unified_id tương ứng
    * Lấy phần crop ở trên áp lên positions_mask của vehicle_detector (nếu không sử dụng track) hoặc vehicle_tracker (nếu sử dụng track)
    * Tìm số lượng giao của unified_id với các vehicle_id trong vùng crop trên và lưu vào 2 dictionary unified_id_to_vehicle_id và vehicle_id_to_unified_id nếu ios thỏa mãn > threshold đặt trước, nếu không thì bỏ qua
* Tạo một unified_id_status_dict = {unified_id: "unknown", ....} tất cả các unified_id có trạng thái ban đầu là unknown
* Đặt một considered_vehicle_id_list = [] chứ các vehicle_id đã được xét với các unified_id
* Duyệt từng unified_id trên unified_id_status_dict:
    * Nếu unified_id không là tồn tại là key trong unified_id_to_vehicle_id thì chuyển trạng thái của unified_id trong
    unified_id_status_dict là "available"
    * unified_id không là tồn tại là key trong unified_id_to_vehicle_id:
        * Xét từng vehicle_id có ios giao trong unified_id_to_vehicle_id[unified_id]:
            * Nếu vehicle_id này không nằm trong considered_vehicle_id_list = [] thì xét tiếp, ngược lại đã ở trong rồi thì bỏ qua chuyển sang vehicle_id tiếp theo:
                * Xét vehicle_id_to_unified_id[vehicle_id] của vehicle_id đang xét này chỉ có đúng một unified_id đang xét:
                    * assert unified_id in vehicle_id_to_unified_id[vehicle_id] (xác nhận lại)
                    * vehicle_id này và unified_id đang xét được match với nhau
                    * unified_id_status_dict[unified_id] = "filled"
                * Nếu vehicle_id_to_unified_id[vehicle_id] của vehicle_id đang xét nhiều hơn một unified_id:
                    * assert unified_id in vehicle_id_to_unified_id[vehicle_id] (xác nhận lại)
                    * Xét từng unified_id này, tạo một pspace_dict lưu trữ các thông tin:
                    south_level, east_level, visited, adjacencies, ios, reversed_considered_orients (tất nhiên phải tương ứng với cam)
                    * Sử dụng đệ quy tương hỗ để tính south_level, east_level của từng unified_id
                    * Tạo reversed_considered_orients = {"orients": [unified_id1, unified_id2, ...], ....}
                    * Xét các hướng nếu trong reversed_considered_orients có: 
                        * Tây Nam, Tây Bắc, Tây thì ưu tiên chọn điểm đỗ ở cực Đông
                        * Đông Nam, Đông Bắc, Đông thì ưu tiên chọn điểm đỗ ở cực Tây
                        * Tây Bắc, Bắc, Đông Bắc thì ưu tiên chọn điểm đỗ ở cực Nam
                        * Tây Nam, Nam, Đông Nam thì ưu tiên chọn điểm đỗ ở cực Bắc
                    * Gộp các trường hợp trên trên lại:
                        * Nếu hợp của hai trường hợp trên là rỗng:
                            * Nếu 1 trong hai trường hợp không rỗng
                                * Chọn unified_id ứng với ios lớn nhất là "filled"
                                * Các unified_id còn lại cái nào ios lớn hơn 0.75 đặt là "unknown" ngược lại là "available"
                            * Nếu cả hai trường hợp trên đều rỗng:
                                Cứ unified_id nào có ios trên 0.65 thì trạng thái là "filled"
                        * Nếu hợp của hai trường hợp trên không rỗng:
                            * Xác nhận chỉ có một điểm đỗ (bug ở chỗ này)
                            * Điểm đỗ này được chọn là "filled"
                            * Khởi tạo filled_list = []. filled_list thêm điểm đỗ vừa rồi
                            * Nếu là max_south (ưu tiên south_level cao nhất):
                                * Nếu điểm đỗ trên có lân cận phía Bắc và ios > 0.6 và loại xe là xe tải thì điểm đỗ lân cận này cũng được điền là "filled"
                            * Nếu là min_south (ưu tiên south_level thấp nhất):
                                * Nếu điểm đỗ trên có lần cận phía Nam và ios > 0.6 và loại xe là xe tải thì điểm đỗ lân cận này cũng được điền là "filled"
                            * filled_list thêm điểm trên vào
                            * Xét các điểm đỗ còn lại (not in filled_list):
                                * Nếu ios > 0.7 thì điểm đỗ này được điền là "unknown"
                                * Nếu không thì điểm đỗ này là "available"
                * considered_vehicle_id_list thêm vehicle_id

* Visualize ảnh sử dụng các mask                      
