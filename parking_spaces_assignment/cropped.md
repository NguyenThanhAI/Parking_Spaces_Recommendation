# Các việc cần làm:
* Mỗi parking space không cần mask nữa, chỉ cần lưu position từng đỉnh tương ứng với từng cam là đủ
* ParkingSpacesInitializer sẽ có thêm một phương thức tạo ra một mask (là một numpy array duy nhất bằng kích thước ảnh đào vào) biểu diễn vị trí các parking space (mỗi cam một mask là mảng numpy array).
Nền của array này là -1. Mask ứng với parking sace id i, những điểm thuộc parking space id i này sẽ có giá trị là i (các giá trị còn lại là -1)
* VehicleDetector cũng vậy, có thêm một phương thức tạo ra một mask biểu diễn vị trí các vehicle. Nền của array này là -1. Mask ứng với vehicle id j, những điểm thuộc vehicle id j này sẽ có giá trị j (các giá trị còn lại là -1)
* VehicleTracker cũng vậy, có thêm một phương thức tạo ra một mask biểu diễn vị trí các tracked vehicle. Nền của array này là -1. Mask ứng với tracked vehicle id k, những điểm thuộc tracked vehicle id k này có giá trị là k (các điểm còn lại giá trị là -1)
* Nghĩ cách tính intersection giữa parking space và vehicle nhanh nhất có thể