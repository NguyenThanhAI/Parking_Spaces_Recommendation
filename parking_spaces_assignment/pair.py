import os
import pickle
from datetime import datetime, timedelta
from load_videos.videos_utils import get_time_amount_from_frames_number, get_start_time_from_video_name
from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis


class Pair(object):

    def __init__(self, unified_id, vehicle_id, class_id, birth_time, inactive_steps_before_removed=10000):
        self.unified_id = unified_id
        self.vehicle_id = vehicle_id
        self.class_id = class_id
        self.birth_time = birth_time
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.end_time = None
        self.inactive_steps = 0

    def delete(self, time):
        self.end_time = time # (Birth_time, end_time), Khi bị xóa sẽ ghi lại khoảng thời gian tồn tại của pair này

    def __str__(self):
        return str(self.__class__) + ":" + str(self.__dict__)


class PairsScheduler(object):

    def __init__(self, time, inactive_steps_before_removed=10000):
        self.start_time = time
        self.time = time
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.active_pairs = [] # Only pair of unified_id and vehicle_id
        self.inactive_pairs = []
        self.deleted_pairs = []
        self.pairs = [] # List của các pair instance

    def update_time_from_frame_numbers(self, num_frames, frame_stride=1, fps=24):
        num_seconds = get_time_amount_from_frames_number(num_frames=num_frames, frame_stride=frame_stride, fps=fps)
        self.time = self.start_time + timedelta(seconds=num_seconds)

    def update_time_from_system(self):
        self.time = datetime.now()

    def get_list_uid_veh_list(self):
        #active_list = list(map(lambda x: (x.unified_id, x.vehicle_id), self.active_pairs))
        #inactive_list = list(map(lambda x: (x.unified_id, x.vehicle_id), self.inactive_pairs))
        active_list = self.active_pairs
        inactive_list = self.inactive_pairs
        deleted_list = self.deleted_pairs

        return active_list, inactive_list, deleted_list

    @timethis
    def step(self, uid_veh_list, num_frames, frame_stride=1, fps=24):
        self.update_time_from_frame_numbers(num_frames, frame_stride, fps) # Cập nhật self.time
        active_list, inactive_list, deleted_list = self.get_list_uid_veh_list()
        inactive_to_active = [] # Những pair đang ở inactive list và ở bước này xuất hiện trong match pair giữa unified id và vehicle id
        active_this_step = [] # Những pair đang ở trong active list và bước này cũng xuất hiện trong match pair giữa unified id và vehicle id
        #active_to_inactive = [] # Những pair đang ở trong active list và bước này không xuất hiện trong match pair giữa unified id và vehicle id
        brand_new = []
        for uid_veh_id in uid_veh_list: # Xét từng uid_veh_id từ kết quả matcher trả về
            #existed = False
            if uid_veh_id in active_list: # Nếu uid_veh_id nằm trong active_list nghĩa là cặp này đang active và tiếp tục bước này vẫn active
                #existed = True
                active_this_step.append(uid_veh_id)
                continue
            if uid_veh_id in inactive_list: # Nếu uid_veh_id nằm trong inactive_list nghĩa là cặp này bước trước đang inactive bước này sẽ được active
                inactive_to_active.append(uid_veh_id)
                continue
            brand_new.append(uid_veh_id) # Những uid_veh_id còn lại được tạo một cặp hoàn toàn mới

        active_to_inactive = [uid_veh_id for uid_veh_id in active_list if uid_veh_id not in active_this_step] # Những cặp active ở bước trước nhưng bước này không nằm trong list các cặp matcher trả về sẽ trở thành inactive ở bước này
        inactive_list = [uid_veh_id for uid_veh_id in inactive_list if uid_veh_id not in inactive_to_active] # Những inactive ở bước này bằng list inactive  loại bỏ đi những cặp trở thành active ở bước này

        active_list = active_this_step + brand_new + inactive_to_active # Tổng hợp lại, những cặp active ở bước này là hợp của những cặp ở trong active list sẵn và vẫn active ở bước này, những cặp mới hoàn toàn, đang là inactive thành active và đang deleted được rebrand
        inactive_list = inactive_list + active_to_inactive # Những cặp inactive ở bước này là hợp của những cặp inactive vẫn inactive ở bước này và những cặp active ở bước trước thành inactive ở bước này

        for uid_veh_id in inactive_to_active: # Những cặp ở trong inactive bước trước, bước này thành active, số bước inactive được đưa về 0
            pair = list(filter(lambda x: x.unified_id == uid_veh_id[0] and x.vehicle_id == uid_veh_id[1] and x.class_id == uid_veh_id[2], self.pairs))
            assert len(pair) == 1
            pair = pair[0]
            pair.inactive_steps = 0

        # Xét các pair trong inactive_list
        # Xem số inactive_step của từng pair lớn hơn ngưỡng thì thêm end_time, filled_period và chuyển sang delete_list
        inactive_to_deleted = [] # List các cặp đang inactive sẽ trở thành deleted do vượt quá số bước là inactive liên tiếp
        for uid_veh_id in inactive_list: # Duyệt từng phần tử của inactive list
            pair = list(filter(lambda x: x.unified_id == uid_veh_id[0] and x.vehicle_id == uid_veh_id[1] and x.class_id == uid_veh_id[2], self.pairs)) # Gọi pair instance ứng với uid_veh_id
            assert len(pair) == 1
            pair = pair[0]
            pair.inactive_steps += 1 # Tăng inactive step liên tiếp thêm 1
            if pair.inactive_steps > pair.inactive_steps_before_removed: # Nếu số bước liên tiếp là inactive của cặp này lớn hơn ngưỡng cho phép
                num_seconds_inactive = get_time_amount_from_frames_number(num_frames=pair.inactive_steps, # Tính thời gian kể từ khi trở thành cặp inactive
                                                                          frame_stride=frame_stride,
                                                                          fps=fps)
                time = self.time + timedelta(seconds=-num_seconds_inactive) # Thời gian trở thành căp inactive
                pair.delete(time=time) # Tạo period là thời gian khởi tạo pair và thời điểm sang inactive??????
                inactive_to_deleted.append(uid_veh_id) # List các cặp inactive thành deleted thêm phần tử này

        inactive_list = [uid_veh_id for uid_veh_id in inactive_list if uid_veh_id not in inactive_to_deleted] # Cập nhật lại list inactive bằng cách list inactive loại bỏ đi các inactive thành deleted
        deleted_list.extend(inactive_to_deleted) # Cập list các cặp bị deleted bằng cách cộng thêm các cặp ở trong list các cặp từ inactive thành deleted

        self.active_pairs = active_list # Cập nhật lại self.active_pairs
        self.inactive_pairs = inactive_list # Cập nhật lại self.inactive_pairs
        self.deleted_pairs = deleted_list # Cập nhật lại self.deleted_pairs

        for uid_veh_id in brand_new: # Xét từng phần tử trong các cặp hoàn toàn mới
            self.pairs.append(Pair(unified_id=uid_veh_id[0], vehicle_id=uid_veh_id[1], class_id=uid_veh_id[2], birth_time=self.time, inactive_steps_before_removed=self.inactive_steps_before_removed)) # Thêm các pair instance tương ứng với các uid_veh_id tương ứng vào self.pairs (list các pair instance)

    def reset(self, time):
        self.start_time = time
        self.time = time
        self.active_pairs = []  # Only pair of unified_id and vehicle_id
        self.inactive_pairs = []
        self.deleted_pairs = []
        self.pairs = []  # List của các pair instance

    def verify(self):
        # Kiểm tra các active_pairs, inactive_pairs, deleted_pairs phải không có phần tử chung
        # Hợp của ba list trên phải đủ trong self.pairs
        assert len(list(set(self.active_pairs))) == len(list(self.active_pairs))
        assert len(list(set(self.inactive_pairs))) == len(list(self.inactive_pairs))
        assert len(list(set(self.deleted_pairs))) == len(list(self.deleted_pairs))

        inter = set(self.active_pairs).intersection(self.inactive_pairs)
        inter = list(inter.intersection(self.deleted_pairs))
        assert len(inter) == 0

        union = self.active_pairs + self.inactive_pairs + self.deleted_pairs
        assert len(union) == len(self.pairs)
        for uid_veh_id in union:
            pair = list(filter(lambda x: x.unified_id == uid_veh_id[0] and x.vehicle_id == uid_veh_id[1] and x.class_id == uid_veh_id[2], self.pairs))
            assert len(pair) == 1

    def get_pairs_instances(self):
        return self.pairs

    def save_pairs_to_db(self, save_path=None, save_dir="../database"):
        if not save_path:
            save_path = self.start_time.strftime("%Y-%m-%d") + ".pkl"
        save_path = os.path.join(save_dir, save_path)
        with open(save_path, "wb") as f:
            pickle.dump(self.pairs, f)