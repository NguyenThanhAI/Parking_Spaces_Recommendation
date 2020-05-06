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
        self.active_pairs = {} # Only pair of unified_id and vehicle_id
        self.inactive_pairs = {}
        self.deleted_pairs = {}

    def update_time_from_frame_numbers(self, num_frames, frame_stride=1, fps=24):
        num_seconds = get_time_amount_from_frames_number(num_frames=num_frames, frame_stride=frame_stride, fps=fps)
        self.time = self.start_time + timedelta(seconds=num_seconds)

    def update_time_from_system(self):
        self.time = datetime.now()

    def get_list_uid_veh_dict(self):
        #active_list = list(map(lambda x: (x.unified_id, x.vehicle_id), self.active_pairs))
        #inactive_list = list(map(lambda x: (x.unified_id, x.vehicle_id), self.inactive_pairs))
        active_dict = self.active_pairs
        inactive_dict = self.inactive_pairs
        deleted_dict = self.deleted_pairs

        return active_dict, inactive_dict, deleted_dict

    @timethis
    def step(self, uid_veh_list, num_frames, frame_stride=1, fps=24):
        self.update_time_from_frame_numbers(num_frames, frame_stride, fps) # Cập nhật self.time
        active_dict, inactive_dict, deleted_dict = self.get_list_uid_veh_dict()
        inactive_to_active = {} # Những pair đang ở inactive list và ở bước này xuất hiện trong match pair giữa unified id và vehicle id
        active_this_step = {} # Những pair đang ở trong active list và bước này cũng xuất hiện trong match pair giữa unified id và vehicle id
        #active_to_inactive = [] # Những pair đang ở trong active list và bước này không xuất hiện trong match pair giữa unified id và vehicle id
        brand_new = {}
        for uid_veh_id in uid_veh_list: # Xét từng uid_veh_id từ kết quả matcher trả về
            #existed = False
            if uid_veh_id in active_dict: # Nếu uid_veh_id nằm trong active_dict nghĩa là cặp này đang active và tiếp tục bước này vẫn active
                #existed = True
                active_this_step[uid_veh_id] = active_dict[uid_veh_id]
                continue
            if uid_veh_id in inactive_dict: # Nếu uid_veh_id nằm trong inactive_dict nghĩa là cặp này bước trước đang inactive bước này sẽ được active
                inactive_to_active[uid_veh_id] = inactive_dict[uid_veh_id]
                inactive_to_active[uid_veh_id].inactive_steps = 0 # Những cặp ở trong inactive bước trước, bước này thành active, số bước inactive được đưa về 0
                continue
            brand_new[uid_veh_id] = Pair(unified_id=uid_veh_id[0], vehicle_id=uid_veh_id[1], class_id=uid_veh_id[2], birth_time=self.time, inactive_steps_before_removed=self.inactive_steps_before_removed) # Những uid_veh_id còn lại được tạo một cặp hoàn toàn mới

        active_to_inactive = dict(filter(lambda x: x[0] not in active_this_step, active_dict.items())) # Những cặp active ở bước trước nhưng bước này không nằm trong list các cặp matcher trả về sẽ trở thành inactive ở bước này
        inactive_dict = dict(filter(lambda x: x[0] not in inactive_to_active, inactive_dict.items())) # Những inactive ở bước này bằng list inactive  loại bỏ đi những cặp trở thành active ở bước này

        self.verify_dicts(active_this_step, brand_new, inactive_to_active)
        active_dict = {**active_this_step, **brand_new, **inactive_to_active} # Tổng hợp lại, những cặp active ở bước này là hợp của những cặp ở trong active list sẵn và vẫn active ở bước này, những cặp mới hoàn toàn, đang là inactive thành active và đang deleted được rebrand
        inactive_dict = {**inactive_dict, **active_to_inactive} # Những cặp inactive ở bước này là hợp của những cặp inactive vẫn inactive ở bước này và những cặp active ở bước trước thành inactive ở bước này

        # Xét các pair trong inactive_dict
        # Xem số inactive_step của từng pair lớn hơn ngưỡng thì thêm end_time, filled_period và chuyển sang delete_list
        inactive_to_deleted = {}
        for uid_veh_id in inactive_dict: # Duyệt từng phần tử của inactive list
            assert inactive_dict[uid_veh_id].unified_id == uid_veh_id[0] and inactive_dict[uid_veh_id].vehicle_id == uid_veh_id[1] and inactive_dict[uid_veh_id].class_id == uid_veh_id[2] # Gọi pair instance ứng với uid_veh_id
            inactive_dict[uid_veh_id].inactive_steps += 1 # Tăng inactive step liên tiếp thêm 1
            if inactive_dict[uid_veh_id].inactive_steps > inactive_dict[uid_veh_id].inactive_steps_before_removed: # Nếu số bước liên tiếp là inactive của cặp này lớn hơn ngưỡng cho phép
                num_seconds_inactive = get_time_amount_from_frames_number(num_frames=inactive_dict[uid_veh_id].inactive_steps, # Tính thời gian kể từ khi trở thành cặp inactive
                                                                          frame_stride=frame_stride,
                                                                          fps=fps)
                time = self.time + timedelta(seconds=-num_seconds_inactive) # Thời gian trở thành căp inactive
                inactive_dict[uid_veh_id].delete(time=time) # Tạo period là thời gian khởi tạo pair và thời điểm sang inactive??????
                inactive_to_deleted[uid_veh_id] = inactive_dict[uid_veh_id] # List các cặp inactive thành deleted thêm phần tử này

        inactive_dict = dict(filter(lambda x: x[0] not in inactive_to_deleted, inactive_dict.items())) # Cập nhật lại list inactive bằng cách list inactive loại bỏ đi các inactive thành deleted
        deleted_dict.update(inactive_to_deleted) # Cập list các cặp bị deleted bằng cách cộng thêm các cặp ở trong list các cặp từ inactive thành deleted

        self.active_pairs = active_dict # Cập nhật lại self.active_pairs
        self.inactive_pairs = inactive_dict # Cập nhật lại self.inactive_pairs
        self.deleted_pairs = deleted_dict # Cập nhật lại self.deleted_pairs

    def reset(self, time):
        self.start_time = time
        self.time = time
        self.active_pairs.clear()  # Only pair of unified_id and vehicle_id
        self.inactive_pairs.clear()
        self.deleted_pairs.clear()

    def verify(self):
        # Kiểm tra các active_pairs, inactive_pairs, deleted_pairs phải không có phần tử chung
        # Hợp của ba list trên phải đủ trong self.pairs
        assert len(list(set(self.active_pairs.keys()))) == len(list(self.active_pairs.keys()))
        assert len(list(set(self.inactive_pairs.keys()))) == len(list(self.inactive_pairs.keys()))
        assert len(list(set(self.deleted_pairs.keys()))) == len(list(self.deleted_pairs.keys()))

        inter = set(self.active_pairs.keys()).intersection(self.inactive_pairs.keys())
        inter = list(inter.intersection(self.deleted_pairs.keys()))
        assert len(inter) == 0

        union = {**self.active_pairs, **self.inactive_pairs, **self.deleted_pairs}
        assert len(union) == (len(self.active_pairs.keys()) + len(self.inactive_pairs.keys()) + len(self.deleted_pairs.keys()))
        for uid_veh_id in union:
            assert union[uid_veh_id].unified_id == uid_veh_id[0] and union[uid_veh_id].vehicle_id == uid_veh_id[1] and union[uid_veh_id].class_id == uid_veh_id[2]

    @staticmethod
    def verify_dicts(*args):
        assert len(args) == 3
        dict_1, dict_2, dict_3 = args
        assert len(list(set(dict_1.keys()))) == len(list(dict_1.keys()))
        assert len(list(set(dict_2.keys()))) == len(list(dict_2.keys()))
        assert len(list(set(dict_3.keys()))) == len(list(dict_3.keys()))
        inter = set(dict_1.keys()).intersection(dict_2.keys())
        inter = list(inter.intersection(dict_3.keys()))
        assert len(inter) == 0
        union = {**dict_1, **dict_2, **dict_3}
        assert len(union) == (len(dict_1.keys()) + len(dict_2.keys()) + len(dict_3.keys()))
        for uid_veh_id in union:
            assert union[uid_veh_id].unified_id == uid_veh_id[0] and union[uid_veh_id].vehicle_id == uid_veh_id[1] and union[uid_veh_id].class_id == uid_veh_id[2]

    def get_pairs_instances(self):
        return {**self.active_pairs, **self.inactive_pairs}

    def save_pairs_to_db(self, save_path=None, save_dir="../database"):
        if not save_path:
            save_path = self.start_time.strftime("%Y-%m-%d") + ".pkl"
        save_path = os.path.join(save_dir, save_path)
        pairs = {**self.active_pairs, **self.inactive_pairs, **self.deleted_pairs}
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                pickle.dump(pairs, f)
        else:
            with open(save_path, "rb") as f:
                db_pairs = pickle.load(f)
            db_pairs.update(pairs)
            with open(save_path, "wb") as f:
                pickle.dump(db_pairs, f)

        self.deleted_pairs.clear()