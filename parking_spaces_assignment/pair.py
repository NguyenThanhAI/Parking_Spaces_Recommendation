import os
import pickle
from datetime import datetime, timedelta
from load_videos.videos_utils import get_time_amount_from_frames_number
from database.sqldatabase import SQLiteDataBase
from database.mysqldatabase import MySQLDataBase
from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis


class TentativePair(object):

    def __init__(self, unified_id, vehicle_id, class_id, type_space, parking_ground, cam, birth_time, tentative_steps_before_accepted=30):
        self.unified_id = unified_id
        self.vehicle_id = vehicle_id
        self.class_id = class_id
        self.type_space = type_space
        self.parking_ground = parking_ground
        self.cam = cam
        self.birth_time = birth_time
        self.tentative_steps = 0
        if unified_id >= 1000:
            assert unified_id >= 1000 and type_space == "outlier"
            self.tentative_steps_before_accepted = 30
        else:
            self.tentative_steps_before_accepted = tentative_steps_before_accepted

    def __str__(self):
        return str(self.__class__) + ":" + str(self.__dict__)


class Pair(object):

    def __init__(self, unified_id, vehicle_id, class_id, type_space, parking_ground, cam, birth_time, inactive_steps_before_removed=10000):
        self.unified_id = unified_id
        self.vehicle_id = vehicle_id
        self.class_id = class_id
        self.type_space = type_space
        self.parking_ground = parking_ground
        self.cam = cam
        self.birth_time = birth_time
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.end_time = None
        self.inactive_steps = 0

    def delete(self, time):
        self.end_time = time # (Birth_time, end_time), Khi bị xóa sẽ ghi lại khoảng thời gian tồn tại của pair này

    def __str__(self):
        return str(self.__class__) + ":" + str(self.__dict__)


class PairsScheduler(object):

    def __init__(self, time, database_dir="../database", database_file=None, use_mysql=False, host="localhost", user="Thanh", passwd="Aimesoft", reset_table=True, save_to_db_period=2, tentative_steps_before_accepted=30, inactive_steps_before_removed=1000):
        self.start_time = time
        self.time = time
        if not database_file:
            if not use_mysql:
                database_file = self.start_time.strftime("%Y-%m-%d") + ".db"
            else:
                database_file = self.start_time.strftime("%Y_%m_%d")
        else:
            if use_mysql:
                database_file = database_file.split(".")[0]
        if not use_mysql:
            self.database = SQLiteDataBase(database_dir=database_dir, database_file=database_file)
        else:
            self.database = MySQLDataBase(host=host, user=user, passwd=passwd, database=database_file, reset_table=reset_table)
        self.save_to_db_period = save_to_db_period
        self.tentative_steps_before_accepted = tentative_steps_before_accepted
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.tentative_pairs = {}
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
        tentative_dict = self.tentative_pairs
        active_dict = self.active_pairs
        inactive_dict = self.inactive_pairs
        deleted_dict = self.deleted_pairs

        return tentative_dict, active_dict, inactive_dict, deleted_dict

    #@timethis
    def step(self, uid_veh_list, num_frames, frame_stride=1, fps=24):
        self.update_time_from_frame_numbers(num_frames, frame_stride, fps) # Cập nhật self.time
        tentative_dict, active_dict, inactive_dict, deleted_dict = self.get_list_uid_veh_dict()
        inactive_to_active = {} # Những pair đang ở inactive list và ở bước này xuất hiện trong match pair giữa unified id và vehicle id
        active_this_step = {} # Những pair đang ở trong active list và bước này cũng xuất hiện trong match pair giữa unified id và vehicle id
        #active_to_inactive = [] # Những pair đang ở trong active list và bước này không xuất hiện trong match pair giữa unified id và vehicle id
        brand_new = {}
        new_tentatives = {}
        for uid_veh_id in uid_veh_list: # Xét từng uid_veh_id từ kết quả matcher trả về
            #existed = False
            if uid_veh_id in tentative_dict:
                tentative_dict[uid_veh_id].tentative_steps += 1
                if tentative_dict[uid_veh_id].tentative_steps > tentative_dict[uid_veh_id].tentative_steps_before_accepted:
                    brand_new[uid_veh_id] = Pair(unified_id=int(uid_veh_id[0]), vehicle_id=int(uid_veh_id[1]), class_id=int(uid_veh_id[2]), type_space=uid_veh_id[3], parking_ground=uid_veh_id[4], cam=uid_veh_id[5], birth_time=tentative_dict[uid_veh_id].birth_time, inactive_steps_before_removed=self.inactive_steps_before_removed)
                continue

            if uid_veh_id in active_dict: # Nếu uid_veh_id nằm trong active_dict nghĩa là cặp này đang active và tiếp tục bước này vẫn active
                #existed = True
                active_this_step[uid_veh_id] = active_dict[uid_veh_id]
                continue

            if uid_veh_id in inactive_dict: # Nếu uid_veh_id nằm trong inactive_dict nghĩa là cặp này bước trước đang inactive bước này sẽ được active
                inactive_to_active[uid_veh_id] = inactive_dict[uid_veh_id]
                inactive_to_active[uid_veh_id].inactive_steps = 0 # Những cặp ở trong inactive bước trước, bước này thành active, số bước inactive được đưa về 0
                continue

            if uid_veh_id in deleted_dict:
                print("{} in deleted_dict".format(uid_veh_id))

            new_tentatives[uid_veh_id] = TentativePair(unified_id=uid_veh_id[0], vehicle_id=uid_veh_id[1], class_id=uid_veh_id[2], type_space=uid_veh_id[3], parking_ground=uid_veh_id[4], cam=uid_veh_id[5], birth_time=self.time, tentative_steps_before_accepted=self.tentative_steps_before_accepted)

        tentative_dict = dict(filter(lambda x: x[0] not in brand_new, tentative_dict.items()))
        assert len(list(set(tentative_dict.keys()).intersection(brand_new.keys()))) == 0
        active_to_inactive = dict(filter(lambda x: x[0] not in active_this_step, active_dict.items())) # Những cặp active ở bước trước nhưng bước này không nằm trong list các cặp matcher trả về sẽ trở thành inactive ở bước này
        inactive_dict = dict(filter(lambda x: x[0] not in inactive_to_active, inactive_dict.items())) # Những inactive ở bước này bằng list inactive  loại bỏ đi những cặp trở thành active ở bước này

        assert len(list(set(tentative_dict.keys()).intersection(new_tentatives.keys()))) == 0
        tentative_dict = {**tentative_dict, **new_tentatives}
        self.verify_dicts(active_this_step, brand_new, inactive_to_active)
        active_dict = {**active_this_step, **brand_new, **inactive_to_active} # Tổng hợp lại, những cặp active ở bước này là hợp của những cặp ở trong active list sẵn và vẫn active ở bước này, những cặp mới hoàn toàn, đang là inactive thành active và đang deleted được rebrand
        assert len(list(set(inactive_dict.keys()).intersection(active_to_inactive.keys()))) == 0
        inactive_dict = {**inactive_dict, **active_to_inactive} # Những cặp inactive ở bước này là hợp của những cặp inactive vẫn inactive ở bước này và những cặp active ở bước trước thành inactive ở bước này

        # Xét các pair trong inactive_dict
        # Xem số inactive_step của từng pair lớn hơn ngưỡng thì thêm end_time, filled_period và chuyển sang delete_list
        inactive_to_deleted = {}
        for uid_veh_id in inactive_dict: # Duyệt từng phần tử của inactive list
            assert inactive_dict[uid_veh_id].unified_id == uid_veh_id[0] and inactive_dict[uid_veh_id].vehicle_id == uid_veh_id[1] and inactive_dict[uid_veh_id].class_id == uid_veh_id[2] and inactive_dict[uid_veh_id].type_space == uid_veh_id[3] and inactive_dict[uid_veh_id].parking_ground == uid_veh_id[4] and inactive_dict[uid_veh_id].cam == uid_veh_id[5] # Gọi pair instance ứng với uid_veh_id
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

        self.tentative_pairs = tentative_dict
        self.active_pairs = active_dict # Cập nhật lại self.active_pairs
        self.inactive_pairs = inactive_dict # Cập nhật lại self.inactive_pairs
        self.deleted_pairs = deleted_dict # Cập nhật lại self.deleted_pairs

        running_time = int((self.time - self.start_time).total_seconds())
        #print(running_time, running_time % self.save_to_db_period)
        if running_time > 0 and (running_time % self.save_to_db_period) == 0:
            self.save_pairs_to_db()

    def reset(self, time):
        self.start_time = time
        self.time = time
        self.tentative_pairs.clear()
        self.active_pairs.clear()  # Only pair of unified_id and vehicle_id
        self.inactive_pairs.clear()
        self.deleted_pairs.clear()

    def verify(self):
        # Kiểm tra các active_pairs, inactive_pairs, deleted_pairs phải không có phần tử chung
        # Hợp của ba list trên phải đủ trong self.pairs
        assert len(list(set(self.active_pairs.keys()))) == len(list(self.active_pairs.keys()))
        assert len(list(set(self.inactive_pairs.keys()))) == len(list(self.inactive_pairs.keys()))
        assert len(list(set(self.tentative_pairs.keys()))) == len(list(self.tentative_pairs.keys()))

        inter = set(self.active_pairs.keys()).intersection(self.inactive_pairs.keys())
        inter = list(inter.intersection(self.tentative_pairs.keys()))
        assert len(inter) == 0

        assert len(list(set(self.active_pairs.keys()).intersection(self.inactive_pairs.keys()))) == 0, print(self.active_pairs.keys(), self.inactive_pairs.keys())
        assert len(list(set(self.active_pairs.keys()).intersection(self.tentative_pairs.keys()))) == 0, print(self.active_pairs.keys(), self.tentative_pairs.keys())
        assert len(list(set(self.inactive_pairs.keys()).intersection(self.tentative_pairs.keys()))) == 0, print(self.inactive_pairs.keys(), self.tentative_pairs.keys())

        union = {**self.active_pairs, **self.inactive_pairs, **self.tentative_pairs}
        assert len(union) == (len(self.active_pairs.keys()) + len(self.inactive_pairs.keys()) + len(self.tentative_pairs.keys())), print(self.active_pairs.keys(), self.inactive_pairs.keys(), self.tentative_pairs.keys())
        for uid_veh_id in union:
            assert union[uid_veh_id].unified_id == uid_veh_id[0] and union[uid_veh_id].vehicle_id == uid_veh_id[1] and union[uid_veh_id].class_id == uid_veh_id[2] and union[uid_veh_id].type_space == uid_veh_id[3] and union[uid_veh_id].parking_ground == uid_veh_id[4] and union[uid_veh_id].cam == uid_veh_id[5]

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
            assert union[uid_veh_id].unified_id == uid_veh_id[0] and union[uid_veh_id].vehicle_id == uid_veh_id[1] and union[uid_veh_id].class_id == uid_veh_id[2] and union[uid_veh_id].type_space == uid_veh_id[3] and union[uid_veh_id].parking_ground == uid_veh_id[4] and union[uid_veh_id].cam == uid_veh_id[5]

    def get_pairs_instances(self):
        return {**self.active_pairs, **self.inactive_pairs}

    def convert_intances_to_list_of_tuple(self, pairs):
        return list(map(lambda x: (x[1].unified_id, x[1].vehicle_id, x[1].class_id, x[1].type_space,
                                   x[1].parking_ground, x[1].cam, x[1].inactive_steps, x[1].birth_time, x[1].end_time), pairs.items()))

    def save_pairs_to_db(self):
        pairs = {**self.active_pairs, **self.inactive_pairs, **self.deleted_pairs} # Các key ở active và deleted pairs có thể trùng nhau nên áp dụng phương thức dưới với từng dict mà không nên gộp cả 3 như này có thể bị mất instance
        pairs_info = self.convert_intances_to_list_of_tuple(pairs)
        self.database.add_pairs(pairs_info=pairs_info)

        self.deleted_pairs.clear()
        print("Save pairs and clear deleted pairs")
