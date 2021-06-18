# coding: utf-8
import os
import time
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd
import pdb

# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
END_DAY = 15
SEED = 2021

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 每个行为的负样本下采样比例(下采样后负样本数/原负样本数)
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.1, "comment": 0.1, "follow": 0.1, "favorite": 0.1}

# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5, "comment": 5, "follow": 5, "favorite": 5}


def create_dir():
    """
    创建所需要的目录
    """
    # 创建data目录
    if not os.path.exists(ROOT_PATH):
        print('Create dir: %s'%ROOT_PATH)
        os.mkdir(ROOT_PATH)
    # data目录下需要创建的子目录
    need_dirs = ["offline_train", "online_train", "evaluate", "submit",
                 "feature", "model", "model/online_train", "model/offline_train"]
    for need_dir in need_dirs:
        need_dir = os.path.join(ROOT_PATH, need_dir)
        if not os.path.exists(need_dir):
            print('Create dir: %s'%need_dir)
            os.mkdir(need_dir)


def check_file():
    '''
    检查数据文件是否存在
    '''
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    flag = True
    not_exist_file = []
    for f in paths:
        if not os.path.exists(f):
            not_exist_file.append(f)
            flag = False
    return flag, not_exist_file


def statis_data():
    """
    统计特征最大，最小，均值
    """
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    pd.set_option('display.max_columns', None)
    for path in paths:
        df = pd.read_csv(path)
        print(path + " statis: ")
        print(df.describe()) # shows a quick statistic summary of your data
        print('Distinct count:') # 统计各个字段的不同值的数量
        print(df.nunique())


def statis_feature(start_day=1, before_day=7, agg='sum'):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数），即时间窗口大小
    :param agg: String. 统计方法
    """
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST] # FEA_COLUMN_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'follow', 'favorite']
    feature_dir = os.path.join(ROOT_PATH, "feature")
    for dim in ["userid", "feedid"]: # dim --> dimention
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in range(start_day, END_DAY-before_day+1):
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start + before_day
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim+"_feature.csv") # ./data/feature/userid_feature.csv
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)


def generate_sample(stage="offline_train"):
    """
    对负样本进行下采样，生成各个阶段所需样本（不包含自定义特征）
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    :return: List of sample df
    """
    day = STAGE_END_DAY[stage]
    if stage == "submit":
        sample_path = TEST_FILE
    else:
        sample_path = USER_ACTION
    stage_dir = os.path.join(ROOT_PATH, stage)
    df = pd.read_csv(sample_path)
    df_arr = []
    if stage == "evaluate":
        # 线下评估
        col = ["userid", "feedid", "date_", "device"] + ACTION_LIST
        df = df[df["date_"] == day][col]
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    elif stage == "submit":
        # 线上提交
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        df["date_"] = 15
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    else:
        # 线下/线上训练
        # 同行为(userid,feedid,action均相同)取按时间最近的样本
        for action in ACTION_LIST:
            df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
        # 负样本下采样
        for action in ACTION_LIST: # ['read_comment', 'like', 'click_avatar', 'forward']
            action_df = df[(df["date_"] <= day) & (df["date_"] >= day - ACTION_DAY_NUM[action] + 1)] # 2882467, 只取后5天的行为用来构造训练数据
            df_neg = action_df[action_df[action] == 0] # 通过判断相应的行为==0，来收集负样本            # 2779890
            df_pos = action_df[action_df[action] == 1] 
            df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False) # 555978 对负样本直接进行1/5随机采样
            df_all = pd.concat([df_neg, df_pos]) # 共658555，包括采样后的555978个负样本和全部102577个正样本=共658555个样本
            col = ["userid", "feedid", "date_", "device"] + [action] # ['userid', 'feedid', 'date_', 'device', 'read_comment']
            file_name = os.path.join(stage_dir, stage + "_" + action + "_" + str(day) + "_generate_sample.csv") # ./data/online_train/online_train_read_comment_14_generate_sample.csv
            print('Save to: %s'%file_name)
            df_all[col].to_csv(file_name, index=False)
            df_arr.append(df_all[col])
    return df_arr


def concat_sample(sample_arr, stage="offline_train"):
    """
    基于样本数据和特征，拼接生成特征数据
    :param sample_arr: List of sample df
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    """
    day = STAGE_END_DAY[stage]
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO) # 106444, FEED_INFO = ./data/wechat_algo_data1/feed_info.csv
    feed_info = feed_info.set_index('feedid') # 将默认的int自增索引改为将feedid作为索引(经过观察，feedid不重复)
    # 在statistics_feature()中，已经基于userid统计的历史行为的次数
    user_date_feature_path = os.path.join(ROOT_PATH, "feature", "userid_feature.csv")
    user_date_feature = pd.read_csv(user_date_feature_path)
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    # 在statistics_feature()中，已经基于feedid统计的历史行为的次数
    feed_date_feature_path = os.path.join(ROOT_PATH, "feature", "feedid_feature.csv") # ./data/feature/feedid_feature.csv
    feed_date_feature = pd.read_csv(feed_date_feature_path)
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    for index, sample in enumerate(sample_arr): # sample_arr长度为5，分别存储了包含read_comment,like,click_avatar,forward的ua，最终生成5个不同action的样本文件
        features = ["userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]
        if stage == "evaluate":
            action = "all"
            features += ACTION_LIST # ACTION_LIST=["read_comment", "like", "click_avatar",  "forward"]
        elif stage == "submit":
            action = "all"
        else:
            action = ACTION_LIST[index]
            features += [action] # 仅将当前的action加入到feature中
        print("action: ", action)
        # 样本拼接
        sample = sample.join(feed_info, on="feedid", how="left", rsuffix="_feed") # DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
        sample = sample.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
        sample = sample.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")
        # 【生成】自定义特征列表(表头)
        feed_feature_col = [b+"sum" for b in FEA_COLUMN_LIST] # ['read_commentsum', 'likesum', 'click_avatarsum', 'forwardsum', 'commentsum', 'followsum', 'favoritesum']
        user_feature_col = [b+"sum_user" for b in FEA_COLUMN_LIST] # ['read_commentsum_user', 'likesum_user', 'click_avatarsum_user', 'forwardsum_user', 'commentsum_user', 'followsum_user', 'favoritesum_user']
        # 【填充】自定义特征列表，并填充空值
        # sample.columns=['userid', 'feedid', 'date_', 'device', 'read_comment', 'authorid','videoplayseconds', 'description', 'ocr', 'asr', 'bgm_song_id', 'bgm_singer_id', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list', 'description_char', 'ocr_char', 'asr_char', 'read_commentsum', 'likesum', 'click_avatarsum', 'forwardsum', 'commentsum', 'followsum', 'favoritesum', 'read_commentsum_user', 'likesum_user', 'click_avatarsum_user',  'forwardsum_user', 'commentsum_user', 'followsum_user','favoritesum_user']
        sample[feed_feature_col] = sample[feed_feature_col].fillna(0.0)
        sample[user_feature_col] = sample[user_feature_col].fillna(0.0)
        sample[feed_feature_col] = np.log(sample[feed_feature_col] + 1.0) # log将偏态分布的样本尽可能转化为正态分布
        sample[user_feature_col] = np.log(sample[user_feature_col] + 1.0)
        features += feed_feature_col
        features += user_feature_col # features=['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id','videoplayseconds','read_comment', 'read_commentsum', 'likesum', 'click_avatarsum', 'forwardsum', 'commentsum', 'followsum', 'favoritesum', 'read_commentsum_user', 'likesum_user', 'click_avatarsum_user', 'forwardsum_user', 'commentsum_user', 'followsum_user', 'favoritesum_user']
        
        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0要用于填未知, 所以有效数据区间应该+1从而避开0值
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
        sample["videoplayseconds"] = np.log(sample["videoplayseconds"] + 1.0) # videoplayseconds>1的情况下取log才为正
        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] = sample[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
        file_name = os.path.join(ROOT_PATH, stage, stage + "_" + action + "_" + str(day) + "_concate_sample.csv") # ./data/online_train/online_train_read_comment_14_concate_sample.csv 以及for循环生成的其他5个action对应的文件
        print('Save to: %s'%file_name) # 加入了自定义统计特征的样本
        sample[features].to_csv(file_name, index=False)


def main():
    t = time.time()
    statis_data()
    logger.info('Create dir and check file')
    create_dir()
    flag, not_exists_file = check_file()
    if not flag:
        print("请检查目录中是否存在下列文件: ", ",".join(not_exists_file))
        return
    logger.info('Generate statistic feature')
    statis_feature()
    for stage in STAGE_END_DAY: # {'online_train': 14, 'offline_train': 12, 'evaluate': 13, 'submit': 15}
        logger.info("Stage: %s"%stage)
        logger.info('Generate sample') # 样本整理(包括负样本下采样)
        sample_arr = generate_sample(stage)
        logger.info('Concat sample with feature')
        concat_sample(sample_arr, stage) # 样本拼接
    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == "__main__":
    main()
    