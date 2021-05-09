import os.path as osp
folder = "tools/sound_cfg"

vote_thresh = 0.6
ensemble_type = 'soft-vote'
conf_type = 'max'
read_from = 'cache'
filter_score = 0.0001
test_set = "/mnt/lustre/share/yaoyongqiang/data/sound/anno/sound_test_all.txt"
# model_path = "/mnt/lustre/yaoyongqiang/workspace/my_project/codebase/mmdetection/exp/wb_base/latest.pth"
model_path = "/mnt/lustre/share/hujiahao1/sound/workdir_51.94/latest.pth"
cfg_path = "/mnt/lustre/yaoyongqiang/workspace/my_project/codebase/mmdetection/exp/hjh_base_2.py"

model_cfg = [{
    "cfg_path": cfg_path,
    "model_path": model_path,
    "scale_range": [0, 10000],
    'prefix': "detectors",
    "scale": (2048, 600),
    # "flip_flag": [False, True],
    },
    {
    "cfg_path": cfg_path,
    "model_path": model_path,
    "scale_range": [0, 10000],
    'prefix': "detectors",
    "scale": (2048, 1000),
    }
]

test_cfg = {
    "vote_thresh": vote_thresh,
    "ensemble_type": ensemble_type,
    "conf_type": conf_type,
    "read_from": read_from,
    "filter_score": filter_score,
    'model_cfg': model_cfg,
    'test_set': test_set
}
