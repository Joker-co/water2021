import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn

from mmcv import Config

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from det_ensemble import DetEnsemble
import os.path as osp
import time
from PIL import Image
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--out_folder', help='output result file in pickle format')
    parser.add_argument(
        '--res_path',
        default='./submit_b.csv',
        help='output result file in pickle format')
    parser.add_argument(
        '--time_path',
        default='./time.txt',
        help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--only_eval', action='store_true', help='only_eval')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


water_classes = ['cube', 'ball', 'cylinder', 'human body', 'tyre', 'circle cage', 'square cage', 'metal bucket']

label_ids = {name: i + 1 for i, name in enumerate(water_classes)}


def get_segmentation(points):

    return [
        points[0], points[1], points[2] + points[0], points[1],
        points[2] + points[0], points[3] + points[1], points[0],
        points[3] + points[1]
    ]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'waterweeds':
            continue
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w * h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [xmin, ymin, w, h],
            "category_id": category_id,
            "id": anno_id,
            "ignore": 0
        })
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file, image_set):
    images = []
    annotations = []
    with open(image_set, "r") as f:
        img_ids = [line.strip() for line in f.readlines()]
    anno_id = 1
    for idx, img_id in enumerate(img_ids):
        img_file_path = osp.join(img_path, img_id + '.jpg')
        xml_file_path = osp.join(xml_path, img_id + '.xml')
        w, h = Image.open(img_file_path).size
        img_name = osp.basename(img_file_path)
        img = {
            "file_name": img_name,
            "height": int(h),
            "width": int(w),
            "id": idx + 1
        }
        images.append(img)
        if not osp.exists(xml_file_path):
            annos = []
        else:
            annos, anno_id = parse_xml(xml_file_path, idx + 1, anno_id)
        annotations.extend(annos)

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    mmcv.dump(final_result, out_file)
    return annotations


def convert_main():
    image_folder = "data/water/test/image"
    test_set = "data/water/test/list/water_test_b.txt"
    test_json = "data/water/test/anno/water_test_b.json"
    ori_image_ids = os.listdir(image_folder)
    image_ids = []
    for i in ori_image_ids:
        if './' or '../' in i:
            continue
        image_ids.append(i.split('.')[0])
    assert len(image_ids) == 1200, "image_ids must == 1200"
    # image_ids = [item.split('.')[0] for item in os.listdir(image_folder)]
    image_ids.sort()

    with open(test_set, "w") as f:
        for item in image_ids:
            print(item.strip(), file=f)

    test_xml_path = 'data/water/test/box'
    test_img_path = 'data/water/test/image'
    cvt_annotations(test_img_path, test_xml_path, test_json, test_set)

    print('Done!')


def pre_data_main():
    pre_define_folder = "./data/water/test/image"
    answer_fodler = "/userhome/answerB"
    if not os.path.exists(answer_fodler):
        os.makedirs(answer_fodler)
    if os.path.exists(pre_define_folder):
        os.system('rm {}'.format(pre_define_folder))
    cmd = "ln -s {} {}".format('/userhome/testB', pre_define_folder)
    os.system(cmd)


def flip_output(outputs, img_infos, cls_num=4):
    flip_output = []
    for idx, output in enumerate(outputs):
        width = img_infos[idx]['width']
        cls_temp = []
        for cls in range(cls_num):
            dets = output[cls]
            flipped = dets.copy()
            flipped[..., 0] = width - dets[..., 2]
            flipped[..., 2] = width - dets[..., 0]
            cls_temp.append(flipped)
        flip_output.append(cls_temp)
    return flip_output


def get_submit_results(outputs,
                       anno_file,
                       img_infos,
                       class_names,
                       out_path="submit_file.csv"):
    im_ids = []
    with open(anno_file, "r") as f:
        for line in f.readlines():
            im_ids.append(line.strip())
    with open(out_path, "w") as f:
        print("name,image_id,confidence,xmin,ymin,xmax,ymax", file=f)
        for idx, output in enumerate(outputs):
            width = img_infos[idx]['width']
            height = img_infos[idx]['height']
            for cls, cls_name in enumerate(class_names):
                dets = output[cls]
                for det in dets:
                    score = round(float(det[4]), 4)
                    xmin = max(1, 1 + round(float(det[0])))
                    ymin = max(1, 1 + round(float(det[1])))
                    xmax = min(1 + round(float(det[2])), width)
                    ymax = min(1 + round(float(det[3])), height)
                    print(
                        cls_name,
                        im_ids[idx],
                        score,
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        sep=',',
                        file=f)


def main():
    from test_cfgs.sound import test_cfg
    test_model_cfg = test_cfg['model_cfg']
    vote_thresh = test_cfg['vote_thresh']
    ensemble_type = test_cfg['ensemble_type']
    conf_type = test_cfg['conf_type']
    read_from = test_cfg['read_from']
    filter_score = test_cfg['filter_score']
    args = parse_args()
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    scale_ranges = []
    out_list = []
    dist_init = False
    res_path = args.res_path
    time_path = args.time_path
    begin = time.time()
    for item in test_model_cfg[:]:
        config = item['cfg_path']
        cfg = Config.fromfile(config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None
        # cfg.model.backbone.depth = item['depth']

        if not dist_init:
            # init distributed env first, since logger depends on the dist info.
            if args.launcher == 'none':
                distributed = False
            else:
                distributed = True
                init_dist(args.launcher, **cfg.dist_params)
            dist_init = True

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)

        # build the model and load checkpoint
        model = build_detector(
            cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        _ = load_checkpoint(model, item['model_path'], map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

        scale = item['scale']
        # flip flag
        flip_flag = item.get('flip_flag', None)
        flip = item.get('flip', False)
        scale_range = item['scale_range']
        scale_ranges.append(scale_range)
        if not flip:
            out_path = item['prefix'] + "_" + str(scale[0]) + "_" + str(
                scale[1]) + '.pkl'
        else:
            out_path = item['prefix'] + "_" + str(scale[0]) + "_" + str(
                scale[1]) + '_flip.pkl'
        val_cfg = cfg.data.test
        val_cfg['pipeline'][1]['img_scale'] = scale
        # flip flag
        val_cfg['pipeline'][1]['flip_flag'] = flip_flag
        img_infos = None
        if flip:  # flip
            val_cfg['pipeline'][1]['transforms'][1] = dict(
                type='DetectFlip', flip_ratio=1.0)
        else:
            val_cfg['pipeline'][1]['transforms'][1] = dict(type='RandomFlip')
        dataset = build_dataset(val_cfg)
        model.CLASSES = dataset.CLASSES
        num_classes = len(dataset.CLASSES)
        if flip:
            img_infos = dataset.load_annotations(val_cfg['ann_file'])
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
        rank, _ = get_dist_info()
        if rank == 0:
            if flip:
                outputs = flip_output(outputs, img_infos, num_classes)
            out_list.append(outputs)
            if args.out_folder:
                if not os.path.exists(args.out_folder):
                    os.makedirs(args.out_folder)
                print('\nwriting results to {}'.format(
                    osp.join(args.out_folder, out_path)))
                mmcv.dump(outputs, osp.join(args.out_folder, out_path))
    rank, _ = get_dist_info()
    if rank == 0:
        val_cfg = cfg.data.test
        dataset = build_dataset(val_cfg)
        model.CLASSES = dataset.CLASSES
        if len(out_list) > 1:
            ensembel = DetEnsemble(
                num_classes=len(dataset.CLASSES),
                scale_ranges=scale_ranges,
                ensemble_type=ensemble_type,
                vote_thresh=vote_thresh,
                conf_type=conf_type,
                read_from=read_from,
                filter_score=filter_score)  # noqa
            outputs = ensembel.vote(out_list)
        else:
            outputs = out_list[0]
        img_infos = dataset.load_annotations(val_cfg['ann_file'])
        test_set = test_cfg['test_set']
        get_submit_results(
            outputs, test_set, img_infos, dataset.CLASSES, out_path=res_path)
        cost_time = time.time() - begin
        with open(time_path, "w") as f:
            print("cost time: ", cost_time, file=f)
        print("cost time", time.time() - begin)


if __name__ == '__main__':
    # pre_data_main()
    # convert_main()
    main()
