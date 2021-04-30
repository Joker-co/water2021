import numpy as np
import pickle as pk

offset = 0

class DetEnsemble(object):
    def __init__(self,
                 num_classes=4,
                 ensemble_type='soft-vote',
                 vote_thresh=0.62,
                 scale_ranges=[],
                 conf_type='max',
                 read_from='disk',
                 filter_score=0.05):
        self.ensemble_type = ensemble_type
        self.num_classes = num_classes
        self.vote_thresh = vote_thresh
        self.conf_type = conf_type
        self.read_from = read_from
        self.filter_score = filter_score
        self.scale_ranges = scale_ranges
        self.vote_func = {
            'vote': self.bbox_vote,
            'soft-vote': self.soft_bbox_vote
        }

    def get_pk_res(self, pk_file):
        with open(pk_file, "rb") as f:
            res = pk.load(f)
        return res

    def filter_boxes_byscale(self, dets, scale_range):
        keep = []
        for i, det in enumerate(dets):
            area = (det[2] - det[0] + offset) * (det[3] - det[1] + offset)
            low = scale_range[0] * scale_range[0]
            high = scale_range[1] * scale_range[1]
            score = det[4]
            if score >= self.filter_score:
                if area > low and area < high:
                    keep.append(i)
        return dets[keep]

    def get_filter_cls_res(self, all_scale_res):
        filter_cls_res = []
        for cls in range(self.num_classes):
            flag = True
            for i, scale_res in enumerate(all_scale_res):
                if flag:
                    cls_res = self.filter_boxes_byscale(
                        scale_res[cls], self.scale_ranges[i])
                    flag = False
                else:
                    temp_res = self.filter_boxes_byscale(
                        scale_res[cls], self.scale_ranges[i])
                    cls_res = np.concatenate((cls_res, temp_res))
            filter_cls_res.append(cls_res)
        return filter_cls_res

    def bbox_vote(self, det):
        if det.shape[0] <= 1:
            return det
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0] + offset) * (det[:, 3] - det[:, 1] + offset)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + offset)
            h = np.maximum(0.0, yy2 - yy1 + offset)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # get needed merge det and delete these  det
            merge_index = np.where(o >= self.vote_thresh)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
                continue
            else:
                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                    det_accu[:, -1:], (1, 4))
                if self.conf_type == 'max':
                    vote_score = np.max(det_accu[:, 4])
                else:
                    vote_score = np.mean(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(
                    det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = vote_score
                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except:
                    dets = det_accu_sum
        return dets

    def soft_bbox_vote(self, det):
        if det.shape[0] <= 1:
            return det
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0] + offset) * (det[:, 3] - det[:, 1] + offset)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + offset)
            h = np.maximum(0.0, yy2 - yy1 + offset)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # get needed merge det and delete these det
            merge_index = np.where(o >= self.vote_thresh)[0]
            det_accu = det[merge_index, :]
            det_accu_iou = o[merge_index]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
                continue
            else:
                soft_det_accu = det_accu.copy()
                soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
                soft_index = np.where(soft_det_accu[:, 4] >= self.filter_score)[0]
                soft_det_accu = soft_det_accu[soft_index, :]

                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                    det_accu[:, -1:], (1, 4))
                if self.conf_type == 'max':
                    vote_score = np.max(det_accu[:, 4])
                else:
                    vote_score = np.mean(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(
                    det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = vote_score

                if soft_det_accu.shape[0] > 0:
                    det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except:
                    dets = det_accu_sum

        order = dets[:, 4].ravel().argsort()[::-1]
        dets = dets[order, :]
        return dets

    def vote(self, pk_file_list):
        pk_file_num = len(pk_file_list)
        print("read form {}".format(self.read_from))
        if pk_file_num == 1:
            return self.get_pk_res(pk_file_list[0])
        if self.read_from == 'disk':
            res_list = [self.get_pk_res(pk_file) for pk_file in pk_file_list]
        else:
            res_list = pk_file_list
        img_num = len(res_list[0])
        scale_res = []
        for i in range(img_num):
            temp = []
            for j in range(pk_file_num):
                temp.append(res_list[j][i])
            scale_res.append(temp)
        out_res = []
        for _, res in enumerate(scale_res):
            filter_cls_res = self.get_filter_cls_res(res)
            cls_res = []
            for j in range(self.num_classes):
                cls_res.append(self.vote_func[self.ensemble_type](filter_cls_res[j]))
            out_res.append(cls_res)
        return out_res
