import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torch.distributed import all_reduce
from torchvision.ops.boxes import nms
import math
from scipy.optimize import linear_sum_assignment

from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils.distributed_utils import is_dist_avail_and_initialized, get_world_size
from collections import defaultdict


class HungarianMatcher(nn.Module):

    def __init__(self,
                 coef_class: float = 2,
                 coef_bbox: float = 5,
                 coef_giou: float = 2):
        super().__init__()
        self.coef_class = coef_class
        self.coef_bbox = coef_bbox
        self.coef_giou = coef_giou
        assert coef_class != 0 or coef_bbox != 0 or coef_giou != 0, "all costs cant be 0"

    def forward(self, pred_logits, pred_boxes, annotations):
        with torch.no_grad():
            bs, num_queries = pred_logits.shape[:2]
            # We flatten to compute the cost matrices in a batch
            pred_logits = pred_logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_class]
            pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
            gt_class = torch.cat([anno["labels"] for anno in annotations]).to(pred_logits.device)
            gt_boxes = torch.cat([anno["boxes"] for anno in annotations]).to(pred_logits.device)
            # Compute the classification cost.
            alpha, gamma = 0.25, 2.0
            neg_cost_class = (1 - alpha) * (pred_logits ** gamma) * (-(1 - pred_logits + 1e-8).log())
            pos_cost_class = alpha * ((1 - pred_logits) ** gamma) * (-(pred_logits + 1e-8).log())
            cost_class = pos_cost_class[:, gt_class] - neg_cost_class[:, gt_class]
            # Compute the L1 cost between boxes
            cost_boxes = torch.cdist(pred_boxes, gt_boxes, p=1)
            # Compute the giou cost between boxes
            cost_giou = - generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
            # Final cost matrix
            cost = self.coef_bbox * cost_boxes + self.coef_class * cost_class + self.coef_giou * cost_giou
            cost = cost.view(bs, num_queries, -1).cpu()
            sizes = [len(anno["boxes"]) for anno in annotations]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):

    def __init__(self,
                 num_classes=9,
                 coef_class=2,
                 coef_boxes=5,
                 coef_giou=2,
                 alpha_focal=0.25,
                 alpha_dt=0.5,
                 gamma_dt=0.9,
                 max_dt=0.45,
                 device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.matcher = HungarianMatcher()
        self.coef_class = coef_class
        self.coef_boxes = coef_boxes
        self.coef_giou = coef_giou
        self.alpha_focal = alpha_focal
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]
        self.alpha_dt = alpha_dt
        self.gamma_dt = gamma_dt
        self.max_dt = max_dt

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    def loss_class(self, pred_logits, annotations, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        gt_classes_o = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)])
        gt_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_o
        gt_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                        dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        gt_classes_onehot.scatter_(2, gt_classes.unsqueeze(-1), 1)
        gt_classes_onehot = gt_classes_onehot[:, :, :-1]
        loss_ce = self.sigmoid_focal_loss(pred_logits,
                                          gt_classes_onehot,
                                          num_boxes,
                                          alpha=self.alpha_focal,
                                          gamma=2) * pred_logits.shape[1]
        return loss_ce

    def loss_boxes(self, pred_boxes, annotations, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none')
        return loss_bbox.sum() / num_boxes

    def loss_giou(self, pred_boxes, annotations, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(gt_boxes)))
        return loss_giou.sum() / num_boxes

    

    def record_positive_logits(self, logits, indices):
        idx = self._get_src_permutation_idx(indices)
        labels = logits[idx].argmax(dim=1)
        pos_logits = logits[idx].max(dim=1).values
        for label, logit in zip(labels, pos_logits):
            self.logits_sum[label] += logit
            self.logits_count[label] += 1


    def clear_positive_logits(self):
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(self.num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.num_classes)]

    @staticmethod
    def _get_src_permutation_idx(indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _discard_empty_labels(out, annotations):
        reserve_index = []
        for anno_idx in range(len(annotations)):
            if torch.numel(annotations[anno_idx]["boxes"]) != 0:
                reserve_index.append(anno_idx)
        for key, value in out.items():
            if key in ['logits_all', 'boxes_all']:
                out[key] = value[:, reserve_index, ...]
            elif key in ['features']:
                continue
            else:
                out[key] = value[reserve_index, ...]
        annotations = [annotations[idx] for idx in reserve_index]
        return out, annotations

    def forward(self, out, annotations=None):
        logits_all = out['logits_all']
        boxes_all = out['boxes_all']
        num_boxes = sum(len(anno["labels"]) for anno in annotations) if annotations is not None else 0
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=logits_all.device)
        if is_dist_avail_and_initialized():
            all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        loss = torch.zeros(1).to(logits_all.device)
        loss_dict = defaultdict(float)
        num_decoder_layers = logits_all.shape[0]
        for i in range(num_decoder_layers):
            # Compute DETR losses
            if annotations is not None:
                # Retrieve the matching between the outputs of the last layer and the targets
                indices = self.matcher(logits_all[i], boxes_all[i], annotations)
                # Compute the DETR losses
                loss_class = self.loss_class(logits_all[i], annotations, indices, num_boxes)
                loss_boxes = self.loss_boxes(boxes_all[i], annotations, indices, num_boxes)
                loss_giou = self.loss_giou(boxes_all[i], annotations, indices, num_boxes)
                loss_dict["loss_class"] += loss_class
                loss_dict["loss_boxes"] += loss_boxes
                loss_dict["loss_giou"] += loss_giou
                loss += self.coef_class * loss_class + self.coef_boxes * loss_boxes + self.coef_giou * loss_giou
                # Record positive logits
                if i == num_decoder_layers - 1:
                    self.record_positive_logits(logits_all[i].sigmoid(), indices)

        # Calculate average for all decoder layers
        loss /= num_decoder_layers
        for k, v in loss_dict.items():
            loss_dict[k] /= num_decoder_layers
        return loss, loss_dict


@torch.no_grad()
def post_process(pred_logits, pred_boxes, image_sizes, topk=100):
    assert len(pred_logits) == len(image_sizes)
    assert image_sizes.shape[1] == 2
    prob = pred_logits.sigmoid()
    prob = prob.view(pred_logits.shape[0], -1)
    topk_values, topk_indexes = torch.topk(prob, topk, dim=1)
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    # From relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(topk_values, labels, boxes)]
    return results


def get_pseudo_labels(pred_logits, pred_boxes, thresholds, nms_threshold=0.75):
    probs = pred_logits.sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    pseudo_labels = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        nms_idx = nms(box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
        scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
    return pseudo_labels
