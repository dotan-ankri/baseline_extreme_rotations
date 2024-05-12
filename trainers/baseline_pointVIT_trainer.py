from time import time
import os
import tqdm
import torch
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.loss_utils import *
from evaluation.evaluation_metrics import *
# from visualization_utils.visualize_pairs import visualize_pair,visualize_pair_w_orig
# from visualization_utils.visualize_panorama import visualize_panorama , visualize_panorama_colmap
# from visualization_utils import visualize_feature
##for vit
# from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
# import pandas as pd
import torchvision.transforms as T
import sys
# sys.path.append("../LoFTR/")
# sys.path.append("/storage/dotanankri/ExtremeRotation_code/LoFTR")
from copy import deepcopy
# t = time()
# from LoFTR.src.loftr import LoFTR, default_cfg
# loftr_time = time() - t
# print(f"\n\nLOFTR TIME: {loftr_time}\n\n")
import itertools
# from einops import rearrange
import cv2
from utils.compute_utils_PointVit import compute_gt_rmat_colmap_sift, qvec2rotmat
# from datasets.dataset_utils import get_crop_square_and_resize_image
from PIL import Image
import pickle
from PointVit.rel_pose.get_pose import get_pose

# from PointVit.rel_pose import 

def center_crop(im, new_shape):
    width, height = im.size   # Get dimensions    
    left = (width - new_shape[1])/2
    top = (height - new_shape[0])/2
    right = (width + new_shape[1])/2
    bottom = (height + new_shape[0])/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


class pointVIT_Baseline():
    def __init__(self,focal:float=1.0,cx:float=0.5,cy:float=0.5
                            ,skew:float=0,mx:float=1,my:float=1,
                            ransac_thr:float=5.0):
        ## Camera intrinsic parameters
        # self.f = focal
        # self.cx, self.cy, self.skew = cx,cy,skew  
        # self.mx, self.my = mx,my 
        # INTRINSIC MATRIX
        self.intrinsic_matrix_1 = self.camera_intrinsic_matrix(focal,cx,cy,skew,mx,my)
        self.intrinsic_matrix_2 = self.camera_intrinsic_matrix(focal,cx,cy,skew,mx,my)
        # Ransac parameters
        self.ransac_thr = ransac_thr

    def get_matches(self, img1:np.ndarray,img2:np.ndarray):
        """Find matching point:
        1. using sift for kp extraction and description
        2. Usig FLANN matcher to mutch the point"""
                        
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return (src_pts, dst_pts)

    def camera_intrinsic_matrix(self, f:float=1.0,cx:float=0.5,cy:float=0.5
                                ,skew:float=0,mx:float=1,my:float=1)->np.array:
        """Build the intrinsic matrix of a camera
        Input:
            f:      focal length [meters]
            cx, cy: principle point [pixels] 
            skew:   skew coefficient between the x and the y axis
            mx, my: inverses of the width and height of a pixel on 
                    the projection plane [1/meters]    
        """
        fx, fy = f*mx,f*my  # Default focal length
        
        K = np.array([[fx, skew, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        
        return K
    def evaluate_rotation_mat_old(self, src_pts:np.ndarray, dst_pts:np.ndarray)->np.array:
        n = min(len(src_pts),len(dst_pts))
        m = min(n,30)
        if n >= 4:
            M, mask = cv2.findHomography(src_pts[:m], dst_pts[:m], cv2.RANSAC, self.ransac_thr)        
            if M is None:
                return np.eye(3,3), None
            nsol, rotations, translations, normals = cv2.decomposeHomographyMat(M, self.intrinsic_matrix)
            return rotations, mask
        else:
            return np.eye(3,3), None
    def evaluate_rotation_mat(self, src_pts:np.ndarray, dst_pts:np.ndarray)->np.array:
        n = min(len(src_pts),len(dst_pts))
        m = min(n,30)        
        if n >= 4:
            # E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=focal, pp=pp, method=cv2.RANSAC)
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.RANSAC)
            if F is None:
                return np.eye(3,3), None
            # R1, R2, t = cv2.decomposeEssentialMat(E)
            K1 = self.intrinsic_matrix_1
            K2 = self.intrinsic_matrix_2
            E = K2.transpose()@F[:3,:3]@K1
            nof_points, R, t, mask2  = cv2.recoverPose(E, src_pts[mask==1], dst_pts[mask==1])
            return R, mask
        else:
            return np.eye(3,3), None
    def get_crop_square_and_resize_image(self, img_raw, img_size=256, df=8, padding=True):
    
        if isinstance(img_raw, np.ndarray):
            img_raw = Image.fromarray(img_raw)

        w, h = img_raw.size
        new_size = min(w, h)

        transform = T.Compose([
            T.CenterCrop((new_size, new_size)),
            # T.Resize((img_size, img_size), antialias=True)
        ])

        image = transform(img_raw)
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
        return image_cv2
    
    def change_and_set_intrinsic(self,img1,img2,fl1,fl2):        
        h1, w1,_ = img1.shape
        h2, w2,_ = img2.shape
        img1, img2 = Image.fromarray(img1), Image.fromarray(img2)
        # fl_factor = fl1/fl2
        # if fl_factor > 1:
        #     h2 *= fl_factor
        #     w2 *= fl_factor
        #     img2 = img2.resize((int(np.round(h2)),int(np.round(w2))))            
        # else:
        #     h1 *= 1/fl_factor
        #     w1 *= 1/fl_factor
        #     img1 = img1.resize((int(np.round(h1)),int(np.round(w1))))  
        h, w = min(h1,h2), min(w1,w2)
        # img1 = np.array(center_crop(img1,(h, w)))
        # img2 = np.array(center_crop(img2,(h, w)))
        cx, cy = w//2, h//2
        self.intrinsic_matrix_1 = [fl1,fl1,cx,cy]
        self.intrinsic_matrix_2 = self.intrinsic_matrix_1
        return np.array(img1), np.array(img2)
    def change_and_set_intrinsic_old(self,img1,img2,fl1,fl2):        
        h1, w1,_ = img1.shape
        h2, w2,_ = img2.shape
        img1, img2 = Image.fromarray(img1), Image.fromarray(img2)
        fl_factor = fl1/fl2
        if fl_factor > 1:
            h2 *= fl_factor
            w2 *= fl_factor
            img2 = img2.resize((int(np.round(h2)),int(np.round(w2))))            
        else:
            h1 *= 1/fl_factor
            w1 *= 1/fl_factor
            img1 = img1.resize((int(np.round(h1)),int(np.round(w1))))  
        h, w = min(h1,h2), min(w1,w2)
        img1 = np.array(center_crop(img1,(h, w)))
        img2 = np.array(center_crop(img2,(h, w)))
        cx, cy = w//2, h//2
        self.intrinsic_matrix_1 = [fl1,fl1,cx,cy]
        self.intrinsic_matrix_2 = self.intrinsic_matrix_1
        return img1, img2
    def get_batch_rmat(self, imgs1,imgs2,data_full):
        fl1 = data_full['fl1']
        fl2 = data_full['fl2']
        rotations = []
        angle_x, angle_y, angle_z = [],[],[]
        for ind,(img1, img2) in enumerate(zip(imgs1,imgs2)):
            img1, img2 = self.change_and_set_intrinsic(img1,img2,fl1[0][ind].item(),fl2[0][ind].item())
            # change to cv2
            img1 =cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 =cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            rmat = get_pose(img1, img2,self.intrinsic_matrix_1,self.intrinsic_matrix_2, interior=False)
            rotations.append(rmat)
            # angle_x.append(ax)
            # angle_y.append(ay)
            # angle_z.append(az)
        return torch.tensor(np.array(rotations))        
    def get_eular_angles_from_rot(self, rot_mat):
        if isinstance(rot_mat,list):
            rot_mat = torch.tensor(rot_mat)
        elif not torch.is_tensor(rot_mat):
            rot_mat = torch.tensor(rot_mat).unsqueeze(axis=0)

        ax, ay, az = compute_euler_angles_from_rotation_matrices(rot_mat)
        return np.rad2deg(ax.cpu().numpy()), np.rad2deg(ay.cpu().numpy()), np.rad2deg(az.cpu().numpy())
    def get_pos(self, img1, img2, intirior=False):        
        pass

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        self.baseline = pointVIT_Baseline()

        # dn_lib = importlib.import_module(cfg.models.rotationnet.type)
        # self.rotation_net = dn_lib.RotationNet(cfg.models.rotationnet)
        # self.rotation_net.cuda()
        # print("rotationnet:")
        # print(self.rotation_net)

        # # The optimizer
        # if not hasattr(self.cfg.trainer, "opt_dn"):
        #     self.cfg.trainer.opt_dn = self.cfg.trainer.opt

        # if getattr(self.cfg.trainer.opt_dn, "scheduler", None) is not None:
        #     self.opt_dn, self.scheduler_dn = get_opt(
        #         list(self.rotation_net.parameters()), self.cfg.trainer.opt_dn)
        # else:
        #     self.opt_dn = get_opt(
        #         list(self.rotation_net.parameters()), self.cfg.trainer.opt_dn)
        #     self.scheduler_dn = None
        
        self.classification = getattr(self.cfg.trainer, "classification", True)
        self.pairwise_type = getattr(self.cfg.trainer, "pairwise_type", "concat")
        self.rotation_parameterization = getattr(self.cfg.trainer, "rotation_parameterization", True)
        self.seg_labels_path = getattr(self.cfg.trainer, "seg_labels_path", None)
        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        ###
        # self.data_type = getattr(self.cfg.data,"data_type","panorama")
        # matcher = LoFTR(config=default_cfg)
        # matcher.load_state_dict(torch.load("/storage/hanibezalel/LoFTR/pretrained/outdoor_ds.ckpt")['state_dict'])
        # self.matcher = matcher.eval().cuda()
        
        # self.segmodel = eval('SegFormer')(
        #     backbone='MiT-B3',
        #     num_classes=150
        # )

        # try:
        #     self.segmodel.load_state_dict(torch.load('/storage/hanibezalel/semantic-segmentation/checkpoints/pretrained/segformer/segformer.b3.ade.pth', map_location='cpu'))
        # except:
        #     print("Download a pretrained model's weights from the result table.")
        # self.segmodel.to('cuda')
        # self.segmodel.eval()

        # print('Loaded Model')
        # self.palette = eval('ADE20K').PALETTE
        # self.labels = eval('ADE20K').CLASSES
        # if self.seg_labels_path:
        #     self.new_labels = old_to_new_labels(self.seg_labels_path)
        #     std = 2.2
        #     new_keys = np.linspace(-std, std, len(self.new_labels.keys()))
        #     self.changed_labels = {old_key : new_key for old_key, new_key in zip(self.new_labels.keys(),new_keys)}
            
        self.feat_wo_transient = getattr(self.cfg.trainer, "feat_wo_transient", True)
        self.normalization = getattr(self.cfg.trainer, "normalization", False)
        self.randomization = getattr(self.cfg.trainer, "randomization", 1.)
        #valid_classes = ["sky","streetlight","road","sidewalk","building"]
        #valid_labels = [np.where(np.asarray(self.labels)==v_class)[0][0] for v_class in valid_classes]
        #valid_classes_dict = dict(zip(valid_classes, valid_labels))
        #self.valid_labels = valid_classes_dict
        #self.road_like_labels = [valid_classes_dict["road"],valid_classes_dict["sidewalk"]]
        ###
 
    def epoch_end(self, epoch, writer=None):
        if self.scheduler_dn is not None:
            self.scheduler_dn.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dn_lr', self.scheduler_dn.get_lr()[0], epoch)
                
    def post_process_seg_map(self,seg,height_feat):
        seg = seg.softmax(1).argmax(1).to(int).unsqueeze(1)
        seg_target = torch.zeros_like(seg).to(torch.float32)
        if self.seg_labels_path:
            for key in self.new_labels.keys():
                mask = torch.isin(seg, torch.tensor(self.new_labels[key]).cuda())
                seg_target[mask] = key
        else:
            seg_target = seg
        unfold = torch.nn.Unfold(kernel_size=(8, 8),stride=(8, 8))
        output = unfold(seg_target.to(torch.float32))
        max_values, _ = torch.mode(output, dim=1)
        out = rearrange(max_values, 'b (h w) -> b h w', h= int(seg.shape[2]/8))
        out = out.unsqueeze(1)
        #mask_valid = torch.isin(seg, torch.tensor(list(self.valid_labels.values())).cuda())
        #seg = torch.where(mask_valid, seg, torch.zeros_like(seg).cuda())
        #mask_road = torch.isin(seg, torch.tensor(self.road_like_labels).cuda())
        #seg = torch.where(mask_road, torch.ones_like(seg)*self.valid_labels["road"],seg)
        #from_values = [self.valid_labels["road"], self.valid_labels["streetlight"]]
        #to_values = [3, 4]
        #for from_val, to_val in zip(from_values, to_values):
        #    seg = torch.where(seg == from_val, torch.tensor(to_val), seg)
        #seg_new = transforms.functional.resize(seg,(height_feat,height_feat),antialias=True,interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        return out
    
    #def post_process_seg_map_old(self,seg,height_feat):
    #    seg = seg.softmax(1).argmax(1).to(int).unsqueeze(1)
    #    mask_valid = torch.isin(seg, torch.tensor(list(self.valid_labels.values())).cuda())
    #    seg = torch.where(mask_valid, seg, torch.zeros_like(seg).cuda())
    #    mask_road = torch.isin(seg, torch.tensor(self.road_like_labels).cuda())
    #    seg = torch.where(mask_road, torch.ones_like(seg)*self.valid_labels["road"],seg)
    #    from_values = [self.valid_labels["road"], self.valid_labels["streetlight"]]
    #    to_values = [3, 4]
    #    for from_val, to_val in zip(from_values, to_values):
    #        seg = torch.where(seg == from_val, torch.tensor(to_val), seg)
    #    #unfold = torch.nn.Unfold(kernel_size=(8, 8),stride=(8, 8))
    #    #temp = unfold(seg.to(torch.float32))
    #    #max_values, _ = torch.mode(temp, dim=0)
    #    #seg_new = max_values.reshape(-1,32,32).to(torch.int32)
    #    seg_new = transforms.functional.resize(seg,(height_feat,height_feat),antialias=True,interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    #    return seg_new
     
    def create_masks_from_keypoints(self,kp0,kp1,confidence,batch_indexes,batch_size,feat_size,scale = 8.0):
        result = np.zeros((batch_size,2,2*feat_size,feat_size))
        for i in range(batch_size):
            rel_kp = batch_indexes== i
            kp0_i = kp0[rel_kp].cpu().numpy()
            kp1_i = kp1[rel_kp].cpu().numpy()
            conf_i = confidence[rel_kp].cpu().numpy()
            if np.sum(conf_i>0.8) == 0:
                matches = None
            else:
                F, mask = cv2.findFundamentalMat(kp0_i[conf_i>0.8],kp1_i[conf_i>0.8],cv2.FM_RANSAC, 1, 0.99)
                if mask is None or F is None:
                    matches = None
                else: 
                    matches = np.array(np.ones((kp0_i.shape[0], 2)) * np.arange(kp0_i.shape[0]).reshape(-1,1)).astype(int)[conf_i>0.8][mask.ravel()==1]
            kp_mask_0,kp_mask_1 = np.zeros((1,feat_size,feat_size)),np.zeros((1,feat_size,feat_size))
            kp_mask_0[0,(kp0_i[:,1]/scale).astype(int),(kp0_i[:,0]/scale).astype(int)] = 1
            kp_mask_1[0,(kp1_i[:,1]/scale).astype(int),(kp1_i[:,0]/scale).astype(int)] = 1
            kp_mask =  np.concatenate((kp_mask_0,kp_mask_1),axis=1)

            matches_mask_0,matches_mask_1 = np.zeros((1,feat_size,feat_size)),np.zeros((1,feat_size,feat_size))
            if matches is not None:
                matches_mask_0[0,(kp0_i[matches,1]/scale).astype(int),(kp0_i[matches,0]/scale).astype(int)] = 1
                matches_mask_1[0,(kp1_i[matches,1]/scale).astype(int),(kp1_i[matches,0]/scale).astype(int)] = 1
            matches_mask =  np.concatenate((matches_mask_0,matches_mask_1),axis=1)
            result[i] = np.concatenate((kp_mask,matches_mask),axis=0)
        return result   

    def update(self, data_full, no_update=False,data_type = "colmap"):
        #img1 = data_full['img1'].cuda()
        #img2 = data_full['img2'].cuda()
        ###
        if data_type == "colmap":
                q1 = data_full['q1']
                q2 = data_full['q2']
        else:
            rotation_x1 = data_full['rotation_x1']
            rotation_y1 = data_full['rotation_y1']
            rotation_x2 = data_full['rotation_x2']
            rotation_y2 = data_full['rotation_y2']
            
        if 'mask1' in data_full:
            batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda(), 'mask0': data_full['mask1'].cuda(), 'mask1': data_full['mask2'].cuda()}
        else:
            batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda()}
            
        with torch.no_grad():
            self.matcher(batch)
            
        image_feature_map1 = batch['feat_c0']
        image_feature_map2 = batch['feat_c1']
        height_feat = int(batch['image0'].shape[2]/8)
        image_feature_map1= rearrange(image_feature_map1, 'b (h w) c -> b c h w', h= height_feat)
        image_feature_map2= rearrange(image_feature_map2, 'b (h w) c -> b c h w', h= height_feat)
        
        kp0 = batch['mkpts0_f']
        kp1 = batch['mkpts1_f']
        confidence = batch['mconf']
        batch_indexes = batch['b_ids']
        mask = self.create_masks_from_keypoints(kp0,kp1,confidence,batch_indexes,batch['bs'],image_feature_map1.shape[2],scale = 8.0)
        
        with torch.no_grad():
            
            seg1 = self.segmodel(data_full['img1'].cuda())
            seg1 = self.post_process_seg_map(seg1,height_feat)
            seg2 = self.segmodel(data_full['img2'].cuda())
            seg2 = self.post_process_seg_map(seg2,height_feat)
        
        batch_size = batch['image0'].size(0)
        
        if self.feat_wo_transient:
            mask1 = torch.isin(seg1.expand(-1, 256,-1,-1), torch.tensor([5]).cuda())
            mask2 = torch.isin(seg2.expand(-1, 256,-1,-1), torch.tensor([5]).cuda())
            image_feature_map1[mask1] = 0.
            image_feature_map2[mask2] = 0.
            
        if self.randomization < 1.0:           
            random_masks = np.random.uniform(size=batch_size)<=self.randomization
            seg1[random_masks] = torch.zeros_like(seg1[0])
            seg2[random_masks] = torch.zeros_like(seg2[0])
            
        if self.normalization:
            for key in self.changed_labels.keys():
                mask1_changed = torch.isin(seg1, torch.tensor([key]).cuda())
                seg1[mask1_changed] = self.changed_labels[key]
                mask2_changed = torch.isin(seg2, torch.tensor([key]).cuda())
                seg2[mask2_changed] = self.changed_labels[key]
            
            
        
            
        if not no_update:
            self.rotation_net.float().train()
            self.opt_dn.zero_grad()

        ###
        if data_type == "colmap":
            gt_rmat = compute_gt_rmat_colmap(q1,q2,batch_size)
        else:
            gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)
        
        if self.rotation_parameterization and not data_type == "colmap":
            angle_x, angle_y, angle_z = compute_angle(rotation_x1, rotation_x2, rotation_y1, rotation_y2)
        else:
            angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(gt_rmat)
        
        input_img = torch.cat([image_feature_map1, image_feature_map2], dim=2)
        input_img = torch.cat([input_img, torch.from_numpy(mask).cuda()], dim=1)
        seg = torch.cat([seg1, seg2], dim=2)
        input_img = torch.cat([input_img, seg.cuda(),torch.zeros_like(seg).cuda()], dim=1)
        

        # loss type
        if not self.classification:
            # regression loss
            print("Not Implement!")
        else:
            # classification loss
            out_rotation_x, out_rotation_y, out_rotation_z = self.rotation_net(input_img.float())
            ###
            loss_x = rotation_loss_class(out_rotation_x, angle_x)
            loss_y = rotation_loss_class(out_rotation_y, angle_y)
            loss_z = rotation_loss_class(out_rotation_z, angle_z)

            loss = loss_x + loss_y + loss_z
            res1 = {"loss": loss, "loss_x": loss_x, "loss_y": loss_y, "loss_z": loss_z}
            #loss = loss_y
            #res1 = {"loss": loss, "loss_x": loss_y, "loss_y": loss_y, "loss_z": loss_y}
            ###


        if not no_update:
            loss.backward()
            self.opt_dn.step()
        else:
            self.opt_dn.zero_grad()
        train_info = {}
        train_info.update(res1)
        train_info.update({"loss": loss})
        return train_info

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False,train_info_second=None,trainers_names = None):
        if writer is not None:
            for k, v in train_info.items():
                if not ('loss' in k) and not ('Error' in k):
                    continue
                if step is not None:
                    if train_info_second is not None:
                        writer.add_scalars(k, {trainers_names[0] : v,
                                               trainers_names[1] : train_info_second[k]} , step)
                    else:
                        writer.add_scalar('train/' + k, v, step)
                else:
                    assert epoch is not None
                    writer.add_scalar('train/' + k, v, epoch)
                    
    def print_scene_error(self, error, gt_angle):
        res = {}
        mean = np.ma.mean(error)
        median = np.ma.median(error)
        error_max = np.ma.max(error)
        std = np.ma.std(error)
        count_10 = (error.compressed()<10).sum(axis=0)
        percent_10 = np.true_divide(count_10, error.compressed().shape[0])
        per_from_all =  np.true_divide(error.compressed().shape[0], error.shape[0])
        
        gt_mean = np.ma.mean(gt_angle)
        gt_median = np.ma.median(gt_angle)
        gt_max = np.ma.max(gt_angle)
        gt_std = np.ma.std(gt_angle)
        gt_count_10 = (gt_angle.compressed()<10).sum(axis=0)
        gt_percent_10 = np.true_divide(gt_count_10, gt_angle.compressed().shape[0])
        res.update({'error_mean': mean, 'error_median': median, 'error_max': error_max, 'error_std': std,
                    'error_10deg': percent_10,'per_from_all': per_from_all,
                    'gt_mean': gt_mean, 'gt_median': gt_median, 'gt_max': gt_max, 'gt_std': gt_std,
                    'gt_10deg': gt_percent_10})
        #print("Rotation Error : " ,res)
        return res
        
    def colmap_error_per_scene(self, scene_array,res_error):
        unique_scenes = set(scene_array)
        res_array=[]
        categories = ["rotation_geodesic_error","rotation_geodesic_error_overlap_large","rotation_geodesic_error_overlap_huge","rotation_geodesic_error_overlap_small","rotation_geodesic_error_overlap_none"]
        names = ["/all","/large","/huge","/small","/none"]
        
        for scene in unique_scenes:
            for category,name in zip(categories,names):
                if name=="/all":
                    mask = np.logical_not(np.asarray(scene_array)==scene) 
                else:
                    mask = np.logical_not(np.asarray(scene_array)==scene) | res_error[category].mask
                res_error_scene = np.ma.masked_array(res_error["rotation_geodesic_error"],mask )
                gt_angle_scene = np.ma.masked_array(res_error["gt_angle"], np.logical_not(np.asarray(scene_array)==scene))
                res = self.print_scene_error(res_error_scene,gt_angle_scene)
                res_array.append(res)
        df = pd.DataFrame(res_array,index= list(itertools.product(unique_scenes, names)))
        df.to_csv(os.path.join(self.cfg.save_dir,"colmap_error_per_scene.csv"), float_format='%.3f')
        return
    
    def save_pair(self, test_loader_id,id,error,out_dir,estimated_mat,gt_mat,gt_mat1,data_type,feature1,feature2,kp_mask,seg1,seg2):
        # if data_type=="panorama":
        #     visualize_pair(test_loader_id,str(id)+'_'+format(error,".2f"),out_dir)
        #     visualize_panorama(test_loader_id,str(id)+'_'+format(error,".2f"),out_dir,estimated_mat,gt_mat,gt_mat1)
        #     visualize_feature.visualize_features(feature1,feature2,id,out_dir)
        #     visualize_segmentation.visualize_seg(seg1,seg2,out_dir,id)
        #     visualize_kp_mask(kp_mask,id,out_dir) 
        # else:
        #     visualize_pair(test_loader_id,str(id)+'_'+test_loader_id['scene']+'_'+format(error,".2f"),out_dir)
        #     visualize_panorama_colmap(test_loader_id,str(id)+'_'+test_loader_id['scene']+'_'+format(error,".2f"),out_dir,estimated_mat,gt_mat,gt_mat1)
        #     visualize_feature.visualize_features(feature1,feature2,id,out_dir)      
        #     visualize_segmentation.visualize_seg(seg1,seg2,out_dir,id)
        #     visualize_kp_mask(kp_mask,id,out_dir) 
        return 
    
    def save_by_precentile(self,out_dir, test_loader, criterion, error, percentile, 
                         estimated_rot,gt_mat,gt_rot1=None, bigger=True,features1_array=None,features2_array=None,
                         kp_mask_array = None,seg1_array=None,seg2_array=None):
        
        mdata = np.ma.filled(criterion, np.nan)
        if bigger:
            idxs = np.ma.where(criterion>=np.nanpercentile(mdata, percentile))
        else:
            idxs = np.ma.where(criterion<=np.nanpercentile(mdata, percentile))
        for id in idxs[0]:
             self.save_pair(test_loader.dataset[id],id,error[id],out_dir,estimated_rot[id],gt_mat[id],
                            gt_rot1[id],test_loader.dataset.data_type,features1_array[id],features2_array[id]
                            ,kp_mask_array[id],seg1_array[id],seg2_array[id])
        return
    
    def save_good_and_bad_examples(self,out_dir, test_loader, criterion, error,estimated_rot,gt_mat,gt_rot1=None,
                                   features1_array=None,features2_array=None,kp_mask_array = None,
                                   seg1_array=None,seg2_array=None):
        bad_precent = 90
        good_precent = 100-bad_precent
        out_dir_bad = os.path.join(out_dir,"bad_images_"+str(bad_precent))
        out_dir_good = os.path.join(out_dir,"good_images_"+str(bad_precent))
        os.makedirs(out_dir_bad)
        os.makedirs(out_dir_good)
        self.save_by_precentile(out_dir_bad, test_loader, criterion, error, bad_precent, estimated_rot, gt_mat, gt_rot1, 
                                bigger=True,features1_array = features1_array,features2_array = features2_array,
                                kp_mask_array=kp_mask_array,seg1_array=seg1_array,seg2_array=seg2_array)
        self.save_by_precentile(out_dir_good, test_loader, criterion, error, good_precent, estimated_rot, gt_mat, gt_rot1, 
                               bigger=False,features1_array = features1_array,features2_array = features2_array,
                               kp_mask_array=kp_mask_array,seg1_array=seg1_array,seg2_array=seg2_array)
        return
    
    def find_yaw_pitch_roll_errors(self,estimated_rot,gt_mat,gt1_mat):
        r2_est = torch.bmm(estimated_rot,gt1_mat)
        #x_est,y_est = compute_viewpoint_from_rotation_matrix(r2_est,1000)
        r2_gt = torch.bmm(gt_mat,gt1_mat)
        #x_gt,y_gt = compute_viewpoint_from_rotation_matrix(r2_gt,1000)
        euler_gt = compute_euler_angles_from_rotation_matrices(r2_gt)
        euler_est = compute_euler_angles_from_rotation_matrices(r2_est)
        euler_diff = tuple(torch.abs(t1 - t2) for t1, t2 in zip(euler_gt, euler_est))
        #euler_diff = torch.abs(euler_gt-euler_est)
        return euler_diff[0],euler_diff[1],euler_diff[2]
    
    def save_gt_angle_vs_error(self,gt_angle,error,out_dir):
        plt.figure()
        plt.scatter(gt_angle,error)
        plt.title("GT angle Vs.Error angle")  
        plt.xlabel("GT angle")
        plt.ylabel("Error angle")
        plt.savefig(os.path.join(out_dir,'error_vs_gt_angle.png'))
        return
    
    def save_error_histograms(self,error,gt_angle, out_dir,estimated_rot,gt_mat,gt_rot1):
        plt.figure()
        mask = np.asarray([False]*len(error))
        if np.ma.is_masked(error):
            mask = error.mask
            error = error.compressed()
        if len(error)>0:
            plt.hist(error,bins=list(np.arange(0,np.max(error),10))) 
            plt.title("Error angle histogram.\n") 
            plt.savefig(os.path.join(out_dir,'geodesic_error_hist.png'))
            self.save_gt_angle_vs_error(gt_angle[~mask],error,out_dir)
            roll,pitch,yaw = self.find_yaw_pitch_roll_errors(estimated_rot,gt_mat,gt_rot1)
            error_euler = {"roll": roll, "pitch": pitch, "yaw": yaw}
            for angle in error_euler:
                plt.figure()
                plt.hist(np.degrees(error_euler[angle].cpu().numpy()),bins=list(np.arange(0,361,10))) 
                plt.title(angle + "_error") 
                plt.savefig(os.path.join(out_dir,angle +'_error_hist.png'))
        return
    
    def save_gt_histograms(self,gt_angle, out_dir,estimated_rot,gt_mat):
        plt.figure()
        plt.hist(gt_angle,bins=list(np.arange(0,61,5))) 
        plt.title("GT angle histogram.\n Number of pairs: " + str(gt_angle.shape[0]))  
        plt.savefig(os.path.join(out_dir,'gt_angle.png'))
        return
    
    def save_pictures(self, test_loader,epoch,res_error,estimated_rot,gt_mat,gt_rot1=None,
                      features1_array=None,features2_array=None,kp_mask_array = None,seg1_array=None,seg2_array=None):
        categories = ["rotation_geodesic_error_overlap_large","rotation_geodesic_error_overlap_huge","rotation_geodesic_error_overlap_small","rotation_geodesic_error_overlap_none"]
        for k in categories:
            out_dir = os.path.join(self.cfg.save_dir,k)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            #self.save_good_and_bad_examples(out_dir, test_loader,res_error[k], res_error["rotation_geodesic_error"],
            #                                estimated_rot,gt_mat,gt_rot1,
            #                                features1_array=features1_array,features2_array=features2_array,
            #                                kp_mask_array=kp_mask_array,seg1_array=seg1_array,seg2_array=seg2_array)
            self.save_error_histograms(res_error[k],res_error["gt_angle"],out_dir,estimated_rot,gt_mat,gt_rot1)
        self.save_gt_histograms(res_error["gt_angle"], self.cfg.save_dir,estimated_rot,gt_mat)
        return

    def validate(self, test_loader, epoch, val_angle=False,save_pictures=False):
        print("Validation")
        out_rmat_array = None
        gt_rmat_array = None
        gt_rmat1_array = None
        out_rmat1_array = None
        gt_r1_array = None
        features1_array = None
        features2_array = None
        kp_mask_array = None
        overlap_amount_array = None
        scene_array = None 
        seg1_array= None
        seg2_array= None
        all_res = {}

        # self.test_sift_rot()

        # count = 0
        
        batch_ind = 0
        if os.path.isfile('test_8pointVIT_rot/outputs/debug_outrot.pkl') and True:
            try:
                with open('test_8pointVIT_rot/outputs/debug_outrot.pkl','rb') as file:
                    last_dict = pickle.load( file)
                batch_ind =last_dict['batch_ind']+1
                out_rmat_array = last_dict['our_rmat']
                gt_rmat_array = last_dict['gt_rmat']
                overlap_amount_array = last_dict['overlap_amount_array']
                
            except:
                batch_ind = 0
                pass
        for ind, data_full in enumerate(tqdm.tqdm(test_loader)):
            if ind >10:
                break
            if ind<batch_ind:
                continue            
            # count += 1
            # if count == 15:
            #     break
                        
            img1 =self. load_imag_batch(data_full['path'],list((data_full['image_height']).numpy()))
            img2 =self. load_imag_batch(data_full['path2'],list((data_full['image_height']).numpy()))            
            ###
            if test_loader.dataset.data_type == "colmap":
                q1 = data_full['q1']
                q2 = data_full['q2']
            else:
                rotation_x1 = data_full['rotation_x1']
                rotation_y1 = data_full['rotation_y1']
                rotation_x2 = data_full['rotation_x2']
                rotation_y2 = data_full['rotation_y2']
            
            batch_size = len(data_full['path'])
            if test_loader.dataset.data_type == "colmap":
                gt_rmat =compute_gt_rmat_colmap_sift(q1,q2,batch_size)
                
                overlap_amount = data_full['overlap_amount']
                if overlap_amount_array is None:
                    overlap_amount_array = overlap_amount
                else:
                    overlap_amount_array = overlap_amount_array + overlap_amount
                    
                scene = data_full['scene']
                if scene_array is None:
                    scene_array = scene
                else:
                    scene_array = scene_array + scene
            else:
                gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)


        
            out_rmat = self.baseline.get_batch_rmat(img1,img2,data_full)
            out_rmat1 = None

            # if self.rotation_parameterization:
            #     out_rmat, out_rmat1 = compute_out_rmat(out_rotation_x, out_rotation_y, out_rotation_z, batch_size)
            # else:                        
            #     out_rmat = compute_rotation_matrix_from_euler_angle(out_rotation_x.float(), out_rotation_y.float(), out_rotation_z.float(), batch_size)
            #     out_rmat1 = None
            
            if gt_rmat_array is None:
                gt_rmat_array = gt_rmat
            else:
                gt_rmat_array = torch.cat((gt_rmat_array, gt_rmat))
            if out_rmat_array is None:
                out_rmat_array = out_rmat
            else:
                out_rmat_array = torch.cat((out_rmat_array, out_rmat))
            ###
            if save_pictures:
                gt_rmat1 = None
                if test_loader.dataset.data_type == "panorama":
                    gt_rmat1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3).cuda()
                else:
                    gt_rmat1 = qvec2rotmat(q1,batch_size).view(batch_size, 3, 3).cuda()
                if gt_r1_array is None:
                    gt_r1_array = gt_rmat1
                else:
                    gt_r1_array = torch.cat((gt_r1_array, gt_rmat1))
                    
                
            ###
            if val_angle:            
                gt_rmat1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3).cuda()
                if gt_rmat1_array is None:
                    gt_rmat1_array = gt_rmat1
                else:
                    gt_rmat1_array = torch.cat((gt_rmat1_array, gt_rmat1))
                if out_rmat1_array is None:
                    out_rmat1_array = out_rmat1
                else:
                    out_rmat1_array = torch.cat((out_rmat1_array, out_rmat1))
        
        
            
            with open('test_8pointVIT_rot/outputs/debug_outrot.pkl','wb') as file:
                pickle.dump({'our_rmat':out_rmat_array,'gt_rmat':gt_rmat_array,'batch_ind':ind,
                             'overlap_amount_array':overlap_amount_array},
                 file)
        
        
        # with open('debug_outrot.pkl','rb') as file:
        #     out_rmat_array,gt_rmat_array =pickle.load( file)
        out_rmat_array = out_rmat_array.cuda()
        if overlap_amount_array is None:
            res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array)
        else:
            res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array,overlap_amount_array)
        if val_angle:
            angle_error = evaluation_metric_rotation_angle(out_rmat_array, gt_rmat_array, gt_rmat1_array, out_rmat1_array)
            res_error.update(angle_error)

        # mean, median, max, std, 10deg
        for k, v in res_error.items():
            if v.size == 0:
                continue
            if np.ma.is_masked(v) and v.mask.all():
                mean, median, error_max, std, percent_10, per_from_all = 0, 0, 0, 0, 0, 0
            else:
                mean = np.ma.mean(v)
                median = np.ma.median(v)
                error_max = np.ma.max(v)
                std = np.ma.std(v)
                count_10 = (v<10).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<10).sum(axis=0)
                percent_10 = np.true_divide(count_10, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_10, v.compressed().shape[0])
                per_from_all = np.true_divide(v.shape[0], res_error["gt_angle"].shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(v.compressed().shape[0], res_error["gt_angle"].shape[0])
            all_res.update({k + '/mean': mean, k + '/median': median, k + '/max': error_max, k + '/std': std,
                            k + '/10deg': percent_10,k + '/per_from_all': per_from_all})
        print("Validation Epoch:%d " % epoch, all_res)
        if save_pictures:
            if test_loader.dataset.data_type == "colmap":
                self.colmap_error_per_scene(scene_array,res_error)
            self.save_pictures(test_loader,epoch,res_error,out_rmat_array,gt_rmat_array,gt_r1_array,features1_array,features2_array,kp_mask_array,seg1_array,seg2_array)
            
                
        print("Validation Epoch:%d " % epoch, all_res)
        self.write_dict_to_file(all_res,'test_8pointVIT_rot/debug/all_res.txt')
        return all_res

    def test_sift_rot(self):
        with open('test_sift_rot/img1.png', 'rb') as f:
                img1 = Image.open(f)
                img1 = img1.convert('RGB')
                img1 = np.array(img1)
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        with open('test_sift_rot/img2.png', 'rb') as f:
                img2 = Image.open(f)
                img2 = img2.convert('RGB')
                img2 = np.array(img2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        with open('test_sift_rot/gt_rot.pkl','rb') as file:
            gt_rot = pickle.load(file)
        # ax, ay, az = compute_euler_angles_from_rotation_matrices(gt_rot)
        # ax, ay, az = np.rad2deg(ax[0].cpu().numpy()), np.rad2deg(ay[0].cpu().numpy()), np.rad2deg(az[0].cpu().numpy())

        baseline = pointVIT_Baseline()
        ax, ay, az = baseline.get_eular_angles_from_rot(gt_rot)
        ax_gt, ay_gt, az_gt = ax[0], ay[0], az[0] 
        
        src_pts, dst_pts = baseline.get_matches(img1,img2)
        rmat, mask = baseline.evaluate_rotation_mat(src_pts, dst_pts)
        src_pts_filtered = src_pts[:len(mask)]
        dst_pts_filtered = dst_pts[:len(mask)]
        src_pts_filtered = src_pts_filtered[mask.squeeze()==1,:,:]    
        dst_pts_filtered = dst_pts_filtered[mask.squeeze()==1,:,:]
        ax, ay, az = baseline.get_eular_angles_from_rot(rmat)
        # ax, ay, az = ax[0], ay[0], az[0] 
        
        print(f'ax_gt: {ax_gt}\tax: {ax}')
        print(f'ay_gt: {ay_gt}\tay: {ay}')
        print(f'az_gt: {az_gt}\taz: {az}')
        # plot kp on images
        out_path = 'test_sift_rot/outputs/'
        plt.figure()
        plt.scatter(src_pts_filtered[:5,0,0], src_pts_filtered[:5,0,1], c=['red','green','blue','yellow','purple'], marker='o', label='Points')                
        plt.imshow(img1[:,:,::-1])
        plt.title('src image')
        plt.savefig(os.path.join(out_path,'src_image_kp.png'))
        plt.figure()
        plt.scatter(dst_pts_filtered[:5,0,0], dst_pts_filtered[:5,0,1], c=['red','green','blue','yellow','purple'], marker='o', label='Points')                
        plt.imshow(img2[:,:,::-1])        
        plt.title('dst image')
        plt.savefig(os.path.join(out_path,'dst_image_kp.png'))
        

        print(f'out rmat :\n {rmat}')
        print(f'gt rmat :\n {gt_rot}')



        
    def log_val(self, val_info, writer=None, step=None, epoch=None,val_info_second=None,tests_names = None):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    if 'vis' in k:
                        writer.add_image(k, v, step)
                    else:
                        if val_info_second is not None:
                            writer.add_scalars(k, {tests_names[0] : v,
                                                   tests_names[1] : val_info_second[k]} , step)
                        else:
                            writer.add_scalar(k, v, step)
                else:
                    if 'vis' in k:
                        writer.add_image(k, v, epoch)
                    else:
                        if val_info_second is not None:
                            writer.add_scalars(k, {tests_names[0] : v,
                                                   tests_names[1] : val_info_second[k]} , step)
                        else:
                            writer.add_scalar(k, v, step)

    def save(self, epoch=None, step=None, appendix=None):
        d = {
            'opt_dn': self.opt_dn.state_dict(),
            'dn': self.rotation_net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        #save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        save_name = "epoch_%s.pt" % (epoch)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
        remove_name = "epoch_%s.pt" % (epoch-1)
        remove_path = os.path.join(self.cfg.save_dir, "checkpoints", remove_name)
        #if os.path.exists(remove_path) and (epoch-1)!=9:
        #    os.remove(remove_path)

    def resume(self, path, strict=True, resume_encoder=False, test=False, **args):
        ckpt = torch.load(path)
        if not resume_encoder:
            self.rotation_net.load_state_dict(ckpt['dn'], strict=strict)
            if False:
                self.opt_dn.load_state_dict(ckpt['opt_dn'])
            start_epoch = ckpt['epoch']
        else:
            start_epoch = 0
        return start_epoch

    def test(self, opt, *arg, **kwargs):
        raise NotImplementedError("Trainer [test] not implemented.")
    
    def load_imag_batch(self,path_list,image_height):
        images = []
        for path, height in zip(path_list,image_height):
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            img = self.baseline.get_crop_square_and_resize_image(img, img_size=height)
            images.append(img)
        return images
    def write_dict_to_file(self, dictionary, file_path):
        with open(file_path, 'w') as file:
            for key, value in dictionary.items():
                file.write(f'{key} = {value}\n')
    
    def start_data_loader_from_batch(self,loader, start_batch):
        # Manually skip the first N batches
        loader = iter(loader)
        for _ in range(start_batch):
            try:
                next(loader)
            except StopIteration:
                # Handle the case where the DataLoader is exhausted
                print("DataLoader is exhausted.")
                break

        return loader

