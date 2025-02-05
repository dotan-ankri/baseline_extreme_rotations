from time import time
import os
import tqdm
import torch
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.loss_utils import *
from evaluation.evaluation_metrics import *

##for vit
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import torchvision.transforms as T
import sys
sys.path.append("../LoFTR/")
sys.path.append("/ExtremeRotation_code/LoFTR")
from copy import deepcopy
t = time()
from LoFTR.src.loftr import LoFTR, default_cfg
loftr_time = time() - t
print(f"\n\nLOFTR TIME: {loftr_time}\n\n")
import itertools
import cv2
from utils.compute_utils_sift import compute_gt_rmat_colmap_sift, qvec2rotmat
# from datasets.dataset_utils import get_crop_square_and_resize_image
from PIL import Image
import pickle
from einops.einops import rearrange

class Loftr_Baseline():
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
            mask = mask.squeeze()
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
    
    def get_batch_rmat(self, kp0,kp1,original_dim0,original_dim1,confidence,batch_indexes,data_full):
        fl1 = data_full['fl1']
        fl2 = data_full['fl2']
        rotations = []
        angle_x, angle_y, angle_z = [],[],[]
        batch_size = data_full['grayimg1'].shape[0]
        for ind in range(batch_size):
            rel_kp = batch_indexes== ind
            kp0_i = kp0[rel_kp].cpu().numpy()
            kp1_i = kp1[rel_kp].cpu().numpy()
            conf_i = confidence[rel_kp].cpu().numpy()
            h0, w0, resize_factor0 = original_dim0[ind]['h'], original_dim0[ind]['w'], original_dim0[ind]['resize_factor']
            h1, w1, resize_factor1 = original_dim1[ind]['h'], original_dim1[ind]['w'], original_dim1[ind]['resize_factor']
            
            cx, cy = w0//2, h0//2
            self.intrinsic_matrix_1 = self.camera_intrinsic_matrix(f=fl1[0][ind].item(),cx=cx,cy=cy)

            cx, cy = w1//2, h1//2
            self.intrinsic_matrix_2 = self.camera_intrinsic_matrix(f=fl2[0][ind].item(),cx=cx,cy=cy)

            rmat, map2 = self.evaluate_rotation_mat(resize_factor0*kp0_i[conf_i>0.9],resize_factor1*kp1_i[conf_i>0.9])
            rotations.append(rmat)
            # angle_x.append(ax)
            # angle_y.append(ay)
            # angle_z.append(az)
        return torch.tensor(np.array(rotations))        
    
    
    def plot_kp(im1,im2,kp1,kp2,nof_lines):
        nof_kp = kp1.shape[0]
        h1,w1 = im1.shape
        full_image = np.concatenate((im1,im2),axis=1)
        # Assuming kp = [rows, cols]
        # Assuming kp = [cols, rows]
        kp2[:,0] += w1 
        plt.imshow(full_image,cmap='gray')
        indices = np.random.choice(np.arange(nof_kp), size=nof_lines, replace=False)
        for idx in indices:
            plt.plot([kp1[idx,0],kp2[idx,0]], [kp1[idx,1],kp2[idx,1]],color='red')
        plt.savefig('/storage/dotanankri/ExtremeRotation_code/test_loftr_rot/debug/kp_matching.png')        

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        self.loftr_baseline = Loftr_Baseline()

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
        self.data_type = getattr(self.cfg.data,"data_type","panorama")
        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(torch.load("LoFTR/outdoor_ds.ckpt")['state_dict'])
        self.matcher = matcher.eval().cuda()
        
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

                
        batch_ind = 0
        if os.path.isfile('mid_run/LoFTR/outputs/debug_outrot_loftr.pkl') and True:
            try:
                with open('mid_run/LoFTR/outputs/debug_outrot_loftr.pkl','rb') as file:
                    last_dict = pickle.load( file)
                batch_ind =last_dict['batch_ind']+1
                out_rmat_array = last_dict['our_rmat']
                gt_rmat_array = last_dict['gt_rmat']
                overlap_amount_array = last_dict['overlap_amount_array']
                
            except:
                batch_ind = 0
                pass
        pairs_path = []
        for ind, data_full in enumerate(tqdm.tqdm(test_loader)):
            if ind<batch_ind:
                continue            
            # count += 1
            # if count == 15:
            #     break                        
            img1 =self. load_imag_batch(data_full['path'],list((data_full['image_height']).numpy()))
            img2 =self. load_imag_batch(data_full['path2'],list((data_full['image_height']).numpy()))
            
            # Extracting the Resize factor
            original_dim0 = []
            original_dim1 = []
            for ind, (im0,im1) in enumerate(zip(img1,img2)):
                new_size0 =data_full['grayimg1'][ind].size(-1)
                h0, w0 = im0.shape[:-1]
                old_size0 = min([h0, w0])
                original_dim0.append({'resize_factor':old_size0/new_size0,
                                      'h':h0,'w':w0})

                new_size1 =data_full['grayimg2'][ind].size(-1)
                h1, w1 = im1.shape[:-1]
                old_size1 = min([h1, w1])
                original_dim1.append({'resize_factor':old_size1/new_size1,
                                      'h':h1,'w':w1}) 

            if 'mask1' in data_full:
                batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda(), 'mask0': data_full['mask1'].cuda(), 'mask1': data_full['mask2'].cuda()}
            else:
                batch = {'image0':  data_full['grayimg1'].cuda(), 'image1': data_full['grayimg2'].cuda()}
                
            

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

            out_rmat = self.loftr_baseline.get_batch_rmat(kp0,kp1,original_dim0,original_dim1,confidence,batch_indexes,data_full)
            out_rmat1 = None

            #list of all the pairs scene and image path, for analisys purpose 
            pairs_path += [(scene,path1, path2) for scene, path1, path2 in zip(data_full['scene'],data_full['path'],data_full['path2'])]

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
        
        
            
            with open('mid_run/LoFTR/outputs/debug_outrot_loftr.pkl','wb') as file:
                pickle.dump({'our_rmat':out_rmat_array,'gt_rmat':gt_rmat_array,'batch_ind':ind,
                             'overlap_amount_array':overlap_amount_array,
                             'pairs_path':pairs_path},
                 file)
        
        
        out_rmat_array = out_rmat_array.cuda()
        if overlap_amount_array is None:
            res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array)
        else:
            res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array,overlap_amount_array)
        if val_angle:
            angle_error = evaluation_metric_rotation_angle(out_rmat_array, gt_rmat_array, gt_rmat1_array, out_rmat1_array)
            res_error.update(angle_error)

        # # save N best examples 
        # best_overlap_huge_idx = np.argmin(res_error['rotation_geodesic_error_overlap_huge'])
        # best_overlap_large_idx = np.argmin(res_error['rotation_geodesic_error_overlap_large'])
        # best_overlap_small_idx = np.argmin(res_error['rotation_geodesic_error_overlap_small'])
        # best_overlap_none_small_idx = np.argmin(res_error['rotation_geodesic_error_overlap_none_small'])
        # best_overlap_large_idx = np.argmin(res_error['rotation_geodesic_error_overlap_none_large'])
        # overlap_huge_example = {'geodesic_error':res_error['rotation_geodesic_error'][best_overlap_huge_idx],
        #                         'gt_rmat':res_error['gt_rmat'][best_overlap_huge_idx],
        #                         'out_rmat':res_error['gt_rmat'][best_overlap_huge_idx]}
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
                count_15 = (v<15).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<15).sum(axis=0)
                percent_15 = np.true_divide(count_15, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_15, v.compressed().shape[0])
                count_30 = (v<30).sum(axis=0) if (k=='rotation_geodesic_error' or k=='gt_angle') else (v.compressed()<30).sum(axis=0)
                percent_30 = np.true_divide(count_30, v.shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(count_30, v.compressed().shape[0])
                per_from_all = np.true_divide(v.shape[0], res_error["gt_angle"].shape[0]) if (k=='rotation_geodesic_error' or k=='gt_angle') else np.true_divide(v.compressed().shape[0], res_error["gt_angle"].shape[0])
            all_res.update({k + '/mean': mean, k + '/median': median, k + '/max': error_max, k + '/std': std,
                            k + '/15deg': percent_15,k + '/30deg': percent_30,k + '/per_from_all': per_from_all})
        print("Validation Epoch:%d " % epoch, all_res)
        if save_pictures:
            if test_loader.dataset.data_type == "colmap":
                self.colmap_error_per_scene(scene_array,res_error)
            self.save_pictures(test_loader,epoch,res_error,out_rmat_array,gt_rmat_array,gt_r1_array,features1_array,features2_array,kp_mask_array,seg1_array,seg2_array)
            
                
        print("Validation Epoch:%d " % epoch, all_res)
        self.write_dict_to_file(all_res,'mid_run/LoFTR/outputs/all_res_loftr.txt')
        return all_res

    def load_imag_batch(self,path_list,image_height):
        images = []
        for path, height in zip(path_list,image_height):
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            img = self.loftr_baseline.get_crop_square_and_resize_image(img, img_size=height)
            images.append(img)
        return images
    def write_dict_to_file(self, dictionary, file_path):
        with open(file_path, 'w') as file:
            for key, value in dictionary.items():
                file.write(f'{key} = {value}\n')
    
    

