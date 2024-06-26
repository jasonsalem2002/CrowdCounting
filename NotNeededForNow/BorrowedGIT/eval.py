#This code is for model evaluation
#Import packages
import os
import glob
import argparse
import pathlib
import h5py
from tqdm import tqdm
from torchvision import transforms
import PIL.Image as Image
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import torch 
from model import MSPSNet 
from sklearn.metrics import auc
from math import sqrt
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


#Argument parser
path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')
parser.add_argument('--model_desc', default='SHB/', help="Set model description")
parser.add_argument('--dataset_path', default='C:/Users/jason/Downloads/crowd_counting-main/crowd_counting-main/ShanghaiTech', help='path to dataset')
parser.add_argument('--exp_sets', default='part_B/test_data')
parser.add_argument('--use_gpu', default=True, help="indicates whether or not to use GPU")
parser.add_argument('--device', default='1', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=path/'Weights/', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=path.parent/'runs/log', type=str, help='log dir')

parser.add_argument('--model_file', default=path/'model.yaml')

parser.add_argument('--best', default=False, type=bool, help='best or last saved checkpoint?') 
parser.add_argument('--vis_patch', default=False, type=bool, help='visualize the patches') 
parser.add_argument('--vis_image', default=False, type=bool, help='visualize the whole image') 
parser.add_argument('--vis_loc', default=False, type=bool, help='visualize the locations') 


#Main function
def eval():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    CUDA = torch.cuda.is_available()
    root = 'ShanghaiTech'  # Corrected directory name
    part_B_train = os.path.join(root, 'part_B/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B/test_data', 'images')
    path_sets = [part_B_train, part_B_test]
    #Configure weight's path
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    #Load model
    model = MSPSNet()

    if CUDA:
        model = model.cuda()
    
    checkpoint = torch.load('Weights/SHB/checkpoint.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    imgs, targets, target_bigs = [], [], []
    average_precision = []
    sum_mae_count_0, sum_mse, sum_mae_count_1, sum_mae_count_2, sum_mae_total_count = 0.0, 0.0, 0.0, 0.0, 0.0
    dataset_length = len(img_paths)
    
    pbar = enumerate(img_paths)
    pbar = tqdm(pbar, total=len(img_paths))

    #Evaluate
    for bi, img_path in pbar: 
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        img_big = Image.open(img_path).convert('RGB')
        img_big = transform(img_big)
        
        gt_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'),'r')
        target_big = np.asarray(gt_file['density'])
        target_big = torch.from_numpy(target_big)

        length_0 = img_big.size(1)
        length_1 = img_big.size(2)

        coord = target_big.nonzero()  # Get the indices of nonzero elements
        bxy = [[yb / length_0, xb / length_1] for yb, xb in zip(coord[0], coord[1])]  # Iterate over indices
        # ...
        targets.append(torch.tensor(bxy))

        img = torch.clone(img_big)

        imgs.append(img)

        target_bigs.append(target_big)

        imgs = torch.stack(imgs, dim=0).squeeze(1)

        targets = [ti for ti in targets if len(ti) != 0]
        if not targets:
            targets.append(torch.tensor([[-1, 0, 0, 0, 0]]))
        targets = torch.cat(targets)

        target_bigs = torch.stack(target_bigs, dim=0).squeeze(1)

        if CUDA:
            imgs = imgs.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            
            predictions0, predictions1, predictions2, predictions3 = model(imgs, training=False)

            average_precision.append(loc_eval(imgs[0,...], predictions2[0,...], target_big))

            predictions3 = torch.sum(predictions3, dim=0).unsqueeze(0)
            # img_name = img_path.replace('.jpg','').replace('ShanghaiTech' + args.exp_sets + '/images/','')

            targets = targets.shape[0]

            pred_count_0 = (predictions0).sum()
            pred_count_1 = (predictions1).sum()
            pred_count_2 = (predictions2).sum()
            total_count = predictions3.sum()
            
            mae_count_0 = abs(pred_count_0 - targets)
            mae_count_1 = abs(pred_count_1 - targets)
            mae_count_2 = abs(pred_count_2 - targets)
            mae_count_T = abs(total_count - targets)
            
            sum_mae_count_0 += mae_count_0
            sum_mae_count_1 += mae_count_1
            sum_mae_count_2 += mae_count_2
            sum_mae_total_count += mae_count_T
            
            mse = (total_count - targets)**2
            sum_mse += mse

            s = str((bi, 'MAE: ', round(mae_count_0.item(), 2), 'Pred: ', round(pred_count_0.item(), 2), 'target: ', targets))
            pbar.set_description(s)

            imgs = []
            targets = []
            target_bigs = []


    print(' * MAE_Count_0 {mae_count_0:.3f} \n * MAE_Count_1 {mae_count_1:.3f} \n * MAE_Count_2 {mae_count_2:.3f} \n * MAE_Count_total {mae_count_t:.3f} \n * MSE {mse:.3f} \n'.\
        format(mae_count_0=(sum_mae_count_0/dataset_length).item(), \
            mae_count_1=(sum_mae_count_1/dataset_length).item(), mae_count_2=(sum_mae_count_2/dataset_length).item(), mae_count_t=(sum_mae_total_count/dataset_length).item(), mse=(sum_mse/dataset_length).sqrt().item()))

    AP = sum(average_precision) / (len(average_precision))
    print(' * AP:', round(AP, 2))


def loc_eval(img, pred, target, vis=False):
    upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    pred = upsample(pred.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).cpu().numpy()
    
    img = img.permute(1, 2, 0).cpu()
    img = cv2.normalize(np.float32(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)

    n_t = target.sum().item()  # Convert target sum to Python scalar
    
    # Convert target tensor to NumPy array, properly scaled to [0, 255] and uint8 data type
    target_np = (target.squeeze().numpy() * 255).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    # Dilate the target density map
    target_dilated = cv2.dilate(target_np, kernel, iterations=1)
    
    h = img.shape[0] - pred.shape[0]
    w = img.shape[1] - pred.shape[1]
    pred = cv2.copyMakeBorder(pred, 0, h, 0, w, cv2.BORDER_CONSTANT, value=0)
    
    precision, recall = [], []
    thresh = np.linspace(0, 1)
    for i, th in enumerate(thresh):
        pred_th = (pred > th)
        n_p, labels_p = cv2.connectedComponents(pred_th.astype(np.uint8))

        overlap = target_dilated & pred_th
        n_o, labels_o = cv2.connectedComponents(overlap.astype(np.uint8))
        
        TP = n_o
        FP = abs(n_p - n_o)
        FN = abs(n_t - n_o)        

        P = TP / (TP + FP)        
        R = TP / (TP + FN)

        p_check, r_check = True, True
        if i > 0:
            p_check = P >= precision[-1]
            r_check = R <= recall[-1]

        if p_check and r_check:
            precision.append(P)
            recall.append(R)

        if R > 0.7 and th != 0 and vis:
            vis_blobs(img, labels_o, target_dilated)

    average_precision = auc(recall, precision)
    
    return average_precision



#Visualize the blobs
def vis_blobs(image, pred, gt):

    im = np.uint8(cv2.bitwise_not(pred) * (-1))

    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)
    im = cv2.bitwise_not(im)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    images = (image, im_with_keypoints)

    plt.imshow(images[0])
    plt.imshow(images[1], alpha=0.5)

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    eval()