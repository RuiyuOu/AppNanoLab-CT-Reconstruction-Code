import SimpleITK as sitk
import numpy as np
import torch
import DemoDataset
import tinycudann as tcnn
import commentjson as json
from torch.utils import data
from torch.optim import lr_scheduler
import DemoUtils 


def train(config_path):

    with open(config_path) as config_file:
        config = json.load(config_file)

    in_dir = config["file"]["in_dir"]
    model_dir = config["file"]["model_dir"]
    SOD = config["data"]["SOD"]
    ODD = config["data"]["ODD"]
    voxel_size = config["data"]["voxel_size"]

    lr = config["train"]["lr"]
    epoch = config["train"]["epoch"]
    gpu = config["train"]["gpu"]
    summary_epoch = config["train"]["summary_epoch"]
    sample_N = config["train"]["sample_N"]
    batch_size = config["train"]["batch_size"]

    num_sv, num_det = sitk.GetArrayFromImage(sitk.ReadImage(in_dir)).shape
    
    train_loader = data.DataLoader(
            dataset=DemoDataset.FanbeamTrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N, SOD=SOD, ODD=ODD, voxel_size=voxel_size),
            batch_size=batch_size,
            shuffle=True
        )

    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    l1_loss_function = torch.nn.L1Loss() 

    MODEL = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                              encoding_config=config["encoding"],
                                              network_config=config["network"]).to(DEVICE)


    optimizer = torch.optim.Adam(params=MODEL.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

    for e in range(epoch):
        MODEL.train()
        loss_train = 0
        for i, (ray_sample, projection_l_sample) in enumerate(train_loader):
            ray_sample = ray_sample.to(DEVICE).float().view(-1, 2) 
            projection_l_sample = projection_l_sample.to(DEVICE).float() 

            pre_intensity = MODEL(ray_sample).view(batch_size, sample_N, num_det, 1).float()  # (N, sample_N, L, 1)
            projection_l_sample_pre = torch.sum(pre_intensity, dim=2) 

            projection_l_sample_pre = projection_l_sample_pre.squeeze(-1).squeeze(-1) 

            loss = l1_loss_function(projection_l_sample_pre, 
                                    projection_l_sample.to(projection_l_sample_pre.dtype))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()

        scheduler.step()

        print('{}, (TRAIN0) Epoch[{}/{}], Lr:{}, Loss:{:.6f}'.
              format(num_sv, e + 1, epoch, scheduler.get_last_lr()[0], loss_train/len(train_loader)))

        if (e + 1) % summary_epoch == 0:
            torch.save(MODEL.state_dict(), model_dir)



def reprojection(config_path):

    with open(config_path) as config_file:
        config = json.load(config_file)

    in_dir = config["file"]["in_dir"]
    out_dir = config["file"]["out_dir"]
    model_dir = config["file"]["model_dir"]
    gpu = config["train"]["gpu"]

    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    MODEL = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                              encoding_config=config["encoding"],
                                              network_config=config["network"]).to(DEVICE)
    MODEL.load_state_dict(torch.load(model_dir))

    sin_original    = sitk.GetArrayFromImage(sitk.ReadImage(in_dir))
    num_sv, num_det = sin_original.shape
    L   = 100/210
    W   = 100/210
    NL  = 1000
    NW  = 1000
    xy  = DemoUtils.grid_coordinate_2d(L,NL,W,NW).to(DEVICE)
    sin_pre = np.zeros(shape=(xy.shape))
    with torch.no_grad():
        sin_pre = MODEL(xy).reshape(NL,NW)
    sin_pre = np.fliplr(sin_pre.cpu().numpy().astype(np.float64))
    sin_pre = sitk.GetImageFromArray(sin_pre)
    sitk.WriteImage(sin_pre, '{}/{}_INR.nii'.format(out_dir, num_sv))
    print('CT Reconstruction Complete')