import SimpleITK as sitk
import numpy as np
import torch
import dataset
import tinycudann as tcnn
import commentjson as json
from torch.utils import data
from torch.optim import lr_scheduler
import utils


def train(config_path):

    # config
    # ------------------------------------
    with open(config_path) as config_file:
        config = json.load(config_file)

    in_dir = config["file"]["in_dir"]
    out_dir = config["file"]["out_dir"]
    model_dir = config["file"]["model_dir"]

    dimensions = config["data"]["dimensions"]
    struc_in = config["data"]["struc_in"]
    source_pos = config["data"]["source_pos"]
    fan_coverage = config["data"]["fan_coverage"]
    sensor_geometry = config["data"]["sensor_geometry"]
    SOD = config["data"]["SOD"]
    ODD = config["data"]["ODD"]
    voxel_size = config["data"]["voxel_size"]

    lr = config["train"]["lr"]
    epoch = config["train"]["epoch"]
    gpu = config["train"]["gpu"]
    summary_epoch = config["train"]["summary_epoch"]
    sample_N = config["train"]["sample_N"]
    batch_size = config["train"]["batch_size"]

    if dimensions == 2:
        num_sv, num_det = sitk.GetArrayFromImage(sitk.ReadImage(in_dir)).shape
    if dimensions == 3:
        num_sv, num_det, num_z = sitk.GetArrayFromImage(sitk.ReadImage(in_dir)).shape

    # data
    # ------------------------------------
    if struc_in == "radon":
        train_loader = data.DataLoader(
            dataset=dataset.RadonTrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N),
            batch_size=batch_size,
            shuffle=True
        )
    elif struc_in == "fanbeam":
        train_loader = data.DataLoader(
            dataset=dataset.FanbeamTrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N, SOD=SOD, ODD=ODD, voxel_size=voxel_size),
            batch_size=batch_size,
            shuffle=True
        )

    # To be implemented
    
    # elif struc_in == "CBCT":
    #     train_loader = data.DataLoader(
    #         dataset=dataset.CBCT_TrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N),
    #         batch_size=batch_size,
    #         shuffle=True
    #     )

    # elif struc_in == "MS-CBCT":
    #     train_loader = data.DataLoader(
    #         dataset=dataset.MS_CBCT_TrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N),
    #         batch_size=batch_size,
    #         shuffle=True
    #     )

    # elif struc_in == "Tomo":
    #     train_loader = data.DataLoader(
    #         dataset=dataset.TomoTrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N),
    #         batch_size=batch_size,
    #         shuffle=True
    #     )

    # elif struc_in == "Helical":
    #     train_loader = data.DataLoader(
    #         dataset=dataset.HelixTrainData(sin_path=in_dir, theta=num_sv, sample_N=sample_N),
    #         batch_size=batch_size,
    #         shuffle=True
    #     )

    # model & optimizer
    # ------------------------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    l1_loss_function = torch.nn.L1Loss() 

    if dimensions == 2:
        MODEL = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                              encoding_config=config["encoding"],
                                              network_config=config["network"]).to(DEVICE)

    elif dimensions == 3:
        MODEL = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1,
                                              encoding_config=config["encoding"],
                                              network_config=config["network"]).to(DEVICE)

    optimizer = torch.optim.Adam(params=MODEL.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # train
    # ------------------------------------
    # A 3D version of this may need to be made
    for e in range(epoch):
        MODEL.train()
        loss_train = 0
        for i, (ray_sample, projection_l_sample) in enumerate(train_loader):
            # the sampled rays and the corresponding projections
            ray_sample = ray_sample.to(DEVICE).float().view(-1, 2)  # (N, sample_N, L, n_dim)
            projection_l_sample = projection_l_sample.to(DEVICE).float()  # (N, sample_N)
            # forward
            pre_intensity = MODEL(ray_sample).view(batch_size, sample_N, num_det, 1).float()  # (N, sample_N, L, 1)
            projection_l_sample_pre = torch.sum(pre_intensity, dim=2)  # (N, sample_N, 1, 1)
            # reshape
            projection_l_sample_pre = projection_l_sample_pre.squeeze(-1).squeeze(-1)  # (N, sample_N)
            # compute loss
            loss = l1_loss_function(projection_l_sample_pre, 
                                    projection_l_sample.to(projection_l_sample_pre.dtype))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record and print loss
            loss_train += loss.item()
        scheduler.step()
        print('{}, (TRAIN0) Epoch[{}/{}], Lr:{}, Loss:{:.6f}'.
              format(num_sv, e + 1, epoch, scheduler.get_last_lr()[0], loss_train/len(train_loader)))

        if (e + 1) % summary_epoch == 0:
            torch.save(MODEL.state_dict(), model_dir)



def reprojection(config_path):

    # config
    # ------------------------------------
    with open(config_path) as config_file:
        config = json.load(config_file)

    in_dir = config["file"]["in_dir"]
    out_dir = config["file"]["out_dir"]
    model_dir = config["file"]["model_dir"]

    struc_in = config["data"]["struc_in"]
    source_pos = config["data"]["source_pos"]
    fan_coverage = config["data"]["fan_coverage"]
    sensor_geometry = config["data"]["sensor_geometry"]
    SOD = config["data"]["SOD"]
    ODD = config["data"]["ODD"]
    voxel_size = config["data"]["voxel_size"]

    batch_size = config["train"]["batch_size"]
    gpu = config["train"]["gpu"]

    dimensions = config["data"]["dimensions"]
    struc_out = config["data"]["struc_out"]
    scale = config["data"]["reproj_density"]

    if dimensions == 2:
        num_sv, num_det = sitk.GetArrayFromImage(sitk.ReadImage(in_dir)).shape
    if dimensions == 3:
        num_sv, num_det, num_z = sitk.GetArrayFromImage(sitk.ReadImage(in_dir)).shape

    # model
    # ------------------------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    if dimensions == 2:
        MODEL = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                              encoding_config=config["encoding"],
                                              network_config=config["network"]).to(DEVICE)
        MODEL.load_state_dict(torch.load(model_dir))

    elif dimensions == 3:
        MODEL = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1,
                                              encoding_config=config["encoding"],
                                              network_config=config["network"]).to(DEVICE)
        MODEL.load_state_dict(torch.load(model_dir))

    # reprojetion
    # ------------------------------------
    if struc_out == "reproj":
        if struc_in == "radon":
            test_loader = data.DataLoader(
            dataset=dataset.RadonTestData(theta=int(num_sv*scale), L=num_det),
            batch_size=batch_size,
            shuffle=False
        )

        elif struc_in == "fanbeam":
            test_loader = data.DataLoader(
            dataset=dataset.FanbeamTestData(theta=int(num_sv*scale), sin_path=in_dir, SOD=SOD, ODD=ODD, voxel_size=voxel_size),
            batch_size=batch_size,
            shuffle=False
        )

        sin_pre = np.zeros(shape=(int(num_sv*scale), num_det))
        with torch.no_grad():
            MODEL.eval()
            for i, (ray_sample) in enumerate(test_loader):
                print(i, len(test_loader))
                # all the parallel rays from each view
                ray_sample = ray_sample.to(DEVICE).float().view(-1, 2)  # (N, L, L, 2)
                # forward
                pre_intensity = MODEL(ray_sample).view(-1, num_det, num_det, 1)  # (N, L, L, 1)
                # projection i.e, Equ. 2
                projection_l_sample_pre = torch.sum(pre_intensity, dim=2)  # (N, L, 1, 1)
                # reshape and store
                projection_l_sample_pre = projection_l_sample_pre.squeeze(-1).squeeze(-1)  # (N, L)
                temp = projection_l_sample_pre.cpu().detach().float().numpy()
                if i == 0:
                    sin_pre = temp
                else:
                    sin_pre = np.concatenate((sin_pre, temp), axis=0)

        # data consistency
        sin_original = sitk.GetArrayFromImage(sitk.ReadImage(in_dir))
        k = 0
        # if struc_in == "fanbeam":
        #     sin_pre = np.fliplr(sin_pre) # the detector numbering is backwards in the code. This is a "bandaid fix" until the fanbeam generator is changed. Or keep it because its not broke. 
        for i in range(len(sin_pre)):
            if i % scale == 0:
                sin_pre[i, :] = sin_original[k, :]
                k = k + 1
        # write dense-view sinogram and model
        sin_pre = sitk.GetImageFromArray(sin_pre)
        sitk.WriteImage(sin_pre, '{}/{}x_sino.nii'.format(out_dir, scale))
    
    elif struc_out == "INR":
        if dimensions == 2:
            sin_original    = sitk.GetArrayFromImage(sitk.ReadImage(in_dir))
            num_det, num_sv = sin_original.shape
            L   = 100/210 # 1/2 * grid size / ODD. Image wanted is 1000 pixels wide of 0.2mm wide pixels => 100mm
            W   = 100/210
            NL  = 1000
            NW  = 1000
            xy = utils.grid_coordinate_2d(L,NL,W,NW).to(DEVICE)
            sin_pre = np.zeros(shape=(xy.shape))
            with torch.no_grad():
                sin_pre = MODEL(xy).reshape(NL,NW)
            sin_pre = np.fliplr(sin_pre.cpu().numpy().astype(np.float64)) #the detector numbering is back-to-front causing a flip across the y-axis. This is a bandaid fix before the fanbeam generator is fixed. 
            sin_pre = sitk.GetImageFromArray(sin_pre)
            sitk.WriteImage(sin_pre, '{}/{}_INR.nii'.format(out_dir, num_sv))
        
        elif dimensions == 3:
            sin_original    = sitk.GetArrayFromImage(sitk.ReadImage(in_dir))
            num_det, num_sv, num_z = sin_original.shape
            L   = 1
            W   = 1
            H   = 1
            NL  = 1000
            NW  = 1000
            NH  = 1000
            xyz = utils.grid_coordinate_3d(L,NL,W,NW,H,NH).to(DEVICE)
            sin_pre = np.zeros(shape=(xyz.shape))
            with torch.no_grad():
                sin_pre = MODEL(xyz).reshape(NL,NW,NH)
            sin_pre = sitk.GetImageFromArray(sin_pre.cpu().numpy().astype(np.float64))
            sitk.WriteImage(sin_pre, '{}/{}_INR.nii'.format(out_dir, num_sv))