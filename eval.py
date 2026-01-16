import utils
import SimpleITK as sitk

if __name__ == '__main__':
    gt = sitk.GetArrayFromImage(sitk.ReadImage('data/gt_32f_masked.nii.gz'))
    recon = sitk.GetArrayFromImage(sitk.ReadImage('data/1454_INR_masked.nii.gz'))

    print('PSNR:', utils.psnr(image=recon, ground_truth=gt), 'SSIM:', utils.ssim(image=recon, ground_truth=gt))