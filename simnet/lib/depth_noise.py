import numpy as np
import IPython
import cv2
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
from skimage import exposure
import imageio
import matplotlib.pyplot as plt


class DepthManager():
    def __init__(self, config=None):

        self.depth_config_ = {
            'multiplicative_denoising': False,
            'gamma_shape': 0.1,

            'image_dropout': True,
            'dropout_poisson_mean': 3000,
            'dropout_radius_shape': 0.8,
            'dropout_radius_scale': 0.8,


            'gradient_dropout': True,
            'gradient_dropout_sigma': 1,
            'gradient_dropout_shape': .8,
            'gradient_dropout_scale': .8,

            'border_distortion': True,
            'border_grad_sigma': 1,
            'border_grad_thresh': 0.1,
            'border_poisson_mean': 1000,
            'border_radius_shape': 0.75,
            'border_radius_scale': 0.75,

            'gaussian_process_denoising': True,
            'gaussian_process_sigma': 0.05,
            'gaussian_process_scaling_factor': 40,


        }
        # self.depth_config_ = {
        #     'multiplicative_denoising': False,
        #     'gamma_shape': 0.1,

        #     'image_dropout': True,
        #     'dropout_poisson_mean': 1500,
        #     'dropout_radius_shape': 0.5,
        #     'dropout_radius_scale': 0.5,


        #     'gradient_dropout': True,
        #     'gradient_dropout_sigma': 1,
        #     'gradient_dropout_shape': .01,
        #     'gradient_dropout_scale': .01,

        #     'border_distortion': True,
        #     'border_grad_sigma': 1,
        #     'border_grad_thresh': 0.01,
        #     'border_poisson_mean': 1000,
        #     'border_radius_shape': 0.5,
        #     'border_radius_scale': 0.5,

        #     'gaussian_process_denoising': True,
        #     'gaussian_process_sigma': 0.05,
        #     'gaussian_process_scaling_factor': 50,


        # }



    def resize_image_and_label(self, depth_image):
	# Resizes to 256,256 , which is the resolution of our network. This needs to be changed for you guys potentially. 
        height, width, c = depth_image.shape
        # Resize image.
        depth_image = cv2.resize(depth_image,(256, 256))

    def depth_255_image(self, d_image):
        # Scale to 255 (Note this handles the case where zero values are marked as invalid)
        mask = d_image > 0
        d_image_scale = (d_image[mask] - np.min(d_image[mask])) / (
            np.max(d_image[mask]) - np.min(d_image[mask]))
        d_image[mask] = d_image_scale * 254 + 1
        d_image[mask] = 255.0 - d_image[mask]
        w, h = d_image.shape
        new_d_image = np.zeros([w, h, 3])
        for i in range(3):
            new_d_image[:, :, i] = d_image
        return new_d_image.astype(np.uint8)

    def convert_depth_image_to_gray_scale(self, depth_image):
	# Turns a depth image to a 3-channel uint8 RGB image with repeated channels.
        depth_image = self.depth_255_image(depth_image)

    def prepare_depth_data(self, orig_depth_image):
	# This is the main function which takes in a rendered depth image which is type float with shape [W,H,1] 
        di = np.copy(orig_depth_image)

        # for i in range(1, 20, 2):
        # self.depth_config_['gamma_shape'] = i
        depth_image = self.distort(np.copy(di))
        # plt.imshow(depth_image, cmap='gray')
        # plt.show()
        return depth_image
        # import IPython; IPython.embed()
        # imageio.imsave("depth_img.jpg", depth_image)
        # import IPython; IPython.embed()
        # self.convert_depth_image_to_gray_scale(depth_image)
        # self.resize_image_and_label(depth_image)
        # return depth_image

    def distort(self, d_img):
        """ Adds noise to a single image """
        width, height = d_img.shape
        
        # denoising and synthetic data generation
        if self.depth_config_['multiplicative_denoising']:
            gamma_scale = 1.0 / self.depth_config_['gamma_shape']
            gamma_shape = self.depth_config_['gamma_shape']
            mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1)
            mult_samples = mult_samples[:, np.newaxis]
            d_img = d_img * np.tile(mult_samples, [width, height])

        # dropout a region around the areas of the image with high gradient
        if self.depth_config_['gradient_dropout']:
            grad_mag = sf.gaussian_gradient_magnitude(
                d_img, sigma=self.depth_config_['gradient_dropout_sigma'])
            thresh = ss.gamma.rvs(
                self.depth_config_['gradient_dropout_shape'],
                scale=self.depth_config_['gradient_dropout_scale'],
                size=100).mean()
            # print(thresh)

            high_gradient_px = np.where(grad_mag > thresh)
            # import IPython; IPython.embed()
            d_img[high_gradient_px[0], high_gradient_px[1]] = 0.0


        # randomly dropout borders of the image for robustness
        if self.depth_config_['border_distortion']:
            grad_mag = sf.gaussian_gradient_magnitude(
                d_img, sigma=self.depth_config_['border_grad_sigma'])
            high_gradient_px = np.where(
                grad_mag > self.depth_config_['border_grad_thresh'])
            high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
            num_nonzero = high_gradient_px.shape[0]
            num_dropout_regions = ss.poisson.rvs(
                self.depth_config_['border_poisson_mean'])

            # sample ellipses
            if num_nonzero == 0:
               return d_img
            dropout_centers = np.random.choice(
                num_nonzero, size=num_dropout_regions)
            x_radii = ss.gamma.rvs(
                self.depth_config_['border_radius_shape'],
                scale=self.depth_config_['border_radius_scale'],
                size=num_dropout_regions)
            y_radii = ss.gamma.rvs(
                self.depth_config_['border_radius_shape'],
                scale=self.depth_config_['border_radius_scale'],
                size=num_dropout_regions)
            # set interior pixels to zero or one
            for j in range(num_dropout_regions):
                ind = dropout_centers[j]
                dropout_center = high_gradient_px[ind, :]
                x_radius = x_radii[j]
                y_radius = y_radii[j]
                dropout_px_y, dropout_px_x = sd.ellipse(
                    dropout_center[0],
                    dropout_center[1],
                    y_radius,
                    x_radius,
                    shape=d_img.shape)
                d_img[dropout_px_y, dropout_px_x] = 0.0


        # randomly dropout regions of the image for robustness
        if self.depth_config_['image_dropout']:
            nonzero_px = np.where(d_img > 0)
            nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
            num_nonzero = nonzero_px.shape[0]
            num_dropout_regions = ss.poisson.rvs(
                self.depth_config_['dropout_poisson_mean'])
            if num_nonzero == 0:
               return d_img
            # sample ellipses
            dropout_centers = np.random.choice(
                num_nonzero, size=num_dropout_regions)
            x_radii = ss.gamma.rvs(
                self.depth_config_['dropout_radius_shape'],
                scale=self.depth_config_['dropout_radius_scale'],
                size=num_dropout_regions)
            y_radii = ss.gamma.rvs(
                self.depth_config_['dropout_radius_shape'],
                scale=self.depth_config_['dropout_radius_scale'],
                size=num_dropout_regions)

            # set interior pixels to zero
            for j in range(num_dropout_regions):
                ind = dropout_centers[j]
                dropout_center = nonzero_px[ind, :]
                x_radius = x_radii[j]
                y_radius = y_radii[j]
                dropout_px_y, dropout_px_x = sd.ellipse(
                    dropout_center[0],
                    dropout_center[1],
                    y_radius,
                    x_radius,
                    shape=d_img.shape)
                d_img[dropout_px_y, dropout_px_x] = 0.0

        # add correlated Gaussian noise
        if self.depth_config_['gaussian_process_denoising']:
            gp_rescale_factor = self.depth_config_[
                'gaussian_process_scaling_factor']
            gp_sample_height = int(height / gp_rescale_factor)
            gp_sample_width = int(width / gp_rescale_factor)
            gp_num_pix = gp_sample_height * gp_sample_width
            gp_sigma = self.depth_config_['gaussian_process_sigma']
            gp_noise = ss.norm.rvs(
                scale=gp_sigma, size=gp_num_pix).reshape(
                    gp_sample_height, gp_sample_width)
            gp_noise = cv2.resize(
                gp_noise, (height, width))
            d_img[d_img > 0] += gp_noise[d_img > 0]

        # randomly dropout borders of the image for robustness
        if self.depth_config_['border_distortion']:
            # plt.imshow(d_img)
            # plt.show()
            high_gradient_px = np.where(
                d_img < 20)
            high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
            num_nonzero = high_gradient_px.shape[0]
            num_dropout_regions = ss.poisson.rvs(
                self.depth_config_['border_poisson_mean'])

            # sample ellipses
            if num_nonzero > 0:
                dropout_centers = np.random.choice(
                    num_nonzero, size=num_dropout_regions)
                x_radii = ss.gamma.rvs(
                    self.depth_config_['border_radius_shape'],
                    scale=self.depth_config_['border_radius_scale'],
                    size=num_dropout_regions)
                y_radii = ss.gamma.rvs(
                    self.depth_config_['border_radius_shape'],
                    scale=self.depth_config_['border_radius_scale'],
                    size=num_dropout_regions)

                # set interior pixels to zero or one
                for j in range(num_dropout_regions):
                    ind = dropout_centers[j]
                    dropout_center = high_gradient_px[ind, :]
                    x_radius = x_radii[j]
                    y_radius = y_radii[j]
                    dropout_px_y, dropout_px_x = sd.ellipse(
                        dropout_center[0],
                        dropout_center[1],
                        y_radius,
                        x_radius,
                        shape=d_img.shape)
                    d_img[dropout_px_y, dropout_px_x] = 0.0



        return d_img

if __name__ == '__main__':
    im = np.load("depth_img.npy")
    DM = DepthManager()
    DM.prepare_depth_data(im)
