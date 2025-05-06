from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import models.siren as siren
import torch.nn.functional as F


class CbNN():
    def __init__(self, clObj, cleanbeam, ci_target, img_target, ci_sigmas=None, stokes=False,
                  device=torch.device("cpu"), imgdim1:int=64, imgdim2:int=64, w0_initial=2.,
                  polar=False):
        self.model = CbNNmodel(imgdim1=imgdim1, imgdim2=imgdim2).to(device)
        if stokes:
            siren_dim_out = 4
        else:
            siren_dim_out = 1

        self.siren = siren.SirenNet(
                        dim_in = 2,                        # input dimension, ex. 2d coor
                        dim_hidden = 512,                  # hidden dimension
                        dim_out = siren_dim_out,                       # output dimension, ex. rgb value
                        num_layers = 5,                    # number of layers
                        final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
                        w0_initial = w0_initial                  # different signals may require different omega_0 in the first layer - this is a hyperparameter
                    ).to(device)
        self.wrapper = siren.SirenWrapper(self.siren, imgdim1, imgdim2).to(device)

        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.nnloss = nn.L1Loss(reduction='mean')
        self.clObj = clObj
        self.cleanbeam = cleanbeam
        self.ci_target = ci_target
        self.img_target = img_target
        self.stokes = stokes
        self.device = device
        self.polar = polar

        if ci_sigmas is None:
            self.ci_sigmas = torch.ones(ci_target.shape[-1])
        else:
            self.ci_sigmas = ci_sigmas #+ torch.zeros(ci_target.shape[-1])+0.01

        fovx = clObj.psize * imgdim1 * 206265*1e6
        fovy = clObj.psize * imgdim2 * 206265*1e6

        coords = np.meshgrid(np.linspace(0, fovx, imgdim1), np.linspace(0, fovy, imgdim2))
        coords = np.stack(coords)
        coords = torch.tensor(coords).float().to(device)
        coords = coords.flatten(1,2).T

        self.coords = coords

        coords_flat = coords.flatten().unsqueeze(0)
        self.coords_flat = coords_flat


    def train(self, siren=False, nepochs:int=500, condition_epochs:int=100, ci_weight=5, init_lr=1e-4, lr_scale=0.999, verbose=False):
        if siren:
            self.trainSiren(nepochs=nepochs, condition_epochs=condition_epochs, ci_weight=ci_weight, init_lr=init_lr, lr_scale=lr_scale, verbose=verbose)
        else:
            self.trainCbNN(nepochs=nepochs, condition_epochs=condition_epochs, ci_weight=ci_weight, init_lr=init_lr, lr_scale=lr_scale, verbose=verbose)

    def trainCbNN(self, nepochs:int=500, condition_epochs:int=100, ci_weight=5, init_lr=1e-4, lr_scale=0.999, verbose=False):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=init_lr)
        criterion = self.loss_function
        l1 = lambda epoch: lr_scale ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l1)
        train_loss = []
        for epoch in tqdm(range(nepochs)):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.coords_flat, stokes=self.stokes)
            loss = criterion(outputs, self.ci_target, self.img_target, epoch=epoch, condition_epochs=condition_epochs, ci_weight=ci_weight)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

            if verbose and ((epoch+1) % 100 == 0 or epoch == 0):
                outputs = self.evaluate(plot=True)
                print(f'Epoch {epoch+1}, lr: {scheduler.get_last_lr()[0]}')
            scheduler.step()

    def trainSiren(self, nepochs:int=1000, condition_epochs:int=500, ci_weight=5, init_lr=1e-4, lr_scale=0.999, verbose=False):
        optimizer = torch.optim.AdamW(self.siren.parameters(), lr=init_lr)
        criterion = self.loss_function
        l1 = lambda epoch: lr_scale ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l1)
        train_loss = []
        for epoch in tqdm(range(nepochs)):
            self.siren.train()
            optimizer.zero_grad()
            outputs = self.siren(self.coords)
            outputs = outputs.reshape(1, self.siren.dim_out, self.imgdim1, self.imgdim2)
            loss = criterion(outputs, self.ci_target, self.img_target, epoch=epoch, condition_epochs=condition_epochs, ci_weight=ci_weight)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
            
            if verbose and ((epoch+1) % 200 == 0 or epoch == 0):
                outputs = self.evaluate(siren=True, plot=True)
                print(f'Epoch {epoch+1}, lr: {scheduler.get_last_lr()[0]}')
            scheduler.step()



    def loss_function(self, outputs, ci_target, img_target, ci_sigmas=None, epoch=0, condition_epochs:int=100, ci_weight=5):
        outputs = self.preprocess(outputs).to(torch.float64)
        img_loss = self.nnloss(outputs, img_target)

        if ci_sigmas is None:
            ci_sigmas = self.ci_sigmas

        if epoch < condition_epochs:
            return img_loss/img_loss.detach()
        else:
            outputs_ci = self.clObj.FTCI(outputs, add_th_noise=False, stokes=self.stokes)
            ci_loss = self.nnloss(outputs_ci/ci_sigmas, ci_target/ci_sigmas)
            # ci_loss = nn.SmoothL1Loss(beta=1e-3)(outputs_ci, ci_target)
            # ci_loss = nn.L1Loss()(outputs_ci, ci_target)
            # ci_loss = nn.MSELoss()(outputs_ci, ci_target)
            return ci_loss/ci_loss.detach() * ci_weight + img_loss/img_loss.detach() #+ self.sl1(output_mask)

    def sl1(self, imvec):
        """L1 norm regularizer
        """
        l1 = torch.sum(torch.abs(imvec))
        return l1

    def preprocess(self, img, threshold=1e-4):
        if self.stokes:
            out_img = torch.zeros_like(img)
            img4 = img[0]
            for ind, img in enumerate(img4):
                norm = torch.max(img)
                img = img/norm
                img = torch.fft.ifft2(torch.fft.fft2(img) * torch.fft.fft2(torch.tensor(self.cleanbeam.imvec.reshape(self.imgdim1, self.imgdim2), dtype=torch.float64).to(self.device)))
                img = torch.fft.fftshift(img)
                img = img.reshape(1, self.imgdim1, self.imgdim2)
                img = img.to(torch.float64)
                if ind == 0:
                    img = F.relu(img)
                    img = F.threshold(img, threshold, 0)
                out_img[:, ind, :, :] = img*norm
            return out_img
        else:
            # if self.polar:
            #     img = self.convertToCartesianImage(img)
            norm = torch.max(img[0])
            img = img/norm
            img = F.threshold(img, threshold, 0)
            img = torch.fft.ifft2(torch.fft.fft2(img[0]) * torch.fft.fft2(torch.tensor(self.cleanbeam.imvec.reshape(self.imgdim1, self.imgdim2), dtype=torch.float64).to(self.device)))
            img = torch.fft.fftshift(img)
            img = img.reshape(1, 1, self.imgdim1, self.imgdim2)
            img = img.to(torch.float64) * norm
            if self.polar:
                img = self.convertToCartesianImage(img.to(torch.float32))
                norm = torch.max(img[0])
                img = img/norm
                img = torch.fft.ifft2(torch.fft.fft2(img[0]) * torch.fft.fft2(torch.tensor(self.cleanbeam.imvec.reshape(self.imgdim1, self.imgdim2), dtype=torch.float64).to(self.device)))
                img = torch.fft.fftshift(img)
                img = img.reshape(1, 1, self.imgdim1, self.imgdim2)
                img = img.to(torch.float64) * norm
            return img
        
    def evaluate(self, siren=False, coords=None, plot=False):
        if siren:
            self.siren.eval()
            if coords is not None:
                outputs = self.siren(coords)
            else:
                outputs = self.siren(self.coords)
                outputs = outputs.reshape(1, self.siren.dim_out, self.imgdim1, self.imgdim2)
                outputs = self.preprocess(outputs)
        else:
            self.model.eval()
            outputs = self.model(self.coords_flat, stokes=self.stokes)
            outputs = self.preprocess(outputs)

        if plot:
            if self.stokes:
                _, axs = plt.subplots(1, 4, figsize=(4,1))
                plt.subplots_adjust(wspace=0, hspace=0)
                for i in range(4):
                    axs[i].imshow(outputs[0, i].detach().cpu().numpy(), cmap='afmhot')
                    axs[i].set_xticks([])
                    axs[i].set_yticks([])
                plt.show()
            else:
                _, ax = plt.subplots(1, 1, figsize=(1,1))
                ax.imshow(outputs[0, 0].detach().cpu().numpy(), cmap='afmhot')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()

        return outputs

    def _replaceInitImg(self, siren=False, blur=False, threshold=1e-2):
        self.img_target = self.evaluate(siren=siren).detach()
        if blur:
            self.img_target = self.preprocess(self.img_target, threshold=threshold)
        return self.img_target
    
    def getPolarPoints2(self, x, y, center):
        cX, cY = x - center[0], y - center[1]
        r = torch.sqrt(cX ** 2 + cY ** 2)
        theta = torch.arctan2(cY, cX)
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        return r, theta

    def convertToCartesianImage(self, image):
        image = image[0][0]
        # This is used to scale the result of the angle to get the appropriate Cartesian value
        scaleAngle = image.shape[-2] / (2 * np.pi)

        imageSize = torch.tensor(image.shape[-2:])
        center = (imageSize) / 2

        # Get list of cartesian x and y coordinate and create a 2D grid of the coordinates using meshgrid
        xs = torch.arange(0, image.shape[1])
        ys = torch.arange(0, image.shape[0])
        x, y = torch.meshgrid(xs, ys, indexing='ij')

        # Take cartesian grid and convert to polar coordinates
        r, theta = self.getPolarPoints2(x, y, center)
        r = r*np.sqrt(2)

        # Offset the theta angle by the initial source angle
        # The theta values may go past 2pi, so they are looped back around by taking modulo with 2pi.
        # Note: This assumes initial source angle is positive
        theta = torch.fmod(theta + 2 * np.pi, 2 * np.pi)

        # Scale the angle from radians to pixels using scale factor
        theta = theta * scaleAngle

        # normlise theta by theta size and r by r size to range [-1, 1]
        theta = (theta / (image.shape[-2])) * 2 - 1
        r = (r / (image.shape[-1])) * 2 - 1

        # Flatten the desired x/y cartesian points into one Nx2 array
        desiredCoords = torch.vstack((r.unsqueeze(0), theta.unsqueeze(0))).T

        # Get the new shape which is the cartesian image shape plus any other dimensions
        # Get the new shape of the cartesian image which is the same shape of the polar image except the last two dimensions
        # (r & theta) are replaced with the cartesian image size
        newShape = image.shape[:-2] + image.shape

        # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D with the 3rd dimension a size of 1
        image = image.reshape((-1,) + image.shape)

        cartesianImages = []


        # Loop through the third dimension and map each 2D slice
        for slice in image:
            imageSlice = torch.nn.functional.grid_sample(slice.unsqueeze(0).unsqueeze(0), desiredCoords.unsqueeze(0).to(self.device), mode='bicubic', align_corners=True)
            cartesianImages.append(imageSlice)

        # Stack all of the slices together and reshape it to what it should be
        cartesianImage = torch.stack(cartesianImages, axis=0).reshape(newShape)

        return cartesianImage.to(self.device).unsqueeze(0).unsqueeze(0)

class CbNNmodel(nn.Module):
    def __init__(self, imgdim1:int=64, imgdim2:int=64):
        super(CbNNmodel, self).__init__()

        self.imgdim1, self.imgdim2 = imgdim1, imgdim2
        self.imgdim = int(imgdim1*imgdim2)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.imgdim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.imgdim),
            nn.ReLU()
        )

        self.mlp_stokes = nn.Sequential(
            nn.Linear(2*self.imgdim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4*self.imgdim),
            # nn.ReLU()
        )

        self.stokes_layers = nn.ModuleList([])
        for i in range(3):
            self.stokes_layers.append(nn.Sequential(
                nn.Linear(2*self.imgdim, 4096),
                nn.ReLU(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.imgdim),
            ))
    
    def forward(self, coords, stokes=False): 
        if stokes:
            result = []
            # for i in range(4):
            #     if i == 0:
            #         result.append(self.mlp(coords))
            #     else:
            #         result.append(self.stokes_layers[i-1](coords))
            # result = torch.stack(result)
            # result = torch.swapaxes(result, 0, 1).reshape(-1, 4, self.imgdim1, self.imgdim2)

            # using the mlp_stokes architecture
            result = self.mlp_stokes(coords)
            result = result.reshape(-1, 4, self.imgdim1, self.imgdim2)
        else:
            result = self.mlp(coords)
            result = result.reshape(-1, 1, self.imgdim1, self.imgdim2)

        return result
    


