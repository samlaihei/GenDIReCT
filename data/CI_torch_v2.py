import numpy as np
import ehtim as eh
from astropy.time import Time
import torch
import pandas as pd
import math

from ClosureInvariants import graphUtils as GU
from ClosureInvariants import scalarInvariants_torch as SI
from ClosureInvariants import vectorInvariants_torch as VI

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Closure_Invariants():

    def __init__(self, ehtarray='EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9], psize=1.7044214966184275e-11,
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5,
                 uvfits_files=None, ehtimAvg=False, avg_timescale=0, 
                 ci_mask=None, ttype=None, device=None):

        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


        array = eh.array.load_txt(ehtarray)
        if subarray is not None:
            # array = array.make_subarray(['ALMA','APEX','LMT','PV','SMT','JCMT','SMA'])
            array = array.make_subarray(subarray)
        
        t = Time(date, format='iso', scale='utc')

        self.psize = psize
        
        self.obslist = []
        if uvfits_files is not None:
            for uvfits in uvfits_files:
                obs = eh.obsdata.load_uvfits(uvfits)
                if ehtimAvg and avg_timescale > 0:
                    # obs.add_scans()
                    obs = obs.avg_coherent(avg_timescale, scan_avg=False)
                self.obslist.append(obs)
        else:
            for rf in bw_hz:
                obs = array.obsdata(ra/360*24, dec, rf, 8e9, tint_sec, tadv_sec, tstart_hr, tstop_hr, mjd=int(t.mjd), timetype='UTC')
                if ehtimAvg and avg_timescale > 0:
                    # obs.add_scans()
                    obs = obs.avg_coherent(avg_timescale, scan_avg=False)                
                self.obslist.append(obs)

        self.set_class_quantities_from_obslist(ehtimAvg, avg_timescale)
        self.ci_mask = ci_mask
        self.ttype = ttype
        self.cached_ftmatrix = []
        self.cached_uv = []

        
    def set_class_quantities_from_obslist(self, ehtimAvg, avg_timescale):
        uvwlist = []
        num_site_pairs = []
        site_pairs = []
        site_pairs_dict = {}
        current_char = 'a'
        timestamps = []
        sigmas = []
        obs_vislist = []
        for obs in self.obslist:
            for tdata in obs.tlist():
                num_antenna = len(np.unique(np.concatenate((tdata['t1'], tdata['t2']))))
                if num_antenna < 3 or len(tdata['t1']) < 3:
                    continue
                u, v, w = tdata['u'], tdata['v'], np.array([0 for i in tdata['u']])
                timestamps.append(tdata['time'])
                uvwlist.append(np.stack((u, v, w), axis=-1))
                pairs = self.recarr_to_ndarr(tdata[['t1', 't2']], 'U32')
                site_pairs.append(pairs)

                unique_pairs = np.unique(pairs.flatten())
                pairs = np.array([np.where(unique_pairs == i)[0][0] for i in pairs.flatten()]).reshape(pairs.shape) # using numerical alias

                if pairs.tobytes() not in site_pairs_dict:
                    current_char = chr(ord(current_char) + 1)
                    site_pairs_dict[pairs.tobytes()] = current_char

                num_site_pairs.append(str(len(pairs))+site_pairs_dict[pairs.tobytes()])
                sigmas.append(tdata['sigma'])
                obs_vislist.append(tdata['vis'])


        sort_idx = np.argsort(num_site_pairs, kind='stable')
        uvwlist = [uvwlist[i] for i in sort_idx]
        site_pairs = [site_pairs[i] for i in sort_idx]
        timestamps = [timestamps[i] for i in sort_idx]
        sigmas = [sigmas[i] for i in sort_idx]
        obs_vislist = [obs_vislist[i] for i in sort_idx]

        uvwlist = np.concatenate(uvwlist, axis=0)
        timestamps = np.concatenate(timestamps, axis=0)

        # group by site pairs
        num_site_pairs = np.array(num_site_pairs)[sort_idx]
        unique_N_site_pairs = np.unique(num_site_pairs)
        N_times = [int(np.sum(num_site_pairs == i)) for i in unique_N_site_pairs]
        N_idx = [int(np.sum(num_site_pairs == i)*int(i[:-1])) for i in unique_N_site_pairs]


        # split site_pairs by times
        site_ids_flat = np.concatenate(site_pairs, axis=0)
        site_pairs = np.array(site_pairs, dtype=object)
        site_pairs = np.split(site_pairs, np.cumsum(N_times)[:-1])

        sigmas = np.array(sigmas, dtype=object)
        sigmas = np.concatenate(sigmas, axis=0)

        obs_vislist = np.array(obs_vislist, dtype=object)
        obs_vislist = np.concatenate(obs_vislist, axis=0)

        self.site_pairs = site_pairs
        self.site_ids_flat = site_ids_flat
        self.sigmas = sigmas
        self.obs_vislist = obs_vislist
        self.uvwlist = uvwlist
        self.timestamps = timestamps
        self.bs = 1

        self.avg_IDs = None
        self.saved_timescale = None
        self.ehtimAvg = ehtimAvg
        if self.ehtimAvg:
            self.avg_timescale = 0
        else:
            self.avg_timescale = avg_timescale

        self.N_times = N_times
        self.N_idx = N_idx

        self.N_independent_CIs = None


    def FTCI_batch(self, batch, imgs, add_th_noise=False,  th_noise_factor=1,
                   fov=None, fovx=None, fovy=None, intensity=0, stokes=False):
        # split images into batches
        imgs_batched = np.array_split(imgs, batch)
        for i, img in enumerate(imgs_batched):
            if img.shape[0] == 0:
                continue
            ci = self.FTCI(img, add_th_noise=add_th_noise, return_uv=False, th_noise_factor=th_noise_factor,
                           fov=fov, fovx=fovx, fovy=fovy, intensity=intensity, stokes=stokes)
            if i == 0:
                ci_batch = ci
            else:
                ci_batch = torch.concatenate((ci_batch, ci), dim=0)
        return ci_batch


    def FTCI(self, imgs, add_th_noise=False, th_noise_factor=1,
             return_uv=False, return_vis=False, return_list=False,
             fov=None, fovx=None, fovy=None, intensity=0, stokes=False,
             useObs=False, avgVis=False, force_recalc_avg=False,
             ttype=None, normwts=None):
        
        self.N_independent_CIs = 0

        if stokes and imgs.shape[1] != 4:
            raise ValueError("Stokes parameter requires 4 channels in the image")
        elif not stokes and imgs.shape[1] == 4:
            imgs = imgs[:, 0, :, :]

        self.bs = imgs.shape[0]
        if intensity > 0:
            imgs = imgs * intensity

        if isinstance(imgs, np.ndarray):
            imgs = torch.tensor(imgs).to(self.device)

        # swap axes to match ehtim
        imgs = torch.swapaxes(imgs, -1, -2)
            
        ci = torch.tensor(torch.empty((self.bs, 0)))
        out_uv = []

        if fov is not None:
            fovx = fov
            fovy = fov
        elif fovx is None or fovy is None:
            fovx = self.psize*206265*1e6 * imgs.shape[-2]
            fovy = self.psize*206265*1e6 * imgs.shape[-1]

        if useObs:
            vis = torch.tensor(self.obs_vislist, dtype=torch.complex128).to(self.device).reshape(1, -1)
            vis = vis.repeat(self.bs, 1)
        else:
            if ttype is not None:
                vis = self.Visibilities(imgs, torch.tensor(self.uvwlist[:,:2], dtype=torch.float64).to(self.device), fovx, fovy, stokes, ttype)
            else:
                vis = self.Visibilities(imgs, torch.tensor(self.uvwlist[:,:2], dtype=torch.float64).to(self.device), fovx, fovy, stokes, self.ttype)

        if return_vis and not return_uv:
            return torch.concat((vis.real, vis.imag), dim=1)
        
        if stokes:
            vis = vis.reshape((len(imgs), 4, -1))
            vis = self.stokes_to_Bmatrix(vis)
        
        if add_th_noise:
            vis = self.add_noise(vis, self.sigmas, th_noise_factor=th_noise_factor, stokes=stokes)

        vis_by_time = torch.split(vis, self.N_idx, dim=1)
        vis_by_time = [i.clone() for i in vis_by_time]
        for ind, (i, times) in enumerate(zip(vis_by_time, self.N_times)):
            if stokes:
                vis_by_time[ind] = i.reshape([self.bs, times] + [-1, 2, 2])
            else:
                vis_by_time[ind] = i.reshape([self.bs, times] + [-1])

        time_by_time = torch.split(torch.tensor(self.timestamps, dtype=torch.float64), self.N_idx, dim=0)
        time_by_time = [i.clone() for i in time_by_time]
        for ind, (i, times) in enumerate(zip(time_by_time, self.N_times)):
            time_by_time[ind] = i.reshape(times, -1)

        uv_by_time = torch.split(torch.tensor(self.uvwlist, dtype=torch.float64), self.N_idx, dim=0)
        uv_by_time = [i.clone() for i in uv_by_time]

        for ind, (i, times) in enumerate(zip(uv_by_time, self.N_times)):
            uv_by_time[ind] = i.reshape(times, -1, 3)
    
        out_list = []
        avg_list = []
        avg_id_list = []
        for ind, (temp_vis, uv, pairs, time) in enumerate(zip(vis_by_time, uv_by_time, self.site_pairs, time_by_time)):

            if return_uv or return_list or (not self.ehtimAvg and self.avg_timescale > 0):
                temp_ci, temp_uv, tnames = self.ClosureInvariants(temp_vis, uv=uv, pairs=pairs, stokes=stokes, normwts=normwts)
                if temp_uv is not None:
                    out_uv.append(temp_uv)
            else:
                temp_ci, _, _ = self.ClosureInvariants(temp_vis, uv=None, pairs=pairs, stokes=stokes, normwts=normwts)

            
            if return_list:
                times = time[:,0]
                element_pairs = np.array([pd.unique(i.ravel()) for i in pairs])
                temp_uv = temp_uv.reshape(1, 3, 3, -1, temp_ci.shape[-1])
                temp_uv = temp_uv[0]
                out_list.append([times, element_pairs, temp_uv, temp_ci.cpu().detach().numpy()])
            
            if not self.ehtimAvg and self.avg_timescale > 0:
                times = time[:,0]
                ci_indices = torch.arange(ci.shape[1], ci.shape[1]+temp_ci.shape[-2]*temp_ci.shape[-1]).reshape(temp_ci.shape[-2], temp_ci.shape[-1]).unsqueeze(0)
                avg_list.append([times, tnames, temp_ci])
                avg_id_list.append([times, tnames, ci_indices])
            
            if len(temp_ci) > 0:
                self.N_independent_CIs += (temp_ci.shape[-1] - 1)*temp_ci.shape[-2]
                temp_ci = temp_ci.reshape(imgs.shape[0], -1)
                ci = torch.cat((ci, temp_ci), dim=1)
        
    
        # Averaging shenanigans
        if not self.ehtimAvg and self.avg_timescale > 0:
            if self.saved_timescale != self.avg_timescale:
                self.saved_timescale = self.avg_timescale
                self.avg_CIs(avg_id_list, stokes=stokes, visData=avgVis)
            if avgVis:
                return self.avg_CIs(avg_list, stokes=stokes, visData=avgVis, indices=self.avg_IDs)
            if force_recalc_avg:
                ci = self.avg_CIs(ci, stokes=stokes, visData=avgVis, indices=self.avg_IDs)
            else:
                ci = self.avg_CIs(ci, stokes=stokes, visData=avgVis, indices=self.avg_IDs)

  

        if return_list:
            return out_list

        if return_uv and return_vis:
            out_uv = np.concatenate(out_uv, axis=-1)
            return ci, vis, out_uv
        
        if return_uv:
            out_uv = np.concatenate(out_uv, axis=-1)
            return ci, out_uv
        
        if return_vis:
            return ci, vis
        

        if self.ci_mask is not None:
            return ci[:, self.ci_mask]
        return ci
    

    def ClosureInvariants(self, vis, uv=None, pairs=None, stokes=False, normwts=None):
        """
        Calculates copolar closure invariants for visibilities assuming an n element 
        interferometer array using method 1.

        Nithyanandan, T., Rajaram, N., Joseph, S. 2022 “Invariants in copolar 
        interferometry: An Abelian gauge theory”, PHYS. REV. D 105, 043019. 
        https://doi.org/10.1103/PhysRevD.105.043019 

        Args:
            vis (torch.Tensor): visibility data sampled by the interferometer array
            n (int): number of antenna as part of the interferometer array

        Returns:
            ci (torch.Tensor): closure invariants
        """
        element_pairs = pairs[0]
        element_ids = pd.unique(np.array(element_pairs).ravel())
        dicts = []
        for p in pairs:
            unique_ids = pd.unique(np.array(p).ravel())
            dicts.append({element_ids[i]: unique_ids[i] for i in range(len(unique_ids))})

        triads_indep = GU.generate_independent_triads(element_ids, baseid=element_ids[0])
        if stokes:
            pol_axes = (-2, -1)
            bl_axis = -3
            vis = vis.to(torch.complex128)
            corrs_lol = VI.corrs_list_on_loops(vis.cpu(), element_pairs, triads_indep, bl_axis=bl_axis, pol_axes=pol_axes)
            advariants = VI.advariants_multiple_loops(corrs_lol, pol_axes=pol_axes)
            if vis.shape[0] == 1:
                advariants = advariants.unsqueeze(0)
            if advariants.dim() == 4:
                advariants = advariants.unsqueeze(2)
            z4 = VI.vector_from_advariant(advariants)
            mdp = VI.complete_minkowski_dots(z4)
            ci = VI.remove_scaling_factor_minkoski_dots(mdp, wts=None)
        else:
            corrs_lol = SI.corrs_list_on_loops(vis.cpu(), element_pairs, triads_indep, bl_axis=-1)
            if len(corrs_lol) == 0:
                return torch.tensor([]), None, None
            advariants = SI.advariants_multiple_loops(corrs_lol)
            ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts=normwts, normpower=2)
            # ci = SI.invariants_from_advariants_method1(advariants, normaxis=-1, normwts='max', normpower=2)


        if uv is not None: 
            uv = uv.swapaxes(1, 2)
            triads_indep = np.array(triads_indep)
            triads_names = np.copy(triads_indep)
            triads_indep = [np.where(np.all(np.sort(np.array(element_pairs)) == np.sort([triad[i], triad[i+1]]), axis=1)) for triad in triads_indep for i in (-1, 0, 1)]
            triads_indep = np.array(triads_indep, dtype=object).reshape(-1, 3)
            keepables = [all(i+1) for i in triads_indep]
            triads_indep = np.array(triads_indep[keepables], dtype=int).reshape(-1, 3)
            triads_names = triads_names[keepables]

            uv0 = uv[:, :, np.array(triads_indep)[:, 0]]
            uv1 = uv[:, :, np.array(triads_indep)[:, 1]]
            uv2 = uv[:, :, np.array(triads_indep)[:, 2]]

            uv = np.dstack((uv0,  uv1,  uv2))
            uv = uv.T.swapaxes(0, 1)
            if stokes:
                uv = uv.reshape(1, 3, 3, advariants.shape[-3], -1)
                triads_names = np.sort(triads_names.reshape(-1, 3))
                if advariants.shape[-3] > 2:
                    basis = uv[:, :, :, :2, :].repeat(5, axis=-2)
                    rest = uv[:, :, :, 2:, :].repeat(8, axis=-2)
                    uv = np.concatenate((basis, rest), axis=-2)
                    t_basis = triads_names[:2].repeat(5, axis=0)
                    t_rest = triads_names[2:].repeat(8, axis=0)
                    triads_names = np.concatenate((t_basis, t_rest), axis=0)
                else:
                    uv = uv.repeat(mdp.shape[-1], axis=-2)
                    triads_names = triads_names.repeat(mdp.shape[-1], axis=0)

            else:
                uv = uv.reshape(1, 3, 3, advariants.shape[-1], -1)
                uv = np.concatenate((uv, uv), axis=-1)
                triads_names = np.sort(triads_names.reshape(-1, 3))
                triads_names = np.concatenate((triads_names, triads_names), axis=0)

            uv = uv.reshape(1, 3, 3, -1)
            triads_names = triads_names.reshape(1, -1, 3)
            triads_names = triads_names.repeat(vis.shape[1], axis=0)

            for t, dict in zip(triads_names, dicts):
                for i, triad in enumerate(t):
                    for j, site in enumerate(triad):
                        t[i][j] = dict[site]
                        
            return ci, uv, triads_names
        
        return ci, None, None

    def add_noise(self, vis, sigmas, th_noise_factor=1, stokes=False):
        sigmas = torch.tensor(sigmas).to(self.device) * th_noise_factor
        if stokes:
            sigmas = sigmas[:, None, None]
            sigmas = sigmas.repeat(1, 2, 2).to(self.device)
        vis_noise = vis + ((torch.randn_like(vis) + 1j*torch.randn_like(vis))* sigmas)
        return vis_noise

    def avg_CIs(self, data_list, stokes=False, visData=False, indices=None): # timescale in seconds

        if indices is not None and not visData:
            # average based on the provided indices
            result = torch.stack([torch.median(data_list[:,ind], dim=-1)[0] for ind in indices], dim=0).T
            return result
        else:
            # Figure out the indices of the triads
            new_data_list = []  
            for i in data_list:
                time = i[0]
                tnames = i[1]
                ci = i[2]
                for ind, t in enumerate(time):
                    new_data_list.append([t, tnames[ind], ci[:, ind, :]])
            new_data_list = sorted(new_data_list, key=lambda x: x[0])

            time_ci_triads = []
            unique_triads = []
            for i in new_data_list:
                time = i[0]
                triads, ci = i[1:]
                time_ci_triads.append([time, ci, triads])

                for t in triads:
                    if tuple(sorted(t)) not in unique_triads:
                        unique_triads.append(tuple(sorted(t)))
            
            unique_triads = np.array(unique_triads)

            unique_triads = tuple(map(tuple, unique_triads))

            triad_list = [[] for _ in range(len(unique_triads))]

            for obj in time_ci_triads:
                time, ci, triads = obj
                ci = ci.T
                for c, t in zip(ci, triads):
                    triad_list[unique_triads.index(tuple(sorted((t))))].append([time, c])
            

            # group by number of times
            visData_list = []
            averaged_ci = []
            triad_indices = []
            for triad in triad_list:
                unique_times, unique_counts = torch.unique(torch.tensor([t[0] for t in triad]), return_counts=True)

                if stokes:
                    # split ci by counts
                    split_ci = torch.split(torch.stack([t[1] for t in triad]), unique_counts.tolist())
                    # reorder by sorted unique times
                    sort = torch.argsort(unique_counts, stable=True)
                    unique_counts = unique_counts[sort]
                    split_ci = [split_ci[i] for i in sort]
                    unique_times = unique_times[sort]
                    _, separate_num = np.unique(unique_counts, return_counts=True)

                    # group split_ci by counts
                    temp = np.ndarray(len(split_ci), dtype=object)
                    for i, s in enumerate(split_ci):
                        temp[i] = s
                    split_ci = np.split(temp, np.cumsum(separate_num)[:-1])
                    split_ci = [torch.stack(tuple(s)) for s in split_ci]
                else:
                    split_ci = torch.stack([t[1] for t in triad])
                    real, imag = split_ci[::2], split_ci[1::2]
                    split_ci = torch.stack([real, imag], dim=1)
                    split_ci = torch.unsqueeze(split_ci, 0)

                time_ind = 0
                _, bin_edges = np.histogram(unique_times, bins=np.arange(unique_times[0]-1e-5, unique_times[-1]+1e-5, self.avg_timescale/3600))
                bin_edges = np.append(bin_edges, unique_times[-1])
                triad_averaged_ci = []
                triad_averaged_ci_visdata = []
                triad_averaged_times = []

                for s in split_ci:
                    times = unique_times[time_ind:time_ind+len(s)]
                    time_ind += len(s)
                    for n, b in enumerate(bin_edges[:-1]):
                        mask = (times >= b) & (times < bin_edges[n+1])
                        if mask.any():
                            res = s[mask]
                            bin_mean = torch.median(res, dim=0)[0]
                            bin_mean_time = torch.mean(times[mask])
                            triad_averaged_times.append(bin_mean_time)
                            triad_averaged_ci_visdata.append(bin_mean)
                            triad_averaged_ci.extend(bin_mean)
                            triad_indices.extend(res.squeeze(-1).T.type(torch.long))
                if len(triad_averaged_ci) != 0:
                    visData_list.append([unique_times, split_ci, np.array(triad_averaged_times), triad_averaged_ci_visdata])
                    triad_averaged_ci = torch.stack(triad_averaged_ci)
                    averaged_ci.extend(triad_averaged_ci)
        
                result = torch.stack(averaged_ci).T # averaged ci
            

            if visData:
                return visData_list
            else:
                self.avg_IDs = triad_indices

            return result


############################################################################################################
 

    def DFT(self, data, uv, xfov=225, yfov=225):
        if data.ndim == 2:
            data = data[None,...]
            out_shape = (uv.shape[0],)
        elif data.ndim > 2:
            data = data.reshape((-1,) + data.shape[-2:])
            out_shape = data.shape[:-2] + (uv.shape[0],)
        ny, nx = data.shape[-2:]
        dx = xfov*4.84813681109536e-12 / nx
        dy = yfov*4.84813681109536e-12 / ny
        angx = (torch.arange(nx, dtype=torch.float64) - nx//2) * dx + (dx/2) # last term added to match ehtim
        angy = (torch.arange(ny, dtype=torch.float64) - ny//2) * dy + (dy/2) # last term added to match ehtim
        lvect = torch.sin(angx)
        mvect = torch.sin(angy)
        l, m = torch.meshgrid(lvect, mvect)
        lm = torch.cat([l.reshape(1,-1), m.reshape(1,-1)], dim=0).to(self.device)
        imgvect = data.reshape((data.shape[0],-1)).to(self.device)
        # uv = uv.to(torch.float64)
        x = -2*torch.pi*torch.matmul(uv,lm)[None, ...].to(self.device)
        visr = torch.sum(imgvect[:, None, :] * torch.cos(x).to(self.device), axis=-1)
        visi = torch.sum(imgvect[:, None, :] * torch.sin(x).to(self.device), axis=-1)
        if data.ndim == 2:
            vis = visr.ravel() + 1j*visi.ravel()
        else:
            vis = visr.ravel() + 1j*visi.ravel()
            vis = vis.reshape(out_shape)
        return vis
    
    def eh_direct(self, data, mat, stokes=False):
        # flatten last two dimensions of data
        if not stokes:
            data = data.reshape((data.shape[0], -1))
        else:
            data = data.reshape((data.shape[0], 4, -1))
        data = data.to(torch.complex128).to(self.device)
        all_vis = torch.tensordot(data, mat, dims=1)
        return all_vis


    def Visibilities(self, imgs, uv=None, xfov=225, yfov=225, stokes=False, ttype='DFT'):
        """
        Samples the visibility plane DFT according to eht uv co-ordinates.

        Args:
            imgs (torch.Tensor): tensor of images

        Returns:
            vis (torch.Tensor): visibilities taken for each image
        """

        if ttype == 'direct': # Method 2: Direct
            mat = None
            for ind, i in enumerate(self.cached_uv):
                if torch.allclose(i, uv):
                    mat = self.cached_ftmatrix[ind]
            if mat == None:
                mat = self.ftmatrix(xfov/1e6/206265/imgs.shape[-1], imgs.shape[-2], imgs.shape[-1], uv).to(self.device).permute(1,0)  # assuming xfov and yfov are the same
                self.cached_ftmatrix.append(mat)
                self.cached_uv.append(uv)
            vis = self.eh_direct(imgs, mat, stokes)
        else: # Method 1: DFT
            vis = self.DFT(imgs, uv, xfov, yfov)

        
        if stokes:
            vis = vis.reshape((len(imgs), 4, -1))
            return self.stokes_to_Bmatrix(vis)
        else:
            return vis.reshape((len(imgs), -1))
        
    def stokes_to_Bmatrix(self, stokes_vis):
        I, Q, U, V = stokes_vis[:, 0], stokes_vis[:, 1], stokes_vis[:, 2], stokes_vis[:, 3]
        B = torch.zeros((2, 2, len(stokes_vis), stokes_vis.shape[-1]), dtype=I.dtype)
        B[0, 0] = I + Q
        B[0, 1] = U + 1j*V
        B[1, 0] = U - 1j*V
        B[1, 1] = I - Q
        B = B.permute(2, 3, 0, 1)
        return B.to(self.device)
    
    def recarr_to_ndarr(self, x, typ):
        """converts a record array x to a normal ndarray with all fields converted to datatype typ
        """
        fields = x.dtype.names
        shape = x.shape + (len(fields),)
        dt = [(name, typ) for name in fields]
        y = x.astype(dt).view(typ).reshape(shape)
        return y
    
    def set_avg_timescale(self, avg_timescale):
        if self.ehtimAvg:
            print("Data is pre-averaged. Cannot set averaging timescale.")
        elif avg_timescale >= 0:
            self.avg_timescale = avg_timescale
        else:
            print("Invalid timescale. Must be greater than or equal to 0. Setting to 0.")
            self.avg_timescale = 0

    def ftmatrix(self, pdim, xdim, ydim, uvlist, mask=[]):
        """Return a DFT matrix for the xdim*ydim image with pixel width pdim
        that extracts spatial frequencies of the uv points in uvlist.
        """

        xlist = torch.arange(0, -xdim, -1, dtype=torch.float64).to(self.device) * pdim + (pdim * xdim) / 2.0 #- pdim / 2.0 # where you set the coordinates, mid-point or edge
        ylist = torch.arange(0, -ydim, -1, dtype=torch.float64).to(self.device) * pdim + (pdim * ydim) / 2.0 #- pdim / 2.0

        ftmatrices = [eh.observing.pulses.deltaPulse2D(2 * torch.pi * uv[0], 2 * torch.pi * uv[1], pdim, dom="F") *
                      torch.outer(torch.exp(2j * torch.pi * xlist * uv[0]), torch.exp(2j * torch.pi * ylist * uv[1]))
                      for uv in uvlist]

        ftmatrices = torch.stack(ftmatrices).reshape(len(uvlist), xdim * ydim)

        if len(mask):
            ftmatrices = ftmatrices[:, mask]

        return ftmatrices


    def get_CI_MCerror(self, img, n=100, add_th_noise=True, th_noise_factor=1, useObs=False,
                        fov=None, fovx=None, fovy=None, intensity=0, stokes=False,
                        ttype=None):
        imgs = torch.tensor(np.array([img for _ in range(n)]))
        noisy_CIs = self.FTCI(imgs, add_th_noise=add_th_noise, th_noise_factor=th_noise_factor, useObs=useObs,
                        fov=fov, fovx=fovx, fovy=fovy, intensity=intensity, stokes=stokes,
                        ttype=ttype)
        CI_sigmas = torch.std(noisy_CIs, dim=0)
        self.CI_sigmas = CI_sigmas
        return CI_sigmas

    def replace_obs_vis(self, img, obs=None, xfov=225, yfov=225, stokes=False, ttype='DFT'): # only implemented I for now
        if obs == None:
            obs = self.obslist[0]
        uv = torch.tensor(np.stack([self.obslist[0].data['u'], self.obslist[0].data['v']])).swapaxes(0,1).to(self.device)
        new_vis = self.Visibilities(img.swapaxes(1,2), uv=uv, xfov=xfov, yfov=yfov, stokes=stokes, ttype=ttype)[0].detach().cpu().numpy()

        # Image = eh.image.Image(img.squeeze(), xfov/1e6/206265/img.shape[-1], obs.ra, obs.dec, rf=obs.rf, mjd=obs.mjd)
        # new_obs = Image.observe_same(obs, ttype='direct')
        
        obs.data['vis'] = new_vis
        # obs.data['sigma'] = new_obs.data['sigma']
        self.obslist[0] = obs
        self.set_class_quantities_from_obslist(self.ehtimAvg, self.avg_timescale)
        return 
