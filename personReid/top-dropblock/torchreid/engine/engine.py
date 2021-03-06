from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import time
import datetime
import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging 

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

import torchreid
from torchreid.utils import AverageMeter, visualize_ranked_results, save_checkpoint, re_ranking, mkdir_if_missing, visualize_ranked_activation_results, visualize_ranked_threshold_activation_results, visualize_ranked_mask_activation_results
from torchreid.losses import DeepSupervision
from torchreid import metrics

from torchreid.utils import read_image
# from astropy.io.ascii.tests.test_ecsv import data

GRID_SPACING = 10
VECT_HEIGTH = 10

from configs import runInfo
from gpuinfo import GPUInfo

class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_gpu=True):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None

        # check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def run(self, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, test_only=False, print_freq=10,
            dist_metric='euclidean', normalize_feature=False, visrank=False, visrankactiv=False, visrankactivthr=False, maskthr=0.7, visrank_topk=10,
            use_metric_cuhk03=False, ranks=[1, 5, 10, 20], rerank=False, visactmap=False, vispartmap=False, visdrop=False, visdroptype='random'):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            print_freq (int, optional): print_frequency. Default is 10.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
            visactmap (bool, optional): visualizes activation maps. Default is False.
        """
        trainloader, testloader = self.datamanager.return_dataloaders()

        if visrank and not test_only:
            raise ValueError('visrank=True is valid only if test_only=True')

        if visrankactiv and not test_only:
            raise ValueError('visrankactiv=True is valid only if test_only=True')

        if visrankactivthr and not test_only:
            raise ValueError('visrankactivthr=True is valid only if test_only=True')

        if visdrop and not test_only:
            raise ValueError('visdrop=True is valid only if test_only=True')

        if test_only:
            self.test(
                0,
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrankactiv=visrankactiv,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                maskthr=maskthr,
                visrankactivthr=visrankactivthr,
                visdrop=visdrop,
                visdroptype=visdroptype
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        if visactmap:
            self.visactmap(testloader, save_dir, self.datamanager.width, self.datamanager.height, print_freq)
            return

        if vispartmap:
            self.vispartmap(testloader, save_dir, self.datamanager.width, self.datamanager.height, print_freq)
            return

        time_start = time.time()
        print('=> Start training')

        for epoch in range(start_epoch, max_epoch):
            self.train(epoch, max_epoch, trainloader, fixbase_epoch, open_layers, print_freq)
            
            if (epoch+1)>=start_eval and eval_freq>0 and (epoch+1)%eval_freq==0 and (epoch+1)!=max_epoch:
                rank1 = self.test(
                    epoch,
                    testloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrankactiv=visrankactiv,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    rerank=rerank,
                    maskthr=maskthr,
                    visrankactivthr=visrankactivthr
                )
                self._save_checkpoint(epoch, rank1, save_dir)

        if max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                epoch,
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrankactiv=visrankactiv,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                maskthr=maskthr,
                visrankactivthr=visrankactivthr
            )
            self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is None:
            self.writer.close()

    def train(self):
        r"""Performs training on source datasets for one epoch.

        This will be called every epoch in ``run()``, e.g.

        .. code-block:: python
            
            for epoch in range(start_epoch, max_epoch):
                self.train(some_arguments)

        .. note::
            
            This must be implemented in subclasses.
        """
        raise NotImplementedError

    def test(self, epoch, testloader, dist_metric='euclidean', normalize_feature=False,
             visrank=False, visrankactiv = False, visrank_topk=10, save_dir='', use_metric_cuhk03=False,
             ranks=[1, 5, 10, 20], rerank=False, maskthr=0.7, visrankactivthr=False, visdrop=False, visdroptype = 'random'):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        targets = list(testloader.keys())
        
        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            queryloader = testloader[name]['query']
            galleryloader = testloader[name]['gallery']
            rank1 = self._evaluate(
                epoch,
                dataset_name=name,
                queryloader=queryloader,
                galleryloader=galleryloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrankactiv=visrankactiv,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
                maskthr=maskthr,
                visrankactivthr=visrankactivthr,
                visdrop=visdrop,
                visdroptype=visdroptype
            )
        
        return rank1

    @torch.no_grad()
    def _evaluate(self, epoch, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False, visrankactiv = False,
                  visrank_topk=10, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                  rerank=False, visrankactivthr = False, maskthr=0.7, visdrop=False, visdroptype='random'):
        batch_time = AverageMeter()

        print('Extracting features from query set ...')
        qf, qa, q_pids, q_camids, qm = [], [], [], [], [] # query features, query activations, query person IDs, query camera IDs and image drop masks
        # for _, data in enumerate(queryloader):
        #     print(_, data)
        #     print(type(data))
        #     imgs, pids, camids = self._parse_data_for_eval(data)
        #     if self.use_gpu:
        #         imgs = imgs.cuda()
        #     end = time.time()
        #     print(imgs.size())
        #     features = self._extract_features(imgs)
        #     activations = self._extract_activations(imgs)
        #     dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
        #     batch_time.update(time.time() - end)
        #     features = features.data.cpu()
        #     qf.append(features)
        #     qa.append(torch.Tensor(activations))
        #     qm.append(torch.Tensor(dropmask))
        #     q_pids.extend(pids)
        #     q_camids.extend(camids)

        query_dir_path = "/home/gram/JCW/covid19_cctv_analyzer/top-dropblock/data/query/"
        query_img_path = ["01.PNG", "002.PNG"]
        
        for img_path in query_img_path:
            imgs = read_image(query_dir_path + img_path)    # imgs type : <class 'PIL.Image.Image'>
            imgs = self.datamanager.transform_te(imgs)      # imgs type : <class 'torch.Tensor'>
            imgs = torch.unsqueeze(imgs, 0)                 # ????????? ????????? ?????????
            pids = [int(img_path.split(".")[0])]            # list ??? ?????????
            # print("pids : ", pids)
            camids = [0]                                    # list??? ????????? , ????????? ????????? ??????
            
            if self.use_gpu:
                imgs = imgs.cuda()
            # print(imgs.size())
            # print("imgs type : t", type(imgs))
            end = time.time()
            features = self._extract_features(imgs)
            activations = self._extract_activations(imgs)
            dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            qa.append(torch.Tensor(activations))
            qm.append(torch.Tensor(dropmask))
            q_pids.extend(pids)
            q_camids.extend(camids)
        
        qf = torch.cat(qf, 0)
        qm = torch.cat(qm, 0)
        qa = torch.cat(qa, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))
        exit(0)
        print('Extracting features from gallery set ...')
        gf, ga, g_pids, g_camids, gm = [], [], [], [], [] # gallery features, gallery activations,  gallery person IDs, gallery camera IDs and image drop masks
        end = time.time()
        for _, data in enumerate(galleryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            activations = self._extract_activations(imgs)
            dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            ga.append(torch.Tensor(activations))
            gm.append(torch.Tensor(dropmask))
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        gm = torch.cat(gm, 0)
        ga = torch.cat(ga, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        #always show results without re-ranking first
        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)
            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                use_metric_cuhk03=use_metric_cuhk03
            )

            print('** Results with Re-Ranking**')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))


        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_'+dataset_name),
                topk=visrank_topk
            )
        if visrankactiv:
            visualize_ranked_activation_results(
                distmat,
                qa,
                ga,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrankactiv_'+dataset_name),
                topk=visrank_topk
            )
        if visrankactivthr:
            visualize_ranked_threshold_activation_results(
                distmat,
                qa,
                ga,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrankactivthr_'+dataset_name),
                topk=visrank_topk,
                threshold=maskthr
            )
        if visdrop:
            visualize_ranked_mask_activation_results(
                distmat,
                qa,
                ga,
                qm,
                gm,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visdrop_{}_{}'.format(visdroptype, dataset_name)),
                topk=visrank_topk
            )

        return cmc[0]

    @torch.no_grad()
    def visactmap(self, testloader, save_dir, width, height, print_freq):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        for target in list(testloader.keys()):
            queryloader = testloader[target]['query']
            # original images and activation maps are saved individually
            actmap_dir = osp.join(save_dir, 'actmap_'+target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(queryloader):
                imgs, paths = data[0], data[3]
                if self.use_gpu:
                    imgs = imgs.cuda()
                
                # forward to get convolutional feature maps
                try:
                    outputs = self.model(imgs, return_featuremaps=True)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_featuremaps". ' \
                                    'Please add return_featuremaps as an input argument to forward(). When ' \
                                    'return_featuremaps=True, return feature maps only.')
                
                if outputs.dim() != 4:
                    raise ValueError('The model output is supposed to have ' \
                                     'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs.dim()))
                
                # compute activation maps
                outputs = (outputs**2).sum(1)
                b, h, w = outputs.size()
                outputs = outputs.view(b, h*w)
                outputs = F.normalize(outputs, p=2, dim=1)
                outputs = outputs.view(b, h, w)
                
                if self.use_gpu:
                    imgs, outputs = imgs.cpu(), outputs.cpu()

                for j in range(outputs.size(0)):
                    # get image name
                    path = paths[j]
                    imname = osp.basename(osp.splitext(path)[0])
                    
                    # RGB image
                    img = imgs[j, ...]
                    for t, m, s in zip(img, imagenet_mean, imagenet_std):
                        t.mul_(s).add_(m).clamp_(0, 1)
                    img_np = np.uint8(np.floor(img.numpy() * 255))
                    img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)
                    
                    # activation map
                    am = outputs[j, ...].numpy()
                    am = cv2.resize(am, (width, height))
                    am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                    am = np.uint8(np.floor(am))
                    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
                    
                    # overlapped
                    overlapped = img_np * 0.4 + am * 0.6
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)

                    # save images in a single figure (add white spacing between images)
                    # from left to right: original image, activation map, overlapped image
                    grid_img = 255 * np.ones((height, 3*width+2*GRID_SPACING, 3), dtype=np.uint8)
                    grid_img[:, :width, :] = img_np[:, :, ::-1]
                    grid_img[:, width+GRID_SPACING: 2*width+GRID_SPACING, :] = am
                    grid_img[:, 2*width+2*GRID_SPACING:, :] = overlapped
                    cv2.imwrite(osp.join(actmap_dir, imname+'.jpg'), grid_img)

                if (batch_idx+1) % print_freq == 0:
                    print('- done batch {}/{}'.format(batch_idx+1, len(queryloader)))

    @torch.no_grad()
    def vispartmap(self, testloader, save_dir, width, height, print_freq):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        for target in list(testloader.keys()):
            queryloader = testloader[target]['query']
            # original images and activation maps are saved individually
            actmap_dir = osp.join(save_dir, 'partmap_'+target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing parts activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(queryloader):
                imgs, paths = data[0], data[3]
                if self.use_gpu:
                    imgs = imgs.cuda()

                # forward to get convolutional feature maps
                try:
                    outputs_list = self.model(imgs, return_partmaps=True)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_partmaps". ' \
                                    'Please add return_partmaps as an input argument to forward(). When ' \
                                    'return_partmaps=True, return feature maps only.')
                if outputs_list[0][0].dim() != 4:
                    raise ValueError('The model output is supposed to have ' \
                                     'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs_list[0][0].dim()))


                #print stats between parts and weights
                print("First image stats")
                w = []
                for i, (_, _, _, weights) in enumerate(outputs_list):
                    print("\tpart{} min {} max {}".format(i, torch.min(weights[0, ...]), torch.max(weights[0, ...])))
                    w.append(weights)
                print("Second image stats")
                for i, (_, _, _, weights) in enumerate(outputs_list):
                    print("\tpart{} min {} max {}".format(i, torch.min(weights[1, ...]), torch.max(weights[1, ...])))
                print("Difference")
                for i, (_, _, _, weights) in enumerate(outputs_list):
                    print("\tpart{} min {} max {} mean {}".format(i, torch.min(weights[0, ...] - weights[1, ...]), torch.max(weights[0, ...] - weights[1, ...]), torch.mean(weights[0, ...] - weights[1, ...])))
                print("\tbetween min {} max {} mean {}".format(torch.min(w[0] - w[1]), torch.max(w[0] - w[1]), torch.mean(w[0] - w[1])))

                for part_ind, (outputs, weights, _, _) in enumerate(outputs_list):
                    # compute activation maps
                    b, c, h, w = outputs.size()
                    outputs = (outputs**2).sum(1)
                    outputs = outputs.view(b, h*w)
                    outputs = F.normalize(outputs, p=2, dim=1)
                    outputs = outputs.view(b, h, w)

                    weights = weights.view(b, c)
                    weights = F.normalize(weights, p=2, dim=1)
                    weights = weights.view(b, 1, c)

                    if self.use_gpu:
                        imgs, outputs, weights = imgs.cpu(), outputs.cpu(), weights.cpu()

                    for j in range(outputs.size(0)):
                        # get image name
                        path = paths[j]
                        imname = osp.basename(osp.splitext(path)[0])

                        # RGB image
                        img = imgs[j, ...].clone()
                        for t, m, s in zip(img, imagenet_mean, imagenet_std):
                            t.mul_(s).add_(m).clamp_(0, 1)
                        img_np = np.uint8(np.floor(img.numpy() * 255))
                        img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)
                        
                        # activation map
                        am = outputs[j, ...].numpy()
                        am = cv2.resize(am, (width, height))
                        am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                        am = np.uint8(np.floor(am))
                        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                        #parts activation map
                        pam = weights[j, ...].numpy()
                        pam = np.resize(pam, (VECT_HEIGTH, c)) #expand to create larger vectors for better visuallization
                        pam = cv2.resize(pam, (3*width+2*GRID_SPACING, VECT_HEIGTH))
                        pam = 255 * (pam - np.max(pam)) / (np.max(pam) - np.min(pam) + 1e-12)
                        pam = np.uint8(np.floor(pam))
                        pam = cv2.applyColorMap(pam, cv2.COLORMAP_JET)

                        # overlapped
                        overlapped = img_np * 0.4 + am * 0.6
                        overlapped[overlapped>255] = 255
                        overlapped = overlapped.astype(np.uint8)

                        # save images in a single figure (add white spacing between images)
                        # from left to right: original image, activation map, overlapped image
                        grid_img = 255 * np.ones((height + GRID_SPACING + VECT_HEIGTH, 3*width+2*GRID_SPACING, 3), dtype=np.uint8)
                        grid_img[:height, :width, :] = img_np[:, :, ::-1]
                        grid_img[:height, width+GRID_SPACING: 2*width+GRID_SPACING, :] = am
                        grid_img[:height, 2*width+2*GRID_SPACING:, :] = overlapped
                        grid_img[height + GRID_SPACING:, :, :] = pam

                        cv2.imwrite(osp.join(actmap_dir, imname+'_{}.jpg'.format(part_ind)), grid_img)

                    if (batch_idx+1) % print_freq == 0:
                        print('- done batch {}/{} part {}/{}'.format(batch_idx+1, len(queryloader), part_ind + 1, len(outputs_list)))

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _extract_activations(self, input):
        self.model.eval()
        outputs = self.model(input, return_featuremaps=True)
        outputs = (outputs**2).sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h*w)
        outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)
        activations = []
        for j in range(outputs.size(0)):
            # activation map
            am = outputs[j, ...].cpu().numpy()
            am = cv2.resize(am, (self.datamanager.width, self.datamanager.height))
            am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
            activations.append(am)
        return np.array(activations)

    def _extract_drop_masks(self, input, visdrop, visdroptype):
        self.model.eval()
        drop_top = (visdroptype == 'top')
        outputs = self.model(input, drop_top=drop_top, visdrop=visdrop)
        outputs = outputs.mean(1)
        masks = []
        for j in range(outputs.size(0)):
            # drop masks
            dm = outputs[j, ...].cpu().numpy()
            dm = cv2.resize(dm, (self.datamanager.width, self.datamanager.height))
            masks.append(dm)
        return np.array(masks)

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        print("_parse :" , imgs.size())
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)
        
    def test_only(self, dist_metric='euclidean', normalize_feature=False,
                visrank=False, visrankactiv = False, visrank_topk=10, save_dir='', use_metric_cuhk03=False,
                ranks=[1, 5, 10, 20], rerank=False, maskthr=0.7, visrankactivthr=False, visdrop=False, visdroptype = 'random',
                gallery_data=None, query_image_path=''):

            rank1 = self._evaluate_test_only(
                    # epoch,
                    # dataset_name=name,
                    # queryloader=queryloader,
                    # galleryloader=galleryloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrankactiv=visrankactiv,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    rerank=rerank,
                    maskthr=maskthr,
                    visrankactivthr=visrankactivthr,
                    visdrop=visdrop,
                    visdroptype=visdroptype,
                    gallery_data=gallery_data, # [(img, pid, camid)]
                    query_image_path=query_image_path,
                )
            
            return rank1

    @torch.no_grad()
    def _evaluate_test_only(self, 
                dist_metric='euclidean', normalize_feature=False, visrank=False, visrankactiv = False,
                visrank_topk=10, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                rerank=False, visrankactivthr = False, maskthr=0.7, visdrop=False, visdroptype='random',
                gallery_data=None, query_image_path=''):
        batch_time = AverageMeter()

        # print('Extracting features from query set ...')
        qf, qa, q_pids, q_camids, qm = [], [], [], [], [] # query features, query activations, query person IDs, query camera IDs and image drop masks

        # ?????? ???????????? ?????? ?????? ??????
        # ~/covid19_cctv_analyzer
        root_path = os.path.dirname(
            os.path.abspath(os.path.dirname(
                os.path.abspath(os.path.dirname(
                    os.path.abspath(os.path.dirname(
                        os.path.abspath(os.path.dirname(__file__)))))))))
        query_dir_path = root_path + "/" + query_image_path
        query_img_path = os.listdir(query_dir_path)
        # print("Num of query set: ", len(query_img_path))
        for img_path in query_img_path:
            # img_path = query_img_path[0]
            imgs = read_image(query_dir_path + img_path)    # imgs type : <class 'PIL.Image.Image'>
            imgs = self.datamanager.transform_te(imgs)      # imgs type : <class 'torch.Tensor'>
            # print("* query size: ", imgs.size())
            imgs = torch.unsqueeze(imgs, 0)                 # ????????? ????????? ?????????
            # print("* unsqueeze query size: ", imgs.size())
            pids = [int(img_path.split("_")[0])]
            camids = [0]                                    # list??? ????????? , ????????? ????????? ??????
            
            # print(" * right before error (imgs.cuda()) * ")
            # GPUInfo.get_users(1)
            # GPUInfo.get_info()
            if self.use_gpu:
                device = torch.device("cuda:{}".format(runInfo.reidGPU))
                imgs = imgs.to(device)
                # imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            # print("* feature size: ", features.size())
            activations = self._extract_activations(imgs)
            dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            qa.append(torch.Tensor(activations))
            qm.append(torch.Tensor(dropmask))
            q_pids.extend(pids)
            q_camids.extend(camids)
        
        qf = torch.cat(qf, 0)
        # print("* query features size: ", qf.size())
        qm = torch.cat(qm, 0)
        qa = torch.cat(qa, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        # print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))
        
        # print('Extracting features from gallery set ...')
        '''
            gf = gallery features, 
            ga = gallery activations,  
            g_pids = gallery person IDs, 
            g_camids = gallery camera IDs,
            gm = image drop masks,
            g_pIdx = gallery's personIdx in shm.data.people -> return value
        '''
        gf, ga, g_pids, g_camids, gm, g_pIdx = [], [], [], [], [], [] 
        end = time.time()
        for data in gallery_data:
            imgs, pids, camids, pIdx = data
            imgs = self.datamanager.transform_te(imgs)      # imgs type : <class 'torch.Tensor'>
            imgs = torch.unsqueeze(imgs, 0)   
            
            if self.use_gpu:
                device = torch.device("cuda:{}".format(runInfo.reidGPU))
                imgs = imgs.to(device)
                # imgs = imgs.cuda()
                
            end = time.time()
            features = self._extract_features(imgs)
            activations = self._extract_activations(imgs)
            dropmask = self._extract_drop_masks(imgs, visdrop, visdroptype)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            ga.append(torch.Tensor(activations))
            gm.append(torch.Tensor(dropmask))
            g_pids.append(pids)
            g_camids.append(camids)
            g_pIdx.append(pIdx)
            
        gf = torch.cat(gf, 0)
        gm = torch.cat(gm, 0)
        ga = torch.cat(ga, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_pIdx = np.asarray(g_pIdx)
        # print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        # print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            # print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        # print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()


        # distmat??? ????????? ?????? sorting
        num_q, num_g = distmat.shape # 5, 36
        #### print distmat top1~10
        mean_top1 = []
        mean_top2 = []
        dist_indices = np.argsort(distmat, axis=1)
        
        # result = open("/home/gram/JCW/covid19_cctv_analyzer_multi_proc/top-dropblock/frame_res_6.txt", "a")
        
        # # Distance ??? ?????? 1 - top10??? distance??? ??????, top10??? gid??? ??????
        # dist_sorting = np.sort(distmat, axis=1)
        # if dist_metric == 'cosine':
        #     dist_sorting = dist_sorting[:, ::-1]
        # for dist in dist_sorting:
        #     mean_top1.append(dist[0])
        #     mean_top2.append(dist[1])
        #     print(" * dist sorting; ", dist[:10]) 
        #     result.write(" * dist sorting; {}\n".format(dist[:10])) 
            
        # print("mean_top1: ", sum(mean_top1)/len(mean_top1))
        # print("mean_top2: ", sum(mean_top2)/len(mean_top2))
        # result.write("mean_top1: {}\n".format(sum(mean_top1)/len(mean_top1)))
        # result.write("mean_top2: {}\n".format(sum(mean_top2)/len(mean_top2)))

        # # Distance ??? ?????? 2 - top10??? distance??? ??????, top10??? gid??? ??????        
        # matches = (g_pids[dist_indices] == q_pids[:, np.newaxis]).astype(np.int32)
        # distmat_sort = np.sort(distmat, axis=1)
        # total_mean = []
        # total_max = []
        # for i, match in enumerate(matches):
        #     equal_dist = (distmat_sort[i]*match)[:20] # top20 ????????? query == gallery??? ????????? dist??? ?????????
        #     mean_dist = sum(equal_dist)/sum(match) # ??? dist?????? mean?????? ??????
        #     total_mean.append(mean_dist)
        #     total_max.append(np.max(equal_dist))
        #     # print("matchng: {}, mean = {}".format(equal_dist, mean_dist ) )
        # print("matching Total mean: ", sum(total_mean)/len(total_mean))
        # print("matching Total max of mean ", max(total_mean))
        # print("matching Total max of equal (not for mean) ", max(total_max))
        # print("matching Total mean of max", sum(total_max)/len(total_max))
        
        # # top10??? gid??? ??????  
        # for nq in range(num_q):
        #     print(" * Query PID #{}'s ranking :".format(q_pids[nq]), g_pids[dist_indices][nq][:15])
        #     result.write(" * Query PID #{}'s ranking :{}\n".format(q_pids[nq], g_pids[dist_indices][nq][:15]))
            
        # result.close()
        
        # #always show results without re-ranking first
        # print('Computing CMC and mAP ...')
        # cmc, mAP = metrics.evaluate_rank(
        #     distmat,
        #     q_pids,
        #     g_pids,
        #     q_camids,
        #     g_camids,
        #     use_metric_cuhk03=use_metric_cuhk03
        # )

        # print('** Results **')
        # print('mAP: {:.1%}'.format(mAP))
        # print('CMC curve')
        # if len(cmc) >= ranks[-1]:
        #     for r in ranks:
        #         print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))

        # if rerank:
        #     print('Applying person re-ranking ...')
        #     distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
        #     distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
        #     distmat = re_ranking(distmat, distmat_qq, distmat_gg)
            
            #### print distmat for top1~10s
            # dist_indices = np.argsort(distmat, axis=1)
            # dist_indices = np.transpose(dist_indices)
            # for dist in np.sort(distmat, axis=1):
            #     print(" * dist sorting; ", dist[:10]) 
                
            # print('Computing CMC and mAP ...')
            # cmc, mAP = metrics.evaluate_rank(
            #     distmat,
            #     q_pids,
            #     g_pids,
            #     q_camids,
            #     g_camids,
            #     use_metric_cuhk03=use_metric_cuhk03
            # )

            # print('** Results with Re-Ranking**')
            # print('mAP: {:.1%}'.format(mAP))
            # print('CMC curve')
            # if len(cmc) >= ranks[-1]:
            #     for r in ranks:
            #         print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))


        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_'+dataset_name),
                topk=visrank_topk
            )
        if visrankactiv:
            visualize_ranked_activation_results(
                distmat,
                qa,
                ga,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrankactiv_'+dataset_name),
                topk=visrank_topk
            )
        if visrankactivthr:
            visualize_ranked_threshold_activation_results(
                distmat,
                qa,
                ga,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrankactivthr_'+dataset_name),
                topk=visrank_topk,
                threshold=maskthr
            )
        if visdrop:
            visualize_ranked_mask_activation_results(
                distmat,
                qa,
                ga,
                qm,
                gm,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visdrop_{}_{}'.format(visdroptype, dataset_name)),
                topk=visrank_topk
            )

        # Get pIdx of the person with the smallest distance
        top_1_indices = np.transpose(dist_indices)[0] # ??? query??? ?????? top1?????? ????????? gallery?????? list
        top_1_indices_idx = np.argmin([distmat[query][idx] for query, idx in enumerate(top_1_indices)])
        top_1_idx = top_1_indices[top_1_indices_idx]
        smallest_dist_pIdx = g_pIdx[top_1_idx]
        
        # Get smallest distance per person
        minDistanceList = np.amin(distmat, axis=0)
        confidenceList = ( 2 - minDistanceList ) / 2
        
        # for debug
        # print('dist_indices: \n{}'.format(dist_indices))
        # print('top_1_indices: \n{}'.format(top_1_indices))
        # print("distmat: \n{}".format(distmat))
        # print('g_pIdx: \n{}'.format(g_pIdx))
        # print('smallest_dist_pIdx: \n{}'.format(smallest_dist_pIdx))
        # print("minDistanceList: \n{}".format(minDistanceList))
        # print("confidenceList: \n{}".format(confidenceList))
        
        return smallest_dist_pIdx, confidenceList