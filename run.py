import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from get_args import get_args
from modules import *
from dataset import CIFAR10, ImageNet, Kodak
from utils import *
from relay_network import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###### Parameter Setting
args = get_args()
args.device = device

job_name = 'JSCC_'+args.channel_mode+'_coop_'+str(args.is_coop)+'_relay_' + args.relay_mode +'_dataset_'+str(args.dataset)+'_cout_'+str(args.cout)+\
            '_sr_link_' + str(args.sr_link) + '_sd_link_' + str(args.sd_link) + '_rd_link_' + str(args.rd_link) #+'_is_adapt_'+ str(args.adapt)

if args.adapt:
    job_name = job_name + '_sr_rng_' + str(args.sr_rng) + '_sd_rng_' + str(args.sd_rng) + '_rd_rng_' + str(args.rd_rng)

print(args)
print(job_name)

frame_size = (32, 32)
src_ratio = args.cout / (3*4*4)

train_set = CIFAR10('datasets/cifar-10-batches-py', 'TRAIN')
valid_set = CIFAR10('datasets/cifar-10-batches-py', 'VALIDATE')
eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')


###### The JSCC Model
source_enc = EncoderCell(c_in=3, c_feat=args.cfeat, c_out=args.cout, attn=args.adapt).to(args.device)

if args.relay_mode == 'AF':
    source_dec = DecoderCell(c_in=args.cout, c_feat=args.cfeat, c_out=3, attn=args.adapt).to(args.device)
    jscc_model = RelayAFR(args, source_enc, source_dec)
else:
    parity_enc = Parity_EncoderCell(c_in=args.cout, c_feat=args.cfeat, c_out=args.cout, attn=args.adapt).to(args.device)
    source_dec = DecoderCell(c_in=2*args.cout, c_feat=args.cfeat, c_out=3, attn=args.adapt).to(args.device)
    jscc_model = RelayPFR1(args, source_enc, source_dec, parity_enc)

# load pre-trained
if args.resume == False:
    pass
else:
    _ = load_weights(job_name, jscc_model)

solver = optim.Adam(jscc_model.parameters(), lr=args.lr)
scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: 0.8)
es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

###### Dataloader
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2
        )

valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
        )

eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2
)


def train_epoch(loader, model, solvers):

    model.train()

    with tqdm(loader, unit='batch') as tepoch:
        for _, (images, _) in enumerate(tepoch):
            
            epoch_postfix = OrderedDict()

            images = images.to(args.device).float()
            
            solvers.zero_grad()
            output = model(images, is_train = True)

            loss = nn.MSELoss()(output, images)
            loss.backward()
            solvers.step()

            epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

            tepoch.set_postfix(**epoch_postfix)


def validate_epoch(loader, model):

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    #msssim_hist = []

    with torch.no_grad():
        with tqdm(loader, unit='batch') as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                images = images.to(args.device).float()

                output = model(images, is_train = False)
                loss = nn.MSELoss()(output, images)

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # Show the snr/loss/psnr/ssim
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
            
            loss_mean = np.nanmean(loss_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std}

        
    return loss_mean, return_aux



if __name__ == '__main__':
    epoch = 0

    while epoch < args.epoch and not args.resume:
        
        epoch += 1
        
        train_epoch(train_loader, jscc_model, solver)

        valid_loss, valid_aux = validate_epoch(valid_loader, jscc_model)

        flag, best, best_epoch, bad_epochs = es.step(torch.Tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, jscc_model)
            break
        else:
            # TODO put this in trainer
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, jscc_model, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % (es.patience//3) == 0:
                scheduler.step()
                print('lr updated: {:.5f}'.format(scheduler.get_last_lr()[0]))



    print('evaluating...')
    #jscc_model.sr_link = 0
    ####### adjust the SNR --- fix sd_link = rd_link
    #jscc_model.sr_link = 8
    for snr in range(0,12,2):
        jscc_model.sd_link, jscc_model.rd_link = snr, snr
        #jscc_model.rd_link = snr
        _, eval_aux = validate_epoch(eval_loader, jscc_model)
        print(eval_aux['psnr'])
        print(eval_aux['ssim'])