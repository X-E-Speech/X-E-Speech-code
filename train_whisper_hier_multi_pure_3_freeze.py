import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
torch.cuda.empty_cache() 
import commons
import utils
from data_utils_whisper_hier_multi_pure import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models_whisper_hier_multi_pure import (
  SynthesizerTrn_3,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
#from text.symbols import symbols
from text_cn.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '65534'
  os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256"
  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [50,100,150,200,250,300,350,400,450,500,600],#,700,800,900,700,800,900,1000
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=True,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  print('len(symbols)',len(symbols))
  net_g = SynthesizerTrn_3(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  # optim_g = torch.optim.AdamW(
  #     [{'params': net_g.parameters(), 'initial_lr': hps.train.learning_rate}],
  #     hps.train.learning_rate, 
  #     betas=hps.train.betas, 
  #     eps=hps.train.eps)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  # net_g = DDP(net_g, device_ids=[rank],find_unused_parameters=True)
  # net_d = DDP(net_d, device_ids=[rank],find_unused_parameters=True)


  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)

    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)

    global_step = (epoch_str - 1) * len(train_loader)
  except:

    epoch_str = 1
    global_step = 0

  for name, param in net_g.named_parameters():
      if "enc_spk" in name:
          #print("freezing",name)
          param.requires_grad = False
      if "dec" in name:
          #print("freezing",name)
          param.requires_grad = False
      # if "enc_q" in name:
      #     #print("freezing",name)
      #     param.requires_grad = False
      # if "enc_whisper" in name:
      #     #print("freezing",name)
      #     param.requires_grad = False
      if "flow_whisper_to_spec" in name:
          #print("freezing",name)
          param.requires_grad = False
      # if "dp" in name:
      #     #print("freezing",name)
      #     param.requires_grad = False
      # if "enc_p" in name:
      #     #print("freezing",name)
      #     param.requires_grad = False
      # if "flow_text_to_whisper" in name:
      #     #print("freezing",name)
      #     param.requires_grad = False
      # if "lang_emb_g" in name:
      #     #print("freezing",name)
      #     param.requires_grad = False
  # for name, param in net_g.named_parameters():
  #     print(name,param.requires_grad)
  # for name, param in net_g.state_dict(keep_vars=True).items():
  #     print(name,param.requires_grad)
  with open('check_grad.txt', 'w') as file:
    for name, param in net_g.named_parameters():
        file.write(f'{name} {param.requires_grad}\n')

  with open('check_grad.txt', 'a') as file:
      file.write('\n')  # Add a newline to separate the two sets of information

      for name, param in net_g.state_dict(keep_vars=True).items():
          file.write(f'{name} {param.requires_grad}\n')

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths,weo,weo_lengths, y, y_lengths,ref_mel, lang_id) in enumerate(train_loader):#这里是需要改的，包括langid和refmel
    torch.cuda.empty_cache()
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    weo, weo_lengths = weo.cuda(rank, non_blocking=True), weo_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    ref_mel=ref_mel.cuda(rank, non_blocking=True)
    lang_id=lang_id.cuda(rank, non_blocking=True)
    # ref_spec=ref_spec.cuda(rank, non_blocking=True)
    # ref_mel = spec_to_mel_torch(
    #       ref_spec, 
    #       hps.data.filter_length, 
    #       hps.data.n_mel_channels, 
    #       hps.data.sampling_rate,
    #       hps.data.mel_fmin, 
    #       hps.data.mel_fmax)
    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,z_mask_weo,\
      (z, z_p, m_p, logs_p, m_q, logs_q,z_p_weo,logs_q_weo,m_q_weo,z_p_p) = net_g(x, x_lengths, spec, spec_lengths,weo,weo_lengths,ref_mel,lang_id)#,z_p_p
      """
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      """
      mel = mel_spectrogram_torch(
          y.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      # if not mel.equal(spec):
      #   print(mel.size(),spec.size())
      #   print('max(mel),min(mel),max(spec),min(spec),max(ref_mel),min(ref_mel)',torch.max(mel),torch.min(mel),torch.max(spec),torch.min(spec),torch.max(ref_mel),torch.min(ref_mel))
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 
      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel  

        loss_kl_text_to_whisper = kl_loss(z_p_weo, logs_q_weo,m_p, logs_p,  z_mask_weo) * hps.train.c_kl * hps.train.c_kl_weo2text#1
        loss_kl_whisper_to_spec = kl_loss(z_p, logs_q,m_q_weo, logs_q_weo,  z_mask) * hps.train.c_kl * hps.train.c_kl_spec2weo#1
        loss_kl_text_to_spec= kl_loss(z_p_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl * hps.train.c_kl_spec2text#1

        loss_fm = feature_loss(fmap_r, fmap_g)# * 0.5
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl_text_to_whisper+loss_kl_whisper_to_spec+loss_kl_text_to_spec

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl_text_to_whisper/1,loss_kl_whisper_to_spec/1,loss_kl_text_to_spec/1]#
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl_text_to_whisper": loss_kl_text_to_whisper,"loss/g/kl_whisper_to_spec": loss_kl_whisper_to_spec})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
      # if global_step % 10 == 0:
      #   evaluate(hps, net_g, eval_loader, writer_eval)
      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths,weo,weo_lengths, y, y_lengths,ref_mel, lang_id) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        weo, weo_lengths = weo.cuda(0), weo_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        ref_mel=ref_mel.cuda(0)
        lang_id=lang_id.cuda(0)

        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        weo = weo[:1]
        weo_lengths = weo_lengths[:1]
        
        
        
        lang_id=lang_id[:1]

        ref_mel_vc=ref_mel[1:2]
        ref_mel=ref_mel[:1]

        y_vc=y[1:2]
        y = y[:1]

        y_lengths_vc = y_lengths[1:2]
        y_lengths = y_lengths[:1]
        break
      y_hat, attn, mask, *_ = generator.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
      """
      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      """
      mel = mel_spectrogram_torch(
        y.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    #if global_step == 0:
    image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
    audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})


    y_hat_vocoder,  mask_vocoder = generator.vocoder(spec, spec_lengths, mel=ref_mel,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)
    y_hat_lengths_vocoder = mask_vocoder.sum([1,2]).long() * hps.data.hop_length

    
    audio_dict.update({"gen/audio_vocoder": y_hat_vocoder[0,:,:y_hat_lengths_vocoder[0]]})


    y_hat, attn, mask, *_ = generator.infer(x, x_lengths, mel=ref_mel_vc,lang=lang_id,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)
    y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
    audio_dict.update({"gen/audio_clone": y_hat[0,:,:y_hat_lengths[0]]})


    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
