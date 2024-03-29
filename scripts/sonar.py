from pathlib import Path
import random
import json
from contextlib import nullcontext
from PIL import Image
from PIL.Image import Image as PILImage
from pprint import pprint as pp
from traceback import print_exc, format_exc
import inspect

import gradio as gr
import torch
import numpy as np
from tqdm.auto import trange

from modules import scripts, devices
from modules.script_callbacks import on_before_image_saved, remove_callbacks_for_function, ImageSaveParams
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
from modules.shared import state, opts, sd_upscalers
from modules.sd_samplers_common import setup_img2img_steps, SamplerData
from modules.sd_samplers_kdiffusion import CFGDenoiser, KDiffusionSampler
from modules.ui import gr_show
from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning
from modules.images import resize_image
from ldm.models.diffusion.ddpm import LatentDiffusion

from typing import List, Tuple, Union, Literal
from torch import Tensor

# local config
SONAR_PATH = Path(scripts.basedir())
CONFIG_FILE = SONAR_PATH / 'config.json'

def load_cfg():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as fh:
        return json.load(fh)
def save_cfg():
    global cfg
    with open(CONFIG_FILE, 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

if CONFIG_FILE.exists():
    cfg = load_cfg()
else:
    cfg = { 'is_scripts': True } ; save_cfg()

if 'global const':
    IS_SCRIPTS: bool = cfg.get('is_scripts', True)
    LABEL_MORPH =  {
        True:  'Move to AlwaysVisible panel at next webui reload',
        False: 'Move to Script dropdown at next webui reload',
    }

    DEFAULT_ENABLE             = False
    DEFAULT_UPSCALE            = False
    DEFAULT_GRID_SEARCH        = False
    DEFAULT_SAMPLER            = 'Euler a'
    DEFAULT_MOMENTUM           = 0.95
    DEFAULT_MOMENTUM_HIST      = 0.75
    DEFAULT_MOMENTUM_GS        = '0.75:0.95:5'
    DEFAULT_MOMENTUM_HIST_GS   = '0.2:0.8:5'
    DEFAULT_MOMENTUM_HIST_INIT = 'zero'
    DEFAULT_MOMENTUM_SIGN      = 'pos'
    DEFAULT_REF_METH           = 'linear'
    DEFAULT_REF_HGF            = 0.2
    DEFAULT_REF_MIN_STEP       = 0.0
    DEFAULT_REF_MAX_STEP       = 0.75
    DEFAULT_REF_IMG            = None
    DEFAULT_UPSCALE_METH       = 'Lanczos'
    DEFAULT_UPSCALE_RATIO      = 1.0
    DEFAULT_UPSCALE_W          = 0
    DEFAULT_UPSCALE_H          = 0

    CHOICE_MOMENTUM_SIGN      = ['pos', 'neg', 'rand']
    CHOICE_MOMENTUM_HIST_INIT = ['zero', 'rand_init', 'rand_new']
    CHOICE_REF_METH           = ['linear', 'euler']
    CHOICE_UPSCALER           = [x.name for x in sd_upscalers]

# debug save latent featmap (when `Euler a`)
#FEAT_MAP_PATH = 'C:\sd-webui_featmaps'
FEAT_MAP_PATH = None

def show_featmap(x, title=''):
    if not FEAT_MAP_PATH: return

    import os
    import matplotlib.pyplot as plt
    os.makedirs(FEAT_MAP_PATH, exist_ok=True)

    x_np = x[0].cpu().numpy()    # [C=4, H=64, W=64]
    x_np_abs = np.abs(x_np)
    print(f'[{title}]')
    print('   x_np:',     x_np    .max(), x_np    .min(), x_np    .mean(), x_np    .std())
    print('   x_np_abs:', x_np_abs.max(), x_np_abs.min(), x_np_abs.mean(), x_np_abs.std())
    for i in range(4):
        plt.axis('off')
        plt.subplot(2, 2, i+1)
        plt.imshow(x_np[i])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FEAT_MAP_PATH, f'{title}.png'))


# the current setting (the wrappers are too deep, we pass it by global var)
settings = {
    'sampler':            DEFAULT_SAMPLER,

    'momentum':           DEFAULT_MOMENTUM,
    'momentum_hist':      DEFAULT_MOMENTUM_HIST,
    'momentum_hist_init': DEFAULT_MOMENTUM_HIST_INIT,
    'momentum_sign':      DEFAULT_MOMENTUM_SIGN,

    'ref_meth':           DEFAULT_REF_METH,
    'ref_hgf':            DEFAULT_REF_HGF,
    'ref_min_step':       DEFAULT_REF_MIN_STEP,
    'ref_max_step':       DEFAULT_REF_MAX_STEP,
    'ref_img':            DEFAULT_REF_IMG,
}

# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓

from modules.processing import *

def StableDiffusionProcessingTxt2Img_sample(self:StableDiffusionProcessingTxt2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    # NOTE: hijack the sampler~
    self.sampler = create_sampler(self.sd_model)

    x = self.rng.next()
    samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
    del x

    if not self.enable_hr:
        return samples

    if self.latent_scale_mode is None:
        decoded_samples = torch.stack(decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(dtype=torch.float32)
    else:
        decoded_samples = None

    with sd_models.SkipWritingToConfig():
        sd_models.reload_model_weights(info=self.hr_checkpoint_info)

    devices.torch_gc()

    return StableDiffusionProcessingTxt2Img_sample_hr_pass(self, samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)

def StableDiffusionProcessingTxt2Img_sample_hr_pass(self:StableDiffusionProcessingTxt2Img, samples, decoded_samples, seeds, subseeds, subseed_strength, prompts):
    if shared.state.interrupted:
        return samples

    self.is_hr_pass = True

    target_width = self.hr_upscale_to_x
    target_height = self.hr_upscale_to_y

    def save_intermediate(image, index):
        """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

        if not self.save_samples() or not opts.save_images_before_highres_fix:
            return

        if not isinstance(image, Image.Image):
            image = sd_samplers.sample_to_image(image, index, approximation=0)

        info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=index)
        images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, info=info, p=self, suffix="-before-highres-fix")

    # NOTE: hijack the sampler~
    self.sampler = create_sampler(self.sd_model)

    if self.latent_scale_mode is not None:
        for i in range(samples.shape[0]):
            save_intermediate(samples, i)

        samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=self.latent_scale_mode["mode"], antialias=self.latent_scale_mode["antialias"])

        # Avoid making the inpainting conditioning unless necessary as
        # this does need some extra compute to decode / encode the image again.
        if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
            image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
        else:
            image_conditioning = self.txt2img_image_conditioning(samples)
    else:
        lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

        batch_images = []
        for i, x_sample in enumerate(lowres_samples):
            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            x_sample = x_sample.astype(np.uint8)
            image = Image.fromarray(x_sample)

            save_intermediate(image, i)

            image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            batch_images.append(image)

        decoded_samples = torch.from_numpy(np.array(batch_images))
        decoded_samples = decoded_samples.to(shared.device, dtype=devices.dtype_vae)

        if opts.sd_vae_encode_method != 'Full':
            self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method
        samples = images_tensor_to_samples(decoded_samples, approximation_indexes.get(opts.sd_vae_encode_method))

        image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

    shared.state.nextjob()

    samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]

    self.rng = rng.ImageRNG(samples.shape[1:], self.seeds, subseeds=self.subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w)
    noise = self.rng.next()

    # GC now before running the next img2img to prevent running out of memory
    devices.torch_gc()

    if not self.disable_extra_networks:
        with devices.autocast():
            extra_networks.activate(self, self.hr_extra_network_data)

    with devices.autocast():
        self.calculate_hr_conds()

    sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))

    if self.scripts is not None:
        self.scripts.before_hr(self)

    samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

    sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())

    self.sampler = None
    devices.torch_gc()

    decoded_samples = decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)

    self.is_hr_pass = False

    return decoded_samples

def StableDiffusionProcessingImg2Img_sample(self:StableDiffusionProcessingImg2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    # NOTE: hijack the sampler~
    self.sampler = create_sampler(self.sd_model)

    x = self.rng.next()

    if self.initial_noise_multiplier != 1.0:
        self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
        x *= self.initial_noise_multiplier

    samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

    if self.mask is not None:
        samples = samples * self.nmask + self.init_latent * self.mask

    del x
    devices.torch_gc()

    return samples

# ↑↑↑ the above is modified from 'modules/processing.py' ↑↑↑


# ↓↓↓ the following is modified from 'k_diffusion/sampling.py' ↓↓↓

from k_diffusion.sampling import to_d, get_ancestral_step, default_noise_sampler

@torch.no_grad()
def sample_naive(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None, *args):
    '''
    采样的本质是寻找降噪模型在某输入维度(图隐层维度)上的不动点，通过反复查询、依近似梯度来优化选点
    你可以基于该模板快速开发 K-Diffusion 采样器 :)
    '''
    # type(model)                                                               modules.sd_samplers.CFGDenoiser
    # type(model.inner_model)                                                   k_diffusion.external.CompVisDenoiser
    # type(model.inner_model.inner_model)                                       ldm.models.diffusion.ddpm.LatentDiffusion
    # type(model.inner_model.inner_model.first_stage_model)                     ldm.models.autoencoder.AutoencoderKL
    # type(model.inner_model.inner_model.cond_stage_model)                      modules.sd_hijack.FrozenCLIPEmbedderWithCustomWords
    # type(model.inner_model.inner_model.cond_stage_model.wrapped)              ldm.modules.encoders.modules.FrozenCLIPEmbedder
    # type(model.inner_model.inner_model.cond_stage_model.wrapped.tokenizer)    transformers.models.clip.tokenization_clip.CLIPTokenizer
    # type(model.inner_model.inner_model.cond_stage_model.wrapped.transformer)  transformers.models.clip.modeling_clip.CLIPTextModel
    # type(model.inner_model.inner_model.model)                                 ldm.models.diffusion.ddpm.DiffusionWrapper
    # type(model.inner_model.inner_model.model.diffusion_model)                 ldm.modules.diffusionmodules.openaimodel.UNetModel
    # x                                                                         Tensor([B, C=4, H, W]), x8 downsampled
    # sigmas                                                                    Tensor([T]), steps
    # extra_args['cond']                                                        MulticondLearnedConditioning, prompt cond
    # extra_args['uncond']                                                      List[List[ScheduledPromptConditioning]], negaivte prompt cond
    # extra_args['image_cond']                                                  Tensor(), mask for img2img; Tensor([1, 5, 1, 1]), dummy for txt2img
    # extra_args['cond_scale']                                                  int, e.g.: 7.0
    # callback                                                                  KDiffusionSampler.callback_state(dict)
    # *args                                                                     see `sampler_extra_params_sonar`

    s_in = x.new_ones([x.shape[0]])         # expand_dim
    for i in trange(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'i': i, 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)    # dy/dx
        dt = sigmas[i + 1] - sigmas[i]      # dx
        x = x + d * dt                      # x += dx * dy/dx
    return x

@torch.no_grad()
def sample_naive_ex(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None):
    sd_model: LatentDiffusion = model.inner_model.inner_model

    momentum           = settings['momentum']
    momentum_sign      = settings['momentum_sign']
    momentum_hist      = settings['momentum_hist']
    momentum_hist_init = settings['momentum_hist_init']
    ref_hgf            = settings['ref_hgf']
    ref_meth           = settings['ref_meth']
    ref_img: PILImage  = settings['ref_img']
    ref_min_step       = settings['ref_min_step']
    ref_max_step       = settings['ref_max_step']

    # init hist momentum memory
    history_d = init_hist_d(momentum_hist_init, x)

    # prepare ref_img latent
    if ref_img is not None:
        ref_img_norm = init_ref_img(sd_model, ref_img, x.shape[0])

    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        if state.interrupted: break

        # denoise step
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'i': i, 'denoised': denoised})
        
        # grad step
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]

        # momentum step
        if momentum < 1.0:
            x, history_d = momentum_step(x, d, dt, momentum, momentum_hist, momentum_sign, history_d)
        else:
            x = x + d * dt      # Euler method original

        # guidance step
        if ref_img is not None and ref_hgf and ref_min_step <= i <= ref_max_step:
            x = guidance_step(x, ref_meth, ref_img_norm, ref_hgf, denoised, sigmas, i)

        # noise step alike ancestral
        if i <= n_steps - 1:
            x = x + torch.randn_like(x) * 1e-5

    return x

@torch.no_grad()
def sample_euler_ex(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    momentum           = settings['momentum']
    momentum_sign      = settings['momentum_sign']
    momentum_hist      = settings['momentum_hist']
    momentum_hist_init = settings['momentum_hist_init']

    history_d = init_hist_d(momentum_hist_init, x)

    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0: x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat

        if momentum < 1.0:
            x, history_d = momentum_step(x, d, dt, momentum, momentum_hist, momentum_sign, history_d)
        else:
            x = x + d * dt      # Euler method original

    return x

@torch.no_grad()
def sample_euler_ancestral_ex(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None, eta=1., s_noise=1., noise_sampler=None):
    momentum           = settings['momentum']
    momentum_sign      = settings['momentum_sign']
    momentum_hist      = settings['momentum_hist']
    momentum_hist_init = settings['momentum_hist_init']

    # x: [1, 4, 64, 64], 外源的高斯随机噪声
    show_featmap(x, 'before sample')

    # 记录梯度历史的惯性
    history_d = init_hist_d(momentum_hist_init, x)

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])     # [B=1]
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        # [1, 4, 64, 64], 一步降噪后，生成的图越来越明显, sigma理解为步长dx
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        show_featmap(denoised, f'denoised (step {i})')

        # 噪声差分 eps 应当服从正太分布
        eps = denoised - x
        show_featmap(eps, f'eps (step {i})')

        # scalar, sigma_down < sigma_up < sigmas[i + 1] < sigmas[i]
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        # # [1, 4, 64, 64], 梯度 = dy/dx = (x - denoised) / sigmas[i]
        d = to_d(x, sigmas[i], denoised)
        show_featmap(d, f'd (step {i}); sigma={sigmas[i]}')

        # ancestral scheduling (down)
        dt = sigma_down - sigmas[i]        # scalar, 沿着梯度移动的步长
        if momentum < 1.0:
            x, history_d = momentum_step(x, d, dt, momentum, momentum_hist, momentum_sign, history_d)
        else:
            x = x + d * dt      # Euler method original
        show_featmap(x, f'x + d x dt (step {i}); sigma_down={sigma_down:.4f}')    # 作画内容逐渐显露
        
        # ancestral scheduling (up)
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            show_featmap(x, f'x + randn (step {i}); sigma_up={sigma_up:.4f}')         # 再被压抑下去

    # x: [1, 4, 64, 64], 采样后是否有语义：有，就是生成图的小图，所以vae decoder做的事基本就是超分
    show_featmap(x, 'after sample')

    return x

# ↑↑↑ the above is modified from 'k_diffusion/sampling.py' ↑↑↑


# ↓↓↓ the following is modified from 'modules/sd_samplers.py' ↓↓↓

all_samplers_sonar = [
    # wrap the well-known samplers
    SamplerData('Euler a', lambda model: KDiffusionSamplerHijack(model, 'sample_euler_ancestral_ex'), ['k_euler_a_ex'], {}),
    SamplerData('Euler',   lambda model: KDiffusionSamplerHijack(model, 'sample_euler_ex'),           ['k_euler_ex'],   {}),
    # my dev-playground
    SamplerData('Naive',   lambda model: KDiffusionSamplerHijack(model, 'sample_naive_ex'),           ['naive_ex'],     {}),
]
all_samplers_sonar_map = {x.name: x for x in all_samplers_sonar}

sampler_extra_params_sonar = {
    # 'sampler_name': ['param1', 'param2', ...]
    'sample_euler_ex': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
}
CHOICE_SAMPLER = [s.name for s in all_samplers_sonar]

def create_sampler(sd_model):
    name = settings['sampler']
    config = all_samplers_sonar_map.get(name, None)
    if not config: raise ValueError(f'implementaion of sampler {name!r} not found')

    sampler = config.constructor(sd_model)
    sampler.config = config
    return sampler

class KDiffusionSamplerHijack(KDiffusionSampler):

    def __init__(self, sd_model, funcname):
        # init the homogenus base sampler
        super().__init__('sample_euler', sd_model)      # the 'funcname' this is dummy

        # NOTE: hijack the sampler object
        self.funcname = funcname
        self.func = globals().get(self.funcname)
        self.extra_params = sampler_extra_params_sonar.get(funcname, [])

    def callback_state(self, d):
        # TODO: exploer later
        return super().callback_state(d)

    def sample(self, p:StableDiffusionProcessing, x:Tensor, 
               conditioning:MulticondLearnedConditioning, unconditional_conditioning:ScheduledPromptConditioning, 
               steps:int=None, image_conditioning:Tensor=None):

        steps = steps or p.steps
        # sigmas: [16=steps+1], sigma[0]=14.6116 && sigma[-1]=0.0 都是常量，中间是递减的插值(指数衰减？)
        sigmas = self.get_sigmas(p, steps)
        # x: [B=1, C=4, H=64, W=64]
        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters
        
        if 'n' in parameters:
            extra_params_kwargs['n'] = steps

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()

        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigmas

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        self.last_latent = x    # [1, 4, 64, 64]
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples

    def sample_img2img(self, p:StableDiffusionProcessing, x:Tensor, noise:Tensor, 
                       conditioning:MulticondLearnedConditioning, unconditional_conditioning:ScheduledPromptConditioning, 
                       steps:int=None, image_conditioning:Tensor=None):
        
        steps, t_enc = setup_img2img_steps(p, steps)
        sigmas = self.get_sigmas(p, steps)
        sigma_sched = sigmas[steps - t_enc - 1:]
        xi = x + noise * sigma_sched[0]
        
        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if 'sigma_min' in parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs['sigma_min'] = sigma_sched[-2]
        if 'sigma_max' in parameters:
            extra_params_kwargs['sigma_max'] = sigma_sched[0]
        if 'n' in parameters:
            extra_params_kwargs['n'] = len(sigma_sched) - 1
        if 'sigma_sched' in parameters:
            extra_params_kwargs['sigma_sched'] = sigma_sched
        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigma_sched

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples

# ↑↑↑ the above is modified from 'modules/sd_samplers.py' ↑↑↑


def init_hist_d(momentum_hist_init:str, x:Tensor) -> Union[Literal[0], Tensor]:
    # memorize delta momentum
    if   momentum_hist_init == 'zero':      history_d = 0
    elif momentum_hist_init == 'rand_init': history_d = x
    elif momentum_hist_init == 'rand_new':  history_d = torch.randn_like(x)
    else: raise ValueError(f'unknown momentum_hist_init: {momentum_hist_init}')
    return history_d

def momentum_step(x:Tensor, d:Tensor, dt:Tensor, momentum:float, momentum_hist:float, momentum_sign:str, history_d):
    # decide correct direction
    sign = momentum_sign
    if sign == 'rand': sign = random.choice(['pos', 'neg'])

    # correct current `d` with momentum
    p = 1.0 - momentum
    if   sign == 'pos': momentum_d = (1.0 - p) * d + p * history_d
    elif sign == 'neg': momentum_d = (1.0 + p) * d - p * history_d
    else: raise ValueError(f'unknown momentum sign {sign}')
    
    # Euler method with momentum
    x = x + momentum_d * dt

    # update momentum history
    q = 1.0 - momentum_hist
    if (isinstance(history_d, int) and history_d == 0):
        history_d = momentum_d
    else:
        if   sign == 'pos': history_d = (1.0 - q) * history_d + q * momentum_d
        elif sign == 'neg': history_d = (1.0 + q) * history_d - q * momentum_d
        else: raise ValueError(f'unknown momentum sign {sign}')

    return x, history_d

def init_ref_img(sd_model:LatentDiffusion, ref_img:PILImage, B:int) -> Tensor:
    x_ref = torch.from_numpy(np.asarray(ref_img)).moveaxis(2, 0)    # [C=3, H, W]
    x_ref = (x_ref / 255) * 2 - 1
    x_ref = x_ref.unsqueeze(dim=0).expand(B, -1, -1, -1)  # [B, C=3, H, W]
    x_ref = x_ref.to(sd_model.first_stage_model.device)

    with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
        latent_ref = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(x_ref))     # [B, C=4, H=64, W=64]

        avg_s = latent_ref.mean(dim=[2, 3], keepdim=True)
        std_s = latent_ref.std (dim=[2, 3], keepdim=True)
        ref_img_norm = (latent_ref - avg_s) / std_s
        ref_img_norm = ref_img_norm.to(x_ref.dtype)

    return ref_img_norm

def guidance_step(x:Tensor, ref_meth:str, ref_img_norm:Tensor, ref_hgf:float, denoised:Tensor, sigmas, i):
    # TODO: make scheduling for hgf?
    if ref_meth == 'euler':
        # rescale `ref_img` to match distribution
        avg_t = denoised.mean(dim=[1, 2, 3], keepdim=True)
        std_t = denoised.std (dim=[1, 2, 3], keepdim=True)
        ref_img_shift = ref_img_norm * std_t + avg_t

        d = to_d(x, sigmas[i], ref_img_shift)
        dt = (sigmas[i + 1] - sigmas[i]) * ref_hgf
        x = x + d * dt

    elif ref_meth == 'linear':
        # rescale `ref_img` to match distribution
        avg_t = x.mean(dim=[1, 2, 3], keepdim=True)
        std_t = x.std (dim=[1, 2, 3], keepdim=True)
        ref_img_shift = ref_img_norm * std_t + avg_t
        x = (1 - ref_hgf) * x + ref_hgf * ref_img_shift
    
    return x


def get_upscale_resolution(p:StableDiffusionProcessing, upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int) -> Tuple[bool, Tuple[int, int]]:
    if upscale_meth == 'None':
        return False, (p.width, p.height)

    if upscale_width == upscale_height == 0:
        if upscale_ratio == 1.0:
            return False, (p.width, p.height)
        else:
            return True, (round(p.width * upscale_ratio), round(p.height * upscale_ratio))
    else:
        if upscale_width  == 0: upscale_width  = round(p.width  * upscale_height / p.height)
        if upscale_height == 0: upscale_height = round(p.height * upscale_width  / p.width)
        return True, (upscale_width, upscale_height)

def parse_gs(seq:str, def_val:float) -> List[float]:
    seq = seq.strip()
    if not seq: return [ def_val ]

    segs = seq.split(':')
    if   len(segs) == 1:
        start = stop = segs[0]
        step = 1
    elif len(segs) == 2:
        start, stop = segs
        step = 3
    elif len(segs) == 3:
        start, stop, step = segs
    else:
        raise ValueError('>> malformatted search list: should follow <start>:<stop>:<step>')

    start, stop = float(start), float(stop)
    assert max(start, stop) <= 1.0
    assert min(start, stop) >= 0.0

    if float(step) >= 1:
        step = int(step)
        return np.linspace(start, stop, step).tolist()
    else:
        step = float(step)
        assert step != 0.0
        assert -1.0 < step < 1.0
        if step > 0.0:
            assert start < stop
            ret = [start]
            while ret[-1] < stop: ret.append(ret[-1] + step)
        if step < 0.0:
            assert start > stop
            ret = [start]
            while ret[-1] > stop: ret.append(ret[-1] + step)
        return ret


class Script(scripts.Script):

    def title(self):
        return 'Sonar'

    def describe(self):
        return "Wrapped samplers with tricks to optimize prompt condition and image latent for better image quality"

    def show(self, is_img2img):
        if IS_SCRIPTS: return True
        else: return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Sonar sampler', open=False) if not IS_SCRIPTS else nullcontext():
            if not IS_SCRIPTS:
                is_enable = gr.Checkbox(label='Enable', value=lambda: DEFAULT_ENABLE)

            with gr.Row(variant='compact', equal_height=True):
                sampler = gr.Radio(label='Base Sampler', value=lambda: DEFAULT_SAMPLER, choices=CHOICE_SAMPLER)
                use_upscale = gr.Checkbox(label='Upscaling', value=lambda: DEFAULT_UPSCALE)

                if IS_SCRIPTS:
                    use_grid_search = gr.Checkbox(label='Grid search', value=lambda: DEFAULT_GRID_SEARCH)

            with gr.Group() as tab_momentum:
                with gr.Row(variant='compact', visible=not DEFAULT_GRID_SEARCH) as tab_m:
                    momentum      = gr.Slider(label='Momentum current', minimum=0.75, maximum=1.0, value=lambda: DEFAULT_MOMENTUM)
                    momentum_hist = gr.Slider(label='Momentum history', minimum=0.0,  maximum=1.0, value=lambda: DEFAULT_MOMENTUM_HIST)

                if IS_SCRIPTS:
                    with gr.Row(variant='compact', visible=DEFAULT_GRID_SEARCH) as tab_m_gs:
                        momentum_gs      = gr.Text(label='Momentum current search list', max_lines=1, value=lambda: DEFAULT_MOMENTUM_GS)
                        momentum_hist_gs = gr.Text(label='Momentum history search list', max_lines=1, value=lambda: DEFAULT_MOMENTUM_HIST_GS)
                    
                    use_grid_search.change(fn=lambda x: [gr_show(not x), gr_show(x)], inputs=use_grid_search, outputs=[tab_m, tab_m_gs])
                
                with gr.Row(variant='compact'):
                    momentum_sign      = gr.Radio(label='Momentum sign (experimental)',         value=lambda: DEFAULT_MOMENTUM_SIGN,      choices=CHOICE_MOMENTUM_SIGN)
                    momentum_hist_init = gr.Radio(label='Momentum history init (experimental)', value=lambda: DEFAULT_MOMENTUM_HIST_INIT, choices=CHOICE_MOMENTUM_HIST_INIT)
                
            with gr.Group(visible=False) as tab_file:
                with gr.Row(variant='compact'):
                    ref_meth = gr.Radio(label='Ref guide step method', value=lambda: DEFAULT_REF_METH, choices=CHOICE_REF_METH)
                    ref_hgf = gr.Slider(label='Ref guide factor', value=lambda: DEFAULT_REF_HGF, minimum=0, maximum=1, step=0.01)
                    ref_min_step = gr.Number(label='Ref start step', value=lambda: DEFAULT_REF_MIN_STEP)
                    ref_max_step = gr.Number(label='Ref stop step', value=lambda: DEFAULT_REF_MAX_STEP)
                
                with gr.Row(variant='compact'):
                    ref_img = gr.Image(label='Reference image file', image_mode=None, type='pil')

            def swith_sampler(sampler:str):
                SHOW_TABS = {
                    # (show_momt, show_file)
                    'Euler a': (True, False),
                    'Euler':   (True, False),
                    'Naive':   (True, True),
                }
                show_momt, show_file = SHOW_TABS[sampler]
                return [
                    gr_show(show_momt),
                    gr_show(show_file),
                ]
            sampler.change(swith_sampler, inputs=[sampler], outputs=[tab_momentum, tab_file])

            with gr.Row(variant='compact', visible=False) as tab_upscale:
                upscale_meth   = gr.Dropdown(label='Upscaler', value=lambda: DEFAULT_UPSCALE_METH,  choices=CHOICE_UPSCALER)
                upscale_ratio  = gr.Slider  (label='Upscale ratio',  value=lambda: DEFAULT_UPSCALE_RATIO, minimum=1.0, maximum=4.0, step=0.1)
                upscale_width  = gr.Slider  (label='Upscale width',  value=lambda: DEFAULT_UPSCALE_W, minimum=0, maximum=2048, step=8)
                upscale_height = gr.Slider  (label='Upscale height', value=lambda: DEFAULT_UPSCALE_H, minimum=0, maximum=2048, step=8)
            use_upscale.change(fn=lambda x: gr_show(x), inputs=use_upscale, outputs=tab_upscale, show_progress=False)

            with gr.Row():
                switch_morph = gr.Checkbox(label=LABEL_MORPH[IS_SCRIPTS])
                def switch_morph_fn():
                    global IS_SCRIPTS
                    IS_SCRIPTS = not IS_SCRIPTS
                    cfg['is_scripts'] = IS_SCRIPTS ; save_cfg()
                switch_morph.change(fn=switch_morph_fn, show_progress=False)

        if IS_SCRIPTS:
            return [
                use_upscale, use_grid_search, sampler, 
                momentum, momentum_hist, momentum_gs, momentum_hist_gs, momentum_hist_init, momentum_sign, 
                ref_meth, ref_hgf, ref_min_step, ref_max_step, ref_img,
                upscale_meth, upscale_ratio, upscale_width, upscale_height
            ]
        else:
            return [
                is_enable,
                use_upscale, sampler, 
                momentum, momentum_hist, momentum_hist_init, momentum_sign, 
                ref_meth, ref_hgf, ref_min_step, ref_max_step, ref_img,
                upscale_meth, upscale_ratio, upscale_width, upscale_height
            ]

    def run(self, p:StableDiffusionProcessing, 
            use_upscale:bool, use_grid_search:bool, sampler:str, 
            momentum:float, momentum_hist:float, momentum_gs:str, momentum_hist_gs:str, momentum_hist_init:str, momentum_sign:str,
            ref_meth:str, ref_hgf:float, ref_min_step:float, ref_max_step:float, ref_img:PILImage,
            upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int
        ):

        # type convert
        if use_grid_search:
            try:
                momentums       = parse_gs(momentum_gs,      momentum)
                momentum_hists  = parse_gs(momentum_hist_gs, momentum_hist)
            except:
                e = format_exc()
                print(e)
                return Processed(p, [], p.seed, f'>> [Sonar] error parse_gs: {e}')
        else:
            momentums       = [momentum]
            momentum_hists  = [momentum_hist]

        n_jobs = len(momentums) * len(momentum_hists)
        if not n_jobs:
            print('momentums:', momentums)
            print('momentum_hists:', momentum_hists)
            return Processed(p, [], p.seed, '>> [Sonar] specified 0 job, are you kidding?')
        state.job_count = n_jobs
        if use_grid_search and n_jobs > 1: print('>> [Sonar] grid search n_jobs:', n_jobs)
        p.seed = get_fixed_seed(p.seed)
        p.extra_generation_params['Sonar sampler'] = sampler

        if ref_img is not None: ref_img = resize_image(1, ref_img, p.width, p.height)

        # save settings to global
        settings['sampler']            = sampler
        settings['momentum']           = momentum
        settings['momentum_hist']      = momentum_hist
        settings['momentum_hist_init'] = momentum_hist_init
        settings['momentum_sign']      = momentum_sign
        settings['ref_meth']           = ref_meth
        settings['ref_min_step']       = int(ref_min_step) if ref_min_step > 1 else round(ref_min_step * p.steps)
        settings['ref_max_step']       = int(ref_max_step) if ref_max_step > 1 else round(ref_max_step * p.steps)
        settings['ref_hgf']            = ref_hgf / 10.0
        settings['ref_img']            = ref_img

        #pp(settings)

        if use_upscale:
            need_upscale, (tgt_w, tgt_h) = get_upscale_resolution(p, upscale_meth, upscale_ratio, upscale_width, upscale_height)
            if need_upscale: print(f'>> [Sonar] upscaling ({p.width}, {p.height}) => ({tgt_w}, {tgt_h})')

        def save_image_hijack(params:ImageSaveParams):
            if use_upscale and need_upscale:
                params.image = resize_image(1, params.image, tgt_w, tgt_h, upscaler_name=upscale_meth)
        on_before_image_saved(save_image_hijack)

        self.sample_saved = p.sample
        if   isinstance(p, StableDiffusionProcessingTxt2Img):
            p.sample = lambda *args, **kwargs: StableDiffusionProcessingTxt2Img_sample(p, *args, **kwargs)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            p.sample = lambda *args, **kwargs: StableDiffusionProcessingImg2Img_sample(p, *args, **kwargs)

        info = None
        imgs = []
        try:
            for m in momentums:
                if state.interrupted: break
                interrupted = False
                for mh in momentum_hists:
                    if state.interrupted: interrupted = True; break

                    p.extra_generation_params['Sonar momentum']      = settings['momentum']      = m
                    p.extra_generation_params['Sonar momentum_hist'] = settings['momentum_hist'] = mh

                    proc = process_images(p)
                    if info is None: info = proc.info
                    imgs.extend(proc.images)
                if interrupted: break

            if use_grid_search and n_jobs > 1:
                grid = images.image_grid(imgs, rows=len(momentums))
                try:
                    grid = images.draw_grid_annotations(
                        grid, 
                        p.width, 
                        p.height, 
                        ver_texts=[[images.GridAnnotation(f'cur: {round(m, 4)}')]   for m  in momentums], 
                        hor_texts=[[images.GridAnnotation(f'hist: {round(mh, 4)}')] for mh in momentum_hists], 
                        margin=4,
                    )
                except:
                    print_exc()
                    print('>> [Sonar] failed to draw_grid_annotations()')
                    print('vertically top to down are momentums_cur:', momentums)
                    print('horizontally left to right are momentum_hists:', momentum_hists)
                imgs.insert(0, grid)
                images.save_image(grid, p.outpath_grids, "sonar_grid_search", info=info, extension=opts.grid_format, prompt=p.prompt, seed=p.seed, grid=True, p=p)
        finally:
            p.sample = self.sample_saved
            remove_callbacks_for_function(save_image_hijack)
        return Processed(p, imgs, p.seed, info)

    def process(self, p:StableDiffusionProcessing, 
            is_enable:bool,
            use_upscale:bool, sampler:str, 
            momentum:float, momentum_hist:float, momentum_hist_init:str, momentum_sign:str,
            ref_meth:str, ref_hgf:float, ref_min_step:float, ref_max_step:float, ref_img:PILImage,
            upscale_meth:str, upscale_ratio:float, upscale_width:int, upscale_height:int
        ):
        if not is_enable: return

        # type convert
        if ref_img is not None: ref_img = resize_image(1, ref_img, p.width, p.height)

        # save settings to global
        settings['sampler']            = sampler
        settings['momentum']           = momentum
        settings['momentum_hist']      = momentum_hist
        settings['momentum_hist_init'] = momentum_hist_init
        settings['momentum_sign']      = momentum_sign
        settings['ref_meth']           = ref_meth
        settings['ref_min_step']       = int(ref_min_step) if ref_min_step > 1 else round(ref_min_step * p.steps)
        settings['ref_max_step']       = int(ref_max_step) if ref_max_step > 1 else round(ref_max_step * p.steps)
        settings['ref_hgf']            = ref_hgf / 10.0
        settings['ref_img']            = ref_img

        p.extra_generation_params['Sonar sampler']       = sampler
        p.extra_generation_params['Sonar momentum']      = momentum
        p.extra_generation_params['Sonar momentum_hist'] = momentum_hist

        #pp(settings)

        if use_upscale:
            need_upscale, (tgt_w, tgt_h) = get_upscale_resolution(p, upscale_meth, upscale_ratio, upscale_width, upscale_height)
            if need_upscale: print(f'>> [Sonar] upscaling: ({p.width}, {p.height}) => ({tgt_w}, {tgt_h})')

        def save_image_hijack(params:ImageSaveParams):
            if use_upscale and need_upscale:
                params.image = resize_image(1, params.image, tgt_w, tgt_h, upscaler_name=upscale_meth)
        on_before_image_saved(save_image_hijack)
        self.save_image_hijack = save_image_hijack

        self.sample_saved = p.sample
        if   isinstance(p, StableDiffusionProcessingTxt2Img):
            p.sample = lambda *args, **kwargs: StableDiffusionProcessingTxt2Img_sample(p, *args, **kwargs)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            p.sample = lambda *args, **kwargs: StableDiffusionProcessingImg2Img_sample(p, *args, **kwargs)

    def postprocess(self, p:StableDiffusionProcessing, processed:Processed, is_enable:bool, *args):
        if not is_enable: return

        p.sample = self.sample_saved
        remove_callbacks_for_function(self.save_image_hijack)
